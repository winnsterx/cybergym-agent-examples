import logging
import os
import re
import shlex
import shutil
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import docker
import docker.errors
import tomli_w
from dotenv import load_dotenv
from simple_parsing import ArgumentGenerationMode, ArgumentParser, flag

from cybergym.task.gen_task import generate_task
from cybergym.task.types import TaskConfig, TaskDifficulty
from cybergym.utils import save_json

# Load .env file if it exists
load_dotenv()

ENVS = ["DOCKER_HOST"]
API_KEY_ENVS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
OPENAI_PREFIXES = ["gpt-", "o3", "o4"]
ANTHROPIC_PREFIXES = ["claude-"]


SCRIPT_DIR = Path(__file__).parent.absolute()

# Setup logger
logger = logging.getLogger(__name__)


class OpenHandsError(Exception):
    """Base class for OpenHands errors"""

    pass


class OpenHandsTimeoutError(OpenHandsError):
    """Exception raised when OpenHands times out"""

    pass


class OpenHandsValidationError(OpenHandsError):
    """Exception raised when OpenHands validation fails"""

    pass


class OpenHandsDockerError(OpenHandsError):
    """Exception raised when OpenHands Docker container fails to start"""

    pass


@dataclass
class LLMArgs:
    model: str
    """Model to use for generation"""

    api_key: str | None = None
    """API key for the model. If None, get from environment."""

    base_url: str = ""
    """Base URL for the model. If None, use the default URL."""

    native_tool_calling: bool | None = None
    """If None, use the default value. If True, use native tool calling."""

    top_p: float = 1.0
    """Top-p sampling value. Default is 1.0."""

    temperature: float = 0.0
    """Temperature value for sampling. Default is 0.0."""

    max_output_tokens: int = 64000
    """Maximum number of output tokens. Default is 64000."""

    seed: int | None = None
    """Random seed for llm. If None, do not set the seed."""


@dataclass
class OpenhandsArgs:
    log_dir: Path
    """Directory to save the logs"""

    tmp_dir: Path
    """Directory to save the temporary files"""

    llm: LLMArgs
    """LLM arguments"""

    max_iter: int = 500
    """Maximum number of iterations to run the agent"""

    repo: Path = SCRIPT_DIR / "openhands-repo"
    """Path to the repo"""

    silent: bool = False
    """If true, suppresses the output of the OpenHands agent"""

    remove_tmp: bool = True
    """If true, remove the tmp directory after running the agent"""

    timeout: int = 2700
    """Timeout for the OpenHands agent in seconds. Default is 45 minutes."""

    debug: bool = flag(default=False)
    """If true, enable debug mode for the OpenHands agent"""


@dataclass
class TaskArgs:
    task_id: str = ""
    """ID of the task to generate"""

    data_dir: Path = Path(".")
    """Directory containing the data files"""

    server: str = ""
    """Server address for the task"""

    difficulty: TaskDifficulty = TaskDifficulty.level1
    """Difficulty level of the task"""

    evaluation_mode: str = "exploit"
    """Evaluation mode: exploit (PoC), reverse_engineering (RE pseudocode), or judge (evaluate RE submission)"""

    rubric: str = "five-point"
    """Rubric to use for RE evaluation: five-point, granular"""

    run_dir: Path | None = None
    """Path to run directory for judge mode (e.g., cybergym_eval_6/arvo_3938/logs/run_0/arvo_3938-xxx)"""

    db_path: Path | None = None
    """Path to database file for judge mode (default: ./poc.db or ./server_poc/poc.db)"""


def validate_output(log_dir: Path):
    """
    Validate that required output files exist.

    Args:
        log_dir: Agent or judge directory (e.g., runs/task/run_0/agent/)

    Returns:
        True if validation passes, False otherwise
    """
    traj_dir = log_dir / "trajectory"
    if not traj_dir.exists():
        logger.warning(f"Trajectory directory not found: {traj_dir}")
        return False
    return True


def model_map(model: str):
    if model.endswith("/thinking"):
        model = model[: -len("/thinking")]

    if model.startswith("claude-"):
        return f"anthropic/{model}"
    elif len(model.split("/")) >= 2:
        return model
    return f"openai/{model}"


def get_api_key(model: str):
    if any(model.startswith(prefix) for prefix in OPENAI_PREFIXES):
        env_var = "OPENAI_API_KEY"
    elif any(model.startswith(prefix) for prefix in ANTHROPIC_PREFIXES):
        env_var = "ANTHROPIC_API_KEY"
    else:
        env_var = "LLM_API_KEY"
    api_key = os.getenv(env_var)
    if api_key is None:
        api_key = "EMPTY"
    return api_key


def get_prompt_file(model: str, evaluation_mode: str = "exploit"):
    if evaluation_mode == "reverse_engineering":
        return "prompt.reverse.md"
    elif evaluation_mode == "judge":
        return "prompt.judge.md"
    elif evaluation_mode == "ctf":
        return "prompt.ctf.md"
    # if "o4-mini" in model or "o3-" in model:
    #     return "prompt.o4-mini.md"
    return "prompt.exploit.md"


def support_native_tool_calling(model: str):
    if "o4-mini" in model:
        return False
    return None


def _cleanup_docker_container(log_dir: Path):
    # Extract container ID from logs
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        logger.warning(f"Log files not found in: {log_dir}")
        return
    # "runtime d1a7102c-cf4e-46df-9483-dbbeb753585d-588e94af345e82b0"
    pat = re.compile(r"runtime ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-[0-9a-f]{16})")
    with open(log_files[0]) as f:
        for line in f:
            match = pat.search(line)
            if match:
                container_id = match.group(1)
                break
        else:
            logger.warning(f"Container ID not found in: {log_files[0]}")
            return

    container_name = f"openhands-runtime-{container_id}"

    try:
        # Step 1: Try to stop container
        logger.debug(f"Attempting to stop container {container_id}...")
        subprocess.run(  # noqa: S603
            ["docker", "stop", "--time=5", container_name],
            timeout=15,
            capture_output=True,
            check=False,
        )

        # Step 2: Try to remove container
        logger.debug(f"Attempting to remove container {container_id}...")
        result = subprocess.run(  # noqa: S603
            ["docker", "rm", "-f", container_name],
            timeout=15,
            capture_output=True,
            check=False,
        )

        if result.returncode == 0:
            logger.info(f"Removed container {container_id}")
            return
        else:
            logger.debug(f"docker rm failed, attempting force kill...")

    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.debug("docker command not available, attempting force kill...")

    # Fallback: Force kill the container process
    try:
        # Get container PID
        result = subprocess.run(  # noqa: S603
            ["docker", "inspect", "--format={{.State.Pid}}", container_name],
            timeout=5,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            pid = result.stdout.strip()
            if pid and pid != "0":
                logger.debug(f"Force killing container process (PID: {pid})")
                subprocess.run(  # noqa: S603
                    ["sudo", "kill", "-9", pid],
                    timeout=5,
                    capture_output=True,
                    check=False,
                )
                # Give process time to die
                time.sleep(0.5)

        # Remove container with sudo
        logger.debug(f"Removing container with sudo...")
        subprocess.run(  # noqa: S603
            ["sudo", "docker", "rm", "-f", container_name],
            timeout=15,
            capture_output=True,
            check=False,
        )
        logger.info(f"Removed container {container_id} with force kill")

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Failed to remove container {container_id}: {e}")


def run_openhands(
    config_path: Path,
    prompt_path: Path,
    log_dir: Path,
    max_iter: int,
    timeout: int,
    model: str,
    llm_api_key: str | None = None,
    repo: Path = SCRIPT_DIR / "openhands-repo",
    silent: bool = False,
    debug: bool = False,
    enable_thinking: bool = False,
):
    poetry_path = Path(shutil.which("poetry")).absolute()
    if not poetry_path.exists():
        raise Exception(f"[*] Poetry not found at {poetry_path}")
    cmd = [
        str(poetry_path), "run", "python",
        "-m", "openhands.core.main",
        "--config-file", str(config_path),
        "--file", str(prompt_path),
        "--max-iterations", str(max_iter),
    ]  # fmt: skip

    # Set up environment variables
    env = {}
    for env_var in ENVS:
        if os.getenv(env_var) is not None:
            env[env_var] = os.getenv(env_var)

    env["LLM_API_KEY"] = llm_api_key or get_api_key(model)
    env["LOG_TO_FILE"] = "1"
    env["LOG_DIR"] = str(log_dir)
    if debug:
        env["DEBUG"] = "1"
    env["LOG_ALL_EVENTS"] = "1"
    env["DEBUG_RUNTIME"] = "1"
    if enable_thinking:
        logger.info(f"enable thinking for the model {model}")
        env["CYBERGYM_ENABLE_THINKING"] = "1"
    if model.startswith("vertex_ai/"):
        env["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        env["VERTEXAI_LOCATION"] = os.getenv("VERTEXAI_LOCATION")

    # Run the command and stream the output
    logger.info(f"Running OpenHands with command: {shlex.join(cmd)}")
    try:
        subprocess.run(  # noqa: S603
            cmd,
            cwd=repo,
            env=env,
            stdout=subprocess.DEVNULL if silent else None,
            stderr=subprocess.DEVNULL if silent else None,
            timeout=timeout,  # Timeout set to 300 seconds (5 minutes)
        )
    except subprocess.TimeoutExpired:
        # TODO: should we retry on timeout?
        logger.error("OpenHands process timed out.")
        raise OpenHandsTimeoutError("OpenHands process timed out.") from None
    except Exception as e:
        logger.error(f"Error running OpenHands: {e}")
    finally:
        _cleanup_docker_container(log_dir=log_dir)


def trigger_judge_evaluation(task_args: TaskArgs, agent_id: str, log_dir: Path) -> bool:
    """
    Trigger judge evaluation for reverse engineering submissions.

    Runs judge runner as subprocess with 10 minute timeout. Synchronous - waits
    for completion. Saves judge output to logs. Non-fatal on failure.

    Args:
        task_args: Task configuration (task_id, data_dir, etc)
        agent_id: Agent ID for logging context
        log_dir: Log directory for saving judge output

    Returns:
        True if judge completed successfully (returncode 0), False otherwise
    """
    # Use database path from server (matches server startup configuration)
    db_path = Path.cwd() / "server_poc" / "poc.db"

    # Fall back to default poc.db if server_poc doesn't exist
    if not db_path.parent.exists():
        db_path = Path.cwd() / "poc.db"

    # Use the judge runner script in the same directory
    judge_script = SCRIPT_DIR / "judge.py"

    cmd = [
        "uv", "run", str(judge_script),
        "--db", str(db_path),
        "--data-dir", str(task_args.data_dir),
        "--task", task_args.task_id,
        "--model", "claude-sonnet-4-5-20250929"
    ]

    logger.info(f"Triggering judge evaluation: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            timeout=1800,  # 30 minutes timeout
            check=False,  # Don't raise on non-zero exit
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Save judge output to log file
        judge_log = log_dir / "judge.log"
        try:
            with open(judge_log, "w") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n=== STDERR ===\n")
                    f.write(result.stderr)
            logger.info(f"Judge output saved to {judge_log}")
        except Exception as e:
            logger.warning(f"Failed to save judge log: {e}")

        if result.returncode == 0:
            logger.info(f"Judge evaluation completed successfully for {task_args.task_id}")
            return True
        else:
            logger.warning(
                f"Judge evaluation failed (non-fatal): returncode={result.returncode}, "
                f"stderr: {result.stderr[:200] if result.stderr else 'none'}"
            )
            return False

    except subprocess.TimeoutExpired:
        logger.warning(f"Judge evaluation timed out after 600s (non-fatal)")
        return False
    except Exception as e:
        logger.warning(f"Judge evaluation error: {e} (non-fatal)")
        return False


def extract_judge_info_from_run_dir(run_dir: Path, db_path: Path | None = None) -> tuple[str, str, str, Path]:
    """
    Extract task_id, agent_id, pseudocode, and tarball_path from a run directory.

    Args:
        run_dir: Path to run directory (e.g., cybergym_eval_6/arvo_3938/logs/run_0/arvo_3938-xxx)
        db_path: Path to database file (optional, will try to find it)

    Returns:
        Tuple of (task_id, agent_id, pseudocode, tarball_path)

    Raises:
        FileNotFoundError: If required files not found
        ValueError: If data cannot be extracted
    """
    import json

    run_dir = run_dir.absolute()

    # 1. Read args.json to get task_id and data_dir
    args_file = run_dir / "args.json"
    if not args_file.exists():
        raise FileNotFoundError(f"args.json not found in {run_dir}")

    with open(args_file) as f:
        args_data = json.load(f)

    task_id = args_data["task_args"]["task_id"]
    data_dir = Path(args_data["task_args"]["data_dir"])

    # 2. Extract agent_id from directory name (format: taskid_underscore-agentid)
    dir_name = run_dir.name
    if "-" not in dir_name:
        raise ValueError(f"Cannot extract agent_id from directory name: {dir_name}")

    agent_id = dir_name.rsplit("-", 1)[1]
    logger.info(f"Extracted from run_dir: task_id={task_id}, agent_id={agent_id}")

    # 3. Find database if not provided
    if db_path is None:
        # Try common locations
        for candidate in [Path.cwd() / "server_poc" / "poc.db", Path.cwd() / "poc.db"]:
            if candidate.exists():
                db_path = candidate
                break
        if db_path is None:
            raise FileNotFoundError("Database not found. Please provide --db_path")

    # 4. Query database for pseudocode
    from sqlalchemy.orm import Session
    from cybergym.server.pocdb import RESubmission, init_engine

    engine = init_engine(db_path)
    with Session(engine) as session:
        submission = (
            session.query(RESubmission)
            .filter(
                RESubmission.task_id == task_id,
                RESubmission.agent_id == agent_id,
            )
            .first()
        )

        if not submission:
            raise ValueError(f"No submission found for task_id={task_id}, agent_id={agent_id}")

        pseudocode = submission.pseudocode

    logger.info(f"Retrieved pseudocode from database ({len(pseudocode)} chars)")

    # 5. Construct tarball path
    project, task_num = task_id.split(":")
    tarball_path = data_dir / project / task_num / "repo-vul.tar.gz"

    if not tarball_path.exists():
        raise FileNotFoundError(f"Tarball not found: {tarball_path}")

    logger.info(f"Found tarball at {tarball_path}")

    return task_id, agent_id, pseudocode, tarball_path


def prepare_judge_workspace(workspace_dir: Path, task_id: str, tarball_path: Path, pseudocode: str, rubric: str = "five-point"):
    """Prepare workspace for judge mode with tarball and pseudocode."""
    import json

    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Copy tarball to workspace
    dest_tarball = workspace_dir / "repo-vul.tar.gz"
    shutil.copy2(tarball_path, dest_tarball)
    dest_tarball.chmod(0o644)

    # Write pseudocode
    pseudocode_path = workspace_dir / "pseudocode.txt"
    with open(pseudocode_path, "w") as f:
        f.write(pseudocode)

    # Create metadata
    metadata = {
        "task_id": task_id,
        "pseudocode_file": "pseudocode.txt",
        "tarball": "repo-vul.tar.gz",
    }
    with open(workspace_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Copy ghidra manual and rubric for judge
    from cybergym.task.arvo_task import SCRIPT_DIR as TASK_SCRIPT_DIR
    from cybergym.task.types import RUBRICS
    ghidra_manual = TASK_SCRIPT_DIR / "ghidra_manual.md"
    if ghidra_manual.exists():
        shutil.copy(ghidra_manual, workspace_dir / "ghidra_manual.md")
    rubric_file = RUBRICS.get(rubric, RUBRICS["five-point"])[0]
    rubric_path = TASK_SCRIPT_DIR / rubric_file
    if rubric_path.exists():
        shutil.copy(rubric_path, workspace_dir / "rubric.md")

    logger.info(f"Judge workspace prepared at {workspace_dir}")


def run_with_configs(openhands_args: OpenhandsArgs, task_args: TaskArgs, judge_pseudocode: str = None, judge_tarball: Path = None, eval_paths=None, run_number: int = 0, judge_number: int = 0):
    openhands_args.log_dir.mkdir(parents=True, exist_ok=True)
    openhands_args.log_dir = openhands_args.log_dir.absolute()

    enable_thinking = openhands_args.llm.model.endswith("/thinking")

    agent_id = uuid4().hex

    # Use eval_paths if provided (new structure), otherwise fall back to old behavior
    if eval_paths:
        # New structure: tmp goes to eval_paths.tmp_base
        tmp_input_dir = eval_paths.tmp_run_dir(task_args.task_id, run_number, agent_id)
        tmp_input_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Legacy behavior for backward compatibility
        if openhands_args.tmp_dir is None:
            raise ValueError("tmp_dir must be set when eval_paths is not provided")
        openhands_args.tmp_dir.mkdir(parents=True, exist_ok=True)
        openhands_args.tmp_dir = openhands_args.tmp_dir.absolute()
        sub_dir = task_args.task_id.replace(":", "_") + "-" + agent_id
        tmp_input_dir = openhands_args.tmp_dir / sub_dir
        tmp_input_dir.mkdir()

    # 1. prepare the challenge inputs

    # 1.1. copy the challenge template to the input directory
    shutil.copytree(
        SCRIPT_DIR / "template",
        tmp_input_dir / "template",
    )

    # 1.2. generate the task or prepare judge workspace
    task_dir = tmp_input_dir / "workspace"
    task_dir.mkdir(parents=True, exist_ok=True)

    if task_args.evaluation_mode == "judge":
        # Judge mode: prepare workspace with pseudocode and tarball
        if not judge_pseudocode or not judge_tarball:
            raise ValueError("judge_pseudocode and judge_tarball are required for judge mode")
        prepare_judge_workspace(task_dir, task_args.task_id, judge_tarball, judge_pseudocode, task_args.rubric)
        task = {"task_id": task_args.task_id, "mode": "judge"}
    else:
        # Normal mode: generate task
        task_config = TaskConfig(
            task_id=task_args.task_id,
            out_dir=task_dir,
            data_dir=task_args.data_dir,
            server=task_args.server,
            difficulty=task_args.difficulty,
            agent_id=agent_id,
            evaluation_mode=task_args.evaluation_mode,
            rubric=task_args.rubric,
        )
        task = generate_task(task_config)

    # 2. prepare the log directory
    # In new structure, openhands_args.log_dir is already the agent or judge directory
    # (e.g., runs/task/run_0/agent/ or runs/task/run_0/judge/)
    log_dir = openhands_args.log_dir
    logger.info(f"Using log directory: {log_dir}")

    # 2.1. save the task info (metadata.json in new structure, args.json for compatibility)
    metadata = {
        "agent": f"openhands:{openhands_args.llm.model}",
        "agent_id": agent_id,
        "task": task,
        "agent_args": openhands_args,
        "task_args": task_args,
    }

    # Save as metadata.json (new structure)
    save_json(metadata, log_dir / "metadata.json", indent=2)
    logger.info(f"Saving metadata to: {log_dir / 'metadata.json'}")

    # 3. prepare the config file
    config_path = tmp_input_dir / "template" / "config.toml"
    with open(config_path) as f:
        config = tomllib.loads(f.read())

    # Create subdirectories in new structure
    # Note: workspace_base should point to task_dir where files are generated
    # The results will be in log_dir subdirectories
    cache_dir = log_dir / "cache"
    file_dir = log_dir / "file"
    trajectory_dir = log_dir / "trajectory"

    # workspace_base is where OpenHands operates (the tmp workspace with task files)
    # Use absolute paths to ensure Docker can find them
    # CRITICAL: workspace_mount_path MUST be set for Docker runtime to mount the directory
    workspace_base_path = str(task_dir.resolve())
    config["core"]["workspace_base"] = workspace_base_path
    config["core"]["workspace_mount_path"] = workspace_base_path  # Required for Docker mount!
    config["core"]["workspace_mount_path_in_sandbox"] = "/workspace"  # Container path
    config["core"]["cache_dir"] = str(cache_dir.resolve())
    config["core"]["file_store_path"] = str(file_dir.resolve())
    config["core"]["save_trajectory_path"] = str(trajectory_dir.resolve())
    config["core"]["runtime"] = os.getenv("OPENHANDS_RUNTIME", "docker")
    config["llm"]["model"] = model_map(openhands_args.llm.model)
    config["llm"]["top_p"] = openhands_args.llm.top_p
    config["llm"]["temperature"] = openhands_args.llm.temperature
    config["llm"]["base_url"] = openhands_args.llm.base_url
    config["llm"]["max_output_tokens"] = openhands_args.llm.max_output_tokens

    native_tool_calling = openhands_args.llm.native_tool_calling
    if native_tool_calling is not None:
        config["llm"]["native_tool_calling"] = native_tool_calling

    if openhands_args.llm.seed is not None:
        config["llm"]["seed"] = openhands_args.llm.seed

    # Set longer timeout for long-running commands (e.g., Ghidra) inside the container
    if "sandbox" not in config:
        config["sandbox"] = {}
    config["sandbox"]["runtime_startup_env_vars"] = {"NO_CHANGE_TIMEOUT_SECONDS": "60"}

    with open(config_path, "w") as f:
        f.write(tomli_w.dumps(config))

    # 4. run the openhands agent
    prompt_file = get_prompt_file(openhands_args.llm.model, task_args.evaluation_mode)
    logs_dir = log_dir / "logs"
    run_openhands(
        config_path=config_path,
        prompt_path=tmp_input_dir / "template" / prompt_file,
        log_dir=logs_dir,
        timeout=openhands_args.timeout,
        repo=openhands_args.repo,
        silent=openhands_args.silent,
        max_iter=openhands_args.max_iter,
        model=openhands_args.llm.model,
        llm_api_key=openhands_args.llm.api_key,
        debug=openhands_args.debug,
        enable_thinking=enable_thinking,
    )

    # 5. Trigger judge evaluation if RE mode (synchronous, non-fatal)
    # Skip if we're already in judge mode or if called from run_eval (which handles judges separately)
    if task_args.evaluation_mode == "reverse_engineering" and eval_paths is None:
        try:
            judge_ok = trigger_judge_evaluation(task_args, agent_id, logs_dir)
            if judge_ok:
                logger.info(f"Judge evaluation completed successfully for {task_args.task_id}")
            else:
                logger.warning(f"Judge evaluation failed (non-fatal), agent still completed")
        except Exception as e:
            logger.warning(f"Error triggering judge: {e} (non-fatal)")

    # 5.5. Copy evaluation.json from workspace to log_dir if in judge mode
    if task_args.evaluation_mode == "judge":
        evaluation_dst = log_dir / "evaluation.json"
        # Check both paths: Modal runtime creates nested workspace dir (/workspace/workspace/)
        evaluation_candidates = [
            task_dir / "outputs" / "evaluation.json",  # New: dedicated outputs directory
            task_dir / "evaluation.json",
            task_dir / "workspace" / "evaluation.json",  # Modal nested workspace (fallback)
        ]
        evaluation_src = None
        for candidate in evaluation_candidates:
            if candidate.exists():
                evaluation_src = candidate
                break

        if evaluation_src:
            shutil.copy2(evaluation_src, evaluation_dst)
            logger.info(f"Copied evaluation.json from {evaluation_src} to {evaluation_dst}")
        else:
            logger.warning(f"evaluation.json not found in workspace at {evaluation_candidates}")

    # 6. remove the tmp directory or save to debug if keep_tmp
    if openhands_args.remove_tmp:
        shutil.rmtree(tmp_input_dir, ignore_errors=True)
        logger.info(f"Removed temporary input directory: {tmp_input_dir}")
    elif eval_paths and eval_paths.keep_tmp:
        # Copy to debug directory
        debug_dir = log_dir / ".debug"
        debug_dir.mkdir(exist_ok=True)
        shutil.copytree(tmp_input_dir, debug_dir / "initial_state", dirs_exist_ok=True)
        logger.info(f"Saved tmp files to: {debug_dir}")

    # 7. validate the output
    is_valid = validate_output(log_dir)

    return agent_id if is_valid else None


def main(raw_args=None):
    parser = ArgumentParser(argument_generation_mode=ArgumentGenerationMode.BOTH)
    parser.add_arguments(OpenhandsArgs, dest="openhands_args")
    parser.add_arguments(TaskArgs, dest="task_args")

    args = parser.parse_args(raw_args)

    # Handle judge mode from run_dir
    if args.task_args.evaluation_mode == "judge" and args.task_args.run_dir:
        logger.info(f"Judge mode: extracting info from run directory: {args.task_args.run_dir}")

        try:
            task_id, agent_id, pseudocode, tarball_path = extract_judge_info_from_run_dir(
                args.task_args.run_dir,
                args.task_args.db_path
            )

            # Override task_id from run_dir
            args.task_args.task_id = task_id

            # Call run_with_configs with extracted pseudocode and tarball
            run_with_configs(
                args.openhands_args,
                args.task_args,
                judge_pseudocode=pseudocode,
                judge_tarball=tarball_path
            )
        except Exception as e:
            logger.error(f"Failed to extract judge info from run_dir: {e}")
            raise

    else:
        # Normal mode (exploit, RE) or programmatic judge mode
        run_with_configs(args.openhands_args, args.task_args)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    main()
