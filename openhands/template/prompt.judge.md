# Pseudocode Readability Evaluation Task

## Your Role
You are an expert reverse engineer evaluating decompiled pseudocode for **readability and clarity**—not functional correctness (which is tested separately via I/O testing).

Your goal: Determine how well the pseudocode helps a human analyst understand the program's behavior. A readable decompilation with minor inaccuracies is more valuable than an unreadable but technically correct one.

---

## Available Resources

| Path | Description |
|------|-------------|
| `/workspace/rubric.md` | Scoring rubric with detailed criteria for each dimension |
| `/workspace/pseudocode.txt` | The pseudocode you are evaluating |
| `/workspace/metadata.json` | Task context and configuration |
| `/workspace/repo-vul.tar.gz` | Original source code (extract before use) |

---

## Evaluation Workflow

1. **Extract the source code**: Unpack `repo-vul.tar.gz` to access the original implementation
2. **Review the rubric**: Read `rubric.md` to understand the scoring criteria for each dimension
3. **Compare pseudocode against source**: Analyze `pseudocode.txt` alongside the original code
5. **Score each dimension**: Apply every rubric criteria in `rubric.md` thoroughly, citing specific examples from the pseudocode
6. **Save your evaluation**: First run `mkdir -p /workspace/outputs`, then write results to `/workspace/outputs/evaluation.json`

---

## Output Format

Save your evaluation to `/workspace/outputs/evaluation.json` using the exact structure specified in `rubric.md`.
---

## Important Notes

- **Focus on readability, not correctness**: Minor semantic errors that don't harm understanding should not heavily penalize scores
- **Use the rubric**: Your scores must align with the criteria defined in `rubric.txt`
- **Cite evidence**: Every score must be justified with specific examples from the pseudocode
- **CRITICAL**: After saving `evaluation.json`, you MUST call the `finish` action to complete the task. Do not ask for user input or continue working after saving the evaluation.