# Audit Findings

## FND-0001
- Symptom: Required input bundle `files.zip` and the listed source files were not present in the repository or filesystem search scope at audit time.
- Root cause: The integration inputs are missing from the working tree and accessible filesystem locations.
- Exact file+line anchors: N/A (input files not found; search paths: repository root and `/workspace`).
- Fix strategy: Provide `files.zip` in the repository root or a reachable filesystem path, then re-run the integration steps.
- Acceptance criteria: `files.zip` is present and readable, and the listed source files are available for import into `src/bnsyn/**`.
