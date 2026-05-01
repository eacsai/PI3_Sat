---
name: no tmp directory for files
description: Never save files to /tmp — always save within the project directory
type: feedback
originSessionId: 8d83b3ca-9845-4245-adf5-af42796924d2
---
All files (eval outputs, intermediate results, scripts, etc.) must be saved within the project directory, not /tmp.

**Why:** User prefers all artifacts to be co-located with the project for easy access and persistence.
**How to apply:** When saving eval outputs, logs, or any temporary files, use `outputs/` or relevant subdirectories within the project instead of /tmp.
