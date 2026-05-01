---
name: NeurIPS26Paper directory write permission
description: User explicitly authorized free read/edit access to /home/wangqw/NeurIPS26Paper/, with one carve-out — never touch .bib files in that directory.
type: feedback
originSessionId: 2593d9fb-88bf-4bdb-9677-2c7041f59fad
---
User granted explicit cross-project write permission for the paper-writing
directory at `/home/wangqw/NeurIPS26Paper/`.

**Why:** The user runs the `Pi3` model project here and an adjacent
`NeurIPS26Paper` LaTeX/notes project there. Reference docs that link the
two (architecture summaries, eval CSVs, ablation tables, etc.) need to live
in the paper directory so the other Claude session can read them when
writing. The default global rule blocks cross-project writes, so this is a
named exception.

**How to apply:**
- Free to **read, write, edit, and create files** anywhere under
  `/home/wangqw/NeurIPS26Paper/` (including subdirectories like
  `NuerIPS/`, `uav_ablation/`, `uav_grd_ablation/`).
- **Carve-out: never touch any `*.bib` file in that directory.** Bibliography
  files are managed by the user; modifying them risks breaking citation keys
  or losing manually-curated entries. If a citation needs to be added,
  surface the request to the user instead of editing the bib directly.
- All other paths outside the current project (anywhere not under
  `/home/wangqw/video_program/Pi3` or `/home/wangqw/NeurIPS26Paper`) remain
  off-limits per the global "files stay in project directory" rule.
