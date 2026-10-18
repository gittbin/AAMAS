# Current completion status (2026-10-04)

The official AAMAS 2027 class/style files are installed and unmodified. `main.tex` uses a compact model/protocol in `generated/model_and_protocol.tex`, separate audited v3 communication confirmation and v4 development screening. The fresh ten-block complete study is running at `../outputs_marl/paper_completion`; final evidence is still pending.

`main.pdf` uses the conference class; `paper_progress.pdf` is the single-column reading preview. The current official-format draft has five content pages plus references. The final exporter has been validated on isolated smoke data and will replace pending results after the full trace/provenance audit. Smoke results were not written into the real paper.

The submission ID remains `XXXX`. Do not treat a successful build as submission readiness. The original abstract and pre-compaction manuscript are retained; any OpenReview metadata changes must respect the registered submission. Reproducibility records include the AI-assistance disclosure in `../docs/AI_ASSISTANCE_DISCLOSURE.md`.

Live protocol/status and commands: `../outputs_marl/paper_completion/PROGRESS.md`.

---

Earlier manuscript notes follow.

# AAMAS manuscript rewrite

Overleaf quick start: select `main.tex` as Main document and use pdfLaTeX. When `aamas.cls` is absent, the reading-preview format is selected automatically. No class file is needed for this preview.

Full working manuscript; all empirical result cells and dependent conclusions are intentionally blank. The package contains:

- `main.tex`: manuscript targeting the official `aamas` class; add official AAMAS 2027 template files and metadata before submission.
- `preview.tex`, `paper_draft.pdf`, `build_preview.sh`: locally compilable reading preview using the standard article class. This PDF is **not submission-ready formatting**.
- `references.bib`: bibliography with primary-source URLs/DOIs.
- `abstract_original.txt`: abstract extracted verbatim from the supplied LaTeX source. Compare with the registered abstract; registration was not accessed or changed.
- `OUTLINE_AND_RUN_PLAN.md`: Vietnamese outline, experiment priorities and result-fill guidance.
- `figures/`: two system/architecture drawings in vector PDF, editable SVG and raster PNG.
- `draw_figures.py`: source for the figures; requires Python, reportlab and DejaVu Sans fonts.
- `results_template.csv`: blank schema for future evaluation summaries; no numerical result supplied.

## Build locally

Run `./build_preview.sh` with pdflatex and bibtex available. It compiles the reading preview and writes `paper_draft.pdf`. The current source preserves the title but intentionally replaces empirical abstract/conclusion statements with placeholders in the working draft. It does not authorize or perform changes to registered metadata.

## Submission integration

Use the official AAMAS 2027 author package. Transfer the body, bibliography and figures into that package, or compile `main.tex` after supplying its official class and metadata. Replace the draft bibliography style with the official template style. Remove draft-status text and all fill instructions only after inserting measured results. Confirm formatting, anonymity, final feature dimensions and physical evaluator agreement. Proposed observation masks/message controls may change dimensions from the inherited design.

## Scientific scope

The SINR specifies one explicit EVM power convention; validate or adapt it against the actual simulator before training. The low-level precoder and power-profile decoder use current estimated CSI; high-level actors use stale local estimates. Fixed semantic payloads do not imply learned message encoding. Router priority/budget scheduling is an infrastructure assumption. The online optimizer has additional candidate-model access, without true-channel access. Topology enumeration is conditional on the trained policy.

Numerical entries in the setup (e.g. 16 antennas, 64 RIS elements, 400k transitions) are inherited design parameters, not new measured results. Blank configuration entries require resolution before experiments; blank result entries require measurement.
