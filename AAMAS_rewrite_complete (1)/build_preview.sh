#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error preview.tex
bibtex preview
pdflatex -interaction=nonstopmode -halt-on-error preview.tex
pdflatex -interaction=nonstopmode -halt-on-error preview.tex
mv preview.pdf paper_draft.pdf
