#!/bin/bash
# Slides PDF: minimal white Beamer (16:9). Usage: bash docs/build/build-slides.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # -> docs/
pandoc slides.md -t beamer -o slides.pdf --pdf-engine=xelatex --pdf-engine-opt=-interaction=nonstopmode \
  --slide-level=1 -V classoption=t -V aspectratio=169 -V fontsize=12pt -V colorlinks=false -H build/slides-plain.tex
echo "built docs/slides.pdf"
