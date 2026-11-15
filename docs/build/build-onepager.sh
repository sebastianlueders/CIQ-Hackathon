#!/bin/bash
# One-pager PDF in a plain LaTeX style (no color, Latin Modern, fully bordered table).
# Usage: bash docs/build/build-onepager.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # -> docs/
tmp=$(mktemp -d)
sed '/^---$/d' one-pager.md | pandoc -f markdown-smart -t latex -s --shift-heading-level-by=-1 --columns=60 \
  -V papersize=letter -V fontsize=10pt -V geometry:"top=0.45in,bottom=0.45in,left=0.6in,right=0.6in" \
  -V colorlinks=false -H build/onepager-plain.tex -o "$tmp/one-pager.tex"
python3 build/grid_tables.py "$tmp/one-pager.tex"
(cd "$tmp" && xelatex -interaction=nonstopmode -halt-on-error one-pager.tex >/dev/null) || { tail -30 "$tmp/one-pager.log"; exit 1; }
cp "$tmp/one-pager.pdf" one-pager.pdf
echo "built docs/one-pager.pdf"
