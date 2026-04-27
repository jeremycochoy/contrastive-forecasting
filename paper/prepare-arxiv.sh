#!/usr/bin/env bash
# Build a clean arXiv submission tarball for the paper.
#
# What it does:
#   1. Stages every file referenced by main.tex into ./arxiv_build/
#      (main.tex, sections/, figures/, neurips_2024.sty, references.bib).
#   2. Runs latexmk to generate main.bbl, then drops references.bib so
#      arXiv uses the precompiled bibliography (per arXiv submission FAQ).
#   3. Appends the arXiv "4 passes" \typeout hint after \end{document} so
#      cross-references resolve on arXiv's build farm.
#   4. Removes intermediate LaTeX artifacts and the rendered PDF.
#   5. Re-extracts the staged tree into a fresh temporary directory and
#      runs pdflatex three times to confirm the package compiles cleanly
#      from scratch (catches missing figures, stale .aux references, etc.).
#   6. Packages the staged tree as ./arxiv_submission.tar.gz.
#
# Output: paper/arxiv_submission.tar.gz, ready to upload to arXiv.
#
# Usage: ./prepare-arxiv.sh

set -euo pipefail

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PAPER_DIR}/arxiv_build"
TARBALL="${PAPER_DIR}/arxiv_submission.tar.gz"
MAIN=main

log() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

command -v latexmk  >/dev/null || die "latexmk not found in PATH"
command -v pdflatex >/dev/null || die "pdflatex not found in PATH"
command -v python3  >/dev/null || die "python3 not found in PATH"
command -v tar      >/dev/null || die "tar not found in PATH"

log "Cleaning previous build outputs"
rm -rf "$BUILD_DIR" "$TARBALL"

log "Staging files into $BUILD_DIR"
mkdir -p "$BUILD_DIR/sections" "$BUILD_DIR/figures"
cp "$PAPER_DIR/$MAIN.tex"        "$BUILD_DIR/"
cp "$PAPER_DIR/neurips_2024.sty" "$BUILD_DIR/"
cp "$PAPER_DIR/references.bib"   "$BUILD_DIR/"
cp "$PAPER_DIR/sections/"*.tex   "$BUILD_DIR/sections/"
cp "$PAPER_DIR/figures/"*.png    "$BUILD_DIR/figures/"

log "Compiling once with latexmk to generate $MAIN.bbl"
(
  cd "$BUILD_DIR"
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$MAIN.tex" \
    >latexmk.log 2>&1 || { tail -40 latexmk.log; die "latexmk failed"; }
)
[ -s "$BUILD_DIR/$MAIN.bbl" ] || die "$MAIN.bbl was not generated"

log "Dropping references.bib (arXiv uses the precompiled .bbl)"
rm -f "$BUILD_DIR/references.bib"

log "Appending arXiv 4-pass typeout hint"
python3 - "$BUILD_DIR/$MAIN.tex" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
src = path.read_text()
hint = r"\typeout{get arXiv to do 4 passes: Label(s) may have changed. Rerun}"
marker = r"\end{document}"
if hint in src:
    sys.exit(0)
idx = src.rfind(marker)
if idx < 0:
    sys.exit(r"could not find \end{document} in main.tex")
end = idx + len(marker)
path.write_text(src[:end] + "\n" + hint + src[end:])
PY

log "Removing LaTeX intermediates and rendered PDF"
(
  cd "$BUILD_DIR"
  rm -f ./*.aux ./*.log ./*.out ./*.blg ./*.fls ./*.fdb_latexmk ./*.pdf \
        ./sections/*.aux latexmk.log
)

log "Final staged tree"
( cd "$BUILD_DIR" && find . -type f | LC_ALL=C sort )

log "Verifying the staged tree compiles in a clean temp dir"
TMP="$(mktemp -d -t arxiv-verify.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
cp -R "$BUILD_DIR/." "$TMP/"
(
  cd "$TMP"
  for pass in 1 2 3; do
    pdflatex -interaction=nonstopmode -halt-on-error "$MAIN.tex" \
      >"pass$pass.log" 2>&1 \
      || { tail -40 "pass$pass.log"; die "pdflatex pass $pass failed"; }
  done
)
[ -s "$TMP/$MAIN.pdf" ] || die "verification PDF not produced"
pdf_kb=$(( $(wc -c <"$TMP/$MAIN.pdf") / 1024 ))
log "Verification PDF built: ${pdf_kb} KB"
[ "$pdf_kb" -gt 500 ] || die "verification PDF is suspiciously small (${pdf_kb} KB) — figures missing?"

log "Building $TARBALL"
( cd "$BUILD_DIR" && tar -czf "$TARBALL" . )

log "Done"
ls -lh "$TARBALL"
echo
echo "Contents:"
tar -tzf "$TARBALL" | LC_ALL=C sort
