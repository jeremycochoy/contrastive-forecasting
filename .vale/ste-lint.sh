#!/usr/bin/env bash
# Check markdown and Python comments against the ASD-STE100 writing rules
# with Vale.
#
#   ste-lint.sh [options] [path ...]
#
# With no path, the script checks the default set for a repository:
#   * README.md at any level
#   * the report of an experiment, in the shape that the agents make:
#     reports/<YYYY-MM-DD>_<name>/<name>.md
#   * two older report shapes, kept only for the files that came before
#     that convention: <dir>/<dir>.md, and report.md in an experiment
#   * every markdown file in docs/
#   * every Python file. Vale reads only the comments and the docstrings.
#
# Options:
#   --level LEVEL   minimum level that fails: suggestion, warning, or error
#                   (default: error)
#   --all           report warnings and suggestions, but do not fail on them
#   --changed REF   check only the default-set files that changed since REF
#   --json          machine-readable output
#   --list          print the files that would be checked, then stop
set -euo pipefail

STYLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEVEL=error
FORMAT=line
LIST_ONLY=0
CHANGED_REF=
PATHS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --level) LEVEL="$2"; shift 2 ;;
    --all) LEVEL=suggestion; shift ;;
    --changed) CHANGED_REF="$2"; shift 2 ;;
    --json) FORMAT=JSON; shift ;;
    --list) LIST_ONLY=1; shift ;;
    -h|--help) sed -n '2,19p' "${BASH_SOURCE[0]}" | cut -c3-; exit 0 ;;
    --) shift; PATHS+=("$@"); break ;;
    *) PATHS+=("$1"); shift ;;
  esac
done

if ! command -v vale >/dev/null 2>&1; then
  echo "ste-lint: vale is not installed." >&2
  echo "  macOS:  brew install vale" >&2
  echo "  other:  https://vale.sh/docs/vale-cli/installation/" >&2
  exit 127
fi

tracked_paths() {
  git ls-files "$1" 2>/dev/null ||
    find . -name "$1" -not -path './.git/*' | sed 's|^\./||'
}

DATE_GLOB='[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'

# The convention that the agents use for a new experiment:
#   reports/<YYYY-MM-DD>_<name>/<name>.md
# See reports/REPORT_STANDARD.md and agents/claude_run_experiment.sh.
is_report() {
  local dir
  case "$1" in reports/${DATE_GLOB}_*/*.md) ;; *) return 1 ;; esac
  dir=$(basename "$(dirname "$1")")
  [ "$(basename "$1" .md)" = "${dir#*_}" ]
}

# Older experiments, made before the convention. Do not use these shapes for
# a new report. They stay here only to keep the earlier files in the set.
is_legacy_report() {
  local dir
  dir=$(dirname "$1")
  [ "$(basename "$1" .md)" = "$(basename "$dir")" ] && return 0
  case "$1" in reports/${DATE_GLOB}_*/report.md) return 0 ;; esac
  return 1
}

# Default set: README.md at any level, docs/*.md, the report files, and *.py
select_defaults() {
  tracked_paths '*.md' | while IFS= read -r f; do
    dir=$(dirname "$f")
    if [ "$(basename "$f")" = "README.md" ] ||
       [ "$dir" = "docs" ] || [ "${dir}" != "${dir#docs/}" ] ||
       is_report "$f" || is_legacy_report "$f"; then
      printf '%s\n' "$f"
    fi
  done
  tracked_paths '*.py'
}

# Keep only the files that changed since a reference, when one is given.
filter_changed() {
  if [ -z "$CHANGED_REF" ]; then
    cat
    return
  fi
  changed=$(git diff --name-only --diff-filter=d "$CHANGED_REF"...HEAD 2>/dev/null || true)
  while IFS= read -r f; do
    printf '%s\n' "$changed" | grep -qxF "$f" && printf '%s\n' "$f"
  done
}

# bash 3.2 on macOS has no mapfile, so read the list line by line
if [ ${#PATHS[@]} -eq 0 ]; then
  while IFS= read -r f; do
    [ -n "$f" ] && PATHS+=("$f")
  done < <(select_defaults | filter_changed)
fi

if [ ${#PATHS[@]} -eq 0 ]; then
  echo "ste-lint: no markdown files to check."
  exit 0
fi

if [ "$LIST_ONLY" -eq 1 ]; then
  printf '%s\n' "${PATHS[@]}"
  exit 0
fi

exec vale --config "$STYLE_DIR/.vale.ini" \
          --minAlertLevel "$LEVEL" \
          --output "$FORMAT" \
          "${PATHS[@]}"
