#!/bin/bash
# #393 — pool the per-machine ladder/decisions CSVs into one file each.
#
# Usage:  RES=<results-dir> bash scripts/merge_pooled.sh [basename ...]
#
# Split out of collect_results.sh so scripts/test_merge_pooled.sh can run it
# against a fixture directory with no ssh in the way. A pooling bug here is
# silent by construction: the output still parses, still sorts, and simply
# holds fewer rows than the machines produced.
#
# It already bit once. The merge ended in
#
#     sort -u -t, -k1,1 -k4,4n
#
# and `sort -u` deduplicates on the KEY FIELDS, not on the line. The key was
# field 1 (cell) and field 4 (stop), so of the two heads scored at every stop
# exactly one row survived: `ladder_all.csv` carried ten student rows and no
# teacher row at all, while the machines' own ladder.csv files held both. The
# head-to-head is the comparison this card exists to make.
#
# Two changes keep it fixed. What makes a row unique is declared per file, by
# column NAME, and resolved against the header at run time — so inserting a
# column cannot silently move the key again. And rows are deduplicated on the
# whole line, with the key used only for ordering, so a row can never be
# discarded on account of a field the key does not mention. Two rows that
# share a key but disagree elsewhere are both kept and reported, because that
# is a rescore or a collision and either way it needs a human.
set -uo pipefail
RES="${RES:?RES must be the absolute path of the results dir}"

# What makes a row unique, per pooled file, by column name. `:n` sorts that
# column numerically so stops read 40000, 100000, 200000 rather than
# 100000, 200000, 40000.
key_spec(){
  case "$1" in
    ladder)    echo "cell,stop:n,head" ;;   # one row per head per stop
    decisions) echo "cell,stop:n" ;;        # one row per stop
    *)         return 1 ;;
  esac
}

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [merge] $*"; }

# Column index (1-based) of a named field in a CSV header line.
col_index(){  # <header> <name>
  awk -F, -v want="$2" '{for(i=1;i<=NF;i++) if($i==want){print i; exit}}' <<<"$1"
}

merge(){  # <basename>
  local base="$1" out="$RES/${1}_all.csv" hdr="" first=1 spec
  spec=$(key_spec "$base") || { echo "ABORT: no key declared for '$base'" >&2; return 3; }

  : > "$out.tmp"
  for f in "$RES/per_machine/${base}_"*.csv; do
    [ -f "$f" ] || continue
    [ $first -eq 1 ] && { hdr=$(head -n 1 "$f"); first=0; }
    [ "$(head -n 1 "$f")" = "$hdr" ] || {
      echo "ABORT: $f has a different header than the others" >&2
      rm -f "$out.tmp"; return 4; }
    tail -n +2 "$f" >> "$out.tmp"
  done
  [ -n "$hdr" ] || { rm -f "$out.tmp"; say "no ${base} rows yet"; return 0; }

  # Resolve the declared key names against this file's actual header.
  local -a sort_key=() idx_list=() ; local field name numeric idx
  for field in ${spec//,/ }; do
    name="${field%%:*}"; numeric=""
    [ "$field" != "$name" ] && numeric="n"
    idx=$(col_index "$hdr" "$name")
    [ -n "$idx" ] || {
      echo "ABORT: ${base}_all.csv key names column '$name', absent from header: $hdr" >&2
      rm -f "$out.tmp"; return 5; }
    sort_key+=( -k"${idx},${idx}${numeric}" )
    idx_list+=( "$idx" )
  done

  # Deduplicate on the whole line — two machines copying the same row collapse,
  # a row differing in any field survives — then order by the key. -s keeps the
  # line order within a key, so the output is deterministic.
  { printf '%s\n' "$hdr"
    sort -u "$out.tmp" | sort -s -t, "${sort_key[@]}"
  } > "$out"
  rm -f "$out.tmp"

  # A surviving duplicate key means the same stop was scored twice with
  # different numbers. Never silently drop one; name it and keep both.
  local dupes
  dupes=$(tail -n +2 "$out" | cut -d, -f"$(IFS=,; echo "${idx_list[*]}")" \
          | sort | uniq -d)
  [ -n "$dupes" ] && {
    say "WARN ${base}_all.csv: $(wc -l <<<"$dupes") key(s) appear twice with differing rows, both kept:"
    while read -r d; do [ -n "$d" ] && say "  $d"; done <<<"$dupes"
  }

  say "$(basename "$out"): $(( $(wc -l < "$out") - 1 )) rows"
}

bases=( "$@" ); [ ${#bases[@]} -gt 0 ] || bases=( ladder decisions )
rc=0
for base in "${bases[@]}"; do merge "$base" || rc=$?; done
exit $rc
