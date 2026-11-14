#!/bin/bash
# #462: the saves of the run in $SAVE, for train.sh and score_stops.sh. The
# caller sets SAVE and NAME. A save is $NAME_<k>k.pth, or $NAME_r<N>_<k>k.pth
# after resume N. The trainer writes <save>_optimizer.pth after the weights,
# so a crash can cut either file. A save counts only when both files load.

loads() {  # <save>: the save and its optimizer file both load
  python3 - "$1" <<'PY' >/dev/null 2>&1
import sys
import torch
path = sys.argv[1]
torch.load(path, map_location="cpu", weights_only=False)
torch.load(path[:-4] + "_optimizer.pth", map_location="cpu", weights_only=False)
PY
}

leg_first() {  # stdin: saves. stdout: the same saves, the latest resume first
  sed -E 's/.*_r([0-9]+)_[0-9]+k\.pth$/\1 &/; t; s/^/0 /' | sort -rn | cut -d' ' -f2
}

stop_save() {  # <k>: the save of step k that loads, from the latest leg
  local c
  for c in $(ls "$SAVE"/${NAME}_"$1"k.pth "$SAVE"/${NAME}_r[0-9]*_"$1"k.pth \
               2>/dev/null | leg_first); do
    if loads "$c"; then echo "$c"; return 0; fi
  done
  return 1
}

last_checkpoint() {  # the numbered save of the highest step that loads
  ls "$SAVE"/${NAME}_[0-9]*k.pth "$SAVE"/${NAME}_r[0-9]*_[0-9]*k.pth \
      2>/dev/null |
    sed -E 's/.*_([0-9]+)k\.pth$/\1 &/' | sort -rn | cut -d' ' -f2 |
    while read -r ckpt; do
      if loads "$ckpt"; then echo "$ckpt"; break; fi
    done
}
