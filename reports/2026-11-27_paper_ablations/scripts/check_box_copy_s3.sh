#!/bin/bash
# The S3 copies of #482 from end to end (s3_box_copy.py), on the S3 of the
# runs, with a local folder as the box (fake_box/): curl runs here with the
# presigned URLs of the tool. The run name is r2_box_copy_check: the check
# writes small torch files under
# s3://model.artifacts/manual/cf_runs/cf-482/r2_box_copy_check/ckpt/ and
# deletes them at its exit. It stops with exit code 1 at the first check
# that fails.
#   1. copy: the saves at 5k and 10k on the box. S3 gets the 10k backbone
#      and its optimizer file.
#   2. copy: a save at 25k. S3 gets the 25k pair. The 10k optimizer file
#      leaves S3, and the 10k backbone stays.
#   3. copy: a save at 50k whose files are not whole. S3 does not get them.
#   4. seed: the 25k pair from S3 on a new box, with the bytes of the box.
#   5. seed again: it refuses a box with a .pth file.
#   6. no output holds a presigned URL.
#   bash check_box_copy_s3.sh <work folder>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$(mkdir -p "${1:?the work folder}" && cd "$1" && pwd)"
RUN=r2_box_copy_check
PREFIX="manual/cf_runs/cf-482/$RUN/ckpt"
PY="${PY:-/home/jupyter/rnd/.venv/bin/python3}"  # boto3
export REAL_RSYNC="$(command -v rsync)" PATH="$HERE/fake_box:$PATH" FAKE_LOG="$W/fake.log"
CKPT="$W/box/$RUN/ckpt"
OUT="$W/out.log"
ok() { echo "ok: $*"; }
no() { echo "FAILED: $*"; exit 1; }
tool() { "$PY" "$HERE/s3_box_copy.py" "$@" root@box 22 >> "$OUT" 2>&1; }

s3_names() {  # the file names under the prefix, on one line
  "$PY" - "$PREFIX" <<'PY'
import sys, boto3
pages = boto3.client("s3").get_paginator("list_objects_v2").paginate(Bucket="model.artifacts", Prefix=sys.argv[1] + "/")
print(" ".join(sorted(o["Key"].rsplit("/", 1)[1] for p in pages for o in p.get("Contents", []))))
PY
}

clean_s3() {  # delete the objects of the check prefix, and only these
  "$PY" - "$PREFIX" <<'PY'
import sys, boto3
prefix = sys.argv[1]
assert prefix == "manual/cf_runs/cf-482/r2_box_copy_check/ckpt", prefix
client = boto3.client("s3")
for page in client.get_paginator("list_objects_v2").paginate(Bucket="model.artifacts", Prefix=prefix + "/"):
    for item in page.get("Contents", []):
        client.delete_object(Bucket="model.artifacts", Key=item["Key"])
PY
}

save() {  # the files of the save at $1 thousand steps on the box
  python3 -c 'import sys, torch; [torch.save({"step": int(sys.argv[1])}, p) for p in sys.argv[2:]]' \
    "$1" "$CKPT/${RUN}_$1k.pth" "$CKPT/${RUN}_$1k_optimizer.pth"
}

copy() { tool copy "$RUN" --stops "10 25 50" --box-h "$W/box" --record "$W/record.tsv"; }

trap clean_s3 EXIT
clean_s3
mkdir -p "$CKPT"
save 5 && save 10 && copy
[ "$(s3_names)" = "${RUN}_10k.pth ${RUN}_10k_optimizer.pth" ] || no "step 1: S3 holds $(s3_names)"
ok "step 1: the 10k pair in S3"
save 25 && copy
[ "$(s3_names)" = "${RUN}_10k.pth ${RUN}_25k.pth ${RUN}_25k_optimizer.pth" ] || no "step 2: S3 holds $(s3_names)"
ok "step 2: the 25k pair in S3, the 10k optimizer file deleted, the 10k backbone kept"
head -c 100 "$CKPT/${RUN}_25k.pth" > "$CKPT/${RUN}_50k.pth"; cp "$CKPT/${RUN}_50k.pth" "$CKPT/${RUN}_50k_optimizer.pth"
copy
[ "$(s3_names)" = "${RUN}_10k.pth ${RUN}_25k.pth ${RUN}_25k_optimizer.pth" ] || no "step 3: S3 holds $(s3_names)"
ok "step 3: a save that is not whole stays out of S3"
mkdir -p "$W/box2" && echo "run $RUN code check" > "$W/box2/setup_done_$RUN"
tool seed "$RUN" --box-h "$W/box2" || no "step 4: the seed failed"
for f in "${RUN}_25k.pth" "${RUN}_25k_optimizer.pth"; do
  cmp -s "$CKPT/$f" "$W/box2/$RUN/ckpt/$f" || no "step 4: $f differs on the new box"
done
[ "$(LC_ALL=C ls "$W/box2/$RUN/ckpt")" = "$(printf '%s\n' "${RUN}_25k.pth" "${RUN}_25k_optimizer.pth")" ] \
  || no "step 4: the new box holds $(ls "$W/box2/$RUN/ckpt")"
ok "step 4: the 25k pair on the new box, the same bytes"
tool seed "$RUN" --box-h "$W/box2" && no "step 5: a second seed ended with code 0"
ok "step 5: no seed on a box that holds a .pth file"
! grep -q "X-Amz-Signature" "$OUT" "$W/fake.log" || no "step 6: an output holds a presigned URL"
ok "step 6: no output holds a presigned URL"
cat "$OUT"
clean_s3
[ -z "$(s3_names)" ] || no "the check files stay in S3: $(s3_names)"
ok "S3 holds no file of $RUN at the end"
