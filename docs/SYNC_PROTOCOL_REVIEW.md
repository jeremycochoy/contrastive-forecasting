# Sync protocol review — design plan

Audit-only doc. No code changed. Plan for a follow-up implementation pass.

Scope: every `sync_<name>/sync_loop.sh` under the repo root,
`experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh`, and the
`Remote Machine Monitoring` runbook in `CLAUDE.md`. Reference incident:
the May 1–2 2026 learnable-τ CSV truncation, in which steps 0–17100 of
`tiny_realonly_4096_smaller_learnable_tau_losses.csv` were destroyed by
the local sync_loop after the operator re-launched on a fresh vast.ai
instance without pushing the existing CSV back.

---

## 1. Post-mortem — May 1–2 learnable-τ CSV truncation

### 1.1 Local fingerprint (verified 2026-05-02)

```
sync_realonly_4096_smaller_learnable_tau/learnable/checkpoints/
  tiny_realonly_4096_smaller_learnable_tau_losses.csv          1,113,937 B   8401 lines
  tiny_realonly_4096_smaller_learnable_tau_losses.csv.prev       636,811 B   4801 lines
```

- Both files have header + data starting at `step,17101,…`.
- The current CSV ends at step 25500.
- The `.prev` file ends at step 21900.
- The on-remote training (resumed after the credit-out crash) loaded a
  `_17k.pth` checkpoint, so step 17100 is exactly the resume boundary.
  Steps 0–17100 from the original instance are therefore not on the new
  remote and not in either local copy.

### 1.2 Sync-log timeline (`sync_realonly_4096_smaller_learnable_tau/learnable/sync.log`)

```
[Ven  1 mai 2026 19:12:30 CEST]  cycle  #1  instance=35970908   csv 167,738 B
[Ven  1 mai 2026 19:31:44 CEST]  cycle  #2                       csv 533,863 B
[Ven  1 mai 2026 19:55:09 CEST]  cycle  #3                       csv 888,267 B
[Ven  1 mai 2026 20:17:10 CEST]  cycle  #4                       csv 1,242,655 B
[Ven  1 mai 2026 20:40:20 CEST]  cycle  #5                       csv 1,613,850 B
[Ven  1 mai 2026 21:03:44 CEST]  cycle  #6                       csv 1,986,121 B   <-- last good pull
[Ven  1 mai 2026 21:27:16 CEST]  cycle  #7    (csv pull missing — instance dying)
[Ven  1 mai 2026 21:49:58 CEST]  cycle  #8    UNREACHABLE
                                  ... credit ran out, 35970908 destroyed ...
[Ven  1 mai 2026 23:12:34 CEST]  loop restart  instance=35980909 (fresh)
                                cycle  #1                       csv 13,392 B   <-- DISASTER
[Ven  1 mai 2026 23:29:03 CEST]  cycle  #2                       csv 517,457 B
[Ven  1 mai 2026 23:57:37 CEST]  cycle  #3                       csv 610,270 B
[Sam  2 mai 2026 00:17:58 CEST]  cycle  #4                       csv 636,811 B  <-- now local .prev
[Sam  2 mai 2026 00:34:29 CEST]  cycle  #5                       csv 1,113,937 B <-- now local .csv
[Sam  2 mai 2026 01:01:25 CEST]  cycle  #6                       (current cycle)
```

### 1.3 What actually destroyed the data

The relevant bash in `sync_realonly_4096_smaller_learnable_tau/sync_loop.sh`
lines 39–44:

```bash
if [ "${sz}" -ge "${min_size}" ]; then
    if [ -s "${local_dest}" ]; then
        mv -f "${local_dest}" "${local_dest}.prev"
    fi
    mv "${tmp}" "${local_dest}"
    ...
```

The CSV's `min_size` is `CSV_MIN=1`. Any non-empty pull is accepted. Step
by step:

1. **Cycle #6 (21:03)**: local `*.csv` = 1,986,121 B (steps 0–~16,000),
   local `.prev` = whatever cycle #5 promoted (slightly smaller).
2. **Instance dies, credit refilled, fresh instance 35980909 provisioned.**
3. **The operator pushed only the `_17k.pth` + `_optimizer.pth` to the
   new remote.** The CSV was not in the resume bundle. Training restarts
   with `--resume` at step 17101 and a brand-new (header-only) CSV is
   created on the new instance. Within seconds it has 1–2 hundred new
   rows.
4. **23:12 cycle #1 on new instance**: `atomic_scp` pulls the brand-new
   13,392-byte CSV. It's "non-empty", so:
   - existing 1,986,121-byte `*.csv` (with steps 0–~16k) is rotated to
     `*.csv.prev`,
   - the previous `.prev` is overwritten silently — gone.
   - the new 13,392-byte CSV becomes the canonical local copy.
5. **23:29 cycle #2**: pulls 517,457 B (steps 17101–~22000 from new
   instance). The 13,392-byte file is rotated to `.prev`, overwriting
   the 1,986,121-byte safety copy. The pre-resume history is now in zero
   places.
6. Subsequent cycles are all post-resume; `.prev` and `.csv` keep growing
   along the same suffix-only history.

The bug pattern is generic: **any time the remote's row count goes
backwards across a sync, the local sync is guaranteed to corrupt itself.**
A single-deep `.prev` cannot recover because the sync_loop ticks faster
than human notice (≤15 min between rotations).

### 1.4 Why operator process didn't catch it

- The `min_size` floor for CSVs is intentionally `1`, because a healthy
  fresh CSV genuinely is ~80 bytes (one header line). There is no
  fingerprint by which sync_loop currently distinguishes "remote CSV
  shrunk because of fresh instance" from "remote CSV shrunk because the
  trainer just rotated it" (the trainer in fact never rotates CSVs — it
  always opens `mode="a"` — but sync_loop has no way to know that).
- The "remote training process: ALIVE" message is per-cycle and was
  green throughout, so the runtime monitor did not flag anything. The
  CSV-shrink event itself produces a `✓ losses.csv (… bytes)` line, not
  a warning.
- `learnable_sync.driver.log` is a copy of the same content so it
  reproduced the silent rotation.

### 1.5 Other sync dirs to check before the fix lands

Each dir below ran across the May 1–2 instance churn. Before patching,
spot-check `losses.csv` step ranges versus `.pth` checkpoint step
suffixes for a similar truncation:

- `sync_realonly_4096_smaller_tau_sweep/{tau005,tau007,tau020}/checkpoints/`
- `sync_realonly_4096_smaller/{revin,ewma128,ewma_span32,ewma_span256}/checkpoints/`
- `sync_compositesynth_v5envboost/{revin,ewma128}/checkpoints/`

`sync_realonly_4096_smaller_tau_sweep/tau005/checkpoints/`: the current
CSV is 30,001 lines (steps 1–30,000); `.prev` is also 30,001 lines.
That's healthy — same arm, but the tau005 instance was not rebilled.
Confirm the same for the others.

---

## 2. Failure-mode catalog

The categories below are exhaustive for the current `atomic_scp`-based
loop (i.e. all sync_loops *except* `sync_v3b/sync_loop.sh`, which still
uses raw `rsync` and is treated separately in §2.7).

### 2.1 Append-style file regression (the May 1–2 case)

**Files**: any append-only file the trainer emits — `*_losses.csv`,
`run_*.log`, eval `summary.txt`.

**Trigger**: re-launch on a fresh remote without pushing the existing
local copy back. Remote starts a new short file. Sync_loop pulls it.
Local rotates the long good copy to `.prev`. A second cycle overwrites
`.prev` with another short pull. Data lost permanently.

**Frequency**: once per resume-without-bundle. Was unrecoverable in
≤30 min of sync_loop ticking on May 1.

**Closes by**: layer 1 (push the bundle) **and** layer 2 (length-guard
on append files).

### 2.2 Numbered periodic snapshot disappearing on the remote

**Files**: `${BB}_${N}k.pth` and matching `_optimizer.pth`.

**Trigger**: a fresh remote instance has not yet hit step `N`, so
`/workspace/app/checkpoints/${BB}_5k.pth` does not exist. The local
copy from the old instance does (e.g. `_5k.pth` from steps before death).

The `list_remote` discovery loop only adds files; it never deletes:

```bash
for rp in $(list_remote "${REMOTE_CKPT_DIR}/${BB}_*k.pth"); do
    fname=$(basename "${rp}")
    if [ ! -f "${LOCAL_CKPT}/${fname}" ]; then
        atomic_scp ...
    fi
done
```

So the local `_5k.pth` survives untouched. **This is benign for
periodic saves**: the local one is an old checkpoint and the new
remote will eventually produce its own `_5k.pth` once it crosses that
step (same filename — and then `[ ! -f … ]` is false so nothing
re-syncs from the new instance, leaving a stale-but-valid local file).

**Real risk**: **`${BB}_5k.pth` from the old instance is at step 5000
of the OLD training trajectory; the new instance will eventually
overwrite step 5000 of the NEW trajectory, but the local copy will
silently stay as the old trajectory's 5k snapshot.** Two different
files share the same name. This was likely fine for the May 1–2 incident
because resume happened at step 17000 on the new remote, well past 5k,
so the old `_5k.pth` is just a useful fossil — but if a fresh-instance
restart happens to roll back past a periodic save (e.g. resume from
step 2k instead of 17k), the local snapshot list silently mixes old- and
new-trajectory checkpoints under identical names.

**Closes by**: layer 3 (archive snapshot on instance change) plus
filename hygiene (suggest stamping `_<instance_id>_5k.pth` going
forward — non-blocking, see §6).

### 2.3 `_best_*.pth` regression silently overwriting a better checkpoint

**Files**: `${BB}_best_loss.pth`, `${BB}_best_gap.pth` and optimizers.

**Trigger**: remote re-launch. The trainer's `--resume` does load
prior `best_loss`/`best_gap` from the optimizer state-dict (per
`train.py:load_training_state`, see lines 286–306) but the *file*
`${BB}_best_loss.pth` is not loaded — it is rewritten from scratch the
moment a new step beats the in-memory `best_loss`. Until that happens,
the file may not even exist on the new remote.

`atomic_scp` pulls the new (worse, but still present) `best_loss.pth`,
rotates the better local one to `.prev`, then on the next cycle
overwrites `.prev` with another worse pull. Same death-by-rotation as
2.1.

**Mitigated by current `min_size`**: the `BB_MIN=40e6` floor catches
truly-empty files but a worse-but-same-size new file will get through.

**Closes by**: layer 3 (archive on instance change) — best-checkpoints
get snapshotted before the fresh instance starts trampling them.
A weight-content fingerprint (e.g. compare embedded `step` from the
optimizer file) is overkill; archive-on-rotation is sufficient.

### 2.4 `run_*.log` truncation

**Files**: `run_${ARM}.log`, `run_learnable_tau.log`, etc.

**Trigger**: same as 2.1 — fresh instance writes its own log file from
empty. The trainer's launcher script (`run_resume_after_pstats_died.sh`
line 11: `exec >> >(tee -a /workspace/app/run_all.log) 2>&1`) does
**append** when launched on the same instance, but a fresh instance has
no such file at all and the `tee -a` re-creates it. So the file is
always reset on a brand-new instance.

The current `LOG_MIN=1` means even a 1-byte log passes. Local good log
gets rotated to `.prev`, second cycle overwrites `.prev`.

**Closes by**: layer 2 (length-guard, generalised to file size for
non-CSV append files) and layer 3 (archive).

### 2.5 `summary.txt` overwrite for a partial GIFT-Eval

**Files**: `results/<eval_name>/summary.txt`, `all_results.csv`.

**Trigger**: a post-train eval (`eval_gift_eval_official.py --resume`)
does support skipping done configs, but if it partway-writes a new
`all_results.csv` after fresh-instance launch and sync_loop pulls,
local good `summary.txt`/CSV is rotated to `.prev` and then trampled.
Same generic regression as 2.1 but on results files.

For results, this matters more than for losses — the CSVs are the
inputs to REPORT.md figures.

**Closes by**: layer 2 (treat every append-only data file the same way)
and layer 3.

### 2.6 Concurrent eval running on the same remote while training has stopped

**Trigger**: `train.py` finishes, `eval_gift_eval_official.py` runs.
The trainer's CSV stops growing but eval doesn't touch it. Sync_loop
continues to re-pull the unchanged CSV. Each tick rotates the same
file to `.prev` (overwriting itself), which is wasteful but not a data
loss.

**Closes by**: a `--no-rotate-if-identical` short-circuit (compare
`wc -l`/sha256 — but this is a quality-of-life fix, not a data-safety
one). Treated as non-goal here unless we add it for free.

### 2.7 Raw rsync in `sync_v3b/sync_loop.sh`

This script (lines 18–22) does:

```bash
rsync -avz --timeout=60 -e "ssh ${SSH_OPTS} -p ${PORT}" \
    --include='tiny_v3b_*' --exclude='*' \
    "${REMOTE}:${REMOTE_DIR}" "${LOCAL_DIR}checkpoints/"
```

`rsync` with no `--partial-dir`/`-T` and without `.tmp` rotation means
a connection drop mid-transfer leaves a partial file in place. CLAUDE.md
already calls this out (`Use scp (not rsync on macOS — it's v2.6.9 …)`).

This script is for an old retired training run. It is dormant. **Out of
scope** for the active fix; flag for deletion (or freeze in place) in
§6 open-questions.

### 2.8 Missing `.prev` for `_best_loss_optimizer.pth` rotation

`atomic_scp` only rotates `${LOCAL_CKPT}/${LOCAL_DEST}` to `.prev` — it
does not couple the optimizer to the model. If the model `.pth` rotates
but the matching optimizer fails the size floor (line 47: `✗ TOO SMALL`),
the local `_best_loss.pth` and `_best_loss_optimizer.pth` go out of
sync (model is new, optimizer is `.prev` of old). Looking at the
sync.log trace from §1.2 cycle #1 (`✗ TOO SMALL …_best_loss_optimizer.pth`),
this happened: model was rotated, optimizer stayed unrotated.

**Severity**: low for safety (the trainer can re-run optimizer warmup),
moderate for resume correctness (loading model+optimizer from
inconsistent files = wrong AdamW momentum).

**Closes by**: a "couple" mode in `atomic_scp` — only rotate when both
the model and the optimizer pass; otherwise rollback both. Non-trivial
because `atomic_scp` is currently per-file. Cheaper alternative: emit a
loud `! INCONSISTENT (…model rotated but optimizer failed)` warning so
the operator notices.

### 2.9 Periodic snapshot syncing the optimizer too late

In each sync_loop:

```bash
atomic_scp "${rp}" "${LOCAL_CKPT}/${fname}" "${BB_MIN}" || true
opt_remote="${rp%.pth}_optimizer.pth"
atomic_scp "${opt_remote}" "${LOCAL_CKPT}/${opt_fname}" "${BB_OPT_MIN}" || true
```

The model is fetched first; if the connection drops between the two
calls, the local periodic snapshot is missing the optimizer. On the
next tick, `[ ! -f "${LOCAL_CKPT}/${fname}" ]` is false (the model is
present), so the optimizer is **never re-attempted**. Result: a
forever-orphan model with no optimizer.

Fingerprint: in §1.2 we observe `_27k.pth` (45 MB present locally) but
`_27k_optimizer.pth.tmp` (63 MB partial, in-progress as of 01:18). If
that connection drops, a follow-up tick will not re-pull the optimizer
unless the operator deletes `_27k.pth` first.

**Closes by**: change the existence check to "both files present" or
re-attempt the optimizer regardless. Trivial fix; should be bundled
with the layer-2 patch.

### 2.10 Multiple sync_loops sharing one local dir

Not currently the case — each `sync_<run>/{arm}/checkpoints/` is per
arm. But if two sync_loops are aimed at the same local target (operator
mistake during cut-over), `atomic_scp`'s `mv -f` is not safe under
concurrency. Out of scope unless the new design introduces a shared
archive dir (see §3.3 — we do introduce one, but per-sync-dir, no
cross-loop conflict).

### 2.11 sync_loop killed mid-rotation

Sequence:

```
1. mv -f "${local_dest}" "${local_dest}.prev"     # ok
2. <kill>
3. (next launch) mv "${tmp}" "${local_dest}"       # never happens
```

Result: local `*.csv` does not exist, `*.csv.prev` has the previous
content. On next sync, the existence check `[ -s "${local_dest}" ]`
is false, so no rotation happens, and the new pull becomes the
canonical file. `.prev` survives. **This is fine** — just noting it for
completeness; the existing rotation logic handles its own
mid-process kill.

### 2.12 Operator "I'll just rsync from remote one more time" footgun

CLAUDE.md already calls this out (`NEVER use raw scp to pull a
checkpoint from a remote`) but the rule lives only in human memory.
Nothing prevents an operator from running `scp root@host:/foo /local/foo`
between sync_loop ticks and trampling a good local copy. **Closes by**:
turning the `safe_pull.sh` rule into a hook (suggest in §6 open
questions, not in the layer-1/2/3 plan).

---

## 3. Proposed design — three layers + runbook

### 3.1 Layer 1 — Resume bundle includes ALL state

When re-launching a training run on a fresh instance (after vast.ai
crash, credit-out, etc.), push **all four artifact classes**, not just
the model. The append-friendly ones are critical because the trainer
doesn't reset them — it appends.

Bundle for backbone:

```
checkpoints/
  ${BB}.pth                   (or whichever specific snapshot to resume from)
  ${BB}_optimizer.pth         (matched optimizer for that .pth)
  ${BB}_losses.csv            (full historical CSV — append-only on remote)
run_${ARM}.log                (full historical log — `tee -a` continues appending)
```

Bundle for head (when continuation needs it):

```
checkpoints/
  ${HEAD}.pth
  ${HEAD}_optimizer.pth
  ${HEAD}_losses.csv
```

Mechanism:

- This is an **operator runbook** rule. There is nothing to add to
  `sync_loop.sh` for this layer; it is a launch-time discipline.
- See §3.4 for the CLAUDE.md text.

Optional automation (deferred — see §6): a `vastrun-resume-bundle <run>`
helper that scp's all four classes from local to fresh-remote in one go.

What this layer closes: 2.1, 2.4, 2.5 (the source of all append-file
truncations). It does not close 2.3 (best.pth regression — that's
trajectory-dependent, not bundle-dependent).

### 3.2 Layer 2 — Length-guard on append files inside `atomic_scp`

For files known to be append-only, refuse a pull that goes backwards.

Definition: a file is "append-only" if either:
- its name ends in `_losses.csv`, or
- its name matches `run_*.log` or is the run log dest, or
- its name matches `*all_results.csv` or `*summary.txt` (eval results
  may legitimately *grow* via `--resume`, but never shrink during
  normal completion).

Pseudocode change to `atomic_scp` (current sync_loop body, around
line 39–44):

```bash
# Existing size-floor check (unchanged):
if [ "${sz}" -lt "${min_size}" ]; then
    rm -f "${tmp}"
    echo "  ✗ TOO SMALL …" | tee -a "${LOG_FILE}"
    return 1
fi

# NEW — append-only regression guard:
case "$(basename "${local_dest}")" in
  *_losses.csv|run_*.log|all_results.csv|summary.txt)
    if [ -s "${local_dest}" ]; then
        local local_lines remote_lines
        local_lines=$(wc -l < "${local_dest}" 2>/dev/null || echo 0)
        remote_lines=$(wc -l < "${tmp}" 2>/dev/null || echo 0)
        if [ "${remote_lines}" -lt "${local_lines}" ]; then
            local archive_dir="${LOCAL_BASE}/archive"
            mkdir -p "${archive_dir}"
            local stamp
            stamp=$(date -u +%Y%m%dT%H%M%SZ)
            local archived="${archive_dir}/$(basename "${local_dest}")_${stamp}_remote_smaller_${remote_lines}lines"
            mv "${tmp}" "${archived}"
            echo "  ⚠ APPEND REGRESSION ${local_dest##*/}: remote=${remote_lines}L < local=${local_lines}L; saved to ${archived#${LOCAL_BASE}/}, local untouched" | tee -a "${LOG_FILE}"
            return 2
        fi
    fi
    ;;
esac

# Existing rotation:
if [ -s "${local_dest}" ]; then
    mv -f "${local_dest}" "${local_dest}.prev"
fi
mv "${tmp}" "${local_dest}"
echo "  ✓ $(basename "${remote_path}") (${sz} bytes)" | tee -a "${LOG_FILE}"
return 0
```

Behavior changes:

- **Append-only file shrinks** → `tmp` archived under
  `<sync-dir>/archive/`, local intact, `.prev` intact, loud `⚠` line in
  the log. Operator sees it on the next look.
- **Append-only file grows** → existing rotation as before.
- **Non-append file (`.pth`)** → unchanged behavior. `_best_loss.pth`
  may legitimately rewrite from a smaller checkpoint; `wc -l` doesn't
  apply. Layer 3 covers this.
- The function now has three return codes (`0` ok, `1` fail/too small,
  `2` regression-archived). All call sites use `|| true` so this is
  safe; the regression message itself is the user-visible signal.

Per-arm `LOG_FILE` is the same one already used; nothing new to plumb.

What this layer closes: 2.1 (definitively — the May 1–2 case becomes a
loud warning line plus an archived shorter-CSV instead of permanent
data destruction), 2.4, 2.5.

Edge cases to call out for the implementer:

- **Header-only counts as 1 line.** A fresh CSV with one header is 1
  line; an old CSV mid-training is thousands of lines. So `1 < 8401`
  trips the regression branch — exactly what we want.
- **First-ever sync (no local file)** → the `[ -s "${local_dest}" ]`
  guard skips the regression check, so first cycle works.
- **`wc -l` on a CR-only file** → the trainer writes Unix newlines via
  Python `csv.writer` (line 245 of `train.py`); not a concern.
- **Ordered call sites**: the regression check must run **before** the
  rotation. The sketch above has this order.

### 3.3 Layer 3 — Archive-on-instance-change

Independent of layer 2; covers the case where layer 2 cannot help (e.g.
non-append files like `_best_loss.pth` whose content can legitimately
change without a length signal).

Mechanism: persist the most-recent instance ID in
`<LOCAL_BASE>/.last_instance_id`. At the *start* of each sync_loop run,
if the file exists and the contents differ from `${INSTANCE_ID}`,
snapshot the entire local sync dir into
`<LOCAL_BASE>/archive/<old_instance_id>_<utc_stamp>/` **before any
pulls**. Then write the new instance id.

Pseudocode at the top of the sync_loop (after `mkdir -p` at line 25,
before the first `cycle_count=0` line):

```bash
LAST_ID_FILE="${LOCAL_BASE}/.last_instance_id"
if [ -s "${LAST_ID_FILE}" ]; then
    LAST_ID=$(cat "${LAST_ID_FILE}")
    if [ "${LAST_ID}" != "${INSTANCE_ID}" ]; then
        STAMP=$(date -u +%Y%m%dT%H%M%SZ)
        ARCHIVE="${LOCAL_BASE}/archive/${LAST_ID}_${STAMP}"
        mkdir -p "${ARCHIVE}"
        # cp -a preserves mtimes; -p preserves perms.
        # Don't archive the archive itself.
        for entry in "${LOCAL_BASE}"/*; do
            base=$(basename "${entry}")
            case "${base}" in
                archive|sync.log|.last_instance_id) continue ;;
            esac
            cp -a "${entry}" "${ARCHIVE}/" 2>/dev/null || true
        done
        echo "[$(date)] Instance changed ${LAST_ID} → ${INSTANCE_ID}; archived prior state to ${ARCHIVE}" | tee -a "${LOG_FILE}"
    fi
fi
echo "${INSTANCE_ID}" > "${LAST_ID_FILE}"
```

Properties:

- Archive happens **once per sync_loop start**, not every cycle. If the
  loop is restarted with the same instance ID, no archive (no-op).
- Archive copies, not moves — local state is preserved unchanged. The
  next cycle's `atomic_scp` proceeds as today against intact files.
- Naming includes the **old** instance ID (the one that produced the
  state being archived) plus a UTC timestamp.
- Disk cost: each archive duplicates the sync dir's `.pth` files.
  Largest single dir today (`sync_realonly_4096_smaller_learnable_tau/learnable/checkpoints/`)
  is ~2.7 GB per the listing in §1.1; one archive per instance change is
  acceptable. Operator can prune `archive/` periodically.

What this layer closes: 2.1 (defense-in-depth with layer 2), 2.2, 2.3,
plus the underlying theme of the May 1–2 incident: "the CSV-shrink was
a *symptom* of an instance change; archiving on the change catches all
artifact classes at once".

Edge cases:

- **First-ever launch (no `.last_instance_id` file)** → no archive,
  just write the file.
- **Operator launches sync_loop twice for the same instance** → second
  launch sees the same id, no archive.
- **Operator launches sync_loop with a wrong instance id** (typo) →
  triggers an archive on the typo and one on the corrected launch.
  Annoying but recoverable; both archives sit under
  `archive/<wrong_id>_<stamp>/` and `archive/<typo_id>_<stamp>/`.
- **Crash partway through the cp** → next launch sees the same
  `.last_instance_id` mismatch and re-archives. Idempotent enough; the
  archive dir name has a fresh timestamp so they don't collide.

### 3.4 Layer 0 — Runbook entry in CLAUDE.md

Add this subsection inside the existing `## Remote Machine Monitoring`
block, immediately after the "EVERY remote training run must have a
sync_loop running …" bullet (so it sits next to the other re-launch
context):

```
- **Resume bundle — what to push when re-launching on a fresh instance.**
  When the original vast.ai instance dies and a fresh instance is
  provisioned to continue training, the resume bundle is *all four
  artifact classes*, not just the model:
    1. `<run_name>.pth` — model weights to resume from (typically a
       periodic save like `<run_name>_17k.pth`).
    2. `<run_name>_optimizer.pth` — matched companion (RNG state, AdamW
       moments, step counter).
    3. `<run_name>_losses.csv` — historical per-step loss CSV. The
       trainer's `CSVLogger` opens with `mode="a"` and only writes a
       header if the file is empty, so an existing CSV cleanly extends.
       **Forgetting this silently destroys the pre-resume rows once the
       sync_loop ticks** (the new remote starts fresh, sync_loop pulls
       the short file, atomic-rotates the long good local copy to
       `.prev`, then overwrites `.prev` on the next cycle). Layer-2
       length-guard in `sync_loop` catches this now, but the cleanest
       outcome is to push the CSV and never trip the guard.
    4. `run_<arm>.log` — historical training log (`tee -a` likewise
       appends).
  **Push all four together.** The CSV and log are append-friendly: the
  trainer extends them in place. The `.pth` and `_optimizer.pth` are
  the resume target itself.
  Reference: May 1–2 2026 learnable-τ run lost steps 0–17100 of its
  loss CSV because only the `_17k.pth` and `_optimizer.pth` were pushed
  on resume; see `docs/SYNC_PROTOCOL_REVIEW.md` §1 for the post-mortem.
```

What this layer closes: documented in human memory — prevents the
operator-side error in the first place.

### 3.5 Additional safety nets noticed during the audit

These are smaller wins; can be folded into the same patch or split as
the implementer prefers.

#### 3.5.1 Couple model+optimizer pulls (closes 2.8 / 2.9)

Replace per-file `atomic_scp` for matched pairs with a `atomic_scp_pair`
that pulls both `${X}.pth` and `${X}_optimizer.pth` to `.tmp`s, only
rotates+commits both if both pass thresholds, otherwise rolls both back.
This is a moderate refactor (changes every `for kind in best_loss …`
block in every sync_loop). Recommend deferring unless layer-2/3 make it
trivial.

A cheaper version that closes 2.9 specifically: when the model exists
locally but the optimizer doesn't, force an `atomic_scp` retry of the
optimizer on every cycle. Three-line change in the periodic block:

```bash
for rp in $(list_remote "${REMOTE_CKPT_DIR}/${BB}_*k.pth"); do
    fname=$(basename "${rp}")
    opt_remote="${rp%.pth}_optimizer.pth"
    opt_fname="${fname%.pth}_optimizer.pth"
    if [ ! -f "${LOCAL_CKPT}/${fname}" ]; then
        atomic_scp "${rp}" "${LOCAL_CKPT}/${fname}" "${BB_MIN}" || true
    fi
    # Always re-attempt the optimizer if it's missing (closes the
    # connection-drop-between-two-scp's failure mode).
    if [ ! -f "${LOCAL_CKPT}/${opt_fname}" ]; then
        atomic_scp "${opt_remote}" "${LOCAL_CKPT}/${opt_fname}" "${BB_OPT_MIN}" || true
    fi
done
```

#### 3.5.2 Better signal for the `! INCONSISTENT` case (closes 2.8)

Right now `atomic_scp` returns `0/1` to a `|| true` caller. After a
`best_loss.pth` succeeds and `best_loss_optimizer.pth` returns `1` (too
small), we silently end up with model rotated and optimizer not. Add a
post-block check after each `for kind in best_loss best_gap FINAL`
block:

```bash
for kind in best_loss best_gap FINAL; do
    atomic_scp "${REMOTE_CKPT_DIR}/${BB}_${kind}.pth"           ".../${BB}_${kind}.pth"           "${BB_MIN}"     || true
    atomic_scp "${REMOTE_CKPT_DIR}/${BB}_${kind}_optimizer.pth" ".../${BB}_${kind}_optimizer.pth" "${BB_OPT_MIN}" || true
    if [ -s "${LOCAL_CKPT}/${BB}_${kind}.pth" ] && [ ! -s "${LOCAL_CKPT}/${BB}_${kind}_optimizer.pth" ]; then
        echo "  ! INCONSISTENT ${BB}_${kind}: model present, optimizer missing — backbone resume from this checkpoint will fail" | tee -a "${LOG_FILE}"
    fi
done
```

#### 3.5.3 Hard-stop hash check on the trainer side

Out of scope for this PR but worth flagging: `train.py` could write a
sidecar `${run_name}_state.json` with `{step, csv_sha, csv_lines}` on
every checkpoint. sync_loop would then validate cross-class consistency
("the step in optimizer and the step at the bottom of CSV match"). Hard
to retrofit; tracked in §6.

---

## 4. Per-file delta plan

The 12 sync_loops fall into three classes by template lineage:

| Class | Files | Relationship |
| --- | --- | --- |
| Modern atomic_scp w/ `.prev` rotation | sync_compositesynth/, sync_compositesynth_v2bseasheavy/, sync_compositesynth_v2pulse/, sync_compositesynth_v3primitives/, sync_compositesynth_v4combined/, sync_compositesynth_v5envboost/, sync_realonly_4096/, sync_realonly_4096_smaller/, sync_realonly_4096_smaller_tau_sweep/, sync_realonly_4096_smaller_learnable_tau/ | All clones with only the run-name / dir / log path differing. Identical `atomic_scp` body, identical `list_remote`, identical loop cadence (900s). |
| Old atomic_scp w/o `.prev` rotation | sync_v3b_final/ | Has `atomic_scp` but does not rotate to `.prev` (line 40 is a plain `mv`). |
| Raw rsync, retired | sync_v3b/ | Pre-PR-#45, no atomic semantics. |

The per-file delta below assumes layer 1 is documentation only, layers
2 and 3 are code, layer 0 is doc. Concrete edits:

### 4.1 `sync_compositesynth/sync_loop.sh`

Single-arm variant (no `${ARM}`). Affected lines:

- **After line 25** (`mkdir -p "${LOCAL_CKPT}" "${LOCAL_RESULTS}"`) and
  **before line 27** (`echo "[$(date)] Sync loop start …"`), insert
  the **layer-3** `.last_instance_id` block (verbatim from §3.3
  pseudocode; substitute `LOCAL_BASE` accordingly — same name).

- **In the `atomic_scp` body, replace lines 39–44** (the `if [ "${sz}"
  -ge "${min_size}" ]; then ... mv -f ... mv "${tmp}" ...` block) with
  the layer-2 length-guard version from §3.2. The `case` pattern list
  is `*_losses.csv|run_*.log|all_results.csv|summary.txt`. The
  `archive_dir` is `${LOCAL_BASE}/archive`.

- **No changes** to `BB_NAMES`/`HEAD_NAMES` arrays, completion regex,
  `SLEEP=900`, sigil lines.

Approx LOC delta: +25 (layer 3 prefix) +18 (layer 2 in `atomic_scp`) =
+43 lines, net.

### 4.2 `sync_compositesynth_v2bseasheavy/sync_loop.sh`

Per-arm variant. Same edits as §4.1 with these substitutions:

- Layer 3 block insertion site: after line 25 (mkdir), before the
  `echo "[$(date)] Sync loop start arm=${ARM}…"` line.
- Layer 2 block: same as §4.1; the function body is identical between
  the two scripts.

The same diff applies cleanly to:

- `sync_compositesynth_v2pulse/sync_loop.sh`
- `sync_compositesynth_v3primitives/sync_loop.sh`
- `sync_compositesynth_v4combined/sync_loop.sh`
- `sync_compositesynth_v5envboost/sync_loop.sh`

(All five have line-for-line identical `atomic_scp` and identical
top-of-loop boilerplate.)

### 4.3 `sync_realonly_4096/sync_loop.sh` and the `_smaller*` family

- `sync_realonly_4096/sync_loop.sh`
- `sync_realonly_4096_smaller/sync_loop.sh`
- `sync_realonly_4096_smaller_tau_sweep/sync_loop.sh`
- `sync_realonly_4096_smaller_learnable_tau/sync_loop.sh`

All four have identical `atomic_scp` to the composite family. Same two
edits.

`sync_realonly_4096_smaller_learnable_tau/sync_loop.sh` is the one that
just lost data; **patch this first** as the smoke test (see §5).

### 4.4 `sync_v3b_final/sync_loop.sh`

This script's `atomic_scp` is a **simpler form** without `.prev`
rotation (line 40 is just `mv "${tmp}" "${local_dest}"`). The fix here
is to **first** add the `.prev` rotation, then layer in the length
guard. Edits:

- **Lines 38–46** in the file: replace with the modern atomic_scp body
  (same as the composite family) plus the layer-2 length-guard.
- **Top-of-loop**: add the layer-3 `.last_instance_id` block after
  line 24 (`mkdir -p "${LOCAL_CKPT}" "${LOCAL_RESULTS}"`).
- **Optional and tangential**: this script's cadence ramps from 300s →
  900s for the first 12 cycles. Not a sync-protocol concern; leave
  alone.

Approx LOC delta: +30 for the rotation rewrite + length guard, +25 for
layer 3.

The dir is for an older run; the patch is mechanical but lower priority
than the actively-running ones in §4.3.

### 4.5 `sync_v3b/sync_loop.sh`

**Recommendation: do not patch.** Add a header comment marking it
deprecated, and a `set -e; exit 1` after the `echo` to refuse to run.
Or, more conservatively, leave alone and call it out in the open
questions (§6).

Reasoning: the script uses raw `rsync` to a directory destination with
no atomic semantics, and CLAUDE.md already prohibits its pattern. The
v3b run completed long ago; nothing references this loop in current
work. Reviving it under the new design is a bigger task than the audit's
scope.

### 4.6 `experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh`

`safe_pull.sh` is the user-facing single-pull tool (not the periodic
loop). It already does `.tmp` + size-check + `.prev` rotation
(lines 31–49). **Edits**:

- After the size check (line 44), and **before** rotation (line 46),
  insert a *file-name-aware* length guard mirroring layer 2:

```bash
case "$(basename "$LOCAL")" in
  *_losses.csv|run_*.log|all_results.csv|summary.txt)
    if [[ -s "$LOCAL" ]]; then
        LL=$(wc -l < "$LOCAL")
        RL=$(wc -l < "${LOCAL}.tmp")
        if (( RL < LL )); then
            ARCH_DIR="$(dirname "$LOCAL")/archive"
            mkdir -p "$ARCH_DIR"
            STAMP=$(date -u +%Y%m%dT%H%M%SZ)
            ARCHIVED="${ARCH_DIR}/$(basename "$LOCAL")_${STAMP}_remote_smaller_${RL}lines"
            mv "${LOCAL}.tmp" "$ARCHIVED"
            echo "⚠ ${REMOTE}: append regression remote=${RL}L < local=${LL}L; saved to ${ARCHIVED}, $LOCAL untouched" >&2
            exit 2
        fi
    fi
    ;;
esac
```

`safe_pull.sh` is already exit-coded (1 = too-small/empty), so adding
exit code 2 = regression-archived is consistent.

There are **three identical copies** of `safe_pull.sh`:

- `experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh` (canonical)
- `.claude/worktrees/feat+composite-synth/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh`
- `.claude/worktrees/feat+source-id-freq-plumb/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh`

Patch the canonical one in `experiments/`. The two worktree copies will
sync naturally on next merge / branch refresh; they should not be
patched independently.

### 4.7 `CLAUDE.md`

Insert the **layer-0** runbook bullet from §3.4 into the
`## Remote Machine Monitoring` block, after the "EVERY remote training
run must have a sync_loop running …" bullet (current line 34) and
before the "NEVER use raw scp …" bullet (current line 35).

The exact text to insert is in §3.4 verbatim (the fenced markdown
block).

### 4.8 No edits needed elsewhere

The repo also has:

- `experiments/2026-04-27_freq-embedding/scripts/train.py` — already correct (line
  226: `open(path, "a", newline="")`; line 228: header only when
  `getsize == 0`). No changes; this is the *reason* layer 1 works.
- `experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py` — provides
  `--resume` to skip done configs. Not in audit scope; included for
  reference because layer 2's `summary.txt` / `all_results.csv` guard
  depends on this trainer being well-behaved.

---

## 5. Migration plan

Goal: roll out layers 0–3 without disrupting any in-flight training run
or losing data during the transition.

### 5.1 Order of operations

1. **PR-1 (doc-only, mergeable independently)**: layer 0 — add the
   resume-bundle bullet to `CLAUDE.md`. Trivial, no behavior change. This
   shrinks the cost of a future operator error while the code patches
   are reviewed.

2. **PR-2 (the core fix)**: layer 2 + layer 3 in
   `sync_realonly_4096_smaller_learnable_tau/sync_loop.sh` only. This
   is the directory that just lost data; patching it first lets us
   smoke-test the new behavior on the live run. Plus, layer 3's
   archive-on-instance-change benefits from being live for the next
   instance event, whenever it comes.

3. **PR-2 smoke test (before merge)**:
   - Create a fake-remote dir locally:
     `/tmp/sync_smoketest_remote/checkpoints/${BB}_losses.csv` with say
     50 lines.
   - Point the patched sync_loop at `localhost`-ish via SSH config or a
     mock by running `atomic_scp` standalone (it's a function — extract
     it for the test).
   - Tick once; verify `*.csv` and `.prev` look as today.
   - Truncate the fake-remote CSV to 10 lines (simulates fresh
     instance).
   - Tick once; expect: local `*.csv` untouched, `.prev` untouched,
     `archive/*_remote_smaller_10lines` created, `⚠` line in sync.log.
   - Same drill for `run.log`.
   - Delete `.last_instance_id` on the local side, change `INSTANCE_ID`
     env, restart sync_loop; expect: full snapshot under
     `archive/<old_id>_<stamp>/`.

4. **PR-3 (fan-out)**: apply the same diff to the remaining 9
   composite/realonly clones (§4.1 - §4.3). One PR, mechanical patch.
   No new functionality beyond PR-2.

5. **PR-4 (cleanup, optional)**: layer 2 + layer 3 in
   `sync_v3b_final/sync_loop.sh` (the older variant). Deprioritized
   because that run is finished — only matters if the user revives it.

6. **PR-5 (decisions)**: address §6 open questions. Each is small enough
   to PR independently.

### 5.2 In-flight run policy during rollout

The in-flight runs as of 2026-05-02 are (from the task list and sync
dirs):

- `sync_realonly_4096_smaller_learnable_tau/learnable/` — patched by
  PR-2.
- `sync_realonly_4096_smaller_tau_sweep/{tau005,tau007,tau020}/` —
  patched by PR-3.

Policy: **do not stop and restart any sync_loop while training is live.**
Each PR's deployment goes:

- Land the patch on `experiments` (after review).
- Wait for a natural sync cycle to complete (the loop sleeps 900s,
  so within 15 min).
- `kill` the running sync_loop (operator).
- `bash sync_<dir>/sync_loop.sh <host> <port> <instance> <arm>` to
  restart with the new code. The `.last_instance_id` file (new in
  layer 3) won't exist on first run, so the first restart triggers no
  spurious archive. Any later instance change triggers the archive
  correctly.
- Verify the next cycle's sync.log shows `Instance changed … → … ;
  archived prior state` if and only if the operator passed a different
  instance ID than the previous run. Otherwise, no archive.

If the patch is rolled out *without* killing the old sync_loop, both
old (no length-guard) and new (length-guard) loops would scribble on
the same files — that's the layer 2.10 hazard. So: **kill old, start
new**.

### 5.3 Rollback

- Revert PR-2 in git → operator restarts the old loop. The `archive/`
  dir and `.last_instance_id` file are no-op clutter for the old loop;
  they don't affect its behavior.
- Forensics: anything archived by layer 2 lives under
  `<sync-dir>/archive/` and is recoverable manually.

### 5.4 Acceptance gate

A PR is mergeable when, on a freshly-launched sync_loop:
- One `✓ <file>` line per expected file class appears (per CLAUDE.md
  rule).
- One simulated regression (truncate the fake remote CSV) produces a
  `⚠ APPEND REGRESSION` line and an `archive/...` file. Local CSV is
  unchanged.
- One simulated instance change (different `INSTANCE_ID` arg) produces
  an `archive/<old_id>_<stamp>/` snapshot at sync_loop start.

---

## 6. Open questions / non-goals

Things the user should weigh in on before implementation. None block
the layered design above; all are scoping decisions.

### 6.1 Multi-deep `.prev` versus a single archive dir

The current design keeps `.prev` single-deep and adds `archive/` as a
forensic store. Alternatives the user might prefer:

- Promote `.prev` → `.prev.1` / `.prev.2` / ... up to N deep. Costs N×
  disk per artifact class, plus another mv on every cycle.
- Keep `.prev` single-deep but add a `last10/` ring buffer — cheaper than
  archive-on-change but more cycles touch disk.
- Archive ONLY on instance change (the proposed design) — minimal
  cost, asymmetric protection (depends on operator passing a different
  instance ID).

Recommendation: stay with the proposed single-deep `.prev` + archive
dir. If the user wants belt-and-suspenders, easy to add a 3-deep ring
later.

### 6.2 Per-file-class versus per-arm archive

The proposal puts archive in `<LOCAL_BASE>/archive/`, where
`LOCAL_BASE` is `sync_<run>/<arm>/` for per-arm loops. So per-arm runs
get separate archives. Acceptable (clean separation), or too noisy
(many small archive dirs)? Easy to centralize to
`sync_<run>/archive/<arm>_<stamp>/` if preferred.

### 6.3 Filename hygiene for periodic snapshots (covers 2.2)

A clean fix for "old-instance `_5k.pth` and new-instance `_5k.pth` are
two different files with the same name" is to stamp filenames:
`${BB}_<instance_id>_5k.pth`. That's a *trainer-side* change (in
`train.py`'s `--save-every` flow). Out of scope for the sync_loop
audit, but worth flagging — it would close the residual stale-snapshot
ambiguity that even layer 3 only partially addresses (layer 3 archives;
it doesn't disambiguate post-archive what the live name refers to).

### 6.4 `sync_v3b/sync_loop.sh` deletion

Recommend deleting (or at least marking deprecated) this script. It's
the only `rsync`-based sync_loop, predates the atomic-scp design, and
its run is long finished. Zero risk in deleting; modest risk in leaving
it as future copy-paste source. **Defer to user.**

### 6.5 Hardening operator runbook

CLAUDE.md already calls out "NEVER use raw scp to pull a checkpoint".
Should we promote that to an actual repo-level `pre-commit` check or a
`safe_pull.sh`-only path enforcement (e.g., a `scripts/sync_kit.sh`
wrapper)? Not in scope here; flagging for a separate discussion.

### 6.6 Do we want a centralized `sync_loop.sh` library?

The 10 modern clones differ only in 4–5 string substitutions
(`REMOTE_RESULTS_DIR`, `REMOTE_LOG`, `LOCAL_BASE`, `BB`, `HEAD`).
Refactoring into a single `scripts/sync_loop.sh` with arguments would
make future fixes one-edit instead of ten-edit. **Not done in this
audit** — the immediate priority is correctness, not factoring. But
the layer-2/3 PR is exactly the kind of fix that argues for it
loudly. Suggest as a follow-up after PR-3 lands.

### 6.7 Does the trainer ever rotate its own CSV?

Re-checked `train.py:CSVLogger.__init__` (line 226):
`open(path, "a", newline="")` — pure append, no rotation. So a
"shrinking remote CSV" is unambiguously a fresh-instance event, not a
trainer rotation. The layer-2 guard is therefore safe.

If a future trainer ever does rotate (e.g. for log-bursting reasons),
layer 2 would false-positive. Flag for review at that time.

### 6.8 Eval `summary.txt` legitimately *changes shape*

Worth noting: `summary.txt` for an eval is rewritten (not appended to)
on each `--resume` round. Its line count is roughly stable
(O(num_metrics) lines), not strictly monotonic. The layer-2 guard for
`summary.txt` may be too aggressive — a fresh-instance eval that
re-emits the same summary will trip the guard at *equal* line count
and pass at *greater*. The proposed condition is `remote_lines <
local_lines` (strict `<`), so equality-count writes pass. Should be
fine in practice; tested in §5.4 acceptance.

If false-positives surface, narrow the case-pattern to only
`*_losses.csv|run_*.log` (the strictly append-only names) and let
`summary.txt` / `all_results.csv` use the existing single-deep `.prev`
plus layer-3 archive on instance change.

### 6.9 Does the runbook entry belong in CLAUDE.md or a separate doc?

CLAUDE.md is already 58 lines of dense rules. Adding the resume-bundle
bullet keeps the rule next to its peers (sync, atomic writes, etc.),
but at the cost of further density. Alternative: a
`docs/REMOTE_MONITORING.md` and a one-line `CLAUDE.md` pointer. Default
to inlining (per the original task spec). User to decide.
