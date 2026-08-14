#!/usr/bin/env python3
"""#373 — how many rollout steps each GIFT-Eval config needs at eval.

Training at `k = 3` teaches the model to read its own output. The eval reads
its own output too, and how far it goes is not the same on every config. This
script counts it.

The eval runs strategy B4. B4 calls `src.forecasting_head.rollout_latent`
once per config with

    n_future_tokens = ceil(prediction_length / W)

and that function takes ONE autoregressive step per token, so the count of
rollout steps a config needs is that ceiling. `W` is the backbone patch
width, 16: `eval_local.sh` passes no `--window` override, so the eval keeps
`BACKBONE_CONFIG["W"]`.

`prediction_length` comes from the GIFT-Eval library itself, per config, so
no horizon here is inferred from a name. The library needs the benchmark
data on disk (`GIFT_EVAL`). Where it is missing this script keeps the
committed CSV and says so, the same way the checkpoint-bound figures do.

The domain of each config comes from this study's own eval CSVs.

Usage: rollout_count.py [--results DIR] [--out results/rollout_count.csv]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
GIFT_SCRIPTS = REPO / "experiments" / "2026-04-13_gift-eval" / "scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(GIFT_SCRIPTS))
import r2_ladder as L                                      # noqa: E402


def domains(res: Path):
    """`{config: domain}` off any finished 97-config eval of this study."""
    for d in sorted((res / "eval").glob("*/all_results.csv")):
        with d.open() as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("domain")]
        if len(rows) == 97:
            return {r["dataset"]: r["domain"] for r in rows}
    return {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=L.RESULTS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = Path(a.results)
    out = Path(a.out) if a.out else res / "rollout_count.csv"

    dom = domains(res)
    if not dom:
        print("no finished 97-config eval — no rollout count")
        return 0

    os.environ.setdefault("GIFT_EVAL",
                          str(Path.home() / "workspaces" / "gift-eval-data"))
    try:
        import eval_gift_eval_official as E
        from gift_eval.data import Dataset as GiftDataset
    except Exception as exc:                                # pragma: no cover
        print(f"no GIFT-Eval library ({exc}) — keeping the committed "
              f"{out.name}")
        return 0

    W = E.BACKBONE_CONFIG["W"]
    rows, missing = [], 0
    for ds_name, term in E.get_all_dataset_configs():
        config, _key = E.get_ds_config_name(ds_name, term)
        try:
            pl = GiftDataset(name=ds_name, term=term,
                             to_univariate=False).prediction_length
        except Exception:
            missing += 1
            continue
        rows.append((config, dom.get(config, ""), term, pl,
                     math.ceil(pl / W)))

    if missing or len(rows) != 97:
        print(f"only {len(rows)} of 97 configs resolved ({missing} without "
              f"data on disk) — keeping the committed {out.name}")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "domain", "term", "prediction_length",
                    "rollout_steps"])
        w.writerows(sorted(rows))
    steps = [r[4] for r in rows]
    print(f"wrote {out}  (97 configs, W = {W}, rollout steps "
          f"{min(steps)}..{max(steps)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
