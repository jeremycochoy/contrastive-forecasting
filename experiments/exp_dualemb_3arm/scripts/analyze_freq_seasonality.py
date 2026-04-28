"""Catalogue every (freq, seasonality) combination present in GIFT-Eval,
and list the ones that would be natural but are absent."""
import os, sys, math, csv
from collections import defaultdict

sys.path.insert(0, '/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting')
sys.path.insert(0, '/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/.claude/worktrees/feat+source-id-freq-plumb/experiments/exp_dualemb_3arm/scripts')

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from gift_eval.data import Dataset as GiftDataset
from gluonts.time_feature import get_seasonality
from plot_worst_configs import _all_configs


def main():
    rows = []
    for ds_name, term in _all_configs():
        try:
            ds = GiftDataset(name=ds_name, term=term, to_univariate=False)
            freq = ds.freq
            season = get_seasonality(freq)
            horizon = ds.prediction_length
            rows.append((ds_name, term, freq, season, horizon))
        except Exception as e:
            rows.append((ds_name, term, "?", -1, -1))

    # Group by (freq, season)
    by_pair = defaultdict(list)
    for ds_name, term, freq, season, h in rows:
        if freq == "?": continue
        by_pair[(freq, season)].append(f"{ds_name}/{term}")

    print("=" * 100)
    print("GIFT-Eval (freq, seasonality) combinations PRESENT")
    print("=" * 100)
    print(f"{'freq':>10}  {'seasonality':>11}  {'count':>5}  examples")
    print("-" * 100)
    for (freq, season), names in sorted(by_pair.items(), key=lambda x: (x[0][0], x[0][1])):
        examples = ", ".join(names[:3]) + (" ..." if len(names) > 3 else "")
        print(f"{freq:>10}  {season:>11}  {len(names):>5}  {examples}")

    # Compute distinct freqs
    freqs_present = sorted(set(f for (f, s) in by_pair))
    seasonalities_present = sorted(set(s for (f, s) in by_pair))

    print()
    print(f"Distinct freqs   ({len(freqs_present):2d}): {freqs_present}")
    print(f"Distinct seas    ({len(seasonalities_present):2d}): {seasonalities_present}")
    print(f"Distinct (f, s)  ({len(by_pair):2d}) combinations covered.")

    # Natural seasonalities the wild has at each freq, beyond gluonts default.
    # The gluonts default usually picks ONE seasonality per freq (e.g. 1H -> 24
    # for daily). Real series at that freq often have multiple natural
    # periodicities:
    #   sub-daily (60-360 minutes), daily, weekly (7d), monthly (~30d),
    #   yearly (365d / 12m / 52w).
    NATURAL = {
        "10S":  [60, 360, 8640],         # 10-min, 1-hour, 1-day
        "5T":   [12, 288, 2016],         # 1-hour, 1-day, 1-week
        "10T":  [6, 144, 1008],          # 1-hour, 1-day, 1-week
        "15T":  [4, 96, 672],            # 1-hour, 1-day, 1-week
        "30T":  [2, 48, 336],            # 1-hour, 1-day, 1-week
        "H":    [24, 168, 8760],         # 1-day, 1-week, 1-year
        "D":    [7, 30, 365],            # 1-week, 1-month, 1-year
        "W-SUN":[4, 13, 26, 52],         # 1-month, 1-quarter, 1-half, 1-year
        "M":    [3, 6, 12],              # 1-quarter, 1-half, 1-year
        "Q-DEC":[4],                     # 1-year
        "A-DEC":[1],                     # nothing meaningful
        "Y-DEC":[1],
    }
    # Also include freq alias variants we see in gift-eval:
    NATURAL.setdefault("60S", NATURAL["1T"] if "1T" in NATURAL else [60])
    NATURAL.setdefault("1T", [60, 1440, 10080])

    print()
    print("=" * 100)
    print("Natural (freq, seasonality) combinations NOT in GIFT-Eval")
    print("=" * 100)
    print(f"{'freq':>10}  {'seasonality':>11}  meaning")
    print("-" * 100)
    missing_count = 0
    for freq in freqs_present:
        # Look up natural seasonalities for this freq (try a few alias forms)
        candidates = NATURAL.get(freq) or NATURAL.get(freq.replace("-SUN",""))
        if candidates is None:
            print(f"{freq:>10}  (no natural list defined)")
            continue
        present_for_freq = sorted(s for (f, s) in by_pair if f == freq)
        for c in candidates:
            if c not in present_for_freq:
                # Decode: for sub-hourly freqs the meaning is "X period samples"
                meaning = describe_period(freq, c)
                print(f"{freq:>10}  {c:>11}  {meaning}")
                missing_count += 1

    # Also flag freqs that don't appear at all but are common in industry
    print()
    print(f"Total missing natural combinations: {missing_count}")

    # Frequency families absent entirely from GIFT-Eval
    common_industry_freqs = ["1S", "30S", "1T", "30T", "2H", "4H", "12H", "M", "Q-DEC"]
    absent = [f for f in common_industry_freqs if f not in freqs_present]
    print(f"Common industry freqs not in GIFT-Eval at all: {absent}")


def describe_period(freq: str, n: int) -> str:
    """Human-readable description of n samples at freq."""
    sec_per = {
        "10S": 10, "30S": 30, "1S": 1,
        "1T": 60, "5T": 300, "10T": 600, "15T": 900, "30T": 1800,
        "H": 3600, "D": 86400, "W-SUN": 604800, "W": 604800,
        "M": 2_628_000, "Q-DEC": 7_884_000, "A-DEC": 31_536_000, "Y-DEC": 31_536_000,
    }
    if freq not in sec_per:
        return f"({n} samples)"
    secs = n * sec_per[freq]
    if secs < 3600:
        return f"~{secs // 60}-min cycle"
    if secs < 86400:
        return f"~{secs // 3600}-hour cycle"
    if secs < 30 * 86400:
        return f"~{secs // 86400}-day cycle"
    if secs < 365 * 86400:
        return f"~{secs // (30 * 86400)}-month cycle"
    return f"~{secs // (365 * 86400)}-year cycle"


if __name__ == "__main__":
    main()
