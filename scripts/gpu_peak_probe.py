#!/usr/bin/env python3
"""Peak GPU memory of ONE run, on a card that other runs also hold.

`nvidia-smi --query-gpu=memory.used` returns the card total, so on elisa —
two 4090s shared between agent sessions — it answers "how much is on the
card", not "how much does this run take". A neighbour that arrives or leaves
mid-probe moves the number by gigabytes. `gpu_mem_probe.sh` of #373 works
around it by subtracting a floor sampled before the run, which still folds in
whatever the neighbour did afterwards.

This samples per process instead. `nvidia-smi --query-compute-apps` gives
`pid, used_gpu_memory` per process, so the run's own figure is the sum over
the processes below one root PID. The peak over the samples is the answer.

The launcher is a shell script and the trainer is its grandchild, so the tree
is walked on every sample: a process that starts after the probe does (a
DataLoader worker, a re-exec) still counts.

Usage:
    gpu_peak_probe.py --root-pid 12345 --gpu 0 --out peak.json
    gpu_peak_probe.py --root-pid 12345 --gpu 0 --interval 1 --duration 600

The probe exits when the root PID exits, or after --duration seconds.
It writes JSON: peak_mib, mean_mib, samples, gpu, root_pid.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

SMI = "nvidia-smi"


def read_ppid_map() -> dict[int, int]:
    """`{pid: ppid}` for every process on this machine."""
    out = subprocess.run(["ps", "-eo", "pid=,ppid="],
                         capture_output=True, text=True)
    ppid = {}
    for line in out.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
            ppid[int(fields[0])] = int(fields[1])
    return ppid


def descendants(root: int, ppid_map: dict[int, int]) -> set[int]:
    """`root` and every process below it. Tolerates a cycle in the map."""
    children: dict[int, list[int]] = {}
    for pid, parent in ppid_map.items():
        children.setdefault(parent, []).append(pid)
    seen = {root}
    stack = [root]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def parse_compute_apps(text: str) -> list[tuple[int, int]]:
    """`[(pid, used_mib)]` from `--query-compute-apps` CSV output.

    A row whose memory reads `[N/A]` is dropped: the driver reports it for a
    process it cannot attribute, and counting it as 0 would be a measurement.
    """
    rows = []
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 2 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        rows.append((int(fields[0]), int(fields[1])))
    return rows


def sample_compute_apps(gpu: int) -> list[tuple[int, int]]:
    """One `nvidia-smi` sample of the processes holding this card."""
    out = subprocess.run(
        [SMI, f"--id={gpu}", "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    return parse_compute_apps(out.stdout)


def peak_used_mib(samples: list[list[tuple[int, int]]], pids: set[int]) -> int:
    """The largest per-sample sum over `pids`, in MiB."""
    return max((sum(mib for pid, mib in s if pid in pids) for s in samples),
               default=0)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError) as exc:
        return isinstance(exc, PermissionError)
    return True


def probe(root_pid: int, gpu: int, interval: float, duration: float) -> dict:
    """Sample until the root exits, and report the run's own peak."""
    samples: list[list[tuple[int, int]]] = []
    owned: set[int] = set()
    started = time.monotonic()
    while alive(root_pid) and time.monotonic() - started < duration:
        owned |= descendants(root_pid, read_ppid_map())
        samples.append(sample_compute_apps(gpu))
        time.sleep(interval)
    sums = [sum(mib for pid, mib in s if pid in owned) for s in samples]
    used = [v for v in sums if v > 0]
    return {
        "root_pid": root_pid,
        "gpu": gpu,
        "peak_mib": peak_used_mib(samples, owned),
        "mean_mib": round(sum(used) / len(used)) if used else 0,
        "samples": len(samples),
        "samples_with_memory": len(used),
        "seconds": round(time.monotonic() - started, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root-pid", type=int, required=True,
                    help="the run's top process; its whole tree is counted")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=86400.0)
    ap.add_argument("--out", help="write the JSON here as well as to stdout")
    args = ap.parse_args(argv)

    result = probe(args.root_pid, args.gpu, args.interval, args.duration)
    text = json.dumps(result)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
