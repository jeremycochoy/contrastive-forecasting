#!/usr/bin/env python3
"""Train the heads of several runs of train_forecasting_head.py on one data
stream (#425).

Each job is one solo run: the flags of ``train_forecasting_head.py``, as one
JSON list per line of the jobs file. Each job keeps its own frozen backbone,
head, optimizer, seed and output files. One stream feeds all the jobs: each
batch goes to the step of each job in turn. So the process reads the stream
once, not once per job. A solo head on GiftEvalPretrain reads about 80 GB.

Each job swaps in its own random state for its step: torch on the CPU (the
draws of patch size and split), on the GPU (dropout) and numpy. So each job
takes the steps of its solo run: the same batches and the same draws.

The jobs of one process must read one data stream (``HeadJob.data_key``):
the same source, seed, vocabulary, batch size, start and number of steps.

Usage:
    python scripts/train_forecasting_heads_shared.py --jobs jobs.jsonl
"""

import argparse
import contextlib
import json
import sys
import time

import numpy as np
import torch

import train_forecasting_head as head_trainer


class RandomState:
    """The random state of one job: torch on the CPU and on the GPU of the
    job, and numpy. A job restores it for its step and saves it after, so
    the steps of the other jobs do not move it."""

    def __init__(self, device):
        self.device = device
        self.save()

    def save(self):
        self.cpu = torch.get_rng_state()
        self.numpy = np.random.get_state()
        self.gpu = (torch.cuda.get_rng_state(self.device)
                    if self.device.type == "cuda" else None)

    def restore(self):
        torch.set_rng_state(self.cpu)
        np.random.set_state(self.numpy)
        if self.gpu is not None:
            torch.cuda.set_rng_state(self.gpu, self.device)


class Prefixed:
    """A text stream that starts each line with ``prefix``."""

    def __init__(self, stream, prefix):
        self.stream, self.prefix, self.at_start = stream, prefix, True

    def write(self, text):
        for piece in text.splitlines(keepends=True):
            if self.at_start:
                self.stream.write(self.prefix)
            self.stream.write(piece)
            self.at_start = piece.endswith("\n")
        return len(text)

    def flush(self):
        self.stream.flush()


class Member:
    """One job of the shared run: its HeadJob, the random state that its
    own setup left, and its log lines, which start with its run name."""

    def __init__(self, argv):
        args = head_trainer.parse_args(argv)
        self.out = Prefixed(sys.stdout, f"[{args.run_name}] ")
        with contextlib.redirect_stdout(self.out):
            self.job = head_trainer.HeadJob(args)
        self.state = RandomState(self.job.device)

    @contextlib.contextmanager
    def turn(self):
        """The random state and the log of this job, for one call."""
        self.state.restore()
        with contextlib.redirect_stdout(self.out):
            yield self.job
        self.state.save()


def read_jobs(path):
    """The flags of each solo run: one JSON list per line."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_one_stream(members):
    """Refuse jobs that read other batches, or that share an output file."""
    keys = {m.job.data_key() for m in members}
    if len(keys) > 1:
        raise SystemExit("The jobs of one process must read one data stream "
                         "for the same steps. These read several:\n  "
                         + "\n  ".join(map(str, sorted(keys, key=str))))
    outputs = {(m.job.args.save_dir, m.job.args.run_name) for m in members}
    if len(outputs) < len(members):
        raise SystemExit("Two jobs write the same files: give each job its "
                         "own --save-dir or --run-name.")


def own_copy(batch):
    """A copy of the batch for one job, so no job sees what another job's
    step did to it."""
    if isinstance(batch, tuple):
        return tuple(t.clone() for t in batch)
    return batch.clone()


def report(done, n_jobs, seconds, recent, device, waited):
    """The step rate of the process, the share of the recent time that it
    waited for a batch of the stream, and its peak GPU memory.

    ``recent`` is ``(steps, seconds)`` since the last report, and ``waited``
    the seconds of them in which the process waited for a batch. A process
    that waits is as fast as its stream, so more jobs cost it no time. A
    process that does not wait is as fast as its GPU."""
    rate, last = done / seconds, recent[0] / recent[1]
    line = (f"[shared] {done} steps: {rate:.2f} steps/s since the start, "
            f"{last:.2f} steps/s over the last {recent[0]}, "
            f"{last * n_jobs:.1f} job steps/s, "
            f"the stream took {100 * waited / recent[1]:.0f}% of that time")
    if device.type == "cuda":
        gib = 2 ** 30
        line += (f", peak {torch.cuda.max_memory_allocated(device) / gib:.2f}"
                 f" GiB allocated, "
                 f"{torch.cuda.max_memory_reserved(device) / gib:.2f} GiB "
                 f"reserved")
    else:
        line += ", 0.00 GiB on the GPU (a CPU run)"
    print(line, flush=True)


def train(members, loader):
    """Each batch of the stream to the step of each job, in turn."""
    first = members[0].job
    data_iter = iter(loader)
    for member in members:
        with member.turn() as job:
            job.start()
    every = first.args.log_every
    t0 = last = time.time()
    last_done, waited = 0, 0.0
    for step in range(first.start_step + 1, first.args.total_steps + 1):
        asked = time.time()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        waited += time.time() - asked
        for member in members:
            with member.turn() as job:
                job.train_step(step, own_copy(batch))
        done = step - first.start_step
        # The first report also tells queue.sh that the memory of the
        # process is in use.
        if step % every == 0 or done == 1:
            now = time.time()
            report(done, len(members), now - t0,
                   (done - last_done, now - last), first.device, waited)
            last, last_done, waited = now, done, 0.0
    for member in members:
        with member.turn() as job:
            job.finish()


def parse_args():
    p = argparse.ArgumentParser(
        description="Train the heads of several runs on one data stream")
    p.add_argument("--jobs", required=True,
                   help="One JSON list per line: the flags of one solo run "
                        "of train_forecasting_head.py.")
    return p.parse_args()


def main():
    path = parse_args().jobs
    argvs = read_jobs(path)
    if not argvs:
        raise SystemExit(f"no job in {path}")
    members = [Member(argv) for argv in argvs]
    check_one_stream(members)
    with members[0].turn() as job:
        loader = job.data_loader()
    print(f"[shared] {len(members)} jobs on one stream: "
          + ", ".join(m.job.args.run_name for m in members), flush=True)
    train(members, loader)
    print("[shared] done", flush=True)


if __name__ == "__main__":
    main()
