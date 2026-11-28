"""The S3 copies of an ablation run of #482, sent by the box itself.

The uplink of elisa is too slow for the checkpoints, so elisa sends no .pth
bytes. For each file, elisa makes a presigned URL for its key, and the box
runs curl with it. The URL reaches curl on its standard input: no command
line, no log and no output of this tool holds it. Then elisa reads the
object with head_object. Its ETag (the MD5 of a single PUT) must equal the
ETag of the box file, which the box computes. A file goes to S3 only when it
is a whole zip file (the format of torch.save).

    python3 s3_box_copy.py copy <run> <box> <port> --record <tsv> [--stops "10 25"]
        One poll of the puller. Each stop backbone that S3 does not hold
        yet goes to S3, then the optimizer file of the newest stop. When the
        record names the two files of the newest stop, the older optimizer
        files of the run leave S3. The backbones stay.
    python3 s3_box_copy.py seed <run> <box> <port>
        Put the newest pair of the run in S3 (a backbone and its optimizer
        file) in the ckpt folder of the run on a new box, with the ETag
        check. It refuses a box with no setup_done_<run> (setup_box.sh) or
        with a .pth file of the run. At a failure it removes both files of
        the pair from the box and ends with exit code 1: start the box only
        after exit code 0.

Keys: s3://model.artifacts/manual/cf_runs/cf-482/<run>/ckpt/<file>. The
client reads the AWS profile of the pullers (upload_and_delete_pth.py).
"""
import argparse
import math
import re
import shlex
import subprocess
import sys
import time

BUCKET = "model.artifacts"
REGION = "us-west-2"
EXPIRY = 900                               # s, the life of a presigned URL
BOX_H = "/workspace/hp64"
STOPS = (10, 25, 50, 75, 100, 125, 150, 166)
CKPT_PREFIX = re.compile(r"manual/cf_runs/cf-482/[^/]+/ckpt")
MIB = 1 << 20
# The part sizes of a multipart upload (upload_and_delete_runs_v2.py): the
# pullers of round 1 sent the stop backbones in parts.
PART_SIZES = (8 * MIB, 64 * MIB, 16 * MIB, 5 * MIB, 32 * MIB, 128 * MIB)
SSH_OPTS = ("-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=20", "-o", "LogLevel=ERROR")
# The box runs this with python3: the S3 ETag of a file (part 0: the MD5 of
# a single PUT) and its size, or exit code 3 when it is not a whole zip file.
BOX_ETAG = r'''
import hashlib, os, sys, zipfile
path, part = sys.argv[1], int(sys.argv[2])
try:
    whole = zipfile.ZipFile(path).testzip() is None
except (zipfile.BadZipFile, OSError):
    whole = False
if not whole:
    sys.exit(3)
plain, parts = hashlib.md5(), []
with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(part or 8 << 20), b""):
        plain.update(chunk)
        parts.append(hashlib.md5(chunk).digest())
etag = (hashlib.md5(b"".join(parts)).hexdigest() + "-" + str(len(parts))
        if part else plain.hexdigest())
print(etag, os.path.getsize(path))
'''


# --- The files of a run ------------------------------------------------------

def checkpoint_order(run, name):
    """(step in thousands, start) of a numbered backbone of the run, or None
    for another file. The first start has no ``_r<n>`` part: start 1."""
    found = re.fullmatch(re.escape(run) + r"(?:_r(\d+))?_(\d+)k\.pth", name)
    return None if found is None else (int(found.group(2)), int(found.group(1) or 1))


def saves(run, names):
    """{(step, start): backbone} of each whole save of the run in
    ``names``: a numbered backbone with its optimizer file. The trainer
    writes the optimizer file after the backbone."""
    names = set(names)
    return {checkpoint_order(run, n): n for n in names
            if checkpoint_order(run, n) and n[:-4] + "_optimizer.pth" in names}


def stop_saves(run, names, stops):
    """{step: backbone} of each stop with a whole save. At one step the last
    start wins, as in the run script and the follower."""
    found = {}
    for (step, _), name in sorted(saves(run, names).items()):
        if step in stops:
            found[step] = name
    return found


def to_copy(run, names, stops):
    """The files to copy, in order: the backbone of each stop, then the
    optimizer file of the newest stop."""
    found = stop_saves(run, names, stops)
    if not found:
        return []
    return [found[k] for k in sorted(found)] + [optimizer_of(found[max(found)])]


def optimizer_of(backbone):
    return backbone[:-4] + "_optimizer.pth"


def ckpt_prefix(run):
    return f"manual/cf_runs/cf-482/{run}/ckpt"


# --- The URLs ------------------------------------------------------------------

def signing_config():
    from botocore.config import Config
    return Config(signature_version="s3v4", retries={"max_attempts": 10})


def put_url(client, key):
    """A presigned PUT of ``key``, valid for EXPIRY seconds."""
    return client.generate_presigned_url(
        "put_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=EXPIRY)


def get_url(client, key):
    """A presigned GET of ``key``, valid for EXPIRY seconds."""
    return client.generate_presigned_url(
        "get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=EXPIRY)


def curl_config(url):
    """The curl config (``-K -``) that gives curl the URL on its input."""
    return f'url = "{url}"\n'


def curl_on_box(box, args, url):
    """Run curl on the box with the URL on its input. Returns the exit code
    and the error text, with no URL in it."""
    out = box.run(f"curl -sS -f {args} -K -", stdin=curl_config(url))
    return out.returncode, out.stderr.replace(url, "<url>").strip()


# --- S3 and the box ----------------------------------------------------------

def head(client, key):
    """(size, etag) of the object ``key``, or None."""
    from botocore.exceptions import ClientError
    try:
        found = client.head_object(Bucket=BUCKET, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    return found["ContentLength"], found["ETag"].strip('"')


def s3_objects(client, prefix):
    """{file name: (size, etag)} of each object under the prefix."""
    pages = client.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix + "/")
    return {item["Key"].rsplit("/", 1)[1]: (item["Size"], item["ETag"].strip('"'))
            for page in pages for item in page.get("Contents", [])}


def box_etag(box, path, part=0):
    """(size, etag) of a whole zip file on the box, in the form of an S3
    ETag (``part`` > 0: parts of this size), or None."""
    out = box.run(f"python3 - {shlex.quote(path)} {part}", stdin=BOX_ETAG)
    if out.returncode:
        return None
    etag, size = out.stdout.split()
    return int(size), etag


def part_sizes(size, etag):
    """The part sizes that give an S3 ETag of this form: 0 for one PUT."""
    if "-" not in etag:
        return [0]
    count = int(etag.split("-")[1])
    return [p for p in PART_SIZES if math.ceil(size / p) == count]


def same_on_box(box, path, held):
    """True when the box file ``path`` has the bytes of the S3 object
    ``held`` (size, etag)."""
    return any(box_etag(box, path, part) == held for part in part_sizes(*held))


class Box:
    """ssh to a rented box. ``run`` returns the CompletedProcess."""

    def __init__(self, target, port, timeout=3600):
        self.argv = ["ssh", "-p", str(port), *SSH_OPTS, target]
        self.timeout = timeout

    def run(self, command, stdin=""):
        return subprocess.run(self.argv + [command], input=stdin, text=True,
                              capture_output=True, timeout=self.timeout)


# --- The copy: one poll of the puller -----------------------------------------

def copy_file(box, client, key, path, log):
    """Copy the box file ``path`` to ``key``, unless S3 holds it. Returns
    (size, etag) when S3 holds the bytes of the box file, else None. S3
    never gets other bytes at a key that it holds."""
    name = key.rsplit("/", 1)[1]
    held = head(client, key)
    if held is not None:
        if same_on_box(box, path, held):
            return held
        log(f"{name}: S3 holds other bytes at {key}; no copy")
        return None
    found = box_etag(box, path)
    if found is None:
        log(f"{name}: not a whole file on the box yet; next poll")
        return None
    code, error = curl_on_box(box, f"-T {shlex.quote(path)}", put_url(client, key))
    if code:
        log(f"{name}: curl exit code {code}: {error}")
        return None
    if head(client, key) != found:
        log(f"{name}: the ETag in S3 is not the ETag of the box file")
        return None
    log(f"{name}: in S3, ETag {found[1]} = the box file")
    return found


def recorded(record):
    """The keys of the record: S3 holds each with the bytes of the box."""
    try:
        with open(record) as f:
            return {line.rstrip("\n").split("\t")[1] for line in f if "\t" in line}
    except FileNotFoundError:
        return set()


def drop_older_optimizers(client, prefix, newest):
    """Delete each optimizer file under the ckpt prefix of a run but the
    file ``newest``. Returns the deleted keys."""
    if not CKPT_PREFIX.fullmatch(prefix):
        raise ValueError(f"{prefix}: not the ckpt prefix of a run of #482")
    older = [f"{prefix}/{name}" for name in s3_objects(client, prefix)
             if name.endswith("_optimizer.pth") and name != newest]
    for key in older:
        client.delete_object(Bucket=BUCKET, Key=key)
    return older


def copy_poll(box, client, run, stops, box_h, record, log):
    """Copy the files of ``to_copy`` that the record does not name, then
    delete the older optimizer files when the record names the newest
    pair."""
    folder, prefix = f"{box_h}/{run}/ckpt", ckpt_prefix(run)
    names = box.run(f"ls {shlex.quote(folder)}").stdout.split()
    done = recorded(record)
    for name in to_copy(run, names, stops):
        key = f"{prefix}/{name}"
        if key not in done and copy_file(box, client, key, f"{folder}/{name}", log):
            with open(record, "a") as f:
                f.write(f"{time.strftime('%F %T')}\t{key}\n")
            done.add(key)
    pair = newest_stop_pair(run, names, stops)
    if pair and all(f"{prefix}/{name}" in done for name in pair):
        older = drop_older_optimizers(client, prefix, pair[1])
        if older:
            log(f"{pair[1]}: the newest pair in S3; deleted {older}")


def newest_stop_pair(run, names, stops):
    """(backbone, optimizer file) of the newest stop with a whole save, or
    None."""
    found = stop_saves(run, names, stops)
    if not found:
        return None
    return found[max(found)], optimizer_of(found[max(found)])


# --- The seed of a new box --------------------------------------------------

def seed(box, client, run, box_h, log):
    """Put the newest pair of the run in S3 in its ckpt folder on the box.
    Returns the name of the pair. At a failure no file of the pair stays
    on the box: with a half pair the run script starts from step 0."""
    folder, prefix = f"{box_h}/{run}/ckpt", ckpt_prefix(run)
    check_new_box(box, run, box_h, folder)
    held = s3_objects(client, prefix)
    whole = saves(run, held)
    if not whole:
        raise SystemExit(f"no pair of {run} in s3://{BUCKET}/{prefix}/")
    pair = (whole[max(whole)], optimizer_of(whole[max(whole)]))
    box.run(f"mkdir -p {shlex.quote(folder)}")
    try:
        for name in pair:
            get_to_box(box, client, f"{prefix}/{name}", f"{folder}/{name}", held[name])
    except SystemExit:
        box.run("rm -f " + " ".join(shlex.quote(f"{folder}/{name}{end}")
                                    for name in pair for end in ("", ".part")))
        raise
    log(f"{pair[0][:-4]}: the pair on the box, each ETag = the S3 object; "
        "the run script resumes from it")
    return pair[0][:-4]


def check_new_box(box, run, box_h, folder):
    """Stop the seed on a box with no successful setup_box.sh of the run, or
    with a .pth file in the ckpt folder of the run."""
    setup = box.run(f"cut -d' ' -f1-2 {box_h}/setup_done_{run}").stdout.strip()
    if setup != f"run {run}":
        raise SystemExit(f"no successful setup_box.sh of {run} on the box")
    if any(n.endswith(".pth") for n in box.run(f"ls {folder}").stdout.split()):
        raise SystemExit(f"the box holds a .pth file of {run} in {folder}")


def get_to_box(box, client, key, path, held):
    """Download ``key`` to ``path`` on the box, through ``path``.part, and
    check its bytes against ``held`` (size, etag)."""
    part = path + ".part"
    code, error = curl_on_box(box, f"-o {shlex.quote(part)}", get_url(client, key))
    if code or not same_on_box(box, part, held):
        box.run(f"rm -f {shlex.quote(part)}")
        raise SystemExit(f"{key}: not the bytes of S3 on the box "
                         f"(curl exit code {code} {error})")
    moved = box.run(f"mv {shlex.quote(part)} {shlex.quote(path)}")
    if moved.returncode:
        raise SystemExit(f"{key}: the move of {part} to {path} failed "
                         f"(exit code {moved.returncode})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("copy", "seed"))
    p.add_argument("run")
    p.add_argument("box", help="the ssh target, root@<ip>")
    p.add_argument("port")
    p.add_argument("--record", help="copy: the record of the S3 copies")
    p.add_argument("--stops", default=" ".join(map(str, STOPS)))
    p.add_argument("--box-h", default=BOX_H, help="the folder of the runs on the box")
    a = p.parse_args(argv)
    import boto3
    client = boto3.client("s3", region_name=REGION, config=signing_config())
    box = Box(a.box, a.port)

    def log(message):
        print(f"{time.strftime('%FT%TZ', time.gmtime())} {a.run}: {message}", flush=True)
    if a.cmd == "copy":
        if not a.record:
            raise SystemExit("copy needs --record")
        copy_poll(box, client, a.run, {int(s) for s in a.stops.split()}, a.box_h,
                  a.record, log)
    else:
        seed(box, client, a.run, a.box_h, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
