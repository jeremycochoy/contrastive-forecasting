"""#482: the S3 copies of an ablation run, sent by the box itself
(reports/2026-11-27_paper_ablations/scripts/s3_box_copy.py).

The uplink of elisa is too slow for the checkpoints. For each file, elisa
makes a presigned URL, the box runs curl with it, and elisa checks the ETag
of the object against the box file. The tests use a box and an S3 in
memory. The URL test signs with test keys and needs no network.
"""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "reports" / "2026-11-27_paper_ablations" / "scripts"
RUN = "g166b_w320e6f6_ab04_instnorm"
STOPS = (10, 25, 50)


def load():
    spec = importlib.util.spec_from_file_location("s3_box_copy_482",
                                                  SCRIPTS / "s3_box_copy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bc = load()
PREFIX = f"manual/cf_runs/cf-482/{RUN}/ckpt"


def save(k, start=1):
    """The two file names of the save at k thousand steps."""
    name = f"{RUN}_{k}k" if start == 1 else f"{RUN}_r{start}_{k}k"
    return [f"{name}.pth", f"{name}_optimizer.pth"]


# ---------------------------------------------------------------------------
# The files to copy
# ---------------------------------------------------------------------------

def test_the_stop_backbones_then_the_optimizer_file_of_the_newest_stop():
    names = save(5) + save(10) + save(15) + save(25) + [f"{RUN}_losses.csv"]
    assert bc.to_copy(RUN, names, STOPS) == [
        f"{RUN}_10k.pth", f"{RUN}_25k.pth", f"{RUN}_25k_optimizer.pth"]


def test_a_save_with_no_optimizer_file_yet_waits():
    names = save(10) + [f"{RUN}_25k.pth"]
    assert bc.to_copy(RUN, names, STOPS) == [
        f"{RUN}_10k.pth", f"{RUN}_10k_optimizer.pth"]


def test_at_one_step_the_last_start_wins():
    names = save(10) + save(10, start=2) + save(25, start=2)
    assert bc.to_copy(RUN, names, STOPS) == [
        f"{RUN}_r2_10k.pth", f"{RUN}_r2_25k.pth", f"{RUN}_r2_25k_optimizer.pth"]


def test_no_stop_gives_no_file():
    assert bc.to_copy(RUN, save(5) + save(15), STOPS) == []


# ---------------------------------------------------------------------------
# The presigned URL (no network)
# ---------------------------------------------------------------------------

def offline_client():
    """An S3 client with test keys: it signs a URL with no network."""
    session = pytest.importorskip("botocore.session").get_session()
    return session.create_client(
        "s3", region_name=bc.REGION, aws_access_key_id="AKIDTEST",
        aws_secret_access_key="test-secret", config=bc.signing_config())


@pytest.mark.parametrize("method, url_of", [("PUT", "put_url"),
                                            ("GET", "get_url")])
def test_the_url_signs_the_key_for_a_short_time(method, url_of):
    key = f"{PREFIX}/{RUN}_10k.pth"
    url = urlsplit(getattr(bc, url_of)(offline_client(), key))
    query = parse_qs(url.query)
    assert url.scheme == "https" and bc.BUCKET in url.netloc + url.path
    assert url.path.endswith(key)
    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    assert "/us-west-2/s3/aws4_request" in query["X-Amz-Credential"][0]
    assert query["X-Amz-Expires"] == [str(bc.EXPIRY)]
    assert query["X-Amz-SignedHeaders"] == ["host"]


def test_the_url_goes_to_curl_on_its_standard_input_only():
    url = "https://s3.example/k?X-Amz-Signature=abc&b=c"
    assert bc.curl_config(url) == f'url = "{url}"\n'


# ---------------------------------------------------------------------------
# The copy of one file, with a box and an S3 in memory
# ---------------------------------------------------------------------------

def etag(data, part=0):
    if not part:
        return hashlib.md5(data).hexdigest()
    parts = [hashlib.md5(data[i:i + part]).digest()
             for i in range(0, len(data), part)]
    return f"{hashlib.md5(b''.join(parts)).hexdigest()}-{len(parts)}"


class MemoryS3:
    """head_object, list_objects_v2, delete_object and the URLs, on a dict
    {key: bytes}. A URL names its key."""

    def __init__(self, objects=None):
        self.objects, self.parts = dict(objects or {}), {}

    def generate_presigned_url(self, op, Params, ExpiresIn, HttpMethod=None):
        return f"https://s3.test/{Params['Key']}?X-Amz-Signature=secret-{op}"

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        data = self.objects[Key]
        return {"ContentLength": len(data),
                "ETag": f'"{etag(data, self.parts.get(Key, 0))}"'}

    def get_paginator(self, name):
        return self

    def paginate(self, Bucket, Prefix):
        return [{"Contents": [{"Key": k, "Size": len(v),
                               "ETag": f'"{etag(v, self.parts.get(k, 0))}"'}
                              for k, v in sorted(self.objects.items())
                              if k.startswith(Prefix)]}]

    def delete_object(self, Bucket, Key):
        del self.objects[Key]


class MemoryBox:
    """The commands of the tool on a box whose files are a dict. curl -T
    puts the file at the key of the URL, curl -o gets it."""

    def __init__(self, s3, files, whole=lambda path: True):
        self.s3, self.files, self.whole = s3, dict(files), whole
        self.commands, self.stdins = [], []
        self.curl_exit = 0

    def run(self, command, stdin=""):
        self.commands.append(command)
        self.stdins.append(stdin)
        words = command.split()
        if words[0] == "ls":
            names = [p.rsplit("/", 1)[1] for p in self.files
                     if p.rsplit("/", 1)[0] == words[1]]
            return self.out("\n".join(names))
        if words[:2] == ["python3", "-"]:
            path, part = words[2], int(words[3])
            if path not in self.files or not self.whole(path):
                return self.out("", 3)
            data = self.files[path]
            return self.out(f"{etag(data, part)} {len(data)}")
        if words[0] == "curl":
            return self.curl(words, stdin)
        if words[0] == "cut":
            return self.out(self.files.get(words[-1], b"").decode()[:4 + len(RUN)])
        if words[0] == "mv":
            self.files[words[2]] = self.files.pop(words[1])
        if words[:2] == ["rm", "-f"]:
            self.files.pop(words[2], None)
        return self.out("")

    def curl(self, words, stdin):
        key = stdin.split('"')[1].split("://", 1)[1].split("/", 1)[1].split("?")[0]
        if self.curl_exit:
            return self.out("", self.curl_exit, "curl: (22) error 403")
        if "-T" in words:
            self.s3.objects[key] = self.files[words[words.index("-T") + 1]]
        else:
            self.files[words[words.index("-o") + 1]] = self.s3.objects[key]
        return self.out("")

    @staticmethod
    def out(stdout, code=0, stderr=""):
        return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)


BOX_DIR = f"/box/{RUN}/ckpt"


def box_with(*saves_of):
    files = {f"{BOX_DIR}/{n}": n.encode() * 50
             for k in saves_of for n in save(k)}
    s3 = MemoryS3()
    return MemoryBox(s3, files), s3


def poll(box, s3, tmp_path, stops=STOPS):
    lines = []
    bc.copy_poll(box, s3, RUN, stops, "/box", tmp_path / "record.tsv",
                 lines.append)
    return lines


def test_a_poll_copies_the_stop_backbones_and_the_newest_optimizer(tmp_path):
    box, s3 = box_with(5, 10, 15, 25)
    lines = poll(box, s3, tmp_path)
    assert sorted(s3.objects) == sorted(
        f"{PREFIX}/{n}" for n in [f"{RUN}_10k.pth", f"{RUN}_25k.pth",
                                  f"{RUN}_25k_optimizer.pth"])
    for key, data in s3.objects.items():
        assert data == box.files[f"{BOX_DIR}/{key.rsplit('/', 1)[1]}"]
    assert all("secret" not in line for line in lines)
    assert all("secret" not in command for command in box.commands)


def test_a_second_poll_copies_nothing_again(tmp_path):
    box, s3 = box_with(10)
    poll(box, s3, tmp_path)
    box.commands.clear()
    poll(box, s3, tmp_path)
    assert [c.split()[0] for c in box.commands] == ["ls"]


def test_a_new_stop_deletes_the_older_optimizer_file_only(tmp_path):
    other = {"manual/cf_runs/cf-482/other/ckpt/x_10k_optimizer.pth": b"o"}
    box, s3 = box_with(10)
    s3.objects.update(other)
    poll(box, s3, tmp_path)
    box.files.update({f"{BOX_DIR}/{n}": n.encode() for n in save(25)})
    lines = poll(box, s3, tmp_path)
    assert sorted(s3.objects) == sorted(
        [f"{PREFIX}/{RUN}_10k.pth", f"{PREFIX}/{RUN}_25k.pth",
         f"{PREFIX}/{RUN}_25k_optimizer.pth", *other])
    assert any("deleted" in line for line in lines)


def test_a_failed_curl_copies_nothing_and_shows_no_url(tmp_path):
    box, s3 = box_with(10)
    box.curl_exit = 22
    lines = poll(box, s3, tmp_path)
    assert s3.objects == {}
    assert any("curl exit code 22" in line for line in lines)
    assert all("secret" not in line for line in lines)
    assert not (tmp_path / "record.tsv").exists()


def test_other_bytes_in_s3_stop_the_copy_of_the_pair(tmp_path):
    box, s3 = box_with(10)
    s3.objects[f"{PREFIX}/{RUN}_10k.pth"] = b"other bytes"
    lines = poll(box, s3, tmp_path)
    assert s3.objects[f"{PREFIX}/{RUN}_10k.pth"] == b"other bytes"
    assert any("other bytes" in line for line in lines)


def test_a_multipart_object_with_the_bytes_of_the_box_counts(tmp_path,
                                                             monkeypatch):
    # The pullers of round 1 sent the stop backbones in parts.
    monkeypatch.setattr(bc, "PART_SIZES", (64,))
    box, s3 = box_with(10)
    key = f"{PREFIX}/{RUN}_10k.pth"
    s3.objects[key], s3.parts[key] = box.files[f"{BOX_DIR}/{RUN}_10k.pth"], 64
    poll(box, s3, tmp_path)
    assert key in (tmp_path / "record.tsv").read_text()
    assert not any(c.startswith("curl") and "_10k.pth " in c
                   for c in box.commands)


def test_a_file_that_is_not_whole_waits(tmp_path):
    box, s3 = box_with(10)
    box.whole = lambda path: not path.endswith("_optimizer.pth")
    poll(box, s3, tmp_path)
    assert sorted(s3.objects) == [f"{PREFIX}/{RUN}_10k.pth"]


def test_the_tool_deletes_nothing_out_of_a_ckpt_prefix():
    with pytest.raises(ValueError):
        bc.drop_older_optimizers(MemoryS3(), "manual/cf_runs/cf-482", "x")


# ---------------------------------------------------------------------------
# The seed of a new box
# ---------------------------------------------------------------------------

def new_box(s3):
    return MemoryBox(s3, {f"/box/setup_done_{RUN}": f"run {RUN} code x".encode()})


def test_the_seed_puts_the_newest_pair_on_the_box():
    s3 = MemoryS3({f"{PREFIX}/{n}": n.encode() * 9 for n in save(10) + save(25)})
    box = new_box(s3)
    assert bc.seed(box, s3, RUN, "/box", print) == f"{RUN}_25k"
    assert {p for p in box.files if p.startswith(BOX_DIR)} == {
        f"{BOX_DIR}/{n}" for n in save(25)}
    assert all("secret" not in command for command in box.commands)


def test_the_seed_refuses_a_box_with_no_setup():
    s3 = MemoryS3({f"{PREFIX}/{n}": b"x" for n in save(10)})
    with pytest.raises(SystemExit, match="setup_box.sh"):
        bc.seed(MemoryBox(s3, {}), s3, RUN, "/box", print)


def test_the_seed_refuses_a_box_with_a_checkpoint():
    s3 = MemoryS3({f"{PREFIX}/{n}": b"x" for n in save(10)})
    box = new_box(s3)
    box.files[f"{BOX_DIR}/{RUN}_5k.pth"] = b"x"
    with pytest.raises(SystemExit, match="holds a .pth"):
        bc.seed(box, s3, RUN, "/box", print)


def test_the_seed_removes_a_download_with_other_bytes():
    s3 = MemoryS3({f"{PREFIX}/{n}": b"x" * 9 for n in save(10)})
    box = new_box(s3)
    box.whole = lambda path: False
    with pytest.raises(SystemExit, match="not the bytes"):
        bc.seed(box, s3, RUN, "/box", print)
    assert not [p for p in box.files if p.startswith(BOX_DIR)]
