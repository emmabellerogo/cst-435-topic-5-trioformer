"""scripts/fetch_models.py: checksummed, atomic, retrying downloads (stdlib only) + manifest integrity."""
from __future__ import annotations

import hashlib
import http.server
import json
import threading
from pathlib import Path

import pytest

from scripts import fetch_models as fm

REPO = Path(__file__).resolve().parents[1]
FILES = {"bert/model.onnx": b"graph-bytes" * 10, "bert/model.onnx.data": b"W" * 5000,
         "bert/tokenizer.json": b'{"tok":1}', "gpt2/model.onnx": b"g" * 100, "gpt2/model.onnx.data": b"G" * 3000,
         "gpt2/tokenizer.json": b"{}", "edu/block_h1.onnx": b"edu-bytes"}


def make_manifest(files, base_url="http://unused"):
    return {"schema": 1, "base_url": base_url, "files": {
        rel: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "asset": rel.replace("/", "__"),
              "bundled": rel.startswith("edu/")} for rel, data in files.items()}}


@pytest.fixture
def release(tmp_path):
    """A local HTTP server acting as a GitHub Release (flat asset names)."""
    serve = tmp_path / "release"
    serve.mkdir()
    for rel, data in FILES.items():
        (serve / rel.replace("/", "__")).write_bytes(data)
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(serve), **k)  # noqa: E731
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    handler.log_message = lambda *a: None
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    dest = tmp_path / "models"
    (dest / "edu").mkdir(parents=True)
    (dest / "edu/block_h1.onnx").write_bytes(FILES["edu/block_h1.onnx"])      # bundled file ships with the repo
    manifest = dest / "manifest.json"
    manifest.write_text(json.dumps(make_manifest(FILES)))
    yield SimpleNs(url=f"http://127.0.0.1:{httpd.server_address[1]}", serve=serve, dest=dest, manifest=manifest)
    httpd.shutdown()


class SimpleNs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def run(rel, **kw):
    lines = []
    rc = fm.fetch(rel.manifest, rel.dest, base_url=rel.url, retries=kw.pop("retries", 2), log=lines.append, **kw)
    return rc, lines


def test_downloads_everything_and_verifies(release):
    rc, lines = run(release)
    assert rc == 0 and lines[-1].startswith("OK: 7/7")
    for rel, data in FILES.items():
        assert (release.dest / rel).read_bytes() == data
    assert not list(release.dest.rglob("*.part"))


def test_second_run_skips_valid_files(release):
    run(release)
    rc, lines = run(release)
    assert rc == 0 and "0 MB downloaded" in lines[-1] and not any("fetched" in x for x in lines)


def test_corrupted_file_is_detected_and_repaired(release):
    run(release)
    (release.dest / "bert/model.onnx.data").write_bytes(b"W" * 4999 + b"X")          # same size, wrong bytes
    assert run(release, verify_only=True)[0] == 1
    rc, lines = run(release)
    assert rc == 0 and any("fetched  bert/model.onnx.data" in x for x in lines)
    assert (release.dest / "bert/model.onnx.data").read_bytes() == FILES["bert/model.onnx.data"]


def test_checksum_mismatch_from_the_server_is_rejected_atomically(release):
    (release.serve / "gpt2__model.onnx.data").write_bytes(b"H" * 3000)               # right size, wrong content
    rc, lines = run(release, only=["gpt2"], retries=1)
    assert rc == 1 and any("SHA-256 mismatch" in x for x in lines)
    assert not (release.dest / "gpt2/model.onnx.data").exists() and not list(release.dest.rglob("*.part"))


def test_truncated_download_is_rejected(release):
    (release.serve / "gpt2__model.onnx.data").write_bytes(b"G" * 100)
    rc, lines = run(release, only=["gpt2"], retries=1)
    assert rc == 1 and any("size mismatch" in x for x in lines) and not (release.dest / "gpt2/model.onnx.data").exists()


def test_missing_asset_404_fails_loudly_after_retries(release):
    (release.serve / "bert__tokenizer.json").unlink()
    rc, lines = run(release, only=["bert"], retries=2)
    assert rc == 1 and sum("retry" in x for x in lines) == 2 and any("404" in x for x in lines)


def test_transient_failure_is_retried(release, monkeypatch):
    real, calls = fm.download, {"n": 0}

    def flaky(url, dest, meta, timeout=60.0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("connection reset")
        return real(url, dest, meta, timeout)

    monkeypatch.setattr(fm, "download", flaky)
    monkeypatch.setattr(fm.time, "sleep", lambda s: None)
    assert run(release, only=["bert"])[0] == 0 and calls["n"] > 1


def test_only_selects_a_model_group(release):
    assert run(release, only=["gpt2"])[0] == 0
    assert (release.dest / "gpt2/model.onnx").exists() and not (release.dest / "bert/model.onnx").exists()


def test_bundled_files_are_never_downloaded_only_verified(release):
    (release.dest / "edu/block_h1.onnx").unlink()
    rc, lines = run(release)
    assert rc == 1 and any("bundled file" in x for x in lines)


def test_unsupported_manifest_is_rejected(tmp_path):
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps({"schema": 2}))
    with pytest.raises(SystemExit):
        fm.load_manifest(bad)


def test_cli_exit_codes(release, capsys):
    assert fm.main(["--manifest", str(release.manifest), "--base-url", release.url]) == 0
    assert fm.main(["--manifest", str(release.manifest), "--verify-only"]) == 0
    (release.dest / "gpt2/tokenizer.json").write_bytes(b"corrupt")
    assert fm.main(["--manifest", str(release.manifest), "--verify-only"]) == 1
    with pytest.raises(SystemExit):
        fm.main(["--only", "nonsense"])


# ---- the committed manifest -------------------------------------------------
def test_committed_manifest_is_well_formed_and_release_compatible():
    m = json.loads((REPO / "models/manifest.json").read_text())
    assert m["schema"] == 1 and m["precision"] == "int8"
    assert m["base_url"].startswith("https://github.com/") and "/releases/download/" in m["base_url"]
    assets = [f["asset"] for f in m["files"].values()]
    assert len(assets) == len(set(assets)) and all("/" not in a for a in assets)       # flat, unique release asset names
    assert max(f["size"] for f in m["files"].values()) < 2 * 1024**3                    # GitHub: each asset < 2 GiB
    assert len(assets) < 1000
    for rel in ("bert/model.onnx", "bert/model.onnx.data", "gpt2/model.onnx.data", "t5-small/decoder_step.onnx.data"):
        assert rel in m["files"]
    assert set(m["huggingface_revisions"]) == {"bert-base-uncased", "gpt2", "t5-small"}


def test_bundled_educational_files_match_the_manifest():
    m = json.loads((REPO / "models/manifest.json").read_text())
    bundled = {k: v for k, v in m["files"].items() if v["bundled"]}
    assert set(bundled) == {"edu/block_h1.onnx", "edu/block_h2.onnx", "edu/block_h4.onnx", "edu/embedding.npy"}
    for rel, meta in bundled.items():
        assert fm.is_valid(REPO / "models" / rel, meta), f"{rel} differs from the manifest (re-run scripts/export_onnx.py)"


# ---- stage -> serve -> fetch round trip (what publishing a release then building on Render does)
def test_staged_release_assets_round_trip_through_the_fetcher(tmp_path):
    from scripts.stage_release_assets import stage

    export = tmp_path / "export"
    for rel, data in FILES.items():
        (export / rel).parent.mkdir(parents=True, exist_ok=True)
        (export / rel).write_bytes(data)
    (export / "manifest.json").write_text(json.dumps(make_manifest(FILES)))
    release = tmp_path / "release"
    assert stage(export / "manifest.json", export, release) == 6                   # edu/* is bundled, not published
    assert sorted(p.name for p in release.iterdir()) == sorted(r.replace("/", "__") for r in FILES if not r.startswith("edu/"))

    dest = tmp_path / "fresh"
    (dest / "edu").mkdir(parents=True)
    (dest / "edu/block_h1.onnx").write_bytes(FILES["edu/block_h1.onnx"])
    (dest / "manifest.json").write_text((export / "manifest.json").read_text())
    rc = fm.fetch(dest / "manifest.json", dest, base_url=release.as_uri(), log=lambda *_: None)   # file:// "release"
    assert rc == 0 and all((dest / r).read_bytes() == d for r, d in FILES.items())


def test_staging_refuses_files_that_do_not_match_the_manifest(tmp_path):
    from scripts.stage_release_assets import stage

    for rel, data in FILES.items():
        (tmp_path / "e" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "e" / rel).write_bytes(data)
    (tmp_path / "e/manifest.json").write_text(json.dumps(make_manifest(FILES)))
    (tmp_path / "e/gpt2/model.onnx.data").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="does not match"):
        stage(tmp_path / "e/manifest.json", tmp_path / "e", tmp_path / "out")
