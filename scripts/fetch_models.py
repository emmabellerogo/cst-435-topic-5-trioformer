"""Download the quantized ONNX model files listed in ``models/manifest.json`` and verify them.

Standard library only (it runs in Render's build step before anything else is installed).

    python scripts/fetch_models.py                 # download missing/corrupt files into ./models
    python scripts/fetch_models.py --verify-only   # just check what is on disk (exit 1 on any problem)
    python scripts/fetch_models.py --only bert,t5  # subset (bert | gpt2 | t5)

Where files come from: ``<base_url>/<asset>`` where ``asset`` is the manifest path with ``/`` replaced
by ``__``. ``base_url`` is the first of: ``--base-url``, env ``TRIOFORMER_MODELS_BASE_URL``, or
``manifest["base_url"]`` (a GitHub Release: https://github.com/<owner>/<repo>/releases/download/<tag>).
Release assets of a *public* repository download anonymously (no token, no paid service).

Every file is streamed to ``<name>.part``, checked against the manifest's size and SHA-256, and only
then renamed into place, so an interrupted/corrupt download never leaves a half-written model behind.
Files already present with the right checksum are skipped. Exit code is non-zero on any failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
GROUPS = {"bert": "bert/", "gpt2": "gpt2/", "t5": "t5-small/"}
CHUNK = 1 << 20
USER_AGENT = "trioformer-fetch-models/1"


def load_manifest(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != 1 or "files" not in manifest:
        raise SystemExit(f"{path}: not a supported manifest (schema 1 expected)")
    return manifest


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid(path: Path, meta: dict) -> bool:
    return path.is_file() and path.stat().st_size == meta["size"] and sha256_of(path) == meta["sha256"]


def download(url: str, dest: Path, meta: dict, timeout: float = 60.0) -> None:
    """Stream ``url`` to ``dest`` (via ``.part``); raise ValueError on size/checksum mismatch."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    n = 0
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as out:
            for chunk in iter(lambda: resp.read(CHUNK), b""):
                out.write(chunk)
                h.update(chunk)
                n += len(chunk)
        if n != meta["size"]:
            raise ValueError(f"size mismatch for {dest.name}: got {n} bytes, manifest says {meta['size']}")
        if h.hexdigest() != meta["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {dest.name}: got {h.hexdigest()[:16]}..., "
                             f"expected {meta['sha256'][:16]}...")
        os.replace(part, dest)
    finally:
        if part.exists():
            part.unlink()


def fetch(manifest_path: Path, dest_dir: Path, base_url: Optional[str] = None, only: Optional[List[str]] = None,
          verify_only: bool = False, retries: int = 3, log=print) -> int:
    manifest = load_manifest(manifest_path)
    base = (base_url or os.environ.get("TRIOFORMER_MODELS_BASE_URL") or manifest["base_url"]).rstrip("/")
    prefixes = [GROUPS[g] for g in only] if only else None
    problems: Dict[str, str] = {}
    total = downloaded = 0
    started = time.time()
    for rel, meta in sorted(manifest["files"].items()):
        bundled = bool(meta.get("bundled"))
        if prefixes is not None and not bundled and not any(rel.startswith(p) for p in prefixes):
            continue
        path = dest_dir / rel
        total += 1
        if is_valid(path, meta):
            log(f"  ok       {rel} ({meta['size'] / 1e6:.1f} MB)")
            continue
        if bundled or verify_only:
            problems[rel] = "missing or corrupt" + (" (bundled file: restore it from git)" if bundled else "")
            log(f"  MISSING  {rel}")
            continue
        url = f"{base}/{meta['asset']}"
        for attempt in range(1, retries + 1):
            try:
                t0 = time.time()
                download(url, path, meta)
                downloaded += meta["size"]
                log(f"  fetched  {rel} ({meta['size'] / 1e6:.1f} MB in {time.time() - t0:.1f}s)")
                problems.pop(rel, None)
                break
            except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
                problems[rel] = f"{type(exc).__name__}: {exc}"
                log(f"  retry {attempt}/{retries} {rel}: {problems[rel]}")
                time.sleep(min(2 ** attempt, 10) if attempt < retries else 0)
    status = "FAILED" if problems else "OK"
    log(f"{status}: {total - len(problems)}/{total} files verified, {downloaded / 1e6:.0f} MB downloaded "
        f"in {time.time() - started:.1f}s")
    for rel, why in problems.items():
        log(f"  problem: {rel}: {why}")
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=REPO / "models" / "manifest.json")
    ap.add_argument("--dest", type=Path, default=None, help="default: the manifest's directory")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--only", default=None, help="comma list from: bert,gpt2,t5")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--retries", type=int, default=3)
    a = ap.parse_args(argv)
    only = [x.strip() for x in a.only.split(",")] if a.only else None
    if only and any(x not in GROUPS for x in only):
        ap.error(f"--only must be a comma list from {sorted(GROUPS)}")
    return fetch(a.manifest, a.dest or a.manifest.parent, a.base_url, only, a.verify_only, a.retries)


if __name__ == "__main__":
    sys.exit(main())
