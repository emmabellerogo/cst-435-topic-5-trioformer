"""Stage the exported model files under the FLAT asset names a GitHub Release needs.

GitHub release assets have no directories, so ``bert/model.onnx.data`` is published as
``bert__model.onnx.data`` (the ``asset`` field in models/manifest.json). This copies every
non-bundled file from the export directory to ``--out`` under that name after re-verifying its
SHA-256 against the manifest, and prints the upload command. Nothing is uploaded by this script.

    python scripts/export_onnx.py --out build/models --local-files-only      # writes build/models/manifest.json
    cp build/models/manifest.json models/manifest.json                       # commit this file
    python scripts/stage_release_assets.py --src build/models --out build/release
    gh release create models-v1 build/release/* --title "ONNX model files v1" --notes "see models/manifest.json"
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.fetch_models import is_valid, load_manifest  # noqa: E402


def stage(manifest_path: Path, src: Path, out: Path) -> int:
    manifest = load_manifest(manifest_path)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for rel, meta in sorted(manifest["files"].items()):
        if meta.get("bundled"):
            continue
        if not is_valid(src / rel, meta):
            raise SystemExit(f"{src / rel}: missing or does not match manifest.json (re-run export_onnx.py?)")
        shutil.copyfile(src / rel, out / meta["asset"])
        n += 1
    print(f"staged {n} assets in {out} (release tag: {manifest.get('release_tag', '?')}, "
          f"repo URL: {manifest.get('base_url', '?').rsplit('/releases', 1)[0]})")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="export directory (contains bert/, gpt2/, ...)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None, help="default: <src>/manifest.json")
    a = ap.parse_args(argv)
    stage(a.manifest or a.src / "manifest.json", a.src, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
