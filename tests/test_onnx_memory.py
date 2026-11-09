"""Opt-in memory regression test: needs the real exported model files.

    TRIOFORMER_MODEL_DIR=<dir with exported files> pytest tests/test_onnx_memory.py -s
    TRIOFORMER_MEMORY_LIMIT_MB=...   # default: 400 on Linux, 500 elsewhere (see below)

Render Free has 512 MB. The default Linux budget (400 MB) leaves headroom; it is measured with VmHWM,
which is what the platform enforces. macOS keeps freed pages resident (peak measured ~450 MB there for
the same code), so the non-Linux default (500 MB) is only a regression guard, not a Render guarantee.

Runs scripts/measure_memory.py in a FRESH process (so the number is the whole interpreter, imports
included) while switching bert -> gpt2 -> t5 -> bert -> gpt2 -> t5 -> gpt2.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from api import onnx_pipelines as op

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not op.backend_status()["ok"],
                                reason="ONNX model files not present (python scripts/fetch_models.py)")


def test_peak_memory_when_switching_among_all_three_models():
    limit = float(os.environ.get("TRIOFORMER_MEMORY_LIMIT_MB", "400" if sys.platform.startswith("linux") else "500"))
    r = subprocess.run([sys.executable, str(REPO / "scripts/measure_memory.py"), "--json"], capture_output=True,
                       text=True, cwd=REPO, env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
    assert r.returncode == 0, r.stderr[-800:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"\nPEAK {res['peak_mb']} MB (limit {limit:.0f}); baseline {res['baseline_rss_mb']} MB; "
          f"per-step peaks {[s['peak_so_far_mb'] for s in res['steps']]}")
    assert res["torch_imported"] is False, "torch/transformers leaked into the production import graph"
    assert res["max_resident_models"] == 1, "more than one model resident at once"
    assert [s["model"] for s in res["steps"]][:3] == ["bert-base-uncased", "gpt2", "t5-small"]
    assert res["peak_mb"] < limit, f"peak {res['peak_mb']} MB exceeds {limit} MB"
