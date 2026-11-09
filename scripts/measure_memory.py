"""Peak process memory while switching among BERT, GPT-2 and T5 on the ONNX backend.

Run in a *fresh* process (it measures the whole interpreter, including the FastAPI + Supabase-client
imports that the real server also carries):

    TRIOFORMER_MODEL_DIR=models python scripts/measure_memory.py            # table
    python scripts/measure_memory.py --json                                 # machine-readable

"peak" is VmHWM from /proc on Linux (what Render's 512 MB limit cares about) and ru_maxrss elsewhere.
"resident" is the current RSS. Note: macOS does not return freed memory to the OS promptly, so
after-release RSS there overstates steady state; the peak is the number to trust on any platform.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ["TRIOFORMER_BACKEND"] = "onnx"      # must be set before api.main is imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARTICLE = ("The city council voted on Tuesday to approve a new bike lane network that will stretch across 40 miles of "
           "downtown streets. Supporters say the plan will reduce traffic and improve safety, while some business owners "
           "worry about losing parking. Construction is expected to begin next spring and finish within two years.")


def _proc_status(key: str):
    try:
        for line in open("/proc/self/status"):
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def rss_mb() -> float:
    v = _proc_status("VmRSS")
    if v is not None:
        return v
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).strip()) / 1024.0


def peak_mb() -> float:
    v = _proc_status("VmHWM")
    if v is not None:
        return v
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1024.0     # bytes on macOS, KB on Linux


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", default="bert,gpt2,t5,bert,gpt2,t5,gpt2")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    import api.main as app_main                      # FastAPI, pydantic, Supabase client, ... like the real server
    mp = app_main.mp
    base, base_peak = rss_mb(), peak_mb()
    calls = {
        "bert": (mp.BERT_MODEL, lambda: mp.bert_fill_mask("The capital of France is [MASK].")),
        "gpt2": (mp.GPT2_MODEL, lambda: mp.gpt2_continue("The meaning of life is", max_new_tokens=40)),
        "t5": (mp.T5_MODEL, lambda: mp.t5_summarize(ARTICLE, max_new_tokens=60)),
    }
    steps = []
    for key in a.sequence.split(","):
        name, run = calls[key]
        t0 = time.perf_counter()
        mp.preload((name,))
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        run()
        first_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        run()
        warm_s = time.perf_counter() - t0
        steps.append({"model": name, "load_ms": round(load_s * 1000), "first_ms": round(first_s * 1000),
                      "warm_ms": round(warm_s * 1000), "rss_mb": round(rss_mb()), "peak_so_far_mb": round(peak_mb()),
                      "resident_models": mp.loaded_models()})
    result = {
        "platform": sys.platform, "baseline_rss_mb": round(base), "baseline_peak_mb": round(base_peak),
        "peak_mb": round(peak_mb()), "max_resident_models": max(len(s["resident_models"]) for s in steps),
        "torch_imported": any(m.split(".")[0] in ("torch", "transformers") for m in sys.modules), "steps": steps,
    }
    if a.json:
        print(json.dumps(result))
    else:
        print(f"baseline after importing api.main (FastAPI+Supabase client+ONNX runtime): rss {result['baseline_rss_mb']} MB")
        for s in steps:
            print(f"  -> {s['model']:18s} load {s['load_ms']:5d} ms | first {s['first_ms']:5d} ms | warm {s['warm_ms']:5d} ms"
                  f" | rss {s['rss_mb']:4d} MB | peak so far {s['peak_so_far_mb']:4d} MB | resident {s['resident_models']}")
        print(f"PEAK {result['peak_mb']} MB | max models resident at once: {result['max_resident_models']} | "
              f"torch/transformers imported: {result['torch_imported']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
