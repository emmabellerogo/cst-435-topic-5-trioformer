"""Run inside a memory-limited container: start uvicorn, drive a scenario over HTTP, report memory + OOM kills.

    python -m scripts.linux_check.run_case NAME bert,gpt,t5 ROUNDS

Prints one line starting with "RESULT " followed by JSON.
"""
import json
import os
import statistics
import subprocess
import sys
import time

import httpx

name, only, rounds = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "scripts.linux_check.mock_db_app:app",
                        "--port", "8765", "--log-level", "warning"])


def cg(path):
    try:
        return open("/sys/fs/cgroup/" + path).read()
    except OSError:
        return None


def peak_mb():
    v = cg("memory.peak") or cg("memory/memory.max_usage_in_bytes")
    return int(v) / 2**20 if v else 0


c = httpx.Client(base_url="http://127.0.0.1:8765", timeout=900)
for _ in range(240):
    try:
        c.get("/healthz")
        break
    except Exception:
        time.sleep(0.5)
ART = ("The Apollo 11 mission landed the first humans on the Moon on July 20, 1969. Commander Neil Armstrong and "
       "lunar module pilot Buzz Aldrin spent about two and a half hours outside the spacecraft, while Michael "
       "Collins orbited above in the command module. The crew returned safely to Earth on July 24 after "
       "collecting 47.5 pounds of lunar material.")
reqs = {"bert": ("/bert_fillmask", {"text": "The capital of France is [MASK].", "top_k": 3}),
        "gpt": ("/gpt2_continue", {"prompt": "The scientist opened the old laboratory door and", "max_new_tokens": 40}),
        "t5": ("/t5_summarize", {"text": ART, "max_new_tokens": 60})}
res = {"case": name, "ok": 0, "failed": 0, "server_died": False, "first_error": None,
       "lat": {k: [] for k in only}, "outputs": {}, "trace": []}
try:
    res["idle_rss_mb"] = round(c.get("/_mem").json()["rss_mb"])
except Exception:
    pass
seq = [k for _ in range(rounds) for k in only] + [k for _ in range(15) for k in only]
for i, k in enumerate(seq):
    path, body = reqs[k]
    try:
        r = c.post(path, json=body)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} {r.text[:200]}")
        j = r.json()
        res["ok"] += 1
        res["lat"][k].append(j["latency_ms"])
        res["outputs"][k] = {"model": j["model"], "output_text": j["output_text"]}
        if i < 3 * len(only) or i % 12 == 0:
            m = c.get("/_mem").json()
            res["trace"].append([i, k, j["cold_start"], j["latency_ms"], round(m["rss_mb"]), round(peak_mb())])
    except Exception as e:  # noqa: BLE001
        res["failed"] += 1
        res["first_error"] = res["first_error"] or f"{type(e).__name__}: {e}"[:300]
        if srv.poll() is not None:
            res["server_died"], res["server_exit_code"] = True, srv.returncode
            break
st = {l.split()[0]: int(l.split()[1]) for l in (cg("memory.stat") or "").splitlines() if l}
res.update(cgroup_end_anon_mb=round(st.get("anon", 0) / 2**20), cgroup_end_file_mb=round(st.get("file", 0) / 2**20),
           cgroup_peak_mb=round(peak_mb()),
           cgroup_events={l.split()[0]: int(l.split()[1]) for l in (cg("memory.events") or "").splitlines() if l},
           warm_median_ms={k: (statistics.median(v[3:]) if len(v) > 3 else None) for k, v in res["lat"].items()})
res["lat"] = {k: ([min(v), max(v)] if v else None) for k, v in res["lat"].items()}
print("RESULT " + json.dumps(res))
srv.terminate()
