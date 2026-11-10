"""Run inside a memory-limited container. Starts uvicorn, drives the scenario, reports peaks + OOM kills."""
import sys, os, time, subprocess, json, statistics, httpx
case, only, rounds = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
cfg = json.loads(os.environ["CASE_MODELS"])
env = dict(os.environ, BERT=cfg["bert"], GPT=cfg["gpt"], T5=cfg["t5"])
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "linux512.scratch_app:app", "--port", "8765", "--log-level", "warning"], env=env)
def cg(name):
    for p in (f"/sys/fs/cgroup/{name}",):
        try: return open(p).read()
        except Exception: pass
    return None
def peak_mb():
    v = cg("memory.peak") or cg("memory/memory.max_usage_in_bytes"); return int(v)/2**20 if v else None
def oom():
    ev = cg("memory.events") or ""
    return {l.split()[0]: int(l.split()[1]) for l in ev.splitlines() if l}
c = httpx.Client(base_url="http://127.0.0.1:8765", timeout=600)
for _ in range(200):
    try: c.get("/healthz"); break
    except Exception: time.sleep(0.5)
ART = ("The Apollo 11 mission landed the first humans on the Moon on July 20, 1969. Commander Neil Armstrong and lunar module pilot Buzz Aldrin spent about two and a half hours outside the spacecraft, while Michael Collins orbited above in the command module. The crew returned safely to Earth on July 24 after collecting 47.5 pounds of lunar material.")
reqs = {"bert": ("/bert_fillmask", {"text": "The capital of France is [MASK].", "top_k": 3}),
        "gpt": ("/gpt2_continue", {"prompt": "The scientist opened the old laboratory door and", "max_new_tokens": 40}),
        "t5": ("/t5_summarize", {"text": ART, "max_new_tokens": 60})}
res = {"case": case, "only": only, "ok": 0, "failed": 0, "server_died": False, "first_error": None, "lat": {k: [] for k in only}, "outputs": {}, "trace": []}
idle = None
try: idle = c.get("/_mem").json()["rss_mb"]
except Exception: pass
res["idle_rss_mb"] = idle
seq = [k for _ in range(rounds) for k in only] + [k for _ in range(15) for k in only]
for i, k in enumerate(seq):
    path, body = reqs[k]
    try:
        r = c.post(path, json=body)
        if r.status_code != 200: raise RuntimeError(f"HTTP {r.status_code} {r.text[:200]}")
        j = r.json(); res["ok"] += 1; res["lat"][k].append(j["latency_ms"]); res["outputs"][k] = {"model": j["model"], "output_text": j["output_text"]}
        if i < 3 * len(only) or i % 12 == 0:
            m = c.get("/_mem").json(); res["trace"].append([i, k, j["cold_start"], j["latency_ms"], round(m["rss_mb"]), round(m["hwm_mb"]), round(peak_mb() or 0)])
    except Exception as e:
        res["failed"] += 1; res["first_error"] = res["first_error"] or f"{type(e).__name__}: {e}"[:300]
        if srv.poll() is not None: res["server_died"] = True; res["server_exit_code"] = srv.returncode; break
try: m = c.get("/_mem").json(); res["final_rss_mb"] = round(m["rss_mb"]); res["server_hwm_mb"] = round(m["hwm_mb"])
except Exception: pass
res["cgroup_peak_mb"] = round(peak_mb() or 0); res["cgroup_events"] = oom()
res["server_alive_at_end"] = srv.poll() is None
res["warm_median_ms"] = {k: (statistics.median(v[3:]) if len(v) > 3 else None) for k, v in res["lat"].items()}
res["lat"] = {k: [min(v), max(v)] if v else None for k, v in res["lat"].items()}
print("RESULT " + json.dumps(res)); srv.terminate()
