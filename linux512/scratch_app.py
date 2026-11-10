"""Test wrapper: the REAL api.main app (torch backend) with swapped checkpoints, a mocked DB,
optional one-model-at-a-time eviction, and a /_mem probe. Repo files are not modified."""
import os, sys, gc, resource, itertools
import psutil
from api import model_pipelines as mp, db
mp.BERT_MODEL = os.environ["BERT"]; mp.GPT2_MODEL = os.environ["GPT"]; mp.T5_MODEL = os.environ["T5"]
if os.environ.get("SEQUENTIAL", "1") == "1":
    _orig = mp._load_bundle
    def load(name, kind):
        mp._CACHE.clear(); gc.collect()
        try:
            import ctypes; ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception: pass
        return _orig(name, kind)
    mp._load_bundle = load
_ids = itertools.count(1)
db.insert_inference = lambda **kw: {"id": next(_ids), "served_at": None}   # mocked Supabase
from api.main import app
@app.get("/_mem")
def _mem():
    return {"rss_mb": psutil.Process().memory_info().rss/2**20,
            "hwm_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, "loaded": mp.loaded_models()}
