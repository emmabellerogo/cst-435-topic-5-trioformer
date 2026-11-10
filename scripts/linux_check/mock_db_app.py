"""The REAL api.main app with only the Supabase insert mocked, plus a /_mem probe (for the memory check)."""
import itertools
import resource

import psutil

from api import db

_ids = itertools.count(1)
db.insert_inference = lambda **kw: {"id": next(_ids), "served_at": None}   # mocked Supabase write
db.ping = lambda: True

from api import model_pipelines as mp  # noqa: E402
from api.main import app  # noqa: E402


@app.get("/_mem")
def _mem():
    return {"rss_mb": psutil.Process().memory_info().rss / 2**20,
            "hwm_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "loaded": mp.loaded_models()}
