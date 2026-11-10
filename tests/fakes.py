"""Torch-free test doubles shared by the API tests: an in-memory stand-in for the Supabase layer."""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from api import db

SECRET = "sb-service-role-SECRET-123"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iso(delta_seconds: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=delta_seconds)).isoformat()


class FakeDb:
    """In-memory stand-in for the TrioFormer functions in api/db.py."""

    def __init__(self) -> None:
        self.inputs: set[str] = set()
        self.inferences: list[dict] = []
        self.fail_insert = self.fail_ping = self.fail_read = self.fail_refresh = False
        self.refresh_calls = 0
        self.refresh_result: dict = {"refreshed": True, "mode": "concurrent", "rows": 2}
        self.rows = [
            {"model_name": "erwanf/gpt2-mini", "n_inferences": 4, "mean_latency_ms": 120.5, "min_latency_ms": 90,
             "p50_latency_ms": 118.0, "p95_latency_ms": 150.0, "p99_latency_ms": 155.0,
             "max_latency_ms": 160, "last_served_at": iso(30)},
            {"model_name": "t5-small", "n_inferences": 2, "mean_latency_ms": 300.0, "min_latency_ms": 280,
             "p50_latency_ms": 300.0, "p95_latency_ms": 318.0, "p99_latency_ms": 319.0,
             "max_latency_ms": 320, "last_served_at": iso(10)},
        ]
        self.status = {"last_refreshed_at": iso(5), "last_duration_ms": 12, "last_row_count": 2,
                       "last_mode": "concurrent"}

    # --- functions patched into api.db
    def ping(self) -> bool:
        return not self.fail_ping

    def insert_inference(self, input_sha256, model_name, output, latency_ms):
        if self.fail_insert:
            raise db.DatabaseError("Database operation 'insert_inference' failed.")
        self.inputs.add(input_sha256)
        row = {"id": len(self.inferences) + 1, "input_sha256": input_sha256, "model_name": model_name,
               "output": output, "latency_ms": latency_ms, "served_at": datetime.now(timezone.utc).isoformat()}
        self.inferences.append(row)
        return row

    def get_benchmarks(self):
        if self.fail_read:
            raise db.DatabaseError("x")
        return self.rows

    def get_benchmark_status(self):
        if self.fail_read:
            raise db.DatabaseError("x")
        return dict(self.status) if self.status is not None else None

    def refresh_benchmarks(self):
        self.refresh_calls += 1
        if self.fail_refresh:
            raise db.DatabaseError("x")
        if self.refresh_result.get("refreshed"):
            self.status = {**(self.status or {}), "last_refreshed_at": iso(0), "last_mode": "concurrent",
                           "last_row_count": 2, "last_duration_ms": 9}
        return self.refresh_result
