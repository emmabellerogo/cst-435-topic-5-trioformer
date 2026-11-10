"""Download the three TrioFormer checkpoints into the Hugging Face cache (run at build time on Render).

    python scripts/download_models.py            # honours HF_HOME
    python scripts/download_models.py --list     # just print the checkpoints

Only the files from_pretrained() needs are fetched, and only ONE weights format per model
(safetensors if the repo has it, else pytorch_model.bin), so the cache is ~0.5 GB, not several GB.
"""
from __future__ import annotations

import sys

from huggingface_hub import list_repo_files, snapshot_download

from api import model_pipelines as mp

CHECKPOINTS = (mp.BERT_MODEL, mp.GPT2_MODEL, mp.T5_MODEL)
SMALL_FILES = ["*.json", "*.txt", "*.model"]


def weights_pattern(files: list[str]) -> str:
    return "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"


def main(argv: list[str]) -> int:
    if "--list" in argv:
        print("\n".join(CHECKPOINTS))
        return 0
    for name in CHECKPOINTS:
        pattern = weights_pattern(list_repo_files(name))
        path = snapshot_download(name, allow_patterns=SMALL_FILES + [pattern])
        print(f"{name}: {pattern} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
