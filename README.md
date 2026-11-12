# Former-It — Three-Cloud Template (From-Scratch Transformer Encoder)

> The same three-cloud architecture as the base template
> (**Streamlit UI ⇄ FastAPI Model API ⇄ Supabase Data**), with the middle box
> swapped for a **from-scratch tiny Transformer encoder** (positional encoding +
> multi-head self-attention) doing sequence classification with per-head
> attention heatmaps. The deploy steps are identical, so follow the main
> [three-cloud TUTORIAL](../Topic_1_three-cloud/TUTORIAL.md).

## Live deployment URLs (fill these in)

| Tier | Platform | URL |
|------|----------|-----|
| **UI** | Streamlit Community Cloud | `https://<your-app>.streamlit.app` |
| **API** | Render.com | `https://<your-api>.onrender.com` |
| **Data** | Supabase | `https://<your-project-ref>.supabase.co` |

> Replace the placeholders above with your real URLs once deployed.

---

## What it does

Former-It decides whether a short symbol sequence is a **palindrome** using a
Transformer encoder built from scratch, and — crucially — returns the
**per-layer, per-head self-attention matrices** so you can see *which positions
attend to which*. Because the palindrome rule compares position `i` with position
`n-1-i`, a self-attention head can learn the **anti-diagonal**, which the heatmaps
make visible. You pick the learning rate, batch size, and epochs; the API trains
with Adam + cross-entropy, reports held-out **accuracy / macro-F1**, and persists
every run. The UI lets you train, type or generate a sequence to classify, view
the attention heatmaps, review run history, and read the model card. Only the
sequence's **sha256 hash** is logged — never the symbols.

## What changed from the base template (the reusable pattern in action)

The three-cloud split and the file layout are identical to *Regress-It*. Only the
middle box changed:

| Aspect | Regress-It | Former-It |
|--------|----------------------|---------------------|
| Model | `nn.Linear(1,1)` + SGD | from-scratch Transformer encoder + Adam |
| Task | regression | palindrome vs random (binary) |
| Metrics | MSE / MAE / R² | accuracy / macro-F1 |
| Input | one scalar `x` | a sequence of symbol ids |
| Explainability | — | **per-head self-attention heatmaps** |
| Extra endpoints | — | `/predict_sample`, `/classes` |
| Privacy invariant | — | store sequence **hash**, never symbols |
| Tables | datasets · runs · predictions | datasets · runs · **run_artifacts** · sequence_metadata |

Everything else — UI as a thin client (no torch), API as the only writer,
Supabase as the single source of truth, service-role vs anon keys, RLS on
`runs`, the four test categories — is unchanged.

## Architecture

```
┌──────────────────────┐   HTTPS/JSON    ┌──────────────────────────┐   service-role   ┌──────────────────┐
│  Streamlit Cloud     │ ──────────────► │  FastAPI on Render        │ ───────────────► │  Supabase        │
│  (ui/app.py)         │   symbol seq    │  (api/main.py)            │   full access    │  Postgres        │
│  thin client, no ML  │                 │  Transformer encoder      │                  │  datasets/runs/  │
│  plots attn heatmaps │                 │  (from scratch, torch)    │                  │  run_artifacts/  │
│                      │ ◄────anon key,  │                          │                  │  sequence_       │
│                      │   read-only ────┼──────────────────────────┼──────────────────►│  metadata        │
└──────────────────────┘   SELECT runs   └──────────────────────────┘                  │  (RLS: anon can  │
                                                                                        │   only SELECT    │
                                                                                        │   runs)          │
                                                                                        └──────────────────┘
```

## Project structure

```
former-it/
├── README.md                 # This file
├── MODEL_CARD.md             # Model details, attention, privacy, limitations
├── shared/
│   ├── schemas.py            # Pydantic API contract
│   └── data.py               # Synthetic palindrome sequence generator + class contract
├── api/                      # FastAPI tier (deploys to Render)
│   ├── main.py               # Endpoints
│   ├── model_pipelines.py    # Pretrained BERT / GPT-2 / T5 inference (one model resident at a time)
│   ├── encoder_demo.py       # Educational attention heatmaps from our own encoder
│   ├── training.py           # From-scratch Transformer encoder + artifact (de)serialization
│   ├── db.py                 # Supabase (service-role) data access
│   ├── configs/default.yaml
│   └── requirements.txt
├── ui/
│   ├── app.py                # 5-tab thin client (type / generate + attention heatmaps)
│   ├── requirements.txt      # No torch (matplotlib only, for heatmaps)
│   └── .streamlit/secrets.toml.example
├── db/
│   ├── migrations/001_init.sql
│   └── seed.py
├── src/transformer.py        # The from-scratch Transformer encoder
├── benchmarks/               # Custom encoder vs torch.nn encoder benchmark
├── scripts/
│   ├── download_models.py    # Fetch the 3 checkpoints from the Hugging Face Hub (Render build step)
│   └── linux_check/          # Render-like 512 MB Linux memory check (run by .github/workflows)
├── tests/                    # pytest suite
├── render.yaml               # Render blueprint
├── requirements-dev.txt
└── .env.example
```

## Quickstart (local)

```bash
cd former-it

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

pytest -q            # tests pass; the live-Supabase test skips without creds.
                     # (the first `pip install` downloads CPU PyTorch, ~200 MB; real-model smoke test is opt-in:
                     #  RUN_HF_SMOKE_DOWNLOAD=1 pytest tests/test_model_pipelines_smoke.py -s)

cp .env.example .env                                   # API: SUPABASE_URL + SERVICE key
cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml

uvicorn api.main:app --reload --port 8000              # terminal 1
streamlit run ui/app.py                                # terminal 2
```

To deploy to the three clouds, follow **Part E** of the main
[TUTORIAL](../Topic_1_three-cloud/TUTORIAL.md): apply
`db/migrations/001_init.sql` in the Supabase SQL Editor → deploy the API from
`render.yaml` on Render → deploy the UI on Streamlit Community Cloud.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/datasets` | Register a synthetic sequence dataset (params only) |
| `POST` | `/train` | Train the Transformer, persist run + model artifact, return metrics |
| `GET`  | `/runs/{run_id}` | Fetch one run |
| `GET`  | `/runs` | List recent runs |
| `POST` | `/predict` | Classify a symbol sequence; return per-head attention; log the hash |
| `POST` | `/predict_sample` | Generate + classify a sample; return attention |
| `GET`  | `/classes` | The class labels the model predicts over |
| `GET`  | `/healthz` | Liveness / DB ping / resident model |
| `GET`  | `/version` | Build SHA + torch / transformers version + checkpoints |

### TrioFormer endpoints (pretrained BERT, GPT-2 and T5 + our own encoder)

| Method | Path | Model | Purpose |
|--------|------|-------|---------|
| `POST` | `/bert_fillmask` | `prajjwal1/bert-small` | Masked-token prediction for one `[MASK]` |
| `POST` | `/gpt2_continue` | `erwanf/gpt2-mini` | Greedy text continuation |
| `POST` | `/t5_summarize` | `t5-small` | Abstractive summarization (`summarize:` prefix, beam search) |
| `POST` | `/encoder_attention` | our `src/transformer.py` (untrained) | Educational per-head attention heatmap |
| `GET`  | `/benchmarks` | – | Per-model latency statistics from Supabase |

Every pretrained call logs only the SHA-256 of the input, the model name, the output and the server-side
latency to Supabase (`inputs` / `inferences`). The models are loaded with plain PyTorch + Hugging Face
Transformers (no ONNX, no quantization, no custom export) and are **deliberately small** so the whole
service fits Render Free:

| Task | Checkpoint | Params | Size (fp32) | Context |
|------|------------|--------|-------------|---------|
| BERT masked-token | `prajjwal1/bert-small` | 28.8 M | 115 MB | 512 |
| GPT-2 continuation | `erwanf/gpt2-mini` | 38.6 M | 154 MB | 512 |
| T5 summarization | `t5-small` | 60.5 M | 242 MB | 512 in |

They are real pretrained checkpoints with working heads (a masked-LM head for BERT, the GPT-2 LM head, and
T5's summarization pre-training), but far smaller than `bert-base` / `gpt2` / `t5-base`, so expect weaker
quality: BERT ranks sensible words with modest confidence, GPT-2 drifts and repeats after a sentence or
two, and T5 mostly copies and lower-cases sentences. Example outputs are in `MODEL_CARD.md`.

### Memory and the Render Free limits (512 MB RAM, 0.1 CPU)

* Only **one** pretrained model is resident at a time: asking for a different model evicts the current one
  (`api/model_pipelines.py::_get_bundle`). Switching models is therefore a *cold start*
  (`cold_start: true`; the load time is excluded from `latency_ms`).
* Measured on Linux (GitHub Actions, Docker `--memory=512m --memory-swap=512m`, real FastAPI app, Supabase
  mocked, 57 requests switching bert→gpt→t5 repeatedly): **no OOM kills, 0 failed requests**, ~320 MB of
  non-reclaimable memory. The OOM-kill threshold was between 352 and 384 MB, leaving roughly 130 MB of
  headroom at 512 MB. Re-run it any time: *Actions → PyTorch Linux check → Run workflow*.
* Expect **slow** responses at 0.1 CPU (measured: GPT-2 ≈ 6 s, T5 ≈ 12–16 s warm) and a slower first request
  after Render's 15-minute idle spin-down.
* Do not add `accelerate` to the requirements (it costs ~80 MB of RAM), and keep `--workers 1`.

### Deploying the API to Render Free

`render.yaml` is a ready blueprint: build =
`pip install -r api/requirements.txt && python -m scripts.download_models` (CPU-only torch from the PyTorch index, weights from the Hugging Face Hub into
`HF_HOME`). Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` as secrets, then apply
`db/migrations/001_init.sql` and `002_trioformer.sql` in Supabase.

## Checklist

- [ ] Three live URLs listed at the top of this README
- [ ] `datasets`, `runs`, `run_artifacts`, `sequence_metadata` tables with RLS
- [ ] 6+ API endpoints
- [ ] 5 Streamlit tabs (Concepts, Train, Predict, Run History, Model Card)
- [ ] Transformer training with held-out accuracy / macro-F1 + per-head attention
- [ ] pytest suite passing
- [ ] `MODEL_CARD.md` completed
