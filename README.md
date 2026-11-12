# TrioFormer — Transformer Comparison Service

> A three-cloud service (**Streamlit UI ⇄ FastAPI Model API ⇄ Supabase Data**) that serves three small
> pretrained transformers — BERT, GPT-2 and T5 — through plain PyTorch and Hugging Face Transformers, plus a
> **from-scratch tiny Transformer encoder** (positional encoding + multi-head self-attention) with per-head
> attention heatmaps. The from-scratch encoder was originally built as the "Former-It" palindrome classifier;
> its endpoints are kept as the legacy API below.

## Live deployment URLs

| Tier | Platform | URL |
|------|----------|-----|
| **UI** | Streamlit Community Cloud | Not yet deployed |
| **API (FastAPI backend)** | Render Free | https://former-it-api.onrender.com |
| **API documentation** | Render Free (`/docs`) | https://former-it-api.onrender.com/docs |
| **Data** | Supabase | Project URL not listed here (kept out of the README) |

> The Render service is deployed and live-tested. The Streamlit UI has not been deployed yet, so no UI URL is listed.

---

## What it does (legacy Former-It encoder)

The original Former-It part of the service decides whether a short symbol sequence is a **palindrome** using a
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

The Former-It encoder was built from an earlier three-cloud regression template (*Regress-It*). The
three-cloud split and file layout are the same; only the middle box changed:

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
│  thin client, no ML  │                 │  Pretrained BERT/GPT-2/T5 │                  │  datasets/runs/  │
│  plots attn heatmaps │                 │  + from-scratch encoder   │                  │  run_artifacts/  │
│                      │ ◄────anon key,  │                          │                  │  sequence_       │
│                      │   read-only ────┼──────────────────────────┼──────────────────►│  metadata        │
└──────────────────────┘   SELECT runs   └──────────────────────────┘                  │  (RLS: anon can  │
                                                                                        │   only SELECT    │
                                                                                        │   runs)          │
                                                                                        └──────────────────┘
```

## Project structure

```
cst-435-topic-5-trioformer/
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
└── requirements-dev.txt
```

## Quickstart (local)

```bash
cd cst-435-topic-5-trioformer

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

pytest -q            # tests pass; the live-Supabase test skips without creds.
                     # (the first `pip install` downloads CPU PyTorch, ~200 MB; real-model smoke test is opt-in:
                     #  RUN_HF_SMOKE_DOWNLOAD=1 pytest tests/test_model_pipelines_smoke.py -s)

# create .env in the repo root with SUPABASE_URL and SUPABASE_SERVICE_KEY (never commit it)
cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml

uvicorn api.main:app --reload --port 8000              # terminal 1
streamlit run ui/app.py                                # terminal 2
```

To deploy to the three clouds: apply `db/migrations/001_init.sql` and `002_trioformer.sql` in the Supabase
SQL Editor, deploy the API from `render.yaml` on Render (see *Deploying the API to Render Free* below), then
deploy `ui/app.py` on Streamlit Community Cloud with the secrets listed in
`ui/.streamlit/secrets.toml.example`.

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
* **Simulated-resource benchmark (not Render):** measured on Linux (GitHub Actions, Docker `--memory=512m --memory-swap=512m`, real FastAPI app, Supabase
  mocked, 57 requests switching bert→gpt→t5 repeatedly): **no OOM kills, 0 failed requests**, ~320 MB of
  non-reclaimable memory. The OOM-kill threshold was between 352 and 384 MB, leaving roughly 130 MB of
  headroom at 512 MB. Re-run it any time: *Actions → PyTorch Linux check → Run workflow*.
* **Simulated 0.1 CPU latency (not Render):** GPT-2 ≈ 6 s and T5 ≈ 12–16 s warm in the Docker test. These come from
  the GitHub Actions simulation, not from the live service.
* **Actual Render Free inference results:** one recorded live request per model (BERT-small 1,809 ms, GPT-2-mini
  3,002 ms, T5-small 7,021 ms); see the Engineering Report. Expect a slower first request after Render's
  15-minute idle spin-down.
* Do not add `accelerate` to the requirements (it costs ~80 MB of RAM), and keep `--workers 1`.

### Deploying the API to Render Free

`render.yaml` is a ready blueprint: build =
`pip install -r api/requirements.txt && python -m scripts.download_models` (CPU-only torch from the PyTorch index, weights from the Hugging Face Hub into
`HF_HOME`). Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` as secrets, then apply
`db/migrations/001_init.sql` and `002_trioformer.sql` in Supabase.

## Engineering Report

### 1. Project Overview

The purpose of TrioFormer was to understand how different transformer architectures work, compare their capabilities, and learn how to deploy them in a real application. The project focuses on three pretrained models: BERT, GPT-2, and T5. BERT uses bidirectional context to predict missing words, GPT-2 generates text by predicting the next token, and T5 uses an encoder-decoder architecture for tasks such as summarization.

Just as important, I learned that a model is only one part of a system. The application must also handle requests, return meaningful results, record performance, and run within available computing resources.

### 2. System Architecture and Implementation

TrioFormer uses a three-cloud architecture: a Streamlit frontend, a FastAPI backend hosted on Render, and a Supabase database. Streamlit provides the interface, FastAPI runs the models, and Supabase stores inference results and latency measurements. The backend has seven TrioFormer endpoints for inference, educational attention demonstrations, performance statistics, and service information. The three model endpoints perform masked-token prediction with BERT, text continuation with GPT-2, and summarization with T5.

The project also includes a custom transformer encoder written in PyTorch with scaled dot-product attention, multi-head attention, positional encoding, and encoder blocks. Scaled dot-product attention compares queries and keys to decide how strongly tokens relate. The scores are divided by the square root of the key dimension and passed through softmax to produce attention weights. Multi-head attention lets the model examine different relationships at once, and positional encoding provides token order, which attention does not represent on its own. Building these parts helped me understand transformers instead of treating pretrained models as black boxes.

### 3. Model Selection and Comparison

The final implementation uses `prajjwal1/bert-small`, `erwanf/gpt2-mini`, and `t5-small`, with approximately 28.8 million, 38.6 million, and 60.5 million parameters, respectively.

BERT-small correctly predicted "paris" for "The capital of France is [MASK]." GPT-2-mini generated readable continuations but sometimes repeated phrases. T5-small produced meaningful summaries, although they often stayed close to the original wording. Smaller pretrained models can still do useful language tasks, but there are tradeoffs between size, output quality, and resource needs. They suit an educational application as long as their limitations are communicated clearly.

### 4. Testing and Benchmarking

The custom transformer was compared against PyTorch's `TransformerEncoderLayer`. Tests examined attention weights, tensor shapes, masking, gradients, and output equivalence. Earlier benchmarks showed the custom version was approximately 1.88 times slower for a smaller configuration, with a much smaller difference for a larger one. The final test suite reported 133 passing tests and one skipped test that requires live Supabase credentials.

Linux testing used a strict 512 MB memory limit and a simulated 0.1 CPU allocation. The sequential model test completed 57 requests without failures or out-of-memory kills, and non-reclaimable anonymous memory was approximately 319 MB at the end. These are simulated-resource results, not Render measurements.

The backend was then deployed to Render Free. Live testing confirmed that all three models generated responses and that their inference records were stored in Supabase.

| Model | Live inference latency |
|---|---:|
| BERT-small | 1,809 ms |
| GPT-2-mini | 3,002 ms |
| T5-small | 7,021 ms |

These are single recorded requests per model and exclude model-loading time, queueing, database operations, and network transfer. More requests would be needed for reliable averages and percentiles. The custom encoder endpoint returned the expected attention shape of 4 × 6 × 6 for four heads and six tokens, with a maximum row-sum error of approximately 1.19 × 10⁻⁷, so the weights summed to approximately one.

### 5. Engineering Challenges and Decisions

The biggest challenge was running BERT, GPT-2, and T5 within Render Free's limited memory. The larger models needed more memory than the free tier could support. We investigated ONNX conversion and quantization, but they added complexity to the implementation and deployment.

We then evaluated smaller pretrained checkpoints with standard PyTorch and Transformers. Testing on macOS suggested memory would still be too high, but Linux testing showed the application could stay within the limit by loading one model at a time and removing the previous one. This showed me the importance of testing in an environment similar to the deployment platform, because macOS did not predict Linux memory behavior.

One-model-at-a-time loading simplifies deployment compared with ONNX, although switching models adds loading overhead and inference is slower on limited CPU, especially for T5. The response times make the service better for educational demonstrations than for high-volume production use.

### 6. Ethical Considerations and Christian Worldview

From a Christian worldview, I believe developers should use technology honestly and carefully. Luke 12:48 teaches that greater responsibility comes with greater trust. I think this applies to artificial intelligence because developers should understand the limitations of their systems and communicate them accurately.

For TrioFormer, this means being transparent about model performance, latency, and output quality. GPT-2 can produce repetitive or inaccurate text, and T5 may oversimplify. Users should understand that outputs are generated predictions, not guaranteed facts. Privacy matters too: TrioFormer logs SHA-256 hashes of inputs instead of raw prompts, which reduces exposure of user text, although generated outputs and operational records still need protection.

### 7. Conclusion

TrioFormer connected the theory behind transformers with the practical challenges of deploying an AI application. Implementing multi-head attention, comparing the three models, and measuring performance improved my understanding of how these architectures work.

The biggest lesson was that choosing a model involves more than its capabilities. Memory, processing time, deployment complexity, and hosting costs also shape engineering decisions. Deploying all three models on Render Free showed how these tradeoffs can be managed while keeping the application's educational functionality.

## Checklist

Only items that were actually completed and verified are checked.

- [ ] Three live URLs listed at the top of this README (API and docs are listed; the Streamlit UI is not deployed yet)
- [ ] `datasets`, `runs`, `run_artifacts`, `sequence_metadata` tables with RLS (migration is in `db/migrations/`; not re-verified in the live Supabase project)
- [x] 6+ API endpoints (the five TrioFormer endpoints plus `/healthz` and `/version`, covered by tests and live testing)
- [ ] 5 Streamlit tabs (Concepts, Train, Predict, Run History, Model Card) (implemented in `ui/app.py`; UI not deployed or live-tested)
- [x] Transformer training with held-out accuracy / macro-F1 + per-head attention (covered by `tests/test_training.py` and `tests/test_transformer.py`)
- [x] pytest suite passing (133 passed, 1 skipped: the live-Supabase test)
- [ ] `MODEL_CARD.md` completed (pending your final review)
