# Individual Contribution Evidence

**Project:** TrioFormer (CST-435, Topic 5, Product 5): a three-cloud transformer comparison service
(Streamlit ⇄ FastAPI on Render ⇄ Supabase) serving pretrained BERT, GPT-2 and T5, plus a from-scratch
transformer encoder.
**Contributor:** Emma Rogoveanu, sole student contributor.
**Repository:** https://github.com/emmabellerogo/cst-435-topic-5-trioformer
**Deployed API:** https://former-it-api.onrender.com (docs: https://former-it-api.onrender.com/docs)

## 1. How I worked (AI-assisted development)

I worked with Claude (Anthropic's AI coding assistant) in my editor. I want to be accurate about who did what.

* **Claude assisted with:** writing and revising code, debugging, writing and running tests, running the Linux
  memory experiments in GitHub Actions, and drafting documentation.
* **I did:** set the project direction and made the technical decisions (for example, dropping ONNX and
  quantization in favour of small pretrained checkpoints with plain PyTorch, and choosing the checkpoints), reviewed
  results and diffs, created and configured the Render service and its environment variables, fixed the Render
  build command after the first deployment failed, ran the live endpoint tests, reviewed and committed changes
  through GitHub Desktop, and completed the submission documentation, including the engineering report.
* **What I am not claiming:** I did not personally write every line of code. All commits are under my Git
  identity, so commit authorship does not show which parts Claude produced. Two commits carry a
  `Co-Authored-By: Claude` trailer; the others do not, but the AI-assisted workflow above applies to the project
  as a whole.

## 2. Product 5 tasks and status

| # | Task | Status | Evidence |
|---|---|---|---|
| 1 | SQL migration (inferences, inputs, benchmarks materialized view) | Done | [`db/migrations/002_trioformer.sql`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/db/migrations/002_trioformer.sql), commit [`5c27fe4`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/commit/5c27fe4) |
| 2 | Custom transformer implementation (attention, multi-head, positional encoding, encoder block) | Done | [`src/transformer.py`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/src/transformer.py), commit [`feda9a1`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/commit/feda9a1) |
| 3 | Unit tests | Done | [`tests/test_transformer.py`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/tests/test_transformer.py) (23 tests), [`tests/test_benchmark.py`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/tests/test_benchmark.py) (9 tests); full suite 133 passed, 1 skipped |
| 4 | Matched-initialization benchmark vs `torch.nn.TransformerEncoderLayer`, with single-step timing | Done | [`benchmarks/benchmark_encoder.py`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/benchmarks/benchmark_encoder.py), [`benchmark_results.json`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/benchmarks/results/benchmark_results.json), commit [`e2629c0`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/commit/e2629c0) |
| 5 | Training-loss curves | Done | [`loss_plot.png`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/benchmarks/results/loss_plot.png), [`loss_history.csv`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/benchmarks/results/loss_history.csv), shown in the README "Verification and Benchmark Evidence" section |
| 6 | Pretrained BERT / GPT-2 / T5 pipelines | Done | [`api/model_pipelines.py`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/api/model_pipelines.py), commits [`dc5b627`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/commit/dc5b627), [`8c9f487`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/commit/8c9f487) |
| 7 | Seven FastAPI endpoints (`/bert_fillmask`, `/gpt2_continue`, `/t5_summarize`, `/encoder_attention`, `/benchmarks`, `/healthz`, `/version`) | Done | [`api/main.py`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/api/main.py), commit [`bc28d29`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/commit/bc28d29), [live `/docs`](https://former-it-api.onrender.com/docs) |
| 8 | Supabase logging and `GET /benchmarks` | Done | [`api/db.py`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/api/db.py), commit [`bc28d29`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/commit/bc28d29); live inference rows and `/benchmarks` tested by me |
| 9 | Render deployment | Done | [`render.yaml`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/render.yaml), commit [`00e3c92`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/commit/00e3c92) (build-command fix), [live API](https://former-it-api.onrender.com) |
| 10 | Documentation (README, model card, derivation, benchmark evidence) | Done | [`README.md`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/README.md), [`MODEL_CARD.md`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/MODEL_CARD.md) |
| 11 | Engineering report | Done | "Engineering Report" section of the README |
| 12 | Individual contribution evidence | Done | This document |
| 13 | Individual presentation video | **Not done** | Not yet recorded |
| n/a | Streamlit UI deployment | **Out of scope for Product 5** | Assigned to Product 5B. The UI code in `ui/` is not part of this submission's deliverables and is not deployed |

## 3. Summary of my work

* Built the project from the course template and chose the TrioFormer scope: three pretrained models, a
  from-scratch encoder, latency benchmarking, and a three-cloud deployment.
* Directed the custom transformer implementation and its tests, and reviewed the matched-initialization
  comparison against PyTorch's encoder layer.
* Decided to move from ONNX/quantization to small pretrained checkpoints (`prajjwal1/bert-small`,
  `erwanf/gpt2-mini`, `t5-small`) with plain PyTorch, after Linux memory tests showed it fit Render Free's 512 MB.
* Approved the one-model-at-a-time loading design and reviewed the Linux test results before merging to `main`.
* Created the Render service from the repository Blueprint, entered the Supabase secrets, and fixed the failed
  first build by changing the build command to `python -m scripts.download_models`.
* Ran live tests of BERT, GPT-2, T5, the custom encoder endpoint, and `/benchmarks`, and recorded the
  latencies.
* Reviewed and committed changes through GitHub Desktop, and wrote and finalized the engineering report and
  submission documentation.

## 4. Verified results

| Result | Value | Where to check |
|---|---|---|
| Automated tests | **133 passed, 1 skipped** (the skipped test needs live Supabase credentials) | `pytest -q` on `main`; reproducible locally |
| Linux memory test (simulated Render limits) | 512 MB limit, 0.1 CPU, no swap: 57 sequential requests across all three models, 0 failures, 0 OOM kills, about 319 MB non-reclaimable memory | [GitHub Actions run 42773886388](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/actions/runs/42773886388), [workflow](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/.github/workflows/torch-linux-check.yml) |
| Live inference on Render Free with Supabase logging | BERT-small 1,809 ms, GPT-2-mini 3,002 ms, T5-small 7,021 ms; records stored in Supabase | My live endpoint tests; `GET /benchmarks` |
| Custom vs torch encoder | Outputs match to about 4.8 × 10⁻⁷ before training; about 1.88× slower (small config), about 1.03× (large config), measured on a laptop | [`benchmark_results.json`](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/blob/main/benchmarks/results/benchmark_results.json) |

**Limits of these results:** the live latencies are one request per model, not a distribution. The Linux test
simulates Render's limits on GitHub Actions and is not a Render measurement. The live Supabase test in the
test suite was skipped because it needs credentials.

## 5. Unfinished and out-of-scope items

* **Unfinished:** the individual presentation video has not been recorded.
* **Out of scope:** Streamlit UI deployment belongs to Product 5B, so it is not counted as unfinished Product 5
  work.
* **Not independently checked:** I have not re-verified the Supabase RLS policies in the live project, and the
  live latency table is one request per model.
