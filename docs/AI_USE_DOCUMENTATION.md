# AI Use Documentation

**Project:** TrioFormer (CST-435, Topic 5, Product 5)
**Student:** Emma Rogoveanu, sole student contributor
**AI tool:** Claude (Anthropic), used as a coding assistant through Claude Code in my editor

## 1. Summary

I used Claude throughout this project. It generated and revised a substantial share of the code, and it helped with
debugging, writing and running tests, running the Linux memory experiments, and drafting documentation. I did not
write every line of code myself.

I set the requirements, approved or rejected Claude's plans, made the architecture and deployment decisions, ran
the cloud setup and live tests, and am responsible for the submitted work. Claude did not make project decisions or
act on its own. Most prompts below limit the scope of a step and end with "do not commit, push, or deploy."

**How to read the evidence.** Every commit is under my Git identity, so authorship alone does not show what Claude
produced. Two commits carry a `Co-Authored-By: Claude` trailer (`8c9f487`, `1c61e25`). The other commits do not, but
the AI assistance described here applies to the project as a whole. Prompt excerpts are verbatim from my local
Claude Code session logs. Their timestamps are UTC, while commit dates are local, so the same evening can fall on
different dates.

## 2. Evidence available and evidence missing

| Source | Status |
|---|---|
| Git history (13 commits, 2026-11-04 to 2026-11-12) | Reviewed |
| Local Claude Code session logs for this repository (3 sessions with prompts, 2026-11-05 to 2026-11-12) | Reviewed. Only my prompts are quoted here. |
| Claude chats outside Claude Code (claude.ai web or app) | **Not accessible to me. None are cited.** |
| Sessions that fall in the gaps listed in Section 11 | **Not found in the available logs** |

I did not reconstruct any conversation. Where a phase has no prompt excerpt, the table says so.

## 3. Architecture

| | |
|---|---|
| **What I directed or decided** | Adapted the course template into TrioFormer. Asked for a plan before any edit, then approved the database design with four decisions: SHA-256 prompt hashes, no anonymous access to benchmarks, refresh time tracked separately from the materialized view, and one additive migration (`002_trioformer.sql`) that leaves `001_init.sql` untouched. Set the constraint of free tiers only (Render, Supabase, Streamlit). Chose the final design of small pretrained checkpoints on plain PyTorch, one model resident at a time, and dropped the ONNX/quantization route. |
| **What Claude assisted with** | Proposed the schema, constraints, indexes and privacy trade-offs for my review. Researched and tested candidate lightweight approaches and small checkpoints, and reported why some were rejected (for example, a model with random weights and a model that produced empty summaries). |
| **How I reviewed or verified** | Gave approval or revisions before each step. Applied the migration to my live Supabase project myself. Accepted the PyTorch design only after the Linux memory test passed (Section 7). |
| **Evidence** | Session `7e7378ca`, 2026-11-05 04:24 UTC: *"Do not run migrations, change files, connect to Supabase, or commit anything. Give me the proposed schema and implementation plan for approval."* Then, 04:26 UTC: *"I approve the TrioFormer database design with these decisions: …"* Session `57d8c0d7`, 2026-11-08 06:34 UTC: *"I do not want to use Hugging Face Spaces, ZeroGPU, a paid hosting plan, or an external model-serving service."* Commits `5c27fe4`, `8c9f487`. |

## 4. Implementation

| | |
|---|---|
| **What I directed or decided** | Scope of each step (custom encoder, pretrained pipelines, seven endpoints, Supabase logging). Required behavior: save the inference record before returning success, return 422 for invalid input and 503 for model or database failures, never store raw prompts, and reuse the existing pipelines rather than duplicating inference logic. |
| **What Claude assisted with** | Wrote and revised the code for `src/transformer.py`, `api/model_pipelines.py`, `api/main.py`, `api/db.py`, `shared/schemas.py`, the migration SQL, and the Linux-check scripts and workflow. This is substantial AI-generated code that I directed and reviewed. |
| **How I reviewed or verified** | Reviewed diffs and test output before each commit. Asked Claude to check diffs for accidental changes, secrets and missing requirements. Ran the live endpoint tests on Render myself. |
| **Evidence** | Session `57d8c0d7`, 2026-11-08 05:59 UTC (endpoint step): *"Reuse the existing inference pipelines. Do not duplicate model inference logic."* and *"Ensure the inference record is saved to Supabase **before** reporting a successful API response."* Commits `feda9a1`, `dc5b627`, `bc28d29`. The prompts that requested the encoder and the pipelines are not in the available logs (Section 11). |

## 5. Tests

| | |
|---|---|
| **What I directed or decided** | Required test areas: all seven endpoints, request validation, SHA-256 consistency, Supabase outage, model-load failure, attention shapes and normalization, health checks, and no real model downloads in ordinary unit tests (mocks only). Required parameter-mapping and output-equivalence tests for the benchmark. |
| **What Claude assisted with** | Wrote the tests (`tests/`) and ran the suite. Updated tests when the PyTorch simplification removed the ONNX backend. |
| **How I reviewed or verified** | Final recorded result: 133 passed, 1 skipped (the skipped test needs live Supabase credentials), as stated in the README and in the `8c9f487` commit message. I did not re-run the suite while writing this document. Run `pytest -q` to reproduce it. |
| **Evidence** | Session `57d8c0d7`, 2026-11-08 05:59 UTC, "Part 3 — Automated Testing". Commits `feda9a1`, `e2629c0`, `dc5b627`, `bc28d29`, `8c9f487`. |

## 6. Benchmarking

| | |
|---|---|
| **What I directed or decided** | The fairness criteria for comparing the custom encoder with `torch.nn.TransformerEncoderLayer`: matched dimensions, dropout 0, pre-norm, copied weights (including PyTorch's fused QKV projection), identical seeds and data, a numerical-equivalence check before training, and warmup and repetition in timing. Required the report to state that single-step runtime does not represent BERT, GPT-2 or T5 inference latency. |
| **What Claude assisted with** | Wrote the benchmark script, tests, loss plot and CSV, and ran the benchmark. Drafted the Linux memory-test workflow I authorized. |
| **How I reviewed or verified** | Reviewed the reported loss and timing results (about 4.8 × 10⁻⁷ output difference before training; about 1.88× slower for the small configuration). Collected the live latencies (BERT 1,809 ms, GPT-2 3,002 ms, T5 7,021 ms) from my own requests to Render, one per model. The README states that these are single observations, not distributions. |
| **Evidence** | Session `57d8c0d7`, 2026-11-08 05:17 UTC: *"Match d_model, num_heads, feed-forward width, dropout=0, activation, layer normalization epsilon, pre-norm behavior (norm_first=True), and batch_first=True."* and *"Before training, verify the two blocks produce numerically comparable outputs on the same input."* Files in `benchmarks/results/`. Commit `e2629c0`. |

## 7. Debugging and memory optimization

| | |
|---|---|
| **What I directed or decided** | Set the hard constraint (Render Free: 512 MB RAM, 0.1 CPU, no cost). Required Claude to stop and report if the Linux test failed rather than change more. Decided to abandon ONNX after the extra complexity, and approved the one-model-at-a-time design after the Linux test. |
| **What Claude assisted with** | Measured memory on macOS, then designed a Docker/cgroup test that simulates the Render limits. Found that macOS memory figures over-reported and that page cache distorts cgroup readings, so the test drops caches and reports non-reclaimable (anonymous) memory. Implemented model eviction with garbage collection and `malloc_trim`. Diagnosed a GitHub Actions dispatch problem (a manually triggered workflow must exist on the default branch). |
| **How I reviewed or verified** | Authorized the temporary test branch and the Actions run. Reviewed the result: 57 sequential requests across all three models, 0 failures, 0 OOM kills, about 319 MB non-reclaimable memory. This is a simulation on GitHub Actions, not a Render measurement. The live Render tests were then the real check. |
| **Evidence** | Session `74129e41`, 2026-11-10 04:41 UTC: *"Do not change the main branch, deploy to Render, modify Supabase, or remove any ONNX code. Use mocked database calls. You have permission to push the temporary test branch and run GitHub Actions, but not to merge it into main."* And 07:13 UTC: *"If the Linux test fails, stop and report the issue before changing anything else."* Commits `10b51bc`, `8c9f487`, `1c61e25`. |

## 8. Deployment

| | |
|---|---|
| **What I directed or decided** | Created the Render service from the repository Blueprint, entered the Supabase secrets, and ran the deployment myself. Told Claude not to create or deploy the service or expose any secrets. Diagnosed and fixed the first failed build by changing the build command to `pip install -r api/requirements.txt && python -m scripts.download_models`. Confirmed in the Supabase dashboard that RLS is enabled on all seven public tables. |
| **What Claude assisted with** | At my instruction, merged the validated branch into `main`, pushed it, and reviewed `render.yaml`, the requirements and the required environment variables. Wrote step-by-step Render instructions. Afterward updated `render.yaml` to match the build command I had fixed in the Render dashboard. |
| **How I reviewed or verified** | Ran live tests of BERT, GPT-2, T5, the custom encoder endpoint and `/benchmarks`. Checked that inference records appeared in Supabase. Reviewed and committed later documentation changes myself through GitHub Desktop. |
| **Evidence** | Session `74129e41`, 2026-11-12 16:47 UTC: *"Do not create or deploy the Render service, modify Supabase, or expose any secrets yet. I want to complete the Render setup myself with your guidance."* And 21:54 UTC: *"I fixed the Render build command manually … Please update `render.yaml` in the repository so its build command matches the working Render configuration. … I will review, commit, and push the change myself through GitHub Desktop."* Commit `00e3c92`. |

## 9. Documentation

| | |
|---|---|
| **What I directed or decided** | Chose the content, scope and rubric targets. Supplied the engineering report text. Corrected scope errors (Streamlit belongs to Product 5B, not Product 5). Set the rules for what may be claimed: no invented results, a checklist item marked complete only with evidence, and the presentation video marked unfinished. Added the live URLs and my RLS verification. |
| **What Claude assisted with** | Drafted and revised README sections, `MODEL_CARD.md` updates and `docs/INDIVIDUAL_CONTRIBUTION.md` from my instructions. Audited the README and report against the Product 5 rubric and reported gaps. Integrated the report into the README. |
| **How I reviewed or verified** | Read each change before committing it. Several prompts require Claude to report exactly what changed and what still needs verification. |
| **Evidence** | Session `74129e41`, 2026-11-12 21:58 UTC: *"Preserve the technical details and results exactly. Do not invent additional benchmark results or deployment measurements."* And 23:02 UTC: *"Clarify that RLS is enabled but a full security-policy audit has not been performed."* Commits `a71af04`, `52f3729`. |

## 10. Responsibility

I am responsible for the submitted code, results and documentation, including the parts Claude drafted. AI output
was not accepted without review, but review does not make me the author of every line, and I do not claim that.

## 11. Evidence I still need to supply or confirm

The available logs do not cover these items. Add the transcript excerpt or a short note for each, or leave the gap visible.

1. **Custom encoder and its tests** (`feda9a1`). The log shows the result (31 passing tests) but not the request that produced it.
2. **Pretrained pipelines** (`dc5b627`), including the original choice of `bert-base-uncased`, `gpt2` and `t5-small`.
3. **First Claude Code session on the Supabase migration**: the logs start on 2026-11-05 and cover only the planning step.
4. **ONNX backend debugging** (`10b51bc`). Only the approval prompt is available.
5. **The PyTorch simplification prompts** between 2026-11-10 04:41 and 07:13 UTC, including the feasibility investigation and the request to implement the change.
6. **Engineering report drafting.** The log shows that I supplied the finished text, but not how it was written. State whether any other tool (for example another chatbot or grammar tool) helped draft it. If one did, add it to this document.
7. **Prompt drafting.** Some of my prompts are long and structured. If any were drafted with another AI tool, say so.
8. **Any claude.ai or other AI chats** about this project, which I cannot retrieve.
9. **Other machines or accounts** that may hold Claude Code sessions for this repository.
