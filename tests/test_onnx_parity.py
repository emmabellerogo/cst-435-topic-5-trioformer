"""ONNX backend vs the PyTorch reference (api/model_pipelines.py) on the REAL checkpoints. Opt-in.

Skipped automatically unless (a) the exported ONNX files are present, (b) torch + transformers are
installed and (c) the three checkpoints are in the local Hugging Face cache (nothing is downloaded):

    HF_HOME=<cache with bert-base-uncased, gpt2, t5-small> \
    TRIOFORMER_MODEL_DIR=<dir from scripts/export_onnx.py>  pytest tests/test_onnx_parity.py -s

Two kinds of check:
* quality bounds for the production int8 files (measured values are printed with ``-s``);
* EXACTNESS of the hand-written generation loops: with fp32 files
  (``TRIOFORMER_FP32_MODEL_DIR=<dir from export_onnx.py --precision fp32>``) the ONNX backend must give
  token-for-token the same BERT / GPT-2 / T5 results as PyTorch.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from api import onnx_pipelines as op

pytest.importorskip("torch")
pytest.importorskip("transformers")


def _cached(repo):
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


pytestmark = pytest.mark.skipif(
    not (all(_cached(r) for r in ("bert-base-uncased", "gpt2", "t5-small"))),
    reason="reference checkpoints not in the local Hugging Face cache")

BERT = ["The capital of France is [MASK].", "I love to eat [MASK] for breakfast.", "She opened the [MASK] and walked inside.",
        "The [MASK] is the largest planet in the solar system.", "He plays the [MASK] in a jazz band.",
        "Water boils at 100 degrees [MASK].", "The dog chased the [MASK] up the tree.", "My favorite color is [MASK].",
        "They traveled to [MASK] last summer.", "The doctor told him to take the [MASK] twice a day.",
        "She was born in the year [MASK].", "The [MASK] shines brightly during the day.",
        "He drove his [MASK] to work every morning.", "The teacher asked the students to open their [MASK].",
        "A triangle has three [MASK].", "The president gave a long [MASK] yesterday.", "We watched a [MASK] at the cinema.",
        "The cat sat on the [MASK].", "Paris is the [MASK] of France.", "He was very [MASK] after the long run.",
        "The [MASK] flew over the mountains.", "She wrote a letter to her [MASK].",
        "The company announced a new [MASK] this week.", "Rain is expected in the [MASK] tomorrow."]
GPT = ["The meaning of life is", "Once upon a time in a small village", "The best way to learn programming is to",
       "In 1969, humans first", "Machine learning models are"]
PASSAGE = ("The history of computing began with simple mechanical devices used for counting. Over the centuries, inventors designed "
           "machines that could add, subtract and eventually follow stored instructions. In the twentieth century, electronic "
           "computers transformed science, business and daily life, and the idea of machines that learn from data moved from "
           "fiction to practice. Today, neural networks power translation, search and speech recognition, although they still "
           "require large amounts of data and energy.")
T5 = ["The city council voted on Tuesday to approve a new bike lane network that will stretch across 40 miles of downtown streets. Supporters say the plan will reduce traffic and improve safety, while some business owners worry about losing parking. Construction is expected to begin next spring and finish within two years.",
      "Scientists announced a new battery design that charges in five minutes and lasts twice as long as current lithium-ion cells. The prototype was tested on 1,000 charge cycles with little loss of capacity. Commercial production could begin in three years if funding is secured.",
      "The museum will reopen on Saturday after a two-year renovation. The project added 20,000 square feet of gallery space and a new education center. Admission will be free for the first month.",
      "A local bakery has won a national award for its sourdough bread. The owner said the secret is a starter that is over 80 years old. Customers now line up before dawn to buy loaves.",
      "The football team won its third consecutive championship on Sunday, beating its rivals 24 to 17 in front of a sold-out crowd. The quarterback threw for 300 yards and two touchdowns. Fans celebrated downtown late into the night.",
      "Health officials reported a drop in flu cases for the second straight week. Hospitals said emergency room visits have returned to normal levels. Officials still urge residents to get vaccinated before winter.",
      "The airline announced it will add 15 new routes next year, including direct flights to three European cities. The expansion will create about 800 jobs. Ticket sales for the new routes begin in January.",
      "A new library opened in the east side of the city on Monday. The building has a rooftop garden, a children's reading room and free internet access. Residents said it was long overdue."]


@pytest.fixture(scope="module")
def ref():
    os.environ["TRIOFORMER_HF_LOCAL_ONLY"] = "1"
    from api import model_pipelines as ref_mod

    yield ref_mod
    ref_mod.clear_model_cache()
    os.environ.pop("TRIOFORMER_HF_LOCAL_ONLY", None)


@pytest.fixture
def onnx_dir(monkeypatch):
    """Use TRIOFORMER_MODEL_DIR (default <repo>/models) and fail clearly if it is incomplete."""
    if not op.backend_status()["ok"]:
        pytest.skip("production ONNX files not present (TRIOFORMER_MODEL_DIR / scripts/fetch_models.py)")
    op.clear_model_cache()
    yield
    op.clear_model_cache()


@pytest.fixture
def fp32_dir(monkeypatch):
    d = os.environ.get("TRIOFORMER_FP32_MODEL_DIR")
    if not d:
        pytest.skip("set TRIOFORMER_FP32_MODEL_DIR to the output of export_onnx.py --precision fp32")
    monkeypatch.setenv("TRIOFORMER_MODEL_DIR", d)
    assert op.backend_status()["ok"], f"incomplete fp32 export in {d}"
    op.clear_model_cache()
    yield
    op.clear_model_cache()


def rouge_l(a: str, b: str) -> float:
    x, y = a.lower().split(), b.lower().split()
    dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(len(x)):
        for j in range(len(y)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if x[i] == y[j] else max(dp[i][j + 1], dp[i + 1][j])
    l = dp[-1][-1]
    return 0.0 if l == 0 else 2 * (l / len(x)) * (l / len(y)) / ((l / len(x)) + (l / len(y)))


def gpt2_nll_and_top1(ids):
    """Teacher-forced next-token NLL and argmax under the ONNX GPT-2 (one cached step per token)."""
    bundle = op._get_bundle(op.GPT2_MODEL, "causal")
    session = bundle.sessions["model"]
    names = [i.name for i in session.get_inputs()][1:]
    past = {n: np.zeros((1, 12, 0, 64), np.float32) for n in names}
    nll, top = [], []
    for i in range(len(ids) - 1):
        out = session.run(None, {"input_ids": np.array([[ids[i]]], dtype=np.int64), **past})
        past = dict(zip(names, out[1:]))
        lp = op._log_softmax(np.asarray(out[0])[0].astype(np.float64))
        nll.append(-lp[ids[i + 1]])
        top.append(int(np.argmax(lp)))
    return np.array(nll), np.array(top)


# ------------------------------------------------------------------ production int8 files: quality bounds
def test_bert_int8_stays_close_to_pytorch(ref, onnx_dir):
    top1 = overlap = diff = 0.0
    for s in BERT:
        r, o = ref.bert_fill_mask(s, 5), op.bert_fill_mask(s, 5)
        rt, ot = [p.token for p in r.predictions], [p.token for p in o.predictions]
        top1 += rt[0] == ot[0]
        overlap += len(set(rt) & set(ot)) / 5
        diff += abs(r.predictions[0].score - o.predictions[0].score)
        assert o.input_tokens == r.input_tokens and o.model == r.model
    n = len(BERT)
    print(f"\nBERT  int8 vs torch: top-1 agreement {top1}/{n} ({100 * top1 / n:.0f}%), top-5 overlap {100 * overlap / n:.0f}%, "
          f"mean |top-1 prob diff| {diff / n:.3f}")
    assert top1 / n >= 0.60 and overlap / n >= 0.70 and diff / n <= 0.10


def test_gpt2_int8_stays_close_to_pytorch(ref, onnx_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    ids = tok(PASSAGE, return_tensors="pt")["input_ids"][:, :120]
    model = AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=True).eval()
    with torch.no_grad():
        lg = model(ids).logits[0, :-1].double()
    lp = torch.log_softmax(lg, -1)
    gold = ids[0, 1:]
    ref_nll = -lp[torch.arange(len(gold)), gold].mean().item()
    nll, top = gpt2_nll_and_top1(ids[0].tolist())
    ppl_ref, ppl_onnx = math.exp(ref_nll), math.exp(nll.mean())
    agree = float((top == lg.argmax(-1).numpy()).mean())
    print(f"\nGPT-2 int8 vs torch: perplexity {ppl_ref:.1f} -> {ppl_onnx:.1f} (x{ppl_onnx / ppl_ref:.2f}), next-token top-1 agreement {100 * agree:.0f}%")
    assert ppl_onnx / ppl_ref <= 1.8 and agree >= 0.5
    for p in GPT:                                             # generation: deterministic, new text only
        a = op.gpt2_continue(p, 30)
        assert a == op.gpt2_continue(p, 30) and a.output_text.strip() and p not in a.output_text
        assert 0 < a.generated_tokens <= 30 and a.prompt_tokens == ref.gpt2_continue(p, 30).prompt_tokens


def test_t5_int8_stays_close_to_pytorch(ref, onnx_dir):
    scores = []
    for article in T5:
        r, o = ref.t5_summarize(article, 40), op.t5_summarize(article, 40)
        assert o.output_text.strip() and len(o.output_text) < len(article) and o.input_tokens == r.input_tokens
        scores.append(rouge_l(r.output_text, o.output_text))
    print(f"\nT5    int8 vs torch: mean ROUGE-L {np.mean(scores):.2f} (min {min(scores):.2f}) over {len(T5)} articles")
    assert np.mean(scores) >= 0.80


# ------------------------------------------------------------------ fp32 files: the generation loops are EXACT
def test_fp32_bert_is_identical_to_pytorch(ref, fp32_dir):
    for s in BERT[:8]:
        r, o = ref.bert_fill_mask(s, 5), op.bert_fill_mask(s, 5)
        assert [p.token for p in r.predictions] == [p.token for p in o.predictions], s
        assert all(abs(a.score - b.score) < 1e-4 for a, b in zip(r.predictions, o.predictions))
        assert r.output_text == o.output_text and r.input_tokens == o.input_tokens


def test_fp32_gpt2_generation_is_identical_to_pytorch(ref, fp32_dir):
    for p in GPT:
        r, o = ref.gpt2_continue(p, 30), op.gpt2_continue(p, 30)
        assert (r.output_text, r.prompt_tokens, r.generated_tokens) == (o.output_text, o.prompt_tokens, o.generated_tokens), p


def test_fp32_t5_beam_search_is_identical_to_pytorch(ref, fp32_dir):
    for article in T5[:6]:
        for kw in ({"max_new_tokens": 40}, {"max_new_tokens": 60, "min_new_tokens": 12}, {"max_new_tokens": 8}):
            r, o = ref.t5_summarize(article, **kw), op.t5_summarize(article, **kw)
            assert (r.output_text, r.input_tokens, r.output_tokens) == (o.output_text, o.input_tokens, o.output_tokens), kw
