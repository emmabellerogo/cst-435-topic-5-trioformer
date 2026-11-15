import pandas as pd
import requests
import streamlit as st
import re
from time import perf_counter
from pathlib import Path
from streamlit.components.v1 import declare_component


def run_model(endpoint, payload, model_name):
    """Send a request to Render and return the model's response."""
    api_url = st.secrets.get("API_URL")
    if not api_url:
        st.error("Add API_URL to your Streamlit secrets.")
        return None

    try:
        with st.spinner(f"Waiting for {model_name}. Render may be loading the model."):
            response = requests.post(
                f"{api_url.rstrip('/')}/{endpoint}",
                json=payload,
                timeout=90,
            )
            response.raise_for_status()
            return response.json()
    except requests.Timeout:
        st.error(f"{model_name} took too long to respond. Wait a moment and try again.")
    except requests.RequestException:
        st.error(f"{model_name} couldn't return a result. Wait a moment and try again.")

    return None

compare_browser = declare_component(
    "compare_browser",
    path=str(Path(__file__).resolve().parent / "compare_browser"),
)

st.title("TrioDeck")

concepts_tab, encoder_tab, bert_tab, gpt_tab, t5_tab, compare_tab, history_tab = st.tabs(
    [
        "Concepts",
        "My Encoder Block",
        "BERT — Fill Mask",
        "GPT-2 — Continue Text",
        "T5 — Summarize",
        "Compare",
        "Latency History",
    ]
)

with bert_tab:
    st.header("BERT — Fill Mask")
    st.markdown("**When to use this archetype**")
    st.write(
        "Use BERT to fill in a blank using clues on both sides, "
        "like in a vocabulary exercise."
    )

    text = st.text_input(
        "Enter a sentence with one [MASK]",
        "The capital of France is [MASK].",
    )

    if st.button("Run BERT"):
        if text.count("[MASK]") != 1:
            st.warning("Include exactly one [MASK] in your sentence.")
        else:
            result = run_model("bert_fillmask", {"text": text, "top_k": 5}, "BERT")
            if result is not None:
                st.write(result["output_text"])
                st.table(result["predictions"])
                st.metric("Server-side inference", f"{result['latency_ms']} ms")

with gpt_tab:
    st.header("GPT-2 — Continue Text")
    st.markdown("**When to use this archetype**")
    st.write("Use GPT-2 to draft what comes next when you already have a starting line.")

    prompt = st.text_area(
        "Enter the beginning of a passage",
        "The students planted tomatoes in the school garden. The next morning,",
    )

    if st.button("Run GPT-2"):
        if not prompt.strip():
            st.warning("Enter a prompt first.")
        else:
            result = run_model(
                "gpt2_continue",
                {"prompt": prompt, "max_new_tokens": 20},
                "GPT-2",
            )
            if result is not None:
                st.write(result["output_text"])
                st.metric("Server-side inference", f"{result['latency_ms']} ms")

with t5_tab:
    st.header("T5 — Summarize")
    st.markdown("**When to use this archetype**")
    st.write("Use T5 to turn a longer passage into a short summary you can review.")

    passage = st.text_area(
        "Enter a passage to summarize",
        "The city council approved a new bike lane network on Tuesday. "
        "The plan covers 40 miles of downtown streets. "
        "Construction will begin next spring and will be completed in phases.",
    )

    if st.button("Run T5"):
        if not passage.strip():
            st.warning("Enter a passage first.")
        else:
            result = run_model(
                "t5_summarize",
                {"text": passage, "max_new_tokens": 24},
                "T5",
            )
            if result is not None:
                st.write(result["output_text"])
                st.metric("Server-side inference", f"{result['latency_ms']} ms")


with encoder_tab:
    st.header("My Encoder Block")
    st.markdown("**When to use this archetype**")
    st.write(
        "Use this demo to see how an encoder combines information from the "
        "words in a sentence. Each row shows where one word puts its attention."
    )

    sentence = st.text_input("Enter a short sentence", "the cat sat on the mat")
    use_positions = st.checkbox("Use positional encoding", value=True)

    if st.button("Show attention"):
        if not 1 <= len(sentence.split()) <= 32:
            st.warning("Enter between 1 and 32 words.")
        else:
            result = run_model(
                "encoder_attention",
                {
                    "text": sentence,
                    "num_heads": 4,
                    "use_positional_encoding": use_positions,
                },
                "the custom encoder",
            )

            if result is not None:
                labels = [
                    f"{index}. {word}"
                    for index, word in enumerate(result["tokens"], start=1)
                ]
                head = 1
                cells = [
                    {"word": labels[row], "looks_at": labels[column], "weight": weight}
                    for row, weights in enumerate(result["attention"][head - 1])
                    for column, weight in enumerate(weights)
                ]
                st.vega_lite_chart(
                    pd.DataFrame(cells),
                    {
                        "mark": "rect",
                        "encoding": {
                            "x": {
                                "field": "looks_at", "type": "ordinal",
                                "sort": labels, "title": "Word receiving attention",
                            },
                            "y": {
                                "field": "word", "type": "ordinal",
                                "sort": labels, "title": "Word being processed",
                            },
                            "color": {
                                "field": "weight", "type": "quantitative",
                                "scale": {"domain": [0, 1]}, "title": "Attention weight",
                            },
                            "tooltip": [
                                {"field": "word", "type": "nominal"},
                                {"field": "looks_at", "type": "nominal"},
                                {"field": "weight", "type": "quantitative", "format": ".3f"},
                            ],
                        },
                    },
                    use_container_width=True,
                )

with concepts_tab:
    st.header("Concepts")
    lesson, guide = st.columns([3, 1])

    with guide:
        st.subheader("When to use this archetype")
        st.write(
            "Use an encoder when a task depends on context on both sides of a word, "
            "like choosing a missing word in a vocabulary exercise."
        )

    with lesson:
        st.subheader("Multi-head attention")
        st.write(
            "Start with T tokens, each represented by a vector of width D. "
            "For each attention head, the input is projected into queries, keys, and values."
        )
        st.latex(r"Q_h=XW_h^Q,\quad K_h=XW_h^K,\quad V_h=XW_h^V")
        st.write(
            "Each projection has shape T × d, where d = D/H and H is the "
            "number of heads. Comparing queries with keys gives a T × T "
            "score table. Scaling keeps the scores from growing with head width, "
            "and softmax turns each row into weights that sum to one."
        )
        st.latex(
            r"A_h=\operatorname{softmax}\left(\frac{Q_hK_h^\top}{\sqrt{d}}\right),"
            r"\qquad Z_h=A_hV_h"
        )
        st.write(
            "Each head returns a T × d output. The heads run in parallel. "
            "Joining them gives a T × D output, and a final projection mixes the heads."
        )
        st.latex(r"\operatorname{MHA}(X)=\operatorname{Concat}(Z_1,\ldots,Z_H)W^O")

        st.subheader("How transformers represent word order")
        st.table([
            {"Method": "Sinusoidal", "How it works": "Adds a fixed wave pattern for each position.",
             "Trade-off": "Uses no extra learned parameters. TrioDeck’s encoder demo uses this."},
            {"Method": "Learned", "How it works": "Learns a separate vector for each position.",
             "Trade-off": "Adapts during training and is usually tied to the trained position range."},
            {"Method": "Rotary", "How it works": "Rotates queries and keys based on their positions.",
             "Trade-off": "Makes their attention score depend on relative position, a reason many modern LLMs use it."},
        ])

with compare_tab:
    st.header("Compare")
    main, guide = st.columns([3, 1])

    with guide:
        st.subheader("When to use each archetype")
        st.write(
            "Choose BERT to fill a blank, GPT-2 to continue a draft, "
            "and T5 to shorten a passage."
        )

    with main:
        passage = st.text_area(
            "Enter one passage for all three models",
            "The school opened a garden this spring. "
            "Students built raised beds and planted tomatoes.",
            key="compare_passage",
            max_chars=4000,
        )
        st.caption(
            "BERT gets the passage with its last word masked. "
            "GPT-2 and T5 get the original passage."
        )

    completed = st.session_state.get("compare_completed")
    run_id = compare_browser(result=completed, key="compare_browser")

    if run_id and (completed is None or completed["id"] != run_id):
        passage = passage.strip()
        masked, substitutions = re.subn(r"\b\w+\b(?=\W*$)", "[MASK]", passage)
        rows = []
        error = None

        if not substitutions or "[MASK]" in passage or len(masked) > 4000:
            error = "Enter a passage with a final word and no existing [MASK]."
        else:
            jobs = [
                ("BERT", "bert_fillmask", {"text": masked, "top_k": 3}),
                ("GPT-2", "gpt2_continue", {"prompt": passage, "max_new_tokens": 20}),
                ("T5", "t5_summarize", {"text": passage, "max_new_tokens": 24}),
            ]

            for name, endpoint, payload in jobs:
                start = perf_counter()
                model_result = run_model(endpoint, payload, name)
                request_ms = round((perf_counter() - start) * 1000)

                if model_result is None:
                    rows.append({"name": name, "output": "No result returned. Try again."})
                else:
                    rows.append({
                        "name": name,
                        "output": model_result["output_text"],
                        "inference_ms": model_result["latency_ms"],
                        "request_ms": request_ms,
                    })

        st.session_state["compare_completed"] = {
            "id": run_id,
            "rows": rows,
            "error": error,
        }
        st.rerun()


with history_tab:
    st.header("Latency History")
    chart_area, guide = st.columns([3, 1])

    with guide:
        st.subheader("When to use each archetype")
        st.write(
            "Use BERT for missing words, GPT-2 for continuations, and T5 "
            "for summaries. Check the request count alongside the timing. "
            "One request gives you an example; more requests give you a better basis for comparison."
        )

    with chart_area:
        st.caption(
            "Hourly median server-side inference time, in UTC. "
            "The chart reflects the latest database refresh."
        )

        if st.button("Load latency history"):
            supabase_url = st.secrets.get("SUPABASE_URL")
            anon_key = st.secrets.get("SUPABASE_ANON_KEY")

            if not supabase_url or not anon_key:
                st.error("Add SUPABASE_URL and SUPABASE_ANON_KEY to Streamlit secrets.")
            else:
                anon_key = anon_key.strip()
                headers = {"apikey": anon_key}
                if anon_key.startswith("eyJ"):
                    headers["Authorization"] = f"Bearer {anon_key}"

                try:
                    response = requests.get(
                        f"{supabase_url.rstrip('/')}/rest/v1/benchmarks_hourly",
                        headers=headers,
                        params={
                            "select": "bucket_start,model_name,n_inferences,p50_latency_ms",
                            "order": "bucket_start.desc,model_name.asc",
                            "limit": 500,
                        },
                        timeout=15,
                    )
                    response.raise_for_status()
                    rows = response.json()
                except requests.RequestException as error:
                    if error.response is not None and error.response.status_code in {401, 403}:
                        st.error("Supabase denied access. Check the anon key and the hourly view's SELECT grant.")
                    else:
                        st.error("Couldn't load latency history from Supabase. Try again in a moment.")
                else:
                    if not rows:
                        st.info("No benchmark history is available yet.")
                    else:
                        history = pd.DataFrame(rows)
                        history["bucket_start"] = pd.to_datetime(history["bucket_start"], utc=True)

                        st.vega_lite_chart(
                            history,
                            {
                                "mark": {"type": "line", "point": True},
                                "encoding": {
                                    "x": {
                                        "field": "bucket_start",
                                        "type": "temporal",
                                        "scale": {"type": "utc"},
                                        "title": "Hour (UTC)",
                                    },
                                    "y": {
                                        "field": "p50_latency_ms",
                                        "type": "quantitative",
                                        "title": "Median inference (ms)",
                                    },
                                    "color": {
                                        "field": "model_name",
                                        "type": "nominal",
                                        "title": "Model",
                                    },
                                    "tooltip": [
                                        {"field": "model_name", "type": "nominal", "title": "Model"},
                                        {
                                            "field": "p50_latency_ms",
                                            "type": "quantitative",
                                            "title": "Median (ms)",
                                            "format": ".1f",
                                        },
                                        {"field": "n_inferences", "type": "quantitative", "title": "Requests"},
                                    ],
                                },
                            },
                            use_container_width=True,
                        )
