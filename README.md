# TrioDeck — Product 5B

TrioDeck gives a learning-and-development team one link to try three transformer models on the same passage. BERT fills in a missing word, GPT‑2 continues the passage, and T5 summarizes it. The app shows the answers and timings side by side. It also includes a custom encoder attention heatmap and an hourly latency history chart.

## Project links

- [Product 5B GitHub branch](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/tree/product-5b-triodeck)
- [Live TrioDeck app on Streamlit Community Cloud](https://cst-435-topic-5-trioformer-trtytkn7i2yhudmfea8bmp.streamlit.app/)
- [Team Product 5 backend repository](https://github.com/emmabellerogo/cst-435-topic-5-trioformer/tree/main)
- [Team FastAPI service on Render](https://former-it-api.onrender.com)
- [Supabase project reference](https://hutifztunklvgvilickm.supabase.co) (`hutifztunklvgvilickm`)

## Deployment boundary

The Streamlit app lives in `ui/triodeck.py`. Every inference request goes over HTTPS to FastAPI on Render, where the models run and inference records are written to Supabase. The Streamlit tier contains no model code and issues no SQL writes. Latency History reads hourly aggregates from the `benchmarks_hourly` materialized view using a read-only anon key. The key and service URLs are configured in Streamlit Cloud Secrets. The service-role key stays on the backend.

## Deployment reflection

I measured one Compare run on the public Streamlit app using a stopwatch. From clicking **Compare all three** to seeing the final result, I waited **23.21 seconds**. The three models reported **13.28 seconds** of combined server-side inference. The three Streamlit-to-Render round trips added up to **22.35 seconds**. That left a **9.93-second gap** between model inference and what I experienced in the browser. The API round trips can include model loading, queueing, network transit, and database logging. My measurements did not isolate those parts or establish how much time came from Render waking up after being idle.

Separating the services made setup harder. I had to configure secrets in Streamlit Cloud and verify that the Supabase anon key could read the hourly aggregates. I ran into a permissions error while setting up that read. The separation also let me deploy the interface using the same Render service as Product 5, with every model running in one place and database writes handled by the backend.

I chose to show the round-trip timing because the inference number alone leaves out much of the wait a person experiences. Proverbs 11:1 connects honest measurement with character. For this project, that means giving the team enough information to decide whether the wait is acceptable and being clear about what I measured.