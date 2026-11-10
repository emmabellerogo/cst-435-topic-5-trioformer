FROM python:3.11-slim
WORKDIR /repo
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1 transformers==4.44.2 huggingface-hub==0.36.2 "numpy<2" sentencepiece psutil httpx \
    fastapi==0.115.0 "uvicorn[standard]==0.30.6" pydantic==2.9.2 supabase==2.7.4 python-dotenv==1.0.1 PyYAML
COPY api /repo/api
COPY shared /repo/shared
COPY src /repo/src
COPY linux512 /repo/linux512
ENV PYTHONPATH=/repo PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 \
    HF_HOME=/hf SUPABASE_URL=x SUPABASE_SERVICE_KEY=x TRIOFORMER_BACKEND=torch
