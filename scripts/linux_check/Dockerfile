# Render-like Linux image for scripts/linux_check: installs EXACTLY api/requirements.txt (no extras).
# Extra packages used only by the test driver: httpx (HTTP client) and psutil.
FROM python:3.11-slim
WORKDIR /repo
COPY api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt httpx psutil
COPY api /repo/api
COPY shared /repo/shared
COPY src /repo/src
COPY scripts /repo/scripts
ENV PYTHONPATH=/repo PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 \
    HF_HOME=/hf SUPABASE_URL=x SUPABASE_SERVICE_KEY=x
