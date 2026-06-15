FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libglpk-dev nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir peft bitsandbytes accelerate

# App code
COPY . .

# Models and data should be mounted at runtime
# - models/ for Qwen/Qwen2___5-7B-Instruct
# - data/ for milvus, bm25_index, onnx_models

ENV PYTHONUNBUFFERED=1
ENV QWEN_API_KEY=""

EXPOSE 8000

CMD ["python", "api_server.py"]
