FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/opt/models \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    HF_HUB_DISABLE_XET=1

# libgl1 / libglib2.0-0 are needed by OpenCV, which the OCR engine uses.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model, tokenizer and OCR models into the image so the
# running container needs no network access except the LLM API.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/models')" \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()"

COPY app ./app
COPY tests ./tests
COPY Makefile pytest.ini ./

ENV HF_HUB_OFFLINE=1
EXPOSE 8000
CMD ["uvicorn", "app.main:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
