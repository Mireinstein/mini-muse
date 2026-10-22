# Stage 1 — build the frontend into static files.
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2 — Python runtime that serves the app.
FROM python:3.12-slim

# Hugging Face Spaces (and some other hosts) run the container as UID 1000, not
# root, so give that user a writable home for the embedding-model cache.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence-transformers \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY mini_muse/ ./mini_muse/
COPY docs/ ./docs/
COPY --from=frontend /app/frontend/dist ./frontend/dist

RUN chown -R user:user /app /home/user
USER user

# Sync the resume/GitHub content, build the vector index, and pull the embedding
# model — all baked into the image so the first real question is fast.
RUN python -m mini_muse --ingest

EXPOSE 7860
CMD ["python", "-m", "mini_muse", "--serve"]
