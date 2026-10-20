# qdrant-lab

An AI/LLM research lab: ingest PDF documents (mainly Vietnamese) into a local [Qdrant](https://qdrant.tech/) instance for semantic search and a RAG chatbot, running entirely on a personal machine.

## Features

- **PDF ingestion**: extracts text, repairs legacy Vietnamese font encodings, re-joins split syllables (`c ủa` → `của`), and chunks by token count at paragraph/sentence/clause/word boundaries.
- **Embeddings**: `BAAI/bge-m3` (1024 dimensions, cosine). Runs on GPU in fp16 when CUDA is available, otherwise on CPU.
- **HTTP bulk upload**: `POST /upload` accepts several PDFs per request and ingests them in the background with a single worker; jobs are tracked in SQLite and grouped into batches.
- **RAG chatbot**: retrieves relevant chunks from Qdrant and passes them to a local LLM (Gemma 2B via llama.cpp's `llama-server`, GPU-accelerated). Answers stream back with sources (file, page, chunk, score).
- If no chunk is relevant enough, the bot replies that it has no answer without calling the LLM.

## Architecture

| File | Role |
| --- | --- |
| `ingest.py` | All ingestion logic: PDF reading, chunking, embedding, upsert into Qdrant |
| `add_document.py` | Thin CLI around `Ingestor` |
| `app.py` | FastAPI app: upload, jobs, chat, conversation history |
| `rag.py` | Retrieval, LLM calls, `llama-server` lifecycle |
| `vi_text.py` | Vietnamese encoding repair and syllable merging |
| `build_vocab.py` | Builds `vi_syllables.txt` from the documents |
| `chat.html` | Chat UI |
| `test_add_to_qdrant.py` | Original lab exercise that loads sample sentences |

## Requirements

- Python 3.10+ (developed on 3.14, Windows)
- Qdrant running at `http://localhost:6333`
- Recommended: an NVIDIA GPU with CUDA (tested on an RTX 2050, 4 GB)
- For the chatbot: a CUDA build of [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) `llama-server` in `bin/llama.cpp/`, and a GGUF model in `models/` (default `gemma-2b-it.Q4_K_M.gguf`). Neither directory is part of this repo.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

For GPU embeddings, install a CUDA build of PyTorch.

## Usage

Put PDFs in `documents/<folder-name>/`, then ingest from the CLI:

```bash
python add_document.py --folder <folder-name>
python add_document.py --all --recreate   # re-ingest everything from scratch
```

Or run the web service (exactly one process; do not use `--workers`):

```bash
uvicorn app:app --port 8000
```

- `http://localhost:8000/` upload test page
- `http://localhost:8000/chat` chat page
- `POST /upload` (form fields `folder`, repeated `files`; the old single `file` field still works) returns 202 with `{batch_id, jobs, rejected}`. Invalid files (not a PDF, empty, over the size limit, duplicate name in the batch) go to `rejected` while the valid ones are still queued; the request fails with 400 only if no file is valid, and with 413 if it exceeds the file-count or total-size limit.
- `GET /batches/{id}` returns the status of a whole batch (per-job state and progress, plus `counts`); `GET /jobs/{id}` returns a single job
- `POST /chat` (JSON `question`, `top_k`, `folder`, `conversation_id`) returns an NDJSON stream

### Main environment variables

`QDRANT_URL`, `QDRANT_COLLECTION`, `EMBED_MODEL`, `EMBED_DEVICE` (`cpu`|`cuda`), `MAX_UPLOAD_MB` (per file, 50), `MAX_FILES_PER_UPLOAD` (20), `MAX_BATCH_MB` (500), `CHAT_MODEL`, `LLM_CTX`, `LLM_GPU_LAYERS`, `LLM_PORT`, `LLAMA_SERVER_BIN`.

## Limitations

- No OCR: scanned image-only PDFs produce 0 chunks.
- Gemma 2B does not reliably emit `[n]` citations; displayed sources come from chunk metadata.
- Tests cover only the upload API (`pip install -r requirements-dev.txt`, then `pytest test_upload.py`); ingestion and RAG are untested.

## Notes

The `documents/` directory, SQLite databases, models and `llama-server` are not in the repo. Prepare them yourself after cloning.
