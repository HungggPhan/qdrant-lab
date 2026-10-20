# qdrant-lab

An AI/LLM research lab: ingest PDF documents (mainly Vietnamese) into a local [Qdrant](https://qdrant.tech/) instance for semantic search and a RAG chatbot, running entirely on a personal machine.

## Features

- **PDF ingestion**: extracts text, repairs legacy Vietnamese font encodings, re-joins split syllables (`c ủa` → `của`), and chunks by token count at paragraph/sentence/clause/word boundaries.
- **Embeddings**: `BAAI/bge-m3` (1024 dimensions, cosine). Runs on GPU in fp16 when CUDA is available, otherwise on CPU.
- **HTTP upload**: `POST /upload` accepts a PDF and ingests it in the background with a single worker; jobs are tracked in SQLite.
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
- `POST /upload` (form fields `folder`, `file`) returns 202 with a `job_id`; check progress at `GET /jobs/{id}`
- `POST /chat` (JSON `question`, `top_k`, `folder`, `conversation_id`) returns an NDJSON stream

### Main environment variables

`QDRANT_URL`, `QDRANT_COLLECTION`, `EMBED_MODEL`, `EMBED_DEVICE` (`cpu`|`cuda`), `MAX_UPLOAD_MB`, `CHAT_MODEL`, `LLM_CTX`, `LLM_GPU_LAYERS`, `LLM_PORT`, `LLAMA_SERVER_BIN`.

## Limitations

- No OCR: scanned image-only PDFs produce 0 chunks.
- Gemma 2B does not reliably emit `[n]` citations; displayed sources come from chunk metadata.
- No automated tests yet.

## Notes

The `documents/` directory, SQLite databases, models and `llama-server` are not in the repo. Prepare them yourself after cloning.
