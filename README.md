![vLLM LLM Backend](https://img.shields.io/badge/vLLM-LLM%20Backend-green)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Lightweight](https://img.shields.io/badge/Image-191MB-blue)

# Agentic Chunker

## Forked

Forked from https://github.com/Ranjith-JS2803/Agentic-Chunker. We replaced the Google Gemini dependency with HTTP calls to our local vLLM inference endpoint via the OpenAI-compatible API. The container doesn't need any API keys for external services — it delegates all LLM work to a dedicated vLLM inference server on the local network.

## Warning — AI-Generated Content Notice

This project was **modified with AI assistance** and should be treated accordingly:

- **Not production-ready**: Created for a specific homelab environment.
- **May contain bugs**: AI-generated code can have subtle issues.
- **Author's Python experience**: The author (modifier) is not an experienced Python programmer.

### AI Tools Used

- GitHub Copilot (Claude models)
- Local vLLM instances for analysis and consolidation

### Licensing Note

The original repository (Ranjith-JS2803/Agentic-Chunker) does not include an explicit licence. This fork inherits that status. Given the AI-generated/modified nature:
- The modifying author makes no claims about originality
- Use at your own risk
- If you discover any copyright concerns, please open an issue

---

## What It Does

Agentic Chunking uses an LLM to decompose text into propositions and then semantically group those propositions into coherent chunks. Unlike statistical chunking approaches (like the Normalized Semantic Chunker), this uses the LLM's understanding of meaning to decide which propositions belong together.

The approach:

1. **Proposition extraction** — Each paragraph is sent to the LLM, which decomposes it into simple, self-contained propositions.
2. **Chunk assignment** — Each proposition is evaluated against existing chunks. The LLM decides whether it belongs to an existing chunk or needs a new one.
3. **Summary & title generation** — As propositions are added, each chunk's summary and title are updated by the LLM.

This is fundamentally different from embedding-based chunkers — it uses the LLM's reasoning to group content, not vector similarity.

## Performance Expectations

This service is **significantly slower** than embedding-based chunkers. It makes 2–3 LLM calls per proposition (find chunk, update summary, update title) plus 1 LLM call per paragraph for proposition extraction. For a 10-paragraph document producing 50 propositions, expect ~160 LLM calls.

Best suited for small, dense documents where high-precision semantic chunking matters — not bulk processing.

## Key Features

- **LLM-driven chunking**: Uses chat completions to semantically group propositions, not statistical similarity.
- **vLLM backend**: All LLM inference delegated to a remote vLLM endpoint via OpenAI-compatible API.
- **Lightweight container**: 191MB Docker image (python:3.11-slim, no ML frameworks).
- **FastAPI REST API**: `POST /chunk` endpoint with automatic documentation at `/docs`.
- **Robust JSON parsing**: Handles markdown-fenced JSON, preamble text, and raw arrays from LLM output.
- **Retry logic**: Exponential backoff for 429, flat retry for 5xx — handles vLLM under load.
- **Think-tag stripping**: Safety net for Qwen3 and similar models that emit `<think>` blocks.
- **Configurable parallelism**: `MAX_WORKERS` environment variable for concurrent proposition extraction.
- **Test-friendly**: `max_propositions` parameter to limit processing during testing.

## API

### `POST /chunk`

```json
{
    "text": "Full document text to chunk agentically.",
    "max_propositions": null
}
```

Response:

```json
{
    "chunks": [
        {
            "chunk_id": "a1b2c",
            "title": "Food Preferences",
            "summary": "This chunk contains information about food preferences.",
            "propositions": ["Greg likes pizza.", "Greg also enjoys pasta."],
            "chunk_index": 0
        }
    ],
    "stats": {
        "total_propositions": 50,
        "total_chunks": 8,
        "total_llm_calls": 162,
        "processing_time_seconds": 45.2,
        "model": "Qwen/Qwen3-235B-A22B"
    }
}
```

### `GET /health`

Returns vLLM connectivity status and detected model.

### `GET /`

Service info — version, model, worker count, status.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_PRIMARY_BASE_URL` | (required) | vLLM OpenAI-compatible base URL (e.g. `http://host:8000/v1`) |
| `VLLM_PRIMARY_API_KEY` | (none) | Bearer token for vLLM authentication |
| `LLM_MODEL` | `auto` | Model name, or `auto` to detect from `/v1/models` |
| `MAX_WORKERS` | `4` | Max concurrent LLM calls |
| `LOG_LEVEL` | `INFO` | Logging level |

## Docker

```bash
docker build -t agentic-chunker .
docker run -p 8102:8000 \
    -e VLLM_PRIMARY_BASE_URL=http://your-vllm-host:8000/v1 \
    -e VLLM_PRIMARY_API_KEY=your-key \
    agentic-chunker
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI entrypoint — `/chunk`, `/health`, `/` endpoints |
| `agentic_chunker.py` | Core chunker — proposition grouping via LLM |
| `llm_client.py` | Stdlib-only OpenAI-compatible LLM client |
| `Dockerfile` | Multi-stage build, python:3.11-slim |
| `requirements.txt` | Minimal deps — fastapi, uvicorn, pydantic |
| `AgenticChunker.py` | Original (Gemini-based) — kept for reference |
| `main.py` | Original CLI — kept for reference |

## Example Test Run

Tested with `Qwen/Qwen3-30B-A3B-GPTQ-Int4` on an RTX 5090 via vLLM.

**Input** — 2 short paragraphs (LangChain text splitting + ChatGPT description), limited to 5 propositions:

```bash
curl -s -X POST http://localhost:8102/chunk \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Text splitting in LangChain is a critical feature that enables the division of large texts into smaller, manageable segments. This capability is essential for improving comprehension and processing efficiency, particularly in tasks that require detailed analysis or the extraction of specific contexts.\n\nChatGPT, developed by OpenAI, represents a significant advancement in natural language processing technologies. As a conversational AI model, ChatGPT is capable of understanding and generating human-like text, facilitating dynamic and engaging interactions.",
    "max_propositions": 5
  }'
```

**Output**:

```json
{
    "chunks": [
        {
            "chunk_id": "4a19f",
            "title": "Text Splitting",
            "summary": "This chunk contains information about text splitting techniques in LangChain, specifically focusing on how text is divided for processing.",
            "propositions": [
                "Text splitting in LangChain is a critical feature."
            ],
            "chunk_index": 0
        },
        {
            "chunk_id": "edaf5",
            "title": "Text Segmentation",
            "summary": "This chunk contains information about text segmentation techniques and their benefits.",
            "propositions": [
                "The critical feature enables the division of large texts into smaller, manageable segments."
            ],
            "chunk_index": 1
        },
        {
            "chunk_id": "f2a3f",
            "title": "Comprehension Skills",
            "summary": "This chunk contains information about essential capabilities that improve comprehension.",
            "propositions": [
                "This capability is essential for improving comprehension."
            ],
            "chunk_index": 2
        },
        {
            "chunk_id": "05d2c",
            "title": "Capabilities & Applications",
            "summary": "This chunk contains information about capabilities that enhance processing efficiency and their applications in detailed analysis tasks.",
            "propositions": [
                "This capability is essential for improving processing efficiency.",
                "This capability is particularly useful in tasks that require detailed analysis."
            ],
            "chunk_index": 3
        }
    ],
    "stats": {
        "total_propositions": 5,
        "total_chunks": 4,
        "total_llm_calls": 14,
        "processing_time_seconds": 2.33,
        "model": "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
    }
}
```

**Observations**:

| Metric | Value |
|--------|-------|
| Propositions extracted | 5 (from 2 paragraphs) |
| Chunks created | 4 |
| LLM calls | 14 |
| Time | 2.33s (~0.17s per LLM call) |
| Model | Qwen/Qwen3-30B-A3B-GPTQ-Int4 (MoE, 3B active params) |
| GPU | NVIDIA RTX 5090 |

The MoE model's low active parameter count (3B) makes per-call latency very low. For a larger document producing ~50 propositions, extrapolated time would be ~25 seconds. Denser models or larger active parameter counts will be proportionally slower.

Note that the chunker was quite aggressive here — creating 4 chunks from 5 closely-related propositions about NLP text processing. For documents with more diverse topics, the chunker should group more effectively. The `max_propositions` parameter is useful for testing without waiting for full document processing.

## Original README

The original project description from [Ranjith-JS2803/Agentic-Chunker](https://github.com/Ranjith-JS2803/Agentic-Chunker):

> Agentic Chunking involves taking a text and organizing its propositions into grouped "chunks." Each chunk is a collection of related propositions that are interconnected, allowing for more efficient processing and retrieval within a RAG system.
>
> ### How does a human go about chunking a text?
>
> 1. You take a pen and paper, and you start at the top of the text, treating the first part as the starting point for a new chunk.
> 2. As you move down the text, you evaluate if a new sentence or piece should be a part of the previous chunk, if not, then create a new chunk.
> 3. You repeat this process, methodically working through the text chunk by chunk until you've covered the entire text.
>
> **Who is the "agent" here?** You're correct - it's the human!!
