# Agentic Chunker — GitHub Copilot Instructions

## Project Overview

Agentic Chunker is a Dockerised FastAPI service that performs LLM-driven text chunking. It breaks documents into semantically coherent propositions using an agentic approach — each proposition is evaluated against existing chunks, merged or assigned to new chunks via LLM calls.

Originally forked from a Google Gemini-based implementation, this version uses a local vLLM endpoint (OpenAI-compatible API) so all processing stays on-network.

## Key Rules

- **Use `python3`** not `python`.
- **British spelling** — `sanitise`, `analyse`, `colour`, etc.
- **No real infrastructure in source** — never put real IPs, hostnames, LXC IDs, or API keys in committed code. All loaded from `.env` (gitignored) or container environment variables.
- **Stdlib LLM client** — `llm_client.py` uses `urllib.request` only. No `requests`, no `httpx` for LLM calls.

## Project Structure

```
Agentic-Chunker/
├── app.py                  # FastAPI service: /, /health, /chunk
├── agentic_chunker.py      # Core chunking logic (AgenticChunker class)
├── AgenticChunker.py        # Legacy entry point (imports from agentic_chunker)
├── llm_client.py           # Stdlib-only LLM client (urllib.request)
├── main.py                 # CLI entry point for local testing
├── requirements.txt        # Python deps (fastapi, uvicorn, pydantic)
├── Dockerfile              # Container build
├── .env                    # Real endpoint/secrets (GITIGNORED — never commit)
├── .env.example            # Template showing variable names
├── .gitignore
├── .github/
│   ├── copilot-instructions.md   # This file
│   └── agents/
│       └── chunker.agent.md      # @chunker agent for health/test
└── tools/
    └── check_service.py          # Health check + test tool
```

## Environment Variables

The service container gets its config from environment variables injected at deploy time:

| Variable | Purpose |
|----------|---------|
| `VLLM_PRIMARY_BASE_URL` | vLLM OpenAI-compatible base URL |
| `VLLM_PRIMARY_API_KEY` | vLLM Bearer token |
| `LLM_MODEL` | Model name (default: `auto` — auto-detect) |
| `MAX_WORKERS` | Max concurrent LLM calls (default: `4`) |
| `LOG_LEVEL` | Logging level (default: `INFO`) |

The client-side `.env` configures the tools for interacting with the deployed service:

| Variable | Purpose |
|----------|---------|
| `AGENTIC_CHUNKER_URL` | URL of the deployed service (e.g. `http://host:8102`) |

## API Endpoints

### GET / — Service info
Returns service name, version, and status.

### GET /health — Health check
Returns vLLM connectivity status and detected model.

### POST /chunk — Agentic chunking
Accepts JSON body with `text` and optional `max_propositions`. Returns LLM-driven semantic chunks with titles, summaries, and propositions.

## Running Locally

```bash
# Set environment variables
export VLLM_PRIMARY_BASE_URL=http://your-vllm:8000/v1
export VLLM_PRIMARY_API_KEY=your-key

# Install deps
pip install -r requirements.txt

# Run
uvicorn app:app --host 0.0.0.0 --port 8102
```

## Health Check

```bash
python3 tools/check_service.py              # Health check
python3 tools/check_service.py --test       # Test chunking
python3 tools/check_service.py --all        # Everything
python3 tools/check_service.py --json       # JSON output
```

## Build & Deploy

The service is built as a Docker image and deployed via stack scripts (paths in per-host docs):

```bash
docker build -t agentic-chunker .
```

If a deployment stack exists (e.g. `build.sh` / `deploy.sh` in a separate deploy directory), use those scripts to push to a local registry and deploy to the Docker host.

## Architecture Notes

- All LLM work is delegated to a remote vLLM endpoint — no local GPU required in the container.
- The agentic approach makes ~3 LLM calls per proposition (extract, evaluate, assign/create).
- Best suited for small dense documents where semantic precision matters.
- Slower than embedding-only chunkers due to multiple LLM round-trips per proposition, but produces more semantically precise chunks.
