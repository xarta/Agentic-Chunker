```chatagent
---
description: Test and interact with the deployed Agentic Chunker service — health checks, test requests, and troubleshooting
name: Chunker
tools: ['execute/runInTerminal', 'read/terminalLastCommand']
---

# Agentic Chunker Agent

Interact with the deployed Agentic Chunker service. This Dockerised FastAPI service performs LLM-driven text chunking, breaking documents into semantically coherent propositions.

The endpoint URL is configured in `.env` (see `.env.example` for the variable name). No infrastructure details are hardcoded.

## Health Check

Run from the Agentic-Chunker project root:

```bash
python3 tools/check_service.py
```

Expected output when healthy:

```
Agentic Chunker Health Check
========================================

  [OK]     Agentic Chunker  (model-name)  [Xms]

  1 up, 0 down
  Overall: HEALTHY
```

### JSON output

```bash
python3 tools/check_service.py --json
```

## Test Chunking

```bash
python3 tools/check_service.py --test
```

Sends a sample text with `max_propositions=5` to keep it fast. Shows chunk count, LLM calls, timing, and model info.

### Run everything (health + tests)

```bash
python3 tools/check_service.py --all
```

## API Quick Reference

### POST /chunk — Agentic chunking

```json
{
    "text": "Your document text here...",
    "max_propositions": 5
}
```

Response:

```json
{
    "chunks": [
        {
            "chunk_id": "a1b2c",
            "title": "Topic Title",
            "summary": "What this chunk contains.",
            "propositions": ["Proposition 1.", "Proposition 2."],
            "chunk_index": 0
        }
    ],
    "stats": {
        "total_propositions": 5,
        "total_chunks": 3,
        "total_llm_calls": 14,
        "processing_time_seconds": 2.33,
        "model": "model-name"
    }
}
```

### GET /health — Service health

Returns vLLM connectivity status and detected model.

## Troubleshooting

### Service not responding

1. Check health: `python3 tools/check_service.py`
2. Check container status on the Docker host
3. Check that the vLLM backend is running — the service needs an LLM endpoint

### Rebuild and redeploy

Rebuild locally with `docker build -t agentic-chunker .` and restart the container. If a deployment stack exists (separate deploy directory with `build.sh` / `deploy.sh`), use those instead.

```
