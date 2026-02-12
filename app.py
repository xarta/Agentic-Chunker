"""
Agentic Chunker — FastAPI service.

Wraps the Agentic Chunker in a REST API, replacing Google Gemini
with the local vLLM endpoint via the OpenAI-compatible client.

Endpoints:
  GET  /        — service info
  GET  /health  — health check with vLLM connectivity test
  POST /chunk   — agentic chunking of input text

Environment variables:
  VLLM_PRIMARY_BASE_URL  — vLLM OpenAI-compatible base URL
  VLLM_PRIMARY_API_KEY   — vLLM Bearer token
  LLM_MODEL              — model name (default: auto-detect)
  MAX_WORKERS            — max concurrent LLM calls (default: 4)
  LOG_LEVEL              — logging level (default: INFO)
"""

import json
import logging
import os
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentic_chunker import AgenticChunker
from llm_client import LLMClient, detect_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("agentic-chunker")

# ---------------------------------------------------------------------------
# LLM client setup
# ---------------------------------------------------------------------------
VLLM_BASE_URL = os.environ.get("VLLM_PRIMARY_BASE_URL", "")
VLLM_API_KEY = os.environ.get("VLLM_PRIMARY_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "auto")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))

llm_client: Optional[LLMClient] = None
detected_model: Optional[str] = None

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Agentic Chunker",
    description="LLM-driven agentic text chunking service",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    """Initialise the LLM client and auto-detect model."""
    global llm_client, detected_model

    if not VLLM_BASE_URL:
        logger.warning("VLLM_PRIMARY_BASE_URL not set — service will not function")
        return

    model = LLM_MODEL
    if model == "auto":
        try:
            model = detect_model(VLLM_BASE_URL, VLLM_API_KEY)
            logger.info("Auto-detected model: %s", model)
        except Exception as exc:
            logger.error("Model auto-detection failed: %s", exc)
            model = None

    detected_model = model
    if model:
        llm_client = LLMClient(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            model=model,
        )
        logger.info("LLM client ready — model=%s, max_workers=%d", model, MAX_WORKERS)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ChunkRequest(BaseModel):
    """Request body for POST /chunk."""
    text: str = Field(..., description="Full document text to chunk.")
    max_propositions: Optional[int] = Field(
        None,
        description="Optional limit on propositions to process (useful for testing).",
    )


class ChunkItem(BaseModel):
    """A single chunk in the response."""
    chunk_id: str
    title: str
    summary: str
    propositions: List[str]
    chunk_index: int


class ChunkStats(BaseModel):
    """Processing statistics."""
    total_propositions: int
    total_chunks: int
    total_llm_calls: int
    processing_time_seconds: float
    model: Optional[str] = None


class ChunkResponse(BaseModel):
    """Response body for POST /chunk."""
    chunks: List[ChunkItem]
    stats: ChunkStats


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Service info."""
    return {
        "service": "agentic-chunker",
        "version": "0.1.0",
        "model": detected_model,
        "max_workers": MAX_WORKERS,
        "status": "ready" if llm_client else "no_llm",
    }


@app.get("/health")
async def health():
    """Health check with vLLM connectivity test."""
    if not llm_client:
        return {
            "status": "unhealthy",
            "vllm_connected": False,
            "error": "LLM client not initialised",
        }

    try:
        # Quick connectivity test — detect model
        detect_model(VLLM_BASE_URL, VLLM_API_KEY)
        return {
            "status": "healthy",
            "vllm_connected": True,
            "model": detected_model,
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "vllm_connected": False,
            "error": str(exc),
        }


@app.post("/chunk", response_model=ChunkResponse)
async def chunk_text(request: ChunkRequest):
    """Agentically chunk the input text."""
    if not llm_client:
        raise HTTPException(
            status_code=503,
            detail="LLM client not initialised — check VLLM_PRIMARY_BASE_URL",
        )

    start = time.monotonic()

    try:
        # Step 1: Extract propositions from paragraphs
        logger.info("Extracting propositions from text (%d chars)", len(request.text))
        propositions = extract_propositions(request.text, llm_client)
        logger.info("Extracted %d propositions", len(propositions))

        # Optionally limit for testing
        if request.max_propositions and request.max_propositions < len(propositions):
            propositions = propositions[: request.max_propositions]
            logger.info("Limited to %d propositions (max_propositions)", len(propositions))

        # Step 2: Agentic chunking
        chunker = AgenticChunker(llm_client=llm_client)
        chunker.add_propositions(propositions)

        elapsed = time.monotonic() - start

        # Build response
        chunks = []
        for chunk_id, chunk_data in chunker.chunks.items():
            chunks.append(ChunkItem(
                chunk_id=chunk_data["chunk_id"],
                title=chunk_data["title"].strip(),
                summary=chunk_data["summary"].strip(),
                propositions=chunk_data["propositions"],
                chunk_index=chunk_data["chunk_index"],
            ))

        stats = ChunkStats(
            total_propositions=len(propositions),
            total_chunks=len(chunks),
            total_llm_calls=chunker.llm_call_count,
            processing_time_seconds=round(elapsed, 2),
            model=detected_model,
        )

        logger.info(
            "Chunking complete: %d propositions → %d chunks in %.1fs (%d LLM calls)",
            stats.total_propositions,
            stats.total_chunks,
            stats.processing_time_seconds,
            stats.total_llm_calls,
        )

        return ChunkResponse(chunks=chunks, stats=stats)

    except Exception as exc:
        logger.exception("Chunking failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Proposition extraction
# ---------------------------------------------------------------------------
PROPOSITION_PROMPT = """Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.

1. Split compound sentences into simple sentences. Maintain the original phrasing from the input whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifiers to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
4. Present the results as a JSON array of strings.

Example:
Input: Greg likes to eat pizza. He also enjoys pasta and salads.
Output: ["Greg likes to eat pizza.", "Greg also enjoys pasta.", "Greg also enjoys salads."]

Decompose the following:
{input}"""


def extract_propositions(text: str, client: LLMClient) -> List[str]:
    """Extract propositions from text paragraphs using the LLM.

    Splits text into paragraphs and sends each to the LLM for
    proposition extraction. Returns a flat list of all propositions.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    all_propositions: List[str] = []

    for i, para in enumerate(paragraphs):
        prompt = PROPOSITION_PROMPT.replace("{input}", para)
        try:
            response = client.generate(prompt)
            props = _parse_json_array(response)
            all_propositions.extend(props)
            logger.debug("Paragraph %d: %d propositions", i + 1, len(props))
        except Exception as exc:
            logger.warning("Failed to extract propositions from paragraph %d: %s", i + 1, exc)
            # Fall back to using the paragraph as a single proposition
            all_propositions.append(para)

    return all_propositions


def _parse_json_array(text: str) -> List[str]:
    """Robustly parse a JSON array from LLM output.

    Handles raw JSON, markdown-fenced JSON, and preamble text.
    """
    import re

    cleaned = text.strip()

    # Strip think tags if present
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # Strip markdown code fences
    fence_match = re.match(
        r"^```(?:json)?\s*\n(.*?)\n\s*```\s*$",
        cleaned,
        flags=re.DOTALL,
    )
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Try direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return [str(item) for item in result]
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array in the text
    array_match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if array_match:
        try:
            result = json.loads(array_match.group())
            if isinstance(result, list):
                return [str(item) for item in result]
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON array from LLM response: {text[:200]}")
