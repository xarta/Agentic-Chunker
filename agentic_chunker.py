"""
Agentic Chunker — LLM-driven proposition-based text chunking.

Forked from https://github.com/Ranjith-JS2803/Agentic-Chunker
Modified to use a local vLLM endpoint via OpenAI-compatible API
instead of Google Gemini.

Each proposition is evaluated by the LLM to determine which chunk
it belongs to. Chunks are created, summarised, and titled by the
LLM as propositions are added.
"""

import logging
import uuid
from typing import Optional

from llm_client import LLMClient

logger = logging.getLogger("agentic-chunker.core")


class AgenticChunker:
    """LLM-driven chunker that groups propositions semantically.

    Args:
        llm_client: An initialised LLMClient for vLLM inference.
        chunk_id_length: Length of truncated UUID chunk IDs.
    """

    def __init__(self, llm_client: LLMClient, chunk_id_length: int = 5):
        self.chunks: dict = {}
        self.llm_client = llm_client
        self.chunk_id_length = chunk_id_length
        self.llm_call_count = 0

    def add_propositions(self, propositions: list[str]) -> None:
        """Add a list of propositions, assigning each to a chunk."""
        for proposition in propositions:
            self.add_proposition(proposition)

    def add_proposition(self, proposition: str) -> None:
        """Add a single proposition — find or create a chunk for it."""
        logger.debug("Adding: %s", proposition[:80])

        if len(self.chunks) == 0:
            logger.debug("No chunks exist, creating first chunk")
            self._create_new_chunk(proposition)
            return

        relevant_chunk_id = self._find_relevant_chunk(proposition)

        if relevant_chunk_id is not None:
            logger.debug(
                "Found chunk %s (%s), adding proposition",
                relevant_chunk_id,
                self.chunks[relevant_chunk_id]["title"].strip(),
            )
            self._add_proposition_to_chunk(relevant_chunk_id, proposition)
        else:
            logger.debug("No relevant chunk found, creating new chunk")
            self._create_new_chunk(proposition)

    # -----------------------------------------------------------------
    # Chunk operations
    # -----------------------------------------------------------------

    def _add_proposition_to_chunk(self, chunk_id: str, proposition: str) -> None:
        """Add proposition to existing chunk and update summary/title."""
        self.chunks[chunk_id]["propositions"].append(proposition)
        self.chunks[chunk_id]["summary"] = self._update_chunk_summary(self.chunks[chunk_id])
        self.chunks[chunk_id]["title"] = self._update_chunk_title(self.chunks[chunk_id])

    def _create_new_chunk(self, proposition: str) -> None:
        """Create a new chunk from a single proposition."""
        new_chunk_id = str(uuid.uuid4())[: self.chunk_id_length]
        new_summary = self._get_new_chunk_summary(proposition)
        new_title = self._get_new_chunk_title(new_summary)

        self.chunks[new_chunk_id] = {
            "chunk_id": new_chunk_id,
            "propositions": [proposition],
            "title": new_title,
            "summary": new_summary,
            "chunk_index": len(self.chunks),
        }

        logger.debug("Created chunk %s: %s", new_chunk_id, new_title.strip())

    # -----------------------------------------------------------------
    # LLM calls — chunk finding
    # -----------------------------------------------------------------

    def _find_relevant_chunk(self, proposition: str) -> Optional[str]:
        """Ask the LLM which existing chunk this proposition belongs to."""
        chunk_outline = self._get_chunk_outline()

        prompt = f"""Determine whether or not the "Proposition" should belong to any of the existing chunks.

A proposition should belong to a chunk if their meaning, direction, or intention are similar.
The goal is to group similar propositions and chunks.

If you think a proposition should be joined with a chunk, return the chunk id.
If you do not think an item should be joined with an existing chunk, just return "No chunks"

Example:
Input:
    - Proposition: "Greg really likes hamburgers"
    - Current Chunks:
        - Chunk ID: 2n4l3
        - Chunk Name: Places in San Francisco
        - Chunk Summary: Overview of the things to do with San Francisco Places

        - Chunk ID: 93833
        - Chunk Name: Food Greg likes
        - Chunk Summary: Lists of the food and dishes that Greg likes
Output: 93833

Current Chunks:
--Start of current chunks--
{chunk_outline}
--End of current chunks--

Determine if the following statement should belong to one of the chunks outlined:
{proposition}"""

        response = self._llm_generate(prompt).strip()

        if len(response) != self.chunk_id_length:
            return None

        # Verify the chunk ID actually exists
        if response not in self.chunks:
            logger.debug("LLM returned non-existent chunk ID: %s", response)
            return None

        return response

    # -----------------------------------------------------------------
    # LLM calls — summaries and titles
    # -----------------------------------------------------------------

    def _update_chunk_summary(self, chunk: dict) -> str:
        prompt = f"""You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Proposition: Greg likes to eat pizza
Output: This chunk contains information about the types of food Greg likes to eat.

Only respond with the chunk new summary, nothing else.

Chunk's propositions:
{chunk['propositions']}

Current chunk summary:
{chunk['summary']}"""
        return self._llm_generate(prompt)

    def _update_chunk_title(self, chunk: dict) -> str:
        prompt = f"""You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

A good title will say what the chunk is about.

Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Summary: This chunk is about dates and times that the author talks about
Output: Date & Times

Only respond with the new chunk title, nothing else.

Chunk's propositions:
{chunk['propositions']}

Chunk summary:
{chunk['summary']}

Current chunk title:
{chunk['title']}"""
        return self._llm_generate(prompt)

    def _get_new_chunk_summary(self, proposition: str) -> str:
        prompt = f"""You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Proposition: Greg likes to eat pizza
Output: This chunk contains information about the types of food Greg likes to eat.

Only respond with the new chunk summary, nothing else.

Determine the summary of the new chunk that this proposition will go into:
{proposition}"""
        return self._llm_generate(prompt)

    def _get_new_chunk_title(self, summary: str) -> str:
        prompt = f"""You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

A good chunk title is brief but encompasses what the chunk is about.

Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Summary: This chunk is about dates and times that the author talks about
Output: Date & Times

Only respond with the new chunk title, nothing else.

Determine the title of the chunk that this summary belongs to:
{summary}"""
        return self._llm_generate(prompt)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _llm_generate(self, prompt: str) -> str:
        """Send prompt to LLM and increment call counter."""
        self.llm_call_count += 1
        return self.llm_client.generate(prompt)

    def _get_chunk_outline(self) -> str:
        """Build a text outline of all current chunks for the LLM."""
        lines = []
        for chunk_id, chunk in self.chunks.items():
            lines.append(
                f"Chunk ID: {chunk_id}\n"
                f"Chunk Name: {chunk['title']}\n"
                f"Chunk Summary: {chunk['summary']}\n"
            )
        return "\n".join(lines)

    def pretty_print_chunks(self) -> None:
        """Display all chunks to stdout."""
        print("\n----- Chunks Created -----\n")
        for _, chunk in self.chunks.items():
            print(f"Chunk ID    : {chunk['chunk_id']}")
            print(f"Title       : {chunk['title'].strip()}")
            print(f"Summary     : {chunk['summary'].strip()}")
            print("Propositions:")
            for prop in chunk["propositions"]:
                print(f"    - {prop}")
            print()
