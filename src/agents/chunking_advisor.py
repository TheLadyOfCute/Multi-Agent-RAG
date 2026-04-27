"""
ChunkingAdvisorAgent — Recommends chunk_size and chunk_overlap for a document.

Called once per document upload, before any chunking takes place.
It receives the first 2000 tokens of the document and asks the LLM to
analyse the content characteristics and recommend suitable flat-chunking
parameters.

Usage
-----
>>> advisor = ChunkingAdvisorAgent(llm=llm)
>>> params  = advisor.advise(document_text)
>>> print(params)   # {"chunk_size": 600, "chunk_overlap": 80}
"""

import json
import re
from typing import Dict

import tiktoken
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.utils.logger import setup_logger
from src.utils.llm_content import message_content_to_text


# Hard bounds for the recommended parameters
_MIN_CHUNK_SIZE    = 100
_MAX_CHUNK_SIZE    = 2000
_MIN_OVERLAP       = 0
# overlap must not exceed half of chunk_size (enforced after LLM response)


_SYSTEM_PROMPT = """\
You are a document chunking expert.
Your task is to analyse a sample of a document and recommend the best
chunk_size (in tokens) and chunk_overlap (in tokens) for flat-text chunking.

Guidelines:
- chunk_size should be between 100 and 2000 tokens.
- chunk_overlap should be between 0 and chunk_size // 2.
- For dense technical / code documents → smaller chunks (200–500 tokens).
- For narrative / prose documents → medium chunks (400–800 tokens).
- For structured data (tables, lists) → medium-to-large chunks (500–1000 tokens).
- For legal / academic long-form text → larger chunks (600–1200 tokens).
- overlap should be roughly 10–20 % of chunk_size to preserve context across boundaries.

You MUST respond with valid JSON only, no additional text:
{"chunk_size": <integer>, "chunk_overlap": <integer>}
"""


class ChunkingAdvisorAgent:
    """
    Recommends flat-chunking parameters by analysing the first 2000 tokens
    of a document with an LLM.

    Parameters
    ----------
    llm : ChatOpenAI
        An OpenAI-compatible LLM client (e.g. Qwen via DashScope).
    preview_tokens : int, optional
        How many tokens of the document to send to the LLM.
        Defaults to 2000.

    Returns from advise()
    ----------------------
    dict with keys ``chunk_size`` (int) and ``chunk_overlap`` (int).
    On any failure the config defaults are returned so the upload is
    never blocked.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        preview_tokens: int = 2000,
    ):
        self.llm            = llm
        self.preview_tokens = preview_tokens
        self.settings       = get_settings()
        self.logger         = setup_logger("agent.chunking_advisor", level="INFO")
        self._encoder       = tiktoken.get_encoding("cl100k_base")

        self.logger.info(
            f"[chunking_advisor] Initialized "
            f"(preview_tokens={preview_tokens})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advise(self, document_text: str) -> Dict[str, int]:
        """
        Analyse the first ``preview_tokens`` tokens of *document_text* and
        return recommended chunking parameters.

        Parameters
        ----------
        document_text : str
            Full or partial document text.

        Returns
        -------
        dict
            ``{"chunk_size": int, "chunk_overlap": int}``
            Falls back to config defaults on LLM error.
        """
        fallback = self._fallback_params()

        if not document_text or not document_text.strip():
            self.logger.warning(
                "[chunking_advisor] Empty document text — using defaults"
            )
            return fallback

        preview = self._extract_preview(document_text)
        self.logger.info(
            f"[chunking_advisor] Sending {len(self._encoder.encode(preview))} "
            f"preview tokens to LLM…"
        )

        try:
            raw_response = self._call_llm(preview)
            params       = self._parse_response(raw_response)
            params       = self._validate_params(params)

            self.logger.info(
                f"[chunking_advisor] Recommended: "
                f"chunk_size={params['chunk_size']}, "
                f"chunk_overlap={params['chunk_overlap']}"
            )
            return params

        except Exception as exc:
            self.logger.warning(
                f"[chunking_advisor] LLM call failed ({exc}); "
                f"using default params: {fallback}"
            )
            return fallback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_preview(self, text: str) -> str:
        """Return at most *preview_tokens* tokens from the start of *text*."""
        tokens = self._encoder.encode(text)
        if len(tokens) <= self.preview_tokens:
            return text
        return self._encoder.decode(tokens[: self.preview_tokens])

    def _call_llm(self, preview: str) -> str:
        """Send the preview to the LLM and return the raw string response."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Below is the beginning of a document. "
                    "Please recommend chunk_size and chunk_overlap.\n\n"
                    f"---\n{preview}\n---"
                )
            ),
        ]
        response = self.llm.invoke(messages)
        return message_content_to_text(response).strip()

    def _parse_response(self, raw: str) -> Dict[str, int]:
        """
        Extract a JSON object from the LLM response.

        Handles both clean JSON responses and cases where the JSON is
        embedded inside markdown fences or extra prose.
        """
        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to extract a JSON object with a regex
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())

        raise ValueError(f"Cannot parse JSON from LLM response: {raw!r}")

    def _validate_params(self, params: Dict) -> Dict[str, int]:
        """
        Clamp and validate the parsed parameters to safe ranges.

        Rules
        -----
        - chunk_size   ∈ [100, 2000]
        - chunk_overlap ∈ [0, chunk_size // 2]
        """
        if not isinstance(params, dict):
            raise ValueError("Parsed params is not a dict")

        chunk_size = int(params.get("chunk_size", self.settings.chunk_size))
        chunk_overlap = int(params.get("chunk_overlap", self.settings.chunk_overlap))

        # Clamp chunk_size
        chunk_size = max(_MIN_CHUNK_SIZE, min(_MAX_CHUNK_SIZE, chunk_size))
        # Clamp chunk_overlap
        max_overlap = chunk_size // 2
        chunk_overlap = max(_MIN_OVERLAP, min(max_overlap, chunk_overlap))

        return {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}

    def _fallback_params(self) -> Dict[str, int]:
        """Return the system config defaults as the fallback."""
        return {
            "chunk_size":    self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
        }
