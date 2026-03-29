"""
Token Usage Tracker

Records per-question LLM token consumption for each API call.
Usage is attached to retrieval results and persisted to JSON.
"""
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class TokenCallRecord:
    """Single LLM API call token record."""
    phase: str          # "retrieval" or "generation"
    function: str       # caller function name, e.g. "extract_query_entities"
    round: Optional[int]  # iteration round for iterative RAG, null otherwise
    in_tokens: int      # prompt_tokens from response.usage
    out_tokens: int     # completion_tokens from response.usage


class TokenTracker:
    """Tracks token usage across multiple LLM calls within a single question."""

    def __init__(self):
        self.calls: List[TokenCallRecord] = []

    def track(self, response, phase: str, function: str, round: int = None) -> TokenCallRecord:
        """Extract usage from an OpenAI API response and record it.

        Args:
            response: OpenAI chat completion response object
            phase: "retrieval" or "generation"
            function: name of the calling function
            round: iteration round (for iterative RAG), None otherwise

        Returns:
            The recorded TokenCallRecord
        """
        usage = response.usage
        record = TokenCallRecord(
            phase=phase,
            function=function,
            round=round,
            in_tokens=usage.prompt_tokens,
            out_tokens=usage.completion_tokens,
        )
        self.calls.append(record)
        return record

    def to_dict(self) -> dict:
        """Serialize all records to a JSON-serializable dict."""
        return {
            "calls": [asdict(r) for r in self.calls],
            "total": self._total(),
        }

    def _total(self) -> dict:
        """Sum across all calls."""
        total_in = sum(r.in_tokens for r in self.calls)
        total_out = sum(r.out_tokens for r in self.calls)
        return {
            "in_tokens": total_in,
            "out_tokens": total_out,
            "total_tokens": total_in + total_out,
        }

    def reset(self):
        """Clear all records (call before processing a new question)."""
        self.calls.clear()
