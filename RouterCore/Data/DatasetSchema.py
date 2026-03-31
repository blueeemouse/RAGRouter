"""Shared constants and lightweight validation helpers for router data schemas."""

from typing import Dict, List


STRATEGY_NAMES: List[str] = [
    "llm_direct",
    "naive_rag",
    "graph_rag",
    "hybrid_rag",
    "iterative_rag_naive",
    "iterative_rag_graph",
]

STRATEGY_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(STRATEGY_NAMES)}
INDEX_TO_STRATEGY: Dict[int, str] = {idx: name for name, idx in STRATEGY_TO_INDEX.items()}


def normalize_strategy_name(method: str, retriever_type: str | None = None) -> str:
    """Normalize benchmark method names into router training strategy names."""
    method = method.strip().lower()
    if method == "iterative_rag":
        if retriever_type not in {"naive", "graph"}:
            raise ValueError("iterative_rag requires retriever_type 'naive' or 'graph'")
        return f"iterative_rag_{retriever_type}"
    if method not in STRATEGY_TO_INDEX:
        raise ValueError(f"Unknown strategy name: {method}")
    return method


def validate_strategy_names(strategy_names: List[str]) -> None:
    """Validate that a strategy list matches the agreed first-stage strategy space."""
    if strategy_names != STRATEGY_NAMES:
        raise ValueError(f"Strategy names must match first-stage strategy space: {STRATEGY_NAMES}")
