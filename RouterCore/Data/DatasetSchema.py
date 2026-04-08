"""Shared constants and lightweight validation helpers for router data schemas.

First-stage router data should treat sample ids as strings (for example: "musique_0002").
This matches the real benchmark data already produced by RAGRouter.
"""

from typing import Any, Dict, List


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
    """Validate strategy names as a non-empty, unique list of strings."""
    if not isinstance(strategy_names, list) or not strategy_names:
        raise ValueError("Strategy names must be a non-empty list")
    if any(not isinstance(name, str) or not name.strip() for name in strategy_names):
        raise ValueError("Each strategy name must be a non-empty string")
    if len(set(strategy_names)) != len(strategy_names):
        raise ValueError(f"Strategy names contain duplicates: {strategy_names}")


def get_strategy_index(strategy_name: str) -> int:
    """Return the integer class index for a normalized strategy name in default v1 space."""
    if strategy_name not in STRATEGY_TO_INDEX:
        raise ValueError(f"Unknown strategy name: {strategy_name}")
    return STRATEGY_TO_INDEX[strategy_name]


def get_strategy_name(index: int) -> str:
    """Return the normalized strategy name for a class index in default v1 space."""
    if index not in INDEX_TO_STRATEGY:
        raise ValueError(f"Unknown strategy index: {index}")
    return INDEX_TO_STRATEGY[index]


def get_strategy_index_from_list(strategy_name: str, strategies: List[str]) -> int:
    """Return class index using a caller-provided strategy list."""
    validate_strategy_names(strategies)
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy name: {strategy_name}")
    return strategies.index(strategy_name)


def get_strategy_name_from_list(index: int, strategies: List[str]) -> str:
    """Return strategy name using a caller-provided strategy list."""
    validate_strategy_names(strategies)
    if index < 0 or index >= len(strategies):
        raise ValueError(f"Unknown strategy index: {index}")
    return strategies[index]


def validate_sample_id(sample_id: str) -> None:
    """Validate that a sample id is a non-empty string."""
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("Router sample id must be a non-empty string")


def build_empty_method_metrics() -> Dict[str, Dict[str, Any]]:
    """Return the canonical per-method metrics skeleton for one query sample.

    Note:
    - `token_usage` is intentionally not part of the mandatory first-stage schema.
    - It may be added later as an optional extension field when cost-aware routing is introduced.
    """
    return {
        strategy: {
            "llm_label": None,
            "llm_reason": None,
            "llm_judge_correct": None,
            "semantic_f1": None,
            "coverage": None,
            "faithfulness_hard": None,
            "faithfulness_soft": None,
        }
        for strategy in STRATEGY_NAMES
    }
