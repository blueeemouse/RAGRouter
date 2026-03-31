"""Aggregate query-level evaluation outputs into a unified router training dataset.

Implementation will be added incrementally in later steps.
"""


class EvaluationAggregator:
    """Aggregate per-method result evaluation files into unified query metrics."""

    def __init__(self):
        pass

    def build(self, dataset_name: str, result_model: str):
        """Build aggregated query-level metrics for a dataset/model pair."""
        raise NotImplementedError("EvaluationAggregator.build is not implemented yet")
