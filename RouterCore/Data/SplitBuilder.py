"""Build train/val/test splits for router datasets.

Implementation will be added incrementally in later steps.
"""


class SplitBuilder:
    """Construct dataset splits for router training data."""

    def __init__(self):
        pass

    def build(self, dataset_name: str, split_name: str = "split_v1"):
        """Build split metadata for a dataset."""
        raise NotImplementedError("SplitBuilder.build is not implemented yet")
