"""Build hard labels from aggregated router query metrics.

Implementation will be added incrementally in later steps.
"""


class HardLabelBuilder:
    """Construct hard-label router supervision data."""

    def __init__(self):
        pass

    def build(self, dataset_name: str, result_model: str):
        """Build hard-label data for a dataset/model pair."""
        raise NotImplementedError("HardLabelBuilder.build is not implemented yet")
