"""Build soft labels from aggregated router query metrics.

Implementation will be added incrementally in later steps.
"""


class SoftLabelBuilder:
    """Construct soft-label router supervision data."""

    def __init__(self):
        pass

    def build(self, dataset_name: str, result_model: str):
        """Build soft-label data for a dataset/model pair."""
        raise NotImplementedError("SoftLabelBuilder.build is not implemented yet")
