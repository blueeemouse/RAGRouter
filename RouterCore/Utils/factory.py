"""Factory helpers for router models, datasets, and trainers.

Implementation will be added incrementally in later steps.
"""


class RouterFactory:
    """Factory for router training components."""

    @staticmethod
    def create_dataset(*args, **kwargs):
        raise NotImplementedError("RouterFactory.create_dataset is not implemented yet")

    @staticmethod
    def create_model(*args, **kwargs):
        raise NotImplementedError("RouterFactory.create_model is not implemented yet")

    @staticmethod
    def create_trainer(*args, **kwargs):
        raise NotImplementedError("RouterFactory.create_trainer is not implemented yet")
