"""Base class for router models.

Concrete model implementations will be added incrementally.
"""

from abc import ABC, abstractmethod


class BaseRouterModel(ABC):
    """Abstract base class for router models."""

    @abstractmethod
    def forward(self, batch):
        """Run the model on a unified batch payload."""
        raise NotImplementedError
