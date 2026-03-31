"""Router datasets.

Concrete dataset implementations will be added incrementally.
"""

from torch.utils.data import Dataset


class RouterSoftLabelDataset(Dataset):
    """Dataset for soft-label router training."""

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError("RouterSoftLabelDataset.__getitem__ is not implemented yet")
