"""Router datasets.

Concrete dataset implementations will be added incrementally.
"""

from torch.utils.data import Dataset


class RouterHardLabelDataset(Dataset):
    """Dataset for hard-label router training."""

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError("RouterHardLabelDataset.__getitem__ is not implemented yet")
