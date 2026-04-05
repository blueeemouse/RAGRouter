"""Router datasets."""

from torch.utils.data import Dataset


class RouterSoftLabelTextDataset(Dataset):
    """Text-based soft-label router dataset.

    This class name is aligned with the current text pipeline v1 naming, even
    though the actual soft-label loading path is not implemented yet.
    """

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError("RouterSoftLabelTextDataset.__getitem__ is not implemented yet")


# Backward-compatible alias kept only during the migration transition.
RouterSoftLabelDataset = RouterSoftLabelTextDataset
