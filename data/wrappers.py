from __future__ import annotations

from torch.utils.data import Dataset


class DatasetSplit(Dataset):
    """
    View a subset of a dataset via an index list.

    idx_return:
      returns (x, y, item_idx) where item_idx is 0..len(idxs)-1

    real_idx_return:
      returns (x, y, item_idx, real_idx) where real_idx is original dataset index
    """

    def __init__(self, dataset, idxs, idx_return=False, real_idx_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return
        self.real_idx_return = real_idx_return

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        x, y = self.dataset[self.idxs[item]]

        if self.idx_return:
            return x, y, item
        if self.real_idx_return:
            return x, y, item, self.idxs[item]
        return x, y


class PairProbDataset(Dataset):
    """
    Returns (x1, x2, y, prob[, item_idx]).
    Used by pair-based methods needing per-sample probability.
    """

    def __init__(self, dataset, idxs, prob, idx_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return
        self.prob = prob

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        x1, y = self.dataset[self.idxs[item]]
        x2, _ = self.dataset[self.idxs[item]]
        p = self.prob[self.idxs[item]]

        if self.idx_return:
            return x1, x2, y, p, item
        return x1, x2, y, p


class PairDataset(Dataset):
    """
    Returns (x1, x2[, y][, item_idx]).
    Used by co-teaching style pipelines.
    """

    def __init__(self, dataset, idxs, idx_return=False, label_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return
        self.label_return = label_return

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        x1, y = self.dataset[self.idxs[item]]
        x2, _ = self.dataset[self.idxs[item]]
        out = (x1, x2)

        if self.label_return:
            out = out + (y,)
        if self.idx_return:
            out = out + (item,)
        return out


class DatasetSplitRFL(Dataset):
    """
    Returns (x, y, real_idx). Some robust FL variants need real indices.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        x, y = self.dataset[self.idxs[item]]
        return x, y, self.idxs[item]