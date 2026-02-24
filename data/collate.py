from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


def dataset_split_collate_fn(batch):
    """
    Collate for both vision tensors and 1D token sequences.

    Supported sample formats:
      (x, y)
      (x, y, item)
      (x, y, item, real_idx)
    """
    if len(batch[0]) == 2:  # (data, label)
        data, labels = zip(*batch)
        if isinstance(data[0], torch.Tensor) and data[0].dim() == 1:
            data_padded = pad_sequence(data, batch_first=True, padding_value=0)
            labels = torch.tensor(labels, dtype=torch.long)
            return data_padded, labels
        data = torch.stack(data)
        labels = torch.tensor(labels, dtype=torch.long)
        return data, labels

    if len(batch[0]) == 3:  # (data, label, item)
        data, labels, items = zip(*batch)
        if isinstance(data[0], torch.Tensor) and data[0].dim() == 1:
            data_padded = pad_sequence(data, batch_first=True, padding_value=0)
            labels = torch.tensor(labels, dtype=torch.long)
            items = torch.tensor(items, dtype=torch.long)
            return data_padded, labels, items
        data = torch.stack(data)
        labels = torch.tensor(labels, dtype=torch.long)
        items = torch.tensor(items, dtype=torch.long)
        return data, labels, items

    if len(batch[0]) == 4:  # (data, label, item, real_idx)
        data, labels, items, real_idxs = zip(*batch)
        if isinstance(data[0], torch.Tensor) and data[0].dim() == 1:
            data_padded = pad_sequence(data, batch_first=True, padding_value=0)
            labels = torch.tensor(labels, dtype=torch.long)
            items = torch.tensor(items, dtype=torch.long)
            real_idxs = torch.tensor(real_idxs, dtype=torch.long)
            return data_padded, labels, items, real_idxs
        data = torch.stack(data)
        labels = torch.tensor(labels, dtype=torch.long)
        items = torch.tensor(items, dtype=torch.long)
        real_idxs = torch.tensor(real_idxs, dtype=torch.long)
        return data, labels, items, real_idxs

    raise ValueError(f"Unsupported batch format with {len(batch[0])} elements.")