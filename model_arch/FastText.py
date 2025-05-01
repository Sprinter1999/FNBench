import torch
import torch.nn as nn
import torch.nn.functional as F


class FastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=4):
        super(FastText, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded_sent = self.embedding(x)
        mask = x != self.embedding.padding_idx
        mask = mask.unsqueeze(-1).expand_as(embedded_sent)
        embedded_sent = embedded_sent * mask
        feat = self.fc1(embedded_sent.sum(1) / mask.sum(1))
        z = self.fc(feat)
        out = F.log_softmax(z, dim=1)
        return out, feat