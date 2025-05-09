import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
import os
from torch.nn.utils.rnn import pad_sequence

class AGNews(Dataset):
    def __init__(self, root, transform=None, mode='train', vocab=None, tokenizer=None):
        self.root = root
        self.mode = mode
        self.num_classes = 4
        self.transform = transform
        
        # load from torchtext
        if mode == 'train':
            self.data = AG_NEWS(split='train')
        else:
            self.data = AG_NEWS(split='test')
            

        if tokenizer is None:
            self.tokenizer = get_tokenizer('basic_english')
        else:
            self.tokenizer = tokenizer
            

        if vocab is None:
            def yield_tokens(data_iter):
                for _, text in data_iter:
                    yield self.tokenizer(text)
                    
            self.vocab = build_vocab_from_iterator(
                yield_tokens(self.data),
                specials=['<unk>', '<pad>', '<bos>', '<eos>']
            )
            self.vocab.set_default_index(self.vocab['<unk>'])
        else:
            self.vocab = vocab
            
        # to list
        self.texts = []
        self.labels = []
        for label, text in self.data:
            self.texts.append(text)
            self.labels.append(label - 1)  
        
        self.train_labels = self.labels  
            
    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        
        if self.transform:
            text = self.transform(text)
            
        tokens = self.tokenizer(text)
        indices = self.vocab(tokens)
        
        # to tensor
        text_tensor = torch.tensor(indices, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return text_tensor, label_tensor
        
    def __len__(self):
        return len(self.texts)
        
    def get_vocab(self):
        return self.vocab
        
    def get_tokenizer(self):
        return self.tokenizer
        
    def get_vocab_size(self):
        return len(self.vocab)
        
    def get_pad_idx(self):
        return self.vocab['<pad>']
        
    def get_unk_idx(self):
        return self.vocab['<unk>']
        
    def get_bos_idx(self):
        return self.vocab['<bos>']
        
    def get_eos_idx(self):
        return self.vocab['<eos>']

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)  # 填充到相同长度
    labels = torch.tensor(labels, dtype=torch.long)
    return texts_padded, labels
