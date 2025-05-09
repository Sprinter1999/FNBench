import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from .cifar import CIFAR10, CIFAR100
import os
import numpy as np
from .agnews import AGNews

def load_dataset(dataset):
    """
    Returns: dataset_train, dataset_test, num_classes, collate_fn
    """
    dataset_train = None
    dataset_test = None
    num_classes = 0
    collate_fn = None

    if dataset == 'cifar10':
        trans_cifar10_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_cifar10_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        #TODO: please load the dataset in advance to /FNBench/data/cifar10, etc
        dataset_train = CIFAR10(
            root='./data/cifar10',
            download=False,
            train=True,
            transform=trans_cifar10_train,
        )
        dataset_test = CIFAR10(
            root='./data/cifar10',
            download=False,
            train=False,
            transform=trans_cifar10_val,
        )
        num_classes = 10

    elif dataset == 'cifar100':
        trans_cifar100_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])],
        )
        trans_cifar100_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])],
        )
        dataset_train = CIFAR100(
            root='./data/cifar100',
            download=False,
            train=True,
            transform=trans_cifar100_train,
        )
        dataset_test = CIFAR100(
            root='./data/cifar100',
            download=False,
            train=False,
            transform=trans_cifar100_val,
        )
        num_classes = 100

    elif dataset == 'AGNews':
        data_path = './data/agnews'
        num_classes = 4
        

        trans_train = transforms.Compose([
            transforms.Lambda(lambda x: x.lower()),  
            transforms.Lambda(lambda x: x.strip()), 
        ])
        trans_val = transforms.Compose([
            transforms.Lambda(lambda x: x.lower()),
            transforms.Lambda(lambda x: x.strip()),
        ])
        

        dataset_train = AGNews(
            root=data_path,
            transform=trans_train,
            mode='train'
        )
        

        dataset_test = AGNews(
            root=data_path,
            transform=trans_val,
            mode='test',
            vocab=dataset_train.get_vocab(),
            tokenizer=dataset_train.get_tokenizer()
        )
        
        # custom collate_fn for text 
        def collate_fn(batch):
            texts, labels = zip(*batch)
            texts_padded = pad_sequence(texts, batch_first=True, padding_value=dataset_train.get_pad_idx())
            labels = torch.tensor(labels, dtype=torch.long)
            return texts_padded, labels

        n_train = len(dataset_train)
        y_train = np.array(dataset_train.labels)
        
        print('[DATA] train labels size:', len(y_train))
        print('[DATA] vocab size:', dataset_train.get_vocab_size())
        # args.vocab_size = dataset_train.get_vocab_size()

    else:
        raise NotImplementedError('Error: unrecognized dataset')

    return dataset_train, dataset_test, num_classes, collate_fn
