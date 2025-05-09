from PIL import Image
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from .sampling import sample_iid, sample_dirichlet, sample_noniid_shard
import torch.utils


def get_dataset(args):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'cifar10':
        data_path = '../data/cifar10'
        args.num_classes = 10
        # args.model = 'resnet18'
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        dataset_train = datasets.CIFAR10(
            data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR10(
            data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)
    elif args.dataset == 'cifar100':
        data_path = '../data/cifar100'
        args.num_classes = 100
        args.model = 'resnet34'
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        dataset_train = datasets.CIFAR100(
            data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR100(
            data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)


    else:
        exit('Error: unrecognized dataset')



    # FIXME: For Main experiments on CIFAR
    if args.iid:
        print("@@@@@@@@@@@@@         IID data partitioning")
        dict_users = sample_iid(n_train, args.num_users, args.seed)
    else:
        if(args.partition == 'dirichlet'):
            print(
                "@@@@@@@@@@@@@         Non-IID data partitioning via Dirichlet distribution")
            dict_users = sample_dirichlet(
                y_train, args.num_classes, args.non_iid_prob_class, args.num_users, args.seed, args.alpha_dirichlet)
        elif(args.partition == 'sharding'):
            print("@@@@@@@@@@@@@         Non-IID data partitioning via Sharding")

            dict_users = sample_noniid_shard(
                y_train, args.num_users, args.num_shards)

    return dataset_train, dataset_test, dict_users



class AGNews(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.texts = []
        self.labels = []
        

        self.num_classes = 4
        

        if mode == 'train':
            data_file = os.path.join(root, 'train.csv')
        else:
            data_file = os.path.join(root, 'test.csv')
            
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                label, title, description = line.strip().split(',', 2)

                text = title + ' ' + description
                self.texts.append(text)
                self.labels.append(int(label) - 1)  
                
    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        
        if self.transform:
            text = self.transform(text)
            
        return text, label
        
    def __len__(self):
        return len(self.texts)
