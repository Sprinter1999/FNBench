import numpy as np
from sklearn.mixture import GaussianMixture

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast as autocast
from torch.nn.utils.rnn import pad_sequence
import copy


from .correctors import SelfieCorrector, JointOptimCorrector
# from .nets import get_model
from model_arch.build_model import build_model as get_model

import torchvision
from torchvision import transforms


def dataset_split_collate_fn(batch):
    if len(batch[0]) == 2:  # (data, label)
        data, labels = zip(*batch)
        if isinstance(data[0], torch.Tensor) and data[0].dim() == 1:  
            data_padded = pad_sequence(data, batch_first=True, padding_value=0)  
            labels = torch.tensor(labels, dtype=torch.long)
            return data_padded, labels
        else:  # images
            data = torch.stack(data)
            labels = torch.tensor(labels, dtype=torch.long)
            return data, labels

    elif len(batch[0]) == 3:  # (data, label, item)
        data, labels, items = zip(*batch)
        if isinstance(data[0], torch.Tensor) and data[0].dim() == 1:  
            data_padded = pad_sequence(data, batch_first=True, padding_value=0)
            labels = torch.tensor(labels, dtype=torch.long)
            items = torch.tensor(items, dtype=torch.long)
            return data_padded, labels, items
        else:  # images
            data = torch.stack(data)
            labels = torch.tensor(labels, dtype=torch.long)
            items = torch.tensor(items, dtype=torch.long)
            return data, labels, items

    elif len(batch[0]) == 4:  # (data, label, item, real_idx)
        data, labels, items, real_idxs = zip(*batch)
        if isinstance(data[0], torch.Tensor) and data[0].dim() == 1:  
            data_padded = pad_sequence(data, batch_first=True, padding_value=0)
            labels = torch.tensor(labels, dtype=torch.long)
            items = torch.tensor(items, dtype=torch.long)
            real_idxs = torch.tensor(real_idxs, dtype=torch.long)
            return data_padded, labels, items, real_idxs
        else:  # images
            data = torch.stack(data)
            labels = torch.tensor(labels, dtype=torch.long)
            items = torch.tensor(items, dtype=torch.long)
            real_idxs = torch.tensor(real_idxs, dtype=torch.long)
            return data, labels, items, real_idxs

    else:
        raise ValueError(f"Unsupported batch format with {len(batch[0])} elements.")


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, idx_return=False, real_idx_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return
        self.real_idx_return = real_idx_return

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label = self.dataset[self.idxs[item]]

        if self.idx_return:
            return image, label, item
        elif self.real_idx_return:
            return image, label, item, self.idxs[item]
        else:
            return image, label


class PairProbDataset(Dataset):
    def __init__(self, dataset, idxs, prob, idx_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return
        self.prob = prob

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image1, label = self.dataset[self.idxs[item]]
        image2, label = self.dataset[self.idxs[item]]
        prob = self.prob[self.idxs[item]]

        if self.idx_return:
            return image1, image2, label, prob, item
        else:
            return image1, image2, label, prob


class PairDataset(Dataset):
    def __init__(self, dataset, idxs, idx_return=False, label_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return
        self.label_return = label_return

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image1, label = self.dataset[self.idxs[item]]
        image2, label = self.dataset[self.idxs[item]]
        sample = (image1, image2,)

        if self.label_return:
            sample += (label,)

        if self.idx_return:
            sample += (item,)

        return sample


class DatasetSplitRFL(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label = self.dataset[self.idxs[item]]

        return image, label, self.idxs[item]


def mixup(inputs, targets, alpha=1.0):
    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)

    idx = torch.randperm(inputs.size(0))

    input_a, input_b = inputs, inputs[idx]
    target_a, target_b = targets, targets[idx]

    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    return mixed_input, mixed_target


def linear_rampup(current, warm_up, lambda_u, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u * float(current)


class SemiLoss:
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, lambda_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        # labeled data loss
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x,
                         dim=1) * targets_x, dim=1))
        # unlabeled data loss
        Lu = torch.mean((probs_u - targets_u) ** 2)

        lamb = linear_rampup(epoch, warm_up, lambda_u)

        return Lx + lamb * Lu


def get_local_update_objects(args, dataset_train, dict_users=None, noise_rates=None, gaussian_noise=None, glob_centroid=None):
    local_update_objects = []
    for idx, noise_rate in zip(range(args.num_users), noise_rates):
        local_update_args = dict(
            args=args,
            user_idx=idx,
            dataset=dataset_train,
            idxs=dict_users[idx],
        )

        # TODO: original federated learning methods
        if args.method == 'fedavg' or args.method == 'median' or args.method == 'krum' or args.method == 'trimmedMean' or args.method == 'fedexp' or args.method =='RFA':
            local_update_object = BaseLocalUpdate(**local_update_args)

        elif args.method == 'fedprox':
            # print(f"generate fedprox clients")
            local_update_object = LocalUpdateFedProx(**local_update_args)

        elif args.method == 'clipping':
            # print(f"generate fedprox clients")
            local_update_object = LocalUpdateClipping(**local_update_args)

        # TODO: noise resilient methods
        elif args.method == 'symmetricce':
            local_update_object = LocalUpdateSymmetric(**local_update_args)

        elif args.method == 'fedlsr':
            local_update_object = LocalUpdateFedLSR(**local_update_args)

        elif args.method == 'robustfl':
            local_update_object = LocalUpdateRFL(**local_update_args)

        elif args.method == 'fedrn':
            local_update_object = LocalUpdateFedRN(
                gaussian_noise=gaussian_noise, **local_update_args)

        elif args.method == 'selfie':
            local_update_object = LocalUpdateSELFIE(
                noise_rate=noise_rate, **local_update_args)

        elif args.method == 'jointoptim':
            local_update_object = LocalUpdateJointOptim(**local_update_args)

        elif args.method in ['coteaching', 'coteaching+']:
            local_update_object = LocalUpdateCoteaching(is_coteaching_plus=bool(args.method == 'coteaching+'),
                                                        **local_update_args)
        elif args.method == 'dividemix':
            local_update_object = LocalUpdateDivideMix(**local_update_args)

        elif args.method == 'fednoro':
            local_update_object = LocalUpdateFedNoRo(**local_update_args)

        elif args.method == 'fedELC':
            local_update_object = LocalUpdateFedELC(**local_update_args)

        else:
            raise NotImplementedError

        local_update_objects.append(local_update_object)

    return local_update_objects


#FIXME: You can add related codes according to your need in your forked code base. This is partially removed in this file to clean up codes :)
class FedDecorrLoss(nn.Module):

    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8
        

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        N, C = x.shape
        if N == 1:
            return 0.0

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / N

        return loss


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(
            labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


# 5:'fedavg', 'krum','median','trimmedMean', 'fedexp'
class BaseLocalUpdate:
    def __init__(
            self,
            args,
            user_idx=None,
            dataset=None,
            idxs=None,
            idx_return=False,
            real_idx_return=False,
    ):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()

        self.dataset = dataset
        self.idxs = idxs
        self.user_idx = user_idx
        self.update_name = args.method

        self.idx_return = idx_return
        self.real_idx_return = real_idx_return
        
        #TODO: Load custom collate_fn
        collate_fn = dataset_split_collate_fn if args.collate_fn else None

        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return,
                         real_idx_return=real_idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn= collate_fn if args.collate_fn else None,
            pin_memory=True,
            drop_last=True,
        )

        self.total_epochs = 0
        self.epoch = 0
        self.batch_idx = 0

        self.net1 = get_model(self.args)
        self.net2 = get_model(self.args)


        self.last_updated = 0

        self.is_svd_loss = args.is_svd_loss
        self.feddecorr = FedDecorrLoss()
        self.decorr_coef = args.decorr_coef

    def train(self, net, net2=None):
        if net2 is None:
            return self.train_single_model(net)
        else:
            return self.train_multiple_models(net, net2)

    def train_single_model(self, net):

        # net.to(self.args.device)
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx

                if(len(batch) == 0):
                    continue

                net.zero_grad()

                # with autocast():
                loss = self.forward_pass(batch, net)
                

                loss.backward()
                optimizer.step()


                batch_loss.append(loss.item())
                self.on_batch_end()

            if(len(batch_loss) > 0):
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        net.to('cpu')
        # del net

        return self.net1.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_multiple_models(self, net1, net2):

        # net1.to(self.args.device)
        # net2.to(self.args.device)

        net1.train()
        net2.train()

        optimizer_args = dict(
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        optimizer1 = torch.optim.SGD(net1.parameters(), **optimizer_args)
        optimizer2 = torch.optim.SGD(net2.parameters(), **optimizer_args)

        epoch_loss1 = []
        epoch_loss2 = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss1 = []
            batch_loss2 = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net1.zero_grad()
                net2.zero_grad()

                # with autocast():
                loss1, loss2 = self.forward_pass(batch, net1, net2)

                loss1.backward()
                loss2.backward()
                optimizer1.step()
                optimizer2.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print(f"Epoch: {epoch} [{batch_idx}/{len(self.ldr_train)}"
                          f"({100. * batch_idx / len(self.ldr_train):.0f}%)]\tLoss: {loss1.item():.6f}"
                          f"\tLoss: {loss2.item():.6f}")

                batch_loss1.append(loss1.item())
                batch_loss2.append(loss2.item())
                self.on_batch_end()

            epoch_loss1.append(sum(batch_loss1) / len(batch_loss1))
            epoch_loss2.append(sum(batch_loss2) / len(batch_loss2))
            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net1.state_dict())
        self.net2.load_state_dict(net2.state_dict())
        self.last_updated = self.args.g_epoch

        # net1.to('cpu')
        # net2.to('cpu')
        # del net1
        # del net2

        return self.net1.state_dict(), sum(epoch_loss1) / len(epoch_loss1), \
            self.net2.state_dict(), sum(epoch_loss2) / len(epoch_loss2)

    def forward_pass(self, batch, net, net2=None):
        images, labels = batch

        # text
        if isinstance(images, torch.Tensor) and images.dim() == 2:  # text (but named with images), bad name!
            images = images.to(self.args.device)
        else:  # images
            images = images.to(self.args.device).float()

        labels = labels.to(self.args.device)

        log_probs, features = net(images)
        loss = self.loss_func(log_probs, labels)
        
        if self.is_svd_loss:
            loss_feddecorr = self.feddecorr(features)
            loss += loss_feddecorr * self.decorr_coef

        if net2 is None:
            return loss

        # 处理第二个模型
        log_probs2, features2 = net2(images)
        loss2 = self.loss_func(log_probs2, labels)

        return loss, loss2

    def on_batch_end(self):
        pass

    def on_epoch_end(self):
        pass





class LocalUpdateSymmetric(BaseLocalUpdate):
    def __init__(
            self,
            args,
            user_idx=None,
            dataset=None,
            idxs=None,
            idx_return=False,
            real_idx_return=False,
    ):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()

        self.dataset = dataset
        self.idxs = idxs
        self.user_idx = user_idx
        self.update_name = 'SCE'

        self.idx_return = idx_return
        self.real_idx_return = real_idx_return
        
        #TODO: Load custom collate_fn
        collate_fn = dataset_split_collate_fn if args.collate_fn else None

        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return,
                         real_idx_return=real_idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers= self.args.num_workers,
            collate_fn= collate_fn if self.args.collate_fn else None,
            pin_memory=True,
            drop_last=True
        )

        self.total_epochs = 0
        self.epoch = 0
        self.batch_idx = 0

        self.net1 = get_model(self.args)


        self.last_updated = 0


    def train(self, net):
        return self.train_single_model(net)

    def train_single_model(self, net):

        # global_params = copy.deepcopy(list(net.parameters()))
        net.to(self.args.device)
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []
        criterion = SCELoss(
            alpha=0.1, beta=1.0, num_classes=self.args.num_classes)
        
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                # with autocast():
                loss = self.forward_pass(batch, net, criterion)

                loss.backward()
                optimizer.step()


                batch_loss.append(loss.item())
                self.on_batch_end()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        net.to('cpu')
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def forward_pass(self, batch, net, criterion):
        
        if self.idx_return:
            images, labels, _ = batch

        elif self.real_idx_return:
            images, labels, _, ids = batch
        else:
            images, labels = batch

        images = images.to(self.args.device)
        labels = labels.to(self.args.device)

        with autocast():
            log_probs, features = net(images)
        # loss = criterion(output, labels)
        loss = criterion(log_probs, labels)


        return loss

    def on_batch_end(self):
        pass

    def on_epoch_end(self):
        pass

#TODO: clipping
class LocalUpdateClipping:
    def __init__(
            self,
            args,
            user_idx=None,
            dataset=None,
            idxs=None,
            idx_return=False,
            real_idx_return=False,
    ):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()

        self.dataset = dataset
        self.idxs = idxs
        self.user_idx = user_idx
        self.update_name = args.method

        self.idx_return = idx_return
        self.real_idx_return = real_idx_return
        
        #TODO: Load custom collate_fn
        collate_fn = dataset_split_collate_fn if args.collate_fn else None

        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return,
                         real_idx_return=real_idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn= collate_fn if self.args.collate_fn else None,
            pin_memory=True,
            drop_last=True,
        )

        self.total_epochs = 0
        self.epoch = 0
        self.batch_idx = 0

        self.net1 = get_model(self.args)
        self.net2 = get_model(self.args)
        # self.net1 = self.net1.to(self.args.device)
        # self.net2 = self.net2.to(self.args.device)

        self.max_grad_norm = args.max_grad_norm
        self.last_updated = 0

        self.feddecorr = FedDecorrLoss()
        self.decorr_coef = args.decorr_coef

    def train(self, net, net2=None):
        if net2 is None:
            return self.train_single_model(net)
        else:
            return self.train_multiple_models(net, net2)

    def train_single_model(self, net):

        global_model = copy.deepcopy(net)

        net.to(self.args.device)
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                with autocast():
                    loss = self.forward_pass(batch, net)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm) 
                optimizer.step()


                batch_loss.append(loss.item())
                self.on_batch_end()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        net.to('cpu')

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def forward_pass(self, batch, net, net2=None):
        if self.idx_return:
            images, labels, _ = batch

        elif self.real_idx_return:
            images, labels, _, ids = batch
        else:
            images, labels = batch

        images = images.to(self.args.device)
        labels = labels.to(self.args.device)

        log_probs, _ = net(images)
        loss = self.loss_func(log_probs, labels)

        if net2 is None:
            return loss

        # 2 models
        log_probs2, _ = net2(images)
        loss2 = self.loss_func(log_probs2, labels)
        return loss, loss2

    def on_batch_end(self):
        pass

    def on_epoch_end(self):
        pass


# 1:'fedprox'
class LocalUpdateFedProx:
    def __init__(
            self,
            args,
            user_idx=None,
            dataset=None,
            idxs=None,
            idx_return=False,
            real_idx_return=False,
    ):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()

        self.dataset = dataset
        self.idxs = idxs
        self.user_idx = user_idx
        self.update_name = args.method

        self.idx_return = idx_return
        self.real_idx_return = real_idx_return
        
        #TODO: Load custom collate_fn
        collate_fn = dataset_split_collate_fn if args.collate_fn else None

        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return,
                         real_idx_return=real_idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn if self.args.collate_fn else None,
            pin_memory=True,
            drop_last=True,
        )

        self.total_epochs = 0
        self.epoch = 0
        self.batch_idx = 0

        self.net1 = get_model(self.args)
        self.net2 = get_model(self.args)


        self.last_updated = 0

        self.decorr_coef = args.decorr_coef

    def train(self, net, net2=None):
        if net2 is None:
            return self.train_single_model(net)
        else:
            return self.train_multiple_models(net, net2)

    def train_single_model(self, net):

        global_model = copy.deepcopy(net)

        net.to(self.args.device)
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                with autocast():
                    loss = self.forward_pass(batch, net)

                    proximal_term = 0.0
                    for w, w_t in zip(net.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)

                    loss += 0.5 * self.args.mu * proximal_term

                loss.backward()
                optimizer.step()


                batch_loss.append(loss.item())
                self.on_batch_end()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        net.to('cpu')

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def forward_pass(self, batch, net, net2=None):
        if self.idx_return:
            images, labels, _ = batch

        elif self.real_idx_return:
            images, labels, _, ids = batch
        else:
            images, labels = batch

        images = images.to(self.args.device)
        labels = labels.to(self.args.device)

        log_probs, feat = net(images)
        loss = self.loss_func(log_probs, labels)

        if net2 is None:
            return loss

        # 2 models
        log_probs2, _ = net2(images)
        loss2 = self.loss_func(log_probs2, labels)
        return loss, loss2

    def on_batch_end(self):
        pass

    def on_epoch_end(self):
        pass


# 1: 'fedlsr'
class LocalUpdateFedLSR(BaseLocalUpdate):
    def __init__(
            self,
            args,
            user_idx=None,
            dataset=None,
            idxs=None,
            idx_return=False,
            real_idx_return=False,
    ):
        self.args = args

        self.dataset = dataset
        self.idxs = idxs
        self.user_idx = user_idx
        self.update_name = 'LSR'

        self.idx_return = idx_return
        self.real_idx_return = real_idx_return
        self.loss_func = nn.CrossEntropyLoss()
        self.total_epochs = 0
        self.net1 = get_model(self.args)
        # self.net1 = self.net1.to(self.args.device)
        
        #TODO: Load custom collate_fn
        collate_fn = dataset_split_collate_fn if args.collate_fn else None

        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return,
                         real_idx_return=real_idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn= collate_fn if self.args.collate_fn else None,
            pin_memory=True,
            drop_last=True
        )

        # by following the original paper
        self.warmup_epochs = args.epochs * args.warm_up_ratio_lsr
        self.gamma_e = self.args.gamma_e
        self.gamma = self.args.gamma
        self.distill_reverse_t = self.args.distill_t

        self.args.g_epoch = 0

        self.s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        self.tt_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)])
        
        self.feddecorr = FedDecorrLoss()
        self.decorr_coef = args.decorr_coef

    def js(self, p_output, q_output):
        # Jensen-Shannon divergence between two distributions
        KLDivLoss = nn.KLDivLoss(reduction='mean')
        log_mean_output = ((p_output + q_output)/2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

    def train(self, net, net2=None, cur_round=0):
        self.args.g_epoch = cur_round
        if net2 is None:
            return self.train_single_model(net)
        else:
            return self.train_multiple_models(net, net2)

    def train_single_model(self, net):

        # net.to(self.args.device)
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                with autocast():
                    loss = self.forward_pass(batch, net)

                loss.backward()
                optimizer.step()


                batch_loss.append(loss.item())
                self.on_batch_end()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        net.to('cpu')
        # del net

        return self.net1.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # return a single loss term

    def forward_pass(self, batch, net, net2=None):

        # net.to(self.args.device)

        if self.idx_return:
            images, labels, _ = batch

        elif self.real_idx_return:
            images, labels, _, ids = batch
        else:
            images, labels = batch

        images = images.to(self.args.device)
        labels = labels.to(self.args.device)
        images_aug = self.tt_transform(images).to(self.args.device)

        sm = torch.nn.Softmax(dim=1)
        lsm = torch.nn.LogSoftmax(dim=1)

        with autocast():

            output1, feat = net(images)  # make a forward pass
            output2, __ = net(images_aug)  # make a forward pass


            mix_1 = np.random.beta(1, 1)  # mixing predict1 and predict2
            mix_2 = 1-mix_1

            logits1, logits2 = torch.softmax(
                output1*self.distill_reverse_t, dim=1), torch.softmax(output2*self.distill_reverse_t, dim=1)
            # for training stability to conduct clamping to avoid exploding gradients
            logits1, logits2 = torch.clamp(
                logits1, min=1e-6, max=1.0), torch.clamp(logits2, min=1e-6, max=1.0)

            # to mix up the two predictions
            p = torch.softmax(output1, dim=1)*mix_1 + \
                torch.softmax(output2, dim=1)*mix_2

            betaa = self.gamma
            if(self.args.g_epoch < self.warmup_epochs):
                betaa = self.gamma * self.args.g_epoch/self.warmup_epochs

            # to get sharpened prediction p_s
            pt = p**(2)
            # normalize the prediction
            pred_mix = pt / pt.sum(dim=1, keepdim=True)

            loss = self.loss_func(pred_mix, labels)
            L_e = - (torch.mean(torch.sum(sm(logits1) * lsm(logits1), dim=1)) +
                     torch.mean(torch.sum(sm(logits1) * lsm(logits1), dim=1))) * 0.5

            loss += self.js(logits1, logits2) * betaa

            loss += L_e * self.gamma_e


            return loss


# 1: 'fedrn'
class LocalUpdateFedRN(BaseLocalUpdate):
    def __init__(self, args, dataset=None, user_idx=None, idxs=None, gaussian_noise=None):
        super().__init__(
            args=args,
            dataset=dataset,
            user_idx=user_idx,
            idxs=idxs,
            real_idx_return=True,
        )
        self.gaussian_noise = gaussian_noise
        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.update_name = 'fedrn'
        
        #TODO: Load custom collate_fn
        self.collate_fn = dataset_split_collate_fn if self.args.collate_fn else None

        self.ldr_eval = DataLoader(
            DatasetSplit(dataset, idxs, real_idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn if self.args.collate_fn else None,
            pin_memory=True,
            # drop_last=True
        )
        self.data_indices = np.array(idxs)
        self.expertise = 0.5
        self.arbitrary_output = torch.rand((1, self.args.num_classes))
        self.net1 = get_model(self.args)
        
                




    def set_expertise(self):
        self.net1.to(self.args.device)
        self.net1.eval()
        correct = 0
        n_total = len(self.ldr_eval.dataset)

        with torch.no_grad():
            for batch_idx, (inputs, targets, items, idxs) in enumerate(self.ldr_eval):
                inputs, targets = inputs.to(
                    self.args.device), targets.to(self.args.device)
                with autocast():
                    outputs,_ = self.net1(inputs)
                y_pred = outputs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(targets.data.view_as(y_pred)
                                     ).float().sum().item()
            expertise = correct / n_total

        self.expertise = expertise


    def set_arbitrary_output(self):
        self.net1.to(self.args.device)
        with autocast():
            arbitrary_output,_ = self.net1(self.gaussian_noise.to(self.args.device))
        self.arbitrary_output = arbitrary_output

    def train_phase1(self, net):
        # local training
        w, loss = self.train_single_model(net)
        self.set_expertise()
        self.set_arbitrary_output()
        self.net1.to('cpu')
        return w, loss

    def fit_gmm(self, net):
        losses = []
        net.eval()
        net.to(self.args.device)
        with torch.no_grad():
            for batch_idx, (inputs, targets, items, idxs) in enumerate(self.ldr_eval):
                inputs, targets = inputs.to(
                    self.args.device), targets.to(self.args.device)
                with autocast():
                    outputs, _ = net(inputs)
                loss = self.CE(outputs, targets)
                losses.append(loss)

        losses = torch.cat(losses).cpu().numpy()
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        input_loss = losses.reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, max_iter=100,
                              tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]

        return prob

    def get_clean_idx(self, prob):
        threshold = self.args.p_threshold
        pred = (prob > threshold)
        pred_clean_idx = pred.nonzero()[0]
        pred_clean_idx = self.data_indices[pred_clean_idx]
        pred_noisy_idx = (1 - pred).nonzero()[0]
        pred_noisy_idx = self.data_indices[pred_noisy_idx]

        if len(pred_clean_idx) == 0:
            pred_clean_idx = pred_noisy_idx
            pred_noisy_idx = np.array([])

        return pred_clean_idx, pred_noisy_idx

    def finetune_head(self, neighbor_list, pred_clean_idx):
        loader = DataLoader(
            DatasetSplit(self.dataset, pred_clean_idx, real_idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.args.collate_fn if self.args.collate_fn else None,
            pin_memory=True,
            # drop_last=True
        )

        optimizer_list = []
        for neighbor_net in neighbor_list:
            neighbor_net.train()
            body_params = [
                p for name, p in neighbor_net.named_parameters() if 'linear' not in name]
            head_params = [
                p for name, p in neighbor_net.named_parameters() if 'linear' in name]

            optimizer = torch.optim.SGD([
                {'params': head_params, 'lr': self.args.lr,
                 'momentum': self.args.momentum,
                 'weight_decay': self.args.weight_decay},
                {'params': body_params, 'lr': 0.0},
            ])
            optimizer_list.append(optimizer)

        for batch_idx, (inputs, targets, items, idxs) in enumerate(loader):
            inputs, targets = inputs.to(
                self.args.device), targets.to(self.args.device)

            for neighbor_net, optimizer in zip(neighbor_list, optimizer_list):
                neighbor_net.to(self.args.device)
                neighbor_net.zero_grad()

                with autocast():
                    outputs, _ = neighbor_net(inputs)
                loss = self.loss_func(outputs, targets)
                loss.backward()
                optimizer.step()

        return neighbor_list

    def train_phase2(self, net, prev_score, neighbor_list, neighbor_score_list):
        # Prev fit GMM & get clean idx
        self.net1.to(self.args.device)
        prob = self.fit_gmm(self.net1)
        pred_clean_idx, pred_noisy_idx = self.get_clean_idx(prob)

        prob_list = [prob]
        neighbor_list = self.finetune_head(neighbor_list, pred_clean_idx)
        for neighbor_net in neighbor_list:
            neighbor_prob = self.fit_gmm(neighbor_net)
            prob_list.append(neighbor_prob)

        # Scores
        score_list = [prev_score] + neighbor_score_list
        score_list = [score / sum(score_list) for score in score_list]

        # Get final prob
        final_prob = np.zeros(len(prob))
        for prob, score in zip(prob_list, score_list):
            final_prob = np.add(final_prob, np.multiply(prob, score))
        # Get final clean idx
        final_clean_idx, final_noisy_idx = self.get_clean_idx(final_prob)

        # Update loader with final clean idxs
        self.ldr_train = DataLoader(DatasetSplit(self.dataset, final_clean_idx, real_idx_return=True),
                                    batch_size=self.args.local_bs,
                                    shuffle=True,
                                    num_workers=self.args.num_workers,
                                    collate_fn=self.collate_fn if self.args.collate_fn else None,
                                    pin_memory=True,
                                    # drop_last=True
                                    )
        # local training
        w, loss = self.train_single_model(net)
        self.set_expertise()
        self.set_arbitrary_output()

        self.net1.to('cpu')
        return w, loss

#FIXME: RobustFL


class LocalUpdateRFL(BaseLocalUpdate):
    def __init__(self, args, dataset=None, user_idx=None, idxs=None):
        super().__init__(
            args=args,
            dataset=dataset,
            user_idx=user_idx,
            idxs=idxs,
            # noise_logger=noise_logger
        )
        self.pseudo_labels = torch.zeros(
            len(self.dataset), dtype=torch.long, device=self.args.device)
        self.sim = torch.nn.CosineSimilarity(dim=1)
        self.loss_func = torch.nn.CrossEntropyLoss(reduce=False)
        
        #TODO: Load custom collate_fn
        self.collate_fn = dataset_split_collate_fn if self.args.collate_fn else None
        
        
        self.ldr_train = DataLoader(
            DatasetSplitRFL(dataset, idxs),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn if self.args.collate_fn else None,
            drop_last=True
        )
        self.ldr_train_tmp = DataLoader(DatasetSplitRFL(
            dataset, idxs), batch_size=1, shuffle=True,collate_fn=self.collate_fn if self.args.collate_fn else None,)
        # self.tmp_true_labels = tmp_true_labels
        self.update_name = 'RobustFL'



    def train(self, model, current_round, f_G, forget_ratee):
        num_classes = self.args.num_classes
        
        criterion = nn.CrossEntropyLoss()
        f_k = torch.zeros(num_classes, self.args.feature_dim, device=self.args.device)
        n_labels = torch.zeros(num_classes, 1, device=self.args.device)
        e_loss = []
        self.args.g_epoch = current_round
        model.cuda()
        optimizer = torch.optim.SGD(model.parameters(
            ), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train_tmp):
                # if torch.cuda.is_available():
                images, labels ,idxs= images.cuda(), labels.cuda(),idxs.cuda()
                with autocast():
                    logit, feature = model(images)
                self.pseudo_labels[idxs] = torch.argmax(logit)
                if self.args.g_epoch == 0:
                    f_k[labels] += feature
                    n_labels[labels] += 1

        if self.args.g_epoch == 0:
            for i in range(len(n_labels)):
                if n_labels[i] == 0:
                    n_labels[i] = 1
            f_k = torch.div(f_k, n_labels)
        else:
            f_k = f_G

        model.train()
        for iter in range(self.args.local_ep): #for epoch in range(self.args.local_ep):
            batch_loss = []
            correct_num = 0
            total = 0
            for batch_idx, batch in enumerate(self.ldr_train):
                model.zero_grad()
                images, labels, idx = batch
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                
                with autocast():
                    logit, feature = model(images)
                    # feature_copy = copy.deepcopy(feature)
                    
                feature = feature.detach()
                if torch.cuda.is_available():
                    f_k = f_k.to(self.args.device)

                small_loss_idxs = self.get_small_loss_samples(
                    logit, labels, forget_ratee)
                

                y_k_tilde = torch.zeros(self.args.local_bs, device=self.args.device)
                mask = torch.zeros(self.args.local_bs, device=self.args.device)
                for i in small_loss_idxs:
                    y_k_tilde[i] = torch.argmax(
                        self.sim(f_k, torch.reshape(feature[i], (1, self.args.feature_dim))))
                    if y_k_tilde[i] == labels[i]:
                        mask[i] = 1

                self.pseudo_labels = self.pseudo_labels.to(self.args.device)
                idx = idx.to(self.args.device)  

                # When to use pseudo-labels
                if self.args.g_epoch < self.args.T_pl:
                    for i in small_loss_idxs:
                        self.pseudo_labels[idx[i]] = labels[i]

                # For loss calculating
                new_labels = mask[small_loss_idxs] * labels[small_loss_idxs] + \
                    (1 - mask[small_loss_idxs]) * self.pseudo_labels[idx[small_loss_idxs]]
                new_labels = new_labels.type(torch.LongTensor).to(self.args.device)

                loss = self.RFLloss(logit, labels, feature,
                                    f_k, mask, small_loss_idxs, new_labels)


                # weight update by minimizing loss: L_total = L_c + lambda_cen * L_cen + lambda_e * L_e
                loss.backward()
                optimizer.step()

                # obtain loss based average features f_k,j_hat from small loss dataset
                f_kj_hat = torch.zeros(num_classes, self.args.feature_dim, device=self.args.device)
                n = torch.zeros(num_classes, 1, device=self.args.device)
                for i in small_loss_idxs:
                    f_kj_hat[labels[i]] += feature[i]
                    n[labels[i]] += 1
                for i in range(len(n)):
                    if n[i] == 0:
                        n[i] = 1
                f_kj_hat = torch.div(f_kj_hat, n)

                # update local centroid f_k
                one = torch.ones(num_classes, 1, device=self.args.device)
                f_k = (one - self.sim(f_k, f_kj_hat).reshape(num_classes, 1) ** 2) * \
                    f_k + (self.sim(f_k, f_kj_hat).reshape(num_classes, 1)
                           ** 2) * f_kj_hat

                batch_loss.append(loss.item())
            
            temp = sum(batch_loss)/len(batch_loss)
            e_loss.append(temp)
            # print(f"loss for this epoch is {temp}")

        return model.state_dict(), sum(e_loss) / len(e_loss), f_k


    def RFLloss(self, logit, labels, feature, f_k, mask, small_loss_idxs, new_labels):
        mse = torch.nn.MSELoss(reduction='none')
        ce = torch.nn.CrossEntropyLoss()
        sm = torch.nn.Softmax(dim=1)
        lsm = torch.nn.LogSoftmax(dim=1)

        L_c = ce(logit[small_loss_idxs], new_labels)
        L_cen = torch.sum(mask[small_loss_idxs] * torch.sum(
            mse(feature[small_loss_idxs], f_k[labels[small_loss_idxs]]), 1))
        L_e = - \
            torch.mean(
                torch.sum(sm(logit[small_loss_idxs]) * lsm(logit[small_loss_idxs]), dim=1))

        # TODO:self.T_pl,lambda_e,lambda_cen
        lambda_e = self.args.lambda_e
        lambda_cen = self.args.lambda_cen
        if self.args.g_epoch < self.args.T_pl:
            lambda_cen = (self.args.lambda_cen * self.args.g_epoch) / self.args.T_pl

        return L_c + (lambda_cen * L_cen) + (lambda_e * L_e)

    def get_small_loss_samples(self, y_pred, y_true, forget_ratee):
        loss = self.loss_func(y_pred, y_true)
        ind_sorted = np.argsort(loss.data.cpu()).to(self.args.device)
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - forget_ratee
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update = ind_sorted[:num_remember]

        return ind_update

# 1: 'fedavg + selfie'
class LocalUpdateSELFIE(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, noise_rate=0):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            real_idx_return=True,
        )
        self.update_name = 'selfie'
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.total_epochs = 0
        self.warmup = args.warmup_epochs
        self.corrector = SelfieCorrector(
            queue_size=args.queue_size,
            uncertainty_threshold=args.uncertainty_threshold,
            noise_rate=noise_rate,
            num_classes=args.num_classes,
        )


    def forward_pass(self, batch, net, net2=None):
        net.to(self.args.device)

        images, labels, _, ids = batch
        images = images.to(self.args.device)
        labels = labels.to(self.args.device)
        ids = ids.numpy()

        log_probs, _ = net(images)
        loss_array = self.loss_func(log_probs, labels)

        # update prediction history
        self.corrector.update_prediction_history(
            ids=ids,
            outputs=log_probs.cpu().detach().numpy(),
        )

        if self.args.g_epoch >= self.args.warmup_epochs:
            # correct labels, remove noisy data
            images, labels, ids = self.corrector.patch_clean_with_corrected_sample_batch(
                ids=ids,
                X=images,
                y=labels,
                loss_array=loss_array.cpu().detach().numpy(),
            )
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            with autocast():
                log_probs, _ = net(images)
            loss_array = self.loss_func(log_probs, labels)



        loss = loss_array.mean()

        # net.to('cpu')

        return loss

    def train(self, net, net2=None):
        if net2 is None:
            return self.train_single_model(net)
        else:
            return self.train_multiple_models(net, net2)

    def train_single_model(self, net):

        net.to(self.args.device)
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                # with autocast():
                loss = self.forward_pass(batch, net)

                loss.backward()
                optimizer.step()


                batch_loss.append(loss.item())
                self.on_batch_end()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        net.to('cpu')
        # del net

        return self.net1.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateJointOptim(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            real_idx_return=True,
        )
        self.corrector = JointOptimCorrector(
            queue_size=args.queue_size,
            num_classes=args.num_classes,
            data_size=len(idxs),
        )
        self.update_name = args.method



    def train(self, net, net2=None):
        if net2 is None:
            return self.train_single_model(net)
        else:
            return self.train_multiple_models(net, net2)

    def train_single_model(self, net):

        net.to(self.args.device)
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                with autocast():
                    loss = self.forward_pass(batch, net)

                loss.backward()
                optimizer.step()


                batch_loss.append(loss.item())
                self.on_batch_end()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        net.to('cpu')
        # del net

        return self.net1.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def forward_pass(self, batch, net, net2=None):

        net.to(self.args.device)
        images, labels, _, ids = batch
        ids = ids.numpy()

        hard_labels, soft_labels = self.corrector.get_labels(ids, labels)
        if self.args.labeling == 'soft':
            labels = soft_labels.to(self.args.device)
        else:
            labels = hard_labels.to(self.args.device)
        images = images.to(self.args.device)

        # with autocast():
        logits, feat = net(images)
        probs = F.softmax(logits, dim=1)


        loss = self.joint_optim_loss(logits, probs, labels)

        
        self.corrector.update_probability_history(ids, probs.cpu().detach())

        # net.to('cpu')
        return loss

    def on_epoch_end(self):
        if self.args.g_epoch >= self.args.warmup_epochs:
            self.corrector.update_labels()

    def joint_optim_loss(self, logits, probs, soft_targets, is_cross_entropy=False):
        if is_cross_entropy:
            loss = - \
                torch.mean(torch.sum(F.log_softmax(
                    logits, dim=1) * soft_targets, dim=1))

        else:
            # We introduce a prior probability distribution p,
            # which is a distribution of classes among all training data.
            p = torch.ones(self.args.num_classes,
                           device=self.args.device) / self.args.num_classes

            avg_probs = torch.mean(probs, dim=0)

            L_c = -torch.mean(torch.sum(F.log_softmax(logits,
                              dim=1) * soft_targets, dim=1))
            L_p = -torch.sum(torch.log(avg_probs) * p)
            L_e = - \
                torch.mean(torch.sum(F.log_softmax(
                    logits, dim=1) * probs, dim=1))

            loss = L_c + self.args.alpha * L_p + self.args.beta * L_e

        return loss


class LocalUpdateCoteaching(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None, is_coteaching_plus=False):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            real_idx_return=True,
        )
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        self.is_coteaching_plus = is_coteaching_plus
        self.update_name = args.method
        self.init_epoch = 10  # only used for coteaching+
        self.args.g_epoch = 0


    def train(self, net, net2, cur_epoch):
        self.args.g_epoch = cur_epoch
        # if net2 is None:
        #     return self.train_single_model(net)
        # else:
        return self.train_multiple_models(net, net2)

    def train_single_model(self, net):

        # net.to(self.args.device)
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                with autocast():
                    loss = self.forward_pass(batch, net)

                loss.backward()
                optimizer.step()


                batch_loss.append(loss.item())
                self.on_batch_end()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        # net.to('cpu')
        # del net

        return self.net1.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_multiple_models(self, net1, net2):

        net1.to(self.args.device)
        net2.to(self.args.device)

        net1.train()
        net2.train()

        optimizer_args = dict(
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        optimizer1 = torch.optim.SGD(net1.parameters(), **optimizer_args)
        optimizer2 = torch.optim.SGD(net2.parameters(), **optimizer_args)

        epoch_loss1 = []
        epoch_loss2 = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss1 = []
            batch_loss2 = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net1.zero_grad()
                net2.zero_grad()

                with autocast():
                    loss1, loss2 = self.forward_pass(batch, net1, net2)

                loss1.backward()
                loss2.backward()
                optimizer1.step()
                optimizer2.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print(f"Epoch: {epoch} [{batch_idx}/{len(self.ldr_train)}"
                          f"({100. * batch_idx / len(self.ldr_train):.0f}%)]\tLoss: {loss1.item():.6f}"
                          f"\tLoss: {loss2.item():.6f}")

                batch_loss1.append(loss1.item())
                batch_loss2.append(loss2.item())
                self.on_batch_end()

            epoch_loss1.append(sum(batch_loss1) / len(batch_loss1))
            epoch_loss2.append(sum(batch_loss2) / len(batch_loss2))
            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net1.state_dict())
        self.net2.load_state_dict(net2.state_dict())
        self.last_updated = self.args.g_epoch

        net1.to('cpu')
        net2.to('cpu')
        # del net1
        # del net2

        return self.net1.state_dict(), sum(epoch_loss1) / len(epoch_loss1), \
            self.net2.state_dict(), sum(epoch_loss2) / len(epoch_loss2)


    def forward_pass(self, batch, net, net2=None):
        net.to(self.args.device)

        # if net2 is not None:
        net2.to(self.args.device)

        images, labels, indices, ids = batch

        images = images.to(self.args.device)
        labels = labels.to(self.args.device)
        log_probs1, feat1 = net(images)
        log_probs2, feat2 = net2(images)

        loss_args = dict(
            y_pred1=log_probs1,
            y_pred2=log_probs2,
            y_true=labels,
            forget_rate=self.args.forget_rate,
        )

        if self.is_coteaching_plus and self.args.g_epoch >= self.init_epoch:
            # print("&&&Co-teaching plus")
            loss1, loss2, indices = self.loss_coteaching_plus(
                indices=indices, step=self.epoch * self.batch_idx, **loss_args)
        else:
            loss1, loss2, indices = self.loss_coteaching(**loss_args)



        return loss1, loss2

    def loss_coteaching(self, y_pred1, y_pred2, y_true, forget_rate):
        loss_1 = self.loss_func(y_pred1, y_true)
        ind_1_sorted = torch.argsort(loss_1)

        loss_2 = self.loss_func(y_pred2, y_true)
        ind_2_sorted = torch.argsort(loss_2)

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(ind_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        # exchange
        loss_1_update = self.loss_func(
            y_pred1[ind_2_update], y_true[ind_2_update])
        loss_2_update = self.loss_func(
            y_pred2[ind_1_update], y_true[ind_1_update])

        ind_1_update = list(ind_1_update.cpu().detach().numpy())

        return torch.sum(loss_1_update) / num_remember, torch.sum(loss_2_update) / num_remember, ind_1_update

    def loss_coteaching_plus(self, y_pred1, y_pred2, y_true, forget_rate, indices, step):
        outputs = F.softmax(y_pred1, dim=1)
        outputs2 = F.softmax(y_pred2, dim=1)

        _, pred1 = torch.max(y_pred1.data, 1)
        _, pred2 = torch.max(y_pred2.data, 1)

        pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

        logical_disagree_id = np.zeros(y_true.size(), dtype=bool)
        disagree_id = []
        for idx, p1 in enumerate(pred1):
            if p1 != pred2[idx]:
                disagree_id.append(idx)
                logical_disagree_id[idx] = True

        temp_disagree = indices * logical_disagree_id.astype(np.int64)
        ind_disagree = np.asarray(
            [i for i in temp_disagree if i != 0]).transpose()
        try:
            assert ind_disagree.shape[0] == len(disagree_id)
        except:
            disagree_id = disagree_id[:ind_disagree.shape[0]]

        if len(disagree_id) > 0:
            update_labels = y_true[disagree_id]
            update_outputs = outputs[disagree_id]
            update_outputs2 = outputs2[disagree_id]
            loss_1, loss_2, indices = self.loss_coteaching(
                update_outputs, update_outputs2, update_labels, forget_rate)
        else:
            update_step = np.logical_or(
                logical_disagree_id, step < 5000).astype(np.float32)
            update_step = Variable(torch.from_numpy(update_step)).cuda()

            cross_entropy_1 = F.cross_entropy(outputs, y_true)
            cross_entropy_2 = F.cross_entropy(outputs2, y_true)

            loss_1 = torch.sum(
                update_step * cross_entropy_1) / y_true.size()[0]
            loss_2 = torch.sum(
                update_step * cross_entropy_2) / y_true.size()[0]
            indices = range(y_true.size()[0])
        return loss_1, loss_2, indices

# 1: 'fedavg + dividemix'


class LocalUpdateDivideMix(BaseLocalUpdate):
    def __init__(self, args, user_idx=None, dataset=None, idxs=None):
        super().__init__(
            args=args,
            user_idx=user_idx,
            dataset=dataset,
            idxs=idxs,
            idx_return=True,
        )
        self.update_name = args.method
        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.CEloss = nn.CrossEntropyLoss()
        self.semiloss = SemiLoss()

        self.loss_history1 = []
        self.loss_history2 = []

        self.ldr_eval = DataLoader(
            DatasetSplit(dataset, idxs, real_idx_return=True),
            batch_size=self.args.local_bs,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=self.args.collate_fn if self.args.collate_fn else None,
            pin_memory=True,
            # drop_last=True
        )

    def train(self, net, net2, cur_epoch):
        self.args.g_epoch = cur_epoch
        if self.args.g_epoch <= self.args.warmup_epochs:
            return self.train_multiple_models(net, net2)
        else:
            return self.train_2_phase(net, net2)

    def train_2_phase(self, net, net2):
        net.to(self.args.device)
        net2.to(self.args.device)

        epoch_loss1 = []
        epoch_loss2 = []

        for ep in range(self.args.local_ep):
            prob_dict1, label_idx1, unlabel_idx1 = self.update_probabilties_split_data_indices(
                net, self.loss_history1)
            prob_dict2, label_idx2, unlabel_idx2 = self.update_probabilties_split_data_indices(
                net2, self.loss_history2)

            # train net1
            loss1 = self.divide_mix(
                net=net,
                net2=net2,
                label_idx=label_idx2,
                prob_dict=prob_dict2,
                unlabel_idx=unlabel_idx2,
                warm_up=self.args.warmup_epochs,
                epoch=self.args.g_epoch,
            )

            # train net2
            loss2 = self.divide_mix(
                net=net2,
                net2=net,
                label_idx=label_idx1,
                prob_dict=prob_dict1,
                unlabel_idx=unlabel_idx1,
                warm_up=self.args.warmup_epochs,
                epoch=self.args.g_epoch,
            )

            self.net1.load_state_dict(net.state_dict())
            self.net2.load_state_dict(net2.state_dict())

            self.total_epochs += 1
            epoch_loss1.append(loss1)
            epoch_loss2.append(loss2)

        loss1 = sum(epoch_loss1) / len(epoch_loss1)
        loss2 = sum(epoch_loss2) / len(epoch_loss2)

        net.to('cpu')
        net2.to('cpu')

        return net.state_dict(), loss1, net2.state_dict(), loss2

    def divide_mix(self, net, net2, label_idx, prob_dict, unlabel_idx, warm_up, epoch):
        net.train()
        net2.eval()  # fix one network and train the other

        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # dataloader
        labeled_trainloader = DataLoader(
            PairProbDataset(self.dataset, label_idx, prob_dict),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.args.collate_fn if self.args.collate_fn else None,
            pin_memory=True,
            # drop_last=True
        )
        unlabeled_trainloader = DataLoader(
            PairDataset(self.dataset, unlabel_idx, label_return=False),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.args.collate_fn if self.args.collate_fn else None,
            pin_memory=True,
            # drop_last=True
        )
        unlabeled_train_iter = iter(unlabeled_trainloader)
        num_iter = len(labeled_trainloader)

        batch_loss = []
        for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
            try:
                inputs_u, inputs_u2 = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2 = unlabeled_train_iter.next()

            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, self.args.num_classes) \
                .scatter_(1, labels_x.view(-1, 1), 1)
            w_x = w_x.view(-1, 1).type(torch.FloatTensor)

            inputs_x = inputs_x.to(self.args.device)
            inputs_x2 = inputs_x2.to(self.args.device)
            labels_x = labels_x.to(self.args.device)
            w_x = w_x.to(self.args.device)

            inputs_u = inputs_u.to(self.args.device)
            inputs_u2 = inputs_u2.to(self.args.device)

            with torch.no_grad():
                # label co-guessing of unlabeled samples
                with autocast():
                    outputs_u11, _ = net(inputs_u)
                    outputs_u12, _ = net(inputs_u2)
                    outputs_u21, _ = net2(inputs_u)
                    outputs_u22, _ = net2(inputs_u2)

                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) +
                      torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
                
                ptu = pu ** (1 / self.args.T)  # temparature sharpening

                targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                targets_u = targets_u.detach()

                # label refinement of labeled samples
                with autocast():
                    outputs_x, _ = net(inputs_x)
                    outputs_x2, _ = net(inputs_x2)

                px = (torch.softmax(outputs_x, dim=1) +
                      torch.softmax(outputs_x2, dim=1)) / 2
                px = w_x * labels_x + (1 - w_x) * px
                ptx = px ** (1 / self.args.T)  # temparature sharpening

                targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                targets_x = targets_x.detach()

            # mixmatch
            all_inputs = torch.cat(
                [inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat(
                [targets_x, targets_x, targets_u, targets_u], dim=0)

            mixed_input, mixed_target = mixup(
                all_inputs, all_targets, alpha=self.args.mm_alpha)

            logits, feat = net(mixed_input)
            # compute loss
            loss = self.semiloss(
                outputs_x=logits[:batch_size * 2],
                targets_x=mixed_target[:batch_size * 2],
                outputs_u=logits[batch_size * 2:],
                targets_u=mixed_target[batch_size * 2:],
                lambda_u=self.args.lambda_u,
                epoch=epoch + batch_idx / num_iter,
                warm_up=warm_up,
            )


            # regularization
            prior = torch.ones(self.args.num_classes,
                               device=self.args.device) / self.args.num_classes
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))
            loss += penalty

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())

        return sum(batch_loss) / len(batch_loss)

    def update_probabilties_split_data_indices(self, model, loss_history):
        model.eval()
        losses_lst = []
        idx_lst = []

        with torch.no_grad():
            for batch_idx, (inputs, targets, items, idxs) in enumerate(self.ldr_eval):
                inputs, targets = inputs.to(
                    self.args.device), targets.to(self.args.device)
                
                with autocast():
                    outputs,_ = model(inputs)

                losses_lst.append(self.CE(outputs, targets))
                idx_lst.append(idxs.cpu().numpy())

        indices = np.concatenate(idx_lst)
        losses = torch.cat(losses_lst).cpu().numpy()
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        loss_history.append(losses)

        # Fit a two-component GMM to the loss
        input_loss = losses.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=100,
                              tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]

        # Split data to labeled, unlabeled dataset
        pred = (prob > self.args.p_threshold)
        label_idx = pred.nonzero()[0]
        label_idx = indices[label_idx]

        unlabel_idx = (1 - pred).nonzero()[0]
        unlabel_idx = indices[unlabel_idx]

        # Data index : probability
        prob_dict = {idx: prob for idx, prob in zip(indices, prob)}

        return prob_dict, label_idx, unlabel_idx



class LocalUpdateFedNoRo(BaseLocalUpdate):
    def __init__(
            self,
            args,
            user_idx=None,
            dataset=None,
            idxs=None,
            idx_return=False,
            real_idx_return=False,
    ):
        self.args = args

        self.dataset = dataset
        self.idxs = idxs
        self.user_idx = user_idx
        self.update_name = 'FedNoRo'

        self.idx_return = idx_return
        self.real_idx_return = real_idx_return
        self.total_epochs = 0
        
        #TODO: Load custom collate_fn
        self.collate_fn = dataset_split_collate_fn if self.args.collate_fn else None

        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return,
                         real_idx_return=real_idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn if self.args.collate_fn else None,
            pin_memory=True,
            drop_last=True
        )

        self.class_sum = np.array([0] * args.num_classes) 
        for idx in self.idxs:
            label = self.dataset.train_labels[idx]
            self.class_sum[label] += 1

        from utils.losses import LogitAdjust, LA_KD
        self.loss_func1 = LogitAdjust(cls_num_list=self.class_sum)
        self.loss_func2 = LA_KD(cls_num_list=self.class_sum)
        self.net1 = get_model(self.args)
        self.last_updated = 0



    def train_stage1(self, net):  # train with LA
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                if self.idx_return:
                    images, labels, _ = batch
                elif self.real_idx_return:
                    images, labels, _, ids = batch
                else:
                    images, labels = batch
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                logits, feat = net(images)    
                loss = self.loss_func1(logits, labels)


                loss.backward()
                optimizer.step()


                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1

        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        net.to('cpu')
        # del net

        return self.net1.state_dict(), sum(epoch_loss) / len(epoch_loss)
        
        
    def train_stage2(self, net, global_net, weight_kd):
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                if self.idx_return:
                    images, labels, _ = batch
                elif self.real_idx_return:
                    images, labels, _, ids = batch
                else:
                    images, labels = batch
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                logits, feat = net(images)
                with torch.no_grad():
                    teacher_output, ____ = global_net(images)
                    soft_label = torch.softmax(teacher_output/0.8, dim=1) 
                loss = self.loss_func2(logits, labels, soft_label, weight_kd)


                loss.backward()
                optimizer.step()


                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1

        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        net.to('cpu')
        # del net

        return self.net1.state_dict(), sum(epoch_loss) / len(epoch_loss)




class LocalUpdateFedELC(BaseLocalUpdate):
    def __init__(
            self,
            args,
            user_idx=None,
            dataset=None,
            idxs=None,
            idx_return=False,
            real_idx_return=True,
    ):
        self.args = args

        self.dataset = dataset
        self.idxs = idxs
        
        

        self.user_idx = user_idx
        self.update_name = 'FedELC'





        self.idx_return = idx_return
        self.real_idx_return = True #Fixed
        self.total_epochs = 0
        
        #TODO: Load custom collate_fn
        self.collate_fn = dataset_split_collate_fn if self.args.collate_fn else None

        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return,
                         real_idx_return=real_idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn if self.args.collate_fn else None,
            pin_memory=True,
            drop_last=True
        )

        self.ldr_train_infer = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return,
                         real_idx_return=real_idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn if self.args.collate_fn else None,
            pin_memory=True,
            drop_last=False
        )

        self.class_sum = np.array([0] * args.num_classes) 
        for idx in self.idxs:
            label = self.dataset.train_labels[idx]
            self.class_sum[label] += 1

        from utils.losses import LogitAdjust, LA_KD, LogitAdjust_soft
        self.loss_func1 = LogitAdjust(cls_num_list=self.class_sum)
        self.loss_func_soft = LogitAdjust_soft(cls_num_list=self.class_sum)
        self.loss_func2 = LA_KD(cls_num_list=self.class_sum)
        self.net1 = get_model(self.args)
        self.last_updated = 0

        self.is_svd_loss = args.is_svd_loss
        self.feddecorr = FedDecorrLoss()
        self.decorr_coef = args.decorr_coef


        self.local_datasize = len(idxs)

        
        self.index_mapper, self.index_mapper_inv = {}, {}

        for i in range(len(self.idxs)):
            self.index_mapper[self.idxs[i]] = i
            self.index_mapper_inv[i] = self.idxs[i]

        self.label_update = torch.index_select(
            args.Soft_labels, 0, torch.tensor(self.idxs))
        # yy = torch.FloatTensor(yy)
        self.label_update = torch.FloatTensor(self.label_update)
        
        self.true_labels_local = torch.index_select(
            args.True_Labels, 0, torch.tensor(self.idxs))

        self.estimated_labels = copy.deepcopy(self.label_update)

        #yield by the local model after E local epochs
        self.final_prediction_labels = copy.deepcopy(self.label_update)

        # self.estimated_labels = F.softmax(self.label_update, dim=1)
        self.lamda = args.lamda_pencil



        for batch_idx, batch in enumerate(self.ldr_train_infer):

            if self.idx_return:
                images, labels, _ = batch
            elif self.real_idx_return:
                images, labels, _, ids = batch
            else:
                images, labels = batch

            indexss = self.indexMapping(ids)
            # self.label_update[indexss].cuda()
            for i in range(len(indexss)):
                self.label_update[indexss[i]][labels[i]] = self.args.K_pencil #fixed 10

        print(f"Initializing the client #{self.user_idx}... Done")





    # from overall index to local index
    def indexMapping(self, indexs):
        indexss = indexs.cpu().numpy().tolist()
        target_mapping = []
        for each in indexss:
            target_mapping.append(self.index_mapper[each])
        return target_mapping

    def label_updating(self, labels_grad):
        self.label_update = self.label_update - self.lamda * labels_grad
        self.estimated_labels = F.softmax(self.label_update, dim=1)



    def pencil_loss(self, outputs, labels_update, labels, feat):

        pred = F.softmax(outputs, dim=1)
        #yd = F.softmax(labels_update, dim=1)

        Lo = -torch.mean(F.log_softmax(labels_update, dim=1)[torch.arange(labels_update.shape[0]),labels])

        Le = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * pred, dim=1))


        Lc = -torch.mean(torch.sum(F.log_softmax(labels_update, dim=1) * pred, dim=1)) - Le
        
        loss_total = Lc/self.args.num_classes + self.args.alpha_pencil* Lo + self.args.beta_pencil* Le/self.args.num_classes 
        
        
        # print(f"lc: {Lc},  le: {Le}, lo: {Lo}")

        return loss_total
    


    def train_stage1(self, net):  # train with LA
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []




        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            


            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                if self.idx_return:
                    images, labels, _ = batch
                elif self.real_idx_return:
                    images, labels, _, ids = batch
                else:
                    images, labels = batch
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                logits, feat = net(images)    
                loss = self.loss_func1(logits, labels)

                if self.is_svd_loss:
                    loss_feddecorr = self.feddecorr(feat)
                    loss += loss_feddecorr * self.decorr_coef

                loss.backward()
                optimizer.step()


                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1

        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        net.to('cpu')
        # del net

        return self.net1.state_dict(), sum(epoch_loss) / len(epoch_loss)
    


    #TODO: for noisy clients in the second phase
    def train_stage2(self, net, global_net, weight_kd):
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []




        #TODO: To begin the local training
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []

            labels_grad = torch.zeros(self.local_datasize, self.args.num_classes, dtype=torch.float32)


            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                if self.idx_return:
                    images, labels, _ = batch

                #TODO: we use the below one
                elif self.real_idx_return:
                    images, labels, _, ids = batch
                else:
                    images, labels = batch


                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                with autocast():
                    logits, feat = net(images)


                indexss = self.indexMapping(ids)

                labels_update = self.label_update[indexss,:].cuda()
                labels_update.requires_grad_()
                # labels_update = torch.autograd.Variable(labels_update,requires_grad = True)

                loss = self.pencil_loss(
                                logits, labels_update, labels, feat)
                
                if self.is_svd_loss:
                    loss_feddecorr = self.feddecorr(feat)
                    loss += loss_feddecorr * self.decorr_coef


                loss.backward()


                labels_grad[indexss] = labels_update.grad.cpu().detach() #.numpy()

                labels_update = labels_update.to('cpu')
                del labels_update

                optimizer.step()


                batch_loss.append(loss.item())
            

            self.label_updating(labels_grad)


            

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1
        
        # After E local epochs
        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch



        #TODO: traverse dataset
        after_correct_predictions = 0
        for batch_idx, batch in enumerate(self.ldr_train_infer):

            if self.idx_return:
                images, labels, _ = batch
            elif self.real_idx_return:
                images, labels, _, ids = batch
            else:
                images, labels = batch

            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            local_index = self.indexMapping(ids)

            # with autocast():
            with torch.no_grad():
                output_final, teacher_feat = net(images)
                output_final = output_final.to('cpu')

                soft_label = torch.softmax(output_final, dim=1) 
                self.final_prediction_labels[local_index]  = soft_label

        
        net.to('cpu')
        del net





        #TODO: merge the softmax(self.label_update) and the prediction after local training(self.final_prediction_labels)
        self.label_update = self.label_update.to('cpu')

        updated_local_labels_tmp = F.softmax(self.label_update, dim=1)
        final_model_prediction_tmp = self.final_prediction_labels
        # average the above two
        merged_local_labels = (updated_local_labels_tmp + final_model_prediction_tmp) / 2
        # the GT labels is self.true_labels_local
        predicted_classes = torch.argmax(merged_local_labels, dim=1)



        # replace the label_update with the merged_local_labels, and rescale by K_pencil
        self.label_update = merged_local_labels * self.args.K_pencil



        
        # del net

        return self.net1.state_dict(), sum(epoch_loss) / len(epoch_loss)
