import copy
import torch
# from torch import nn
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# TODO: This file manages all the clients' state and the aggregation protocals


class LocalModelWeights:
    def __init__(self, all_clients, net_glob, num_users, method, dict_users, args):
        self.all_clients = all_clients
        self.num_users = num_users
        self.method = method
        self.args = args
        self.user_data_size = [len(dict_users[i]) for i in range(num_users)]
        

        if self.user_data_size and \
                all([self.user_data_size[0] == data_size for data_size in self.user_data_size]):
            self.user_data_size = [1] * len(self.user_data_size)

        self.model_ = copy.deepcopy(net_glob)
        w_glob = net_glob.state_dict()
        self.global_w_init = net_glob.state_dict()  # which can be used for FedExp

        if self.all_clients:
            print("Aggregation over all clients")
            self.w_locals = [w_glob for i in range(self.num_users)]
            self.data_size_locals = self.user_data_size
        else:
            self.w_locals = []
            self.data_size_locals = []

    def init(self):
        # Reset local weights if necessary
        if not self.all_clients:
            self.w_locals = []
            self.data_size_locals = []

    def update(self, idx, w):
        if self.all_clients:
            self.w_locals[idx] = copy.deepcopy(w)
        else:
            self.w_locals.append(copy.deepcopy(w))
            self.data_size_locals.append(self.user_data_size[idx])

    def average(self):
        w_glob = None
        # approaches for original methods which not modify the aggregation process
        if self.method == 'fedavg':
            w_glob = FedAvg(self.w_locals, self.data_size_locals)
        
        
        elif self.method == 'fedexp':
            w_glob = FedExp(
                self.w_locals, self.data_size_locals, self.global_w_init,copy.deepcopy(self.model_))
        
        elif self.method == 'RFA':
            w_glob = RFA(self.w_locals)
        
        
        elif self.method == 'fednoro':
            if self.noisy_clients == 0:
                w_glob = FedAvg(self.w_locals, self.data_size_locals)
            else:
                #FIXME:
                assert len(self.client_tag) == self.args.selected_total_clients_num, self.client_tag

                if len(set(self.client_tag)) == 1: # all clean or all noisy, use FedAvg
                    w_glob = FedAvg(self.w_locals, self.data_size_locals)
                else:
                    w_glob = DaAgg(
                        self.w_locals, self.data_size_locals, self.client_tag)

                self.client_tag = [] # initial again for the next round


        elif self.method == 'fedELC':
            if self.noisy_clients == 0:
                w_glob = FedAvg(self.w_locals, self.data_size_locals)
            else:
                #FIXME: 10 is computed by 100*client_selection_ratio(0.1), we can modify it to make it more flexible
                assert len(self.client_tag) == self.args.selected_total_clients_num, self.client_tag

                if len(set(self.client_tag)) == 1: # all clean or all noisy, use FedAvg
                    w_glob = FedAvg(self.w_locals, self.data_size_locals)
                else:
                    w_glob = DaAgg(
                        self.w_locals, self.data_size_locals, self.client_tag)

                self.client_tag = [] # initial again for the next round
                
        
        
        
        # approaches for robust aggregation
        elif self.method == 'krum':
            # add an interface for comprimised number
            compromised_num = int(len(self.w_locals) *
                                  self.args.compromised_rate)
            # we mildly set comprimised number as 20% of all selected clients(which is unknown in practice)
            self.args.device = torch.device('cuda:{}'.format(
                self.args.gpu) if torch.cuda.is_available() and self.args.gpu != -1 else 'cpu')
            w_glob = Krum(self.w_locals, compromised_num, self.args)
        
        
        elif self.method == 'median':
            w_glob = Median(self.w_locals)
        
        
        elif self.method == 'trimmedMean':
            # add an interface for comprimised number
            compromised_num = int(len(self.w_locals) *
                                  self.args.compromised_rate)
            self.args.device = torch.device('cuda:{}'.format(
                self.args.gpu) if torch.cuda.is_available() and self.args.gpu != -1 else 'cpu')
            w_glob = trimmed_mean(
                self.w_locals, compromised_num, self.args)
        
        
        else:
            # default method for aggregation
            w_glob = FedAvg(self.w_locals, self.data_size_locals)

        return w_glob


# average_weights is a list of weights for each client
def FedAvg(w, average_weights):
    global_w_update = copy.deepcopy(w[0])
    for k in global_w_update.keys():
        global_w_update[k] *= average_weights[0]
        for i in range(1, len(w)):
            global_w_update[k] += w[i][k] * average_weights[i]
        global_w_update[k] = torch.div(
            global_w_update[k], sum(average_weights))

    return global_w_update



def RFA(w):
    """
    Weiszfeld
    """
    global_w_update = copy.deepcopy(w[0]) 
    eps = 1e-5  
    max_iter = 10  

    for k in global_w_update.keys():

        param_updates = torch.stack([client_w[k] for client_w in w]).float()  

        median = torch.mean(param_updates, dim=0)

        for _ in range(max_iter):
            diff = param_updates - median
            diff_flat = diff.view(diff.size(0), -1)  
            distances = torch.norm(diff_flat, dim=1) 


            weights = 1.0 / torch.clamp(distances, min=eps)
            weights /= weights.sum() 


            new_shape = [weights.shape[0]] + [1] * (param_updates.ndim - 1)
            weights_reshaped = weights.view(*new_shape)


            new_median = torch.sum(weights_reshaped * param_updates, dim=0)

            if torch.norm(new_median - median) < eps:
                break
            median = new_median

        global_w_update[k] = median  

    return global_w_update



def FedExp(w, average_weights, global_w_init, model):

    eps = 1e-3 # following the original paper

    # transform average_weights to torch tensor on cuda
    average_weights = torch.tensor(average_weights).to('cpu')
    average_weights = average_weights / average_weights.sum()

    nets_this_round = {}

    for i in range(len(w)):
        model_each = copy.deepcopy(model).to('cpu')
        model_each.load_state_dict(w[i])
        nets_this_round[i]=model_each

    global_w_init = copy.deepcopy(model.to('cpu').state_dict())
    global_w_update = copy.deepcopy(global_w_init)
    global_w = model.state_dict()

    for net_id, net in enumerate(nets_this_round.values()):
        net_para = net.state_dict()
        if net_id == 0:
            for key in net_para:
                global_w_update[key] = net_para[key] * average_weights[net_id]
        else:
            for key in net_para:
                global_w_update[key] += net_para[key] * average_weights[net_id]

    # calculate eta_g, the server learning rate defined in FedExp
    # ======================================
    # calculate \|\bar{Delta}^{(t)}\|^{2}
    sqr_avg_delta = 0.0
    for key in global_w_update:
        sqr_avg_delta += ((global_w_update[key] - global_w_init[key])**2).sum()

    # calculate \sum_{i}{p_{i}\|\Delta_{i}^{(t)}\|^{2}} for each client
    avg_sqr_delta = 0.0
    for net_id, net in enumerate(nets_this_round.values()):
        net_para = net.state_dict()
        for key in net_para:
            avg_sqr_delta += average_weights[net_id] * ((net_para[key] - global_w_init[key])**2).sum()

    eta_g = avg_sqr_delta / (2*(sqr_avg_delta + eps))
    eta_g = max(1.0, eta_g.item())

    # log eta_g
    # self.logger.info('eta_g at current round: %f' % eta_g)
    # ======================================

    for key in global_w:
        global_w[key] = global_w_init[key] + eta_g*(global_w_update[key] - global_w_init[key])

    return global_w


def DaAgg(w, dict_len, client_tag):
    client_weight = np.array(dict_len)
    client_weight = client_weight / client_weight.sum()

    clean_clients = []
    noisy_clients = []
    for index, element in enumerate(client_tag):
        if element == 1:
            clean_clients.append(index)
        elif element == 0:
            noisy_clients.append(index)
        else:
            raise
    
    distance = np.zeros(len(dict_len))
    for n_idx in noisy_clients:
        dis = []
        for c_idx in clean_clients:
            dis.append(model_dist(w[n_idx], w[c_idx]))
        distance[n_idx] = min(dis)
    distance = distance / distance.max()
    client_weight = client_weight * np.exp(-distance)
    client_weight = client_weight / client_weight.sum()
    # print(client_weight)

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * client_weight[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * client_weight[i]
    return w_avg


def model_dist(w_1, w_2):
    assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
    dist_total = torch.zeros(1).float()
    for key in w_1.keys():
        if "int" in str(w_1[key].dtype):
            continue
        dist = torch.norm(w_1[key] - w_2[key])
        dist_total += dist.cpu()

    return dist_total.cpu().item()



  
def Median(w):  
    global_w_update = copy.deepcopy(w[0])  
    num_models = len(w)  
  
    for k in global_w_update.keys():  
        parameter_values = [w[i][k] for i in range(num_models)]  
        aggregated_parameter = torch.median(torch.stack(parameter_values, dim=0), dim=0).values  
  
        global_w_update[k] = aggregated_parameter  
  
    return global_w_update


def euclid(v1, v2):
    diff = v1 - v2
    return torch.matmul(diff, diff.T)


def multi_vectorization(w_locals, args):
    vectors = copy.deepcopy(w_locals)

    for i, v in enumerate(vectors):
        for name in v:
            v[name] = v[name].reshape([-1])
        vectors[i] = torch.cat(list(v.values()))

    return vectors


def Krum(w_locals, c, args):
    n = len(w_locals) - c

    distance = pairwise_distance(w_locals, args)
    sorted_idx = distance.sum(dim=0).argsort()[: n]

    chosen_idx = int(sorted_idx[0])

    return copy.deepcopy(w_locals[chosen_idx])


def pairwise_distance(w_locals, args):
    vectors = multi_vectorization(w_locals, args)
    distance = torch.zeros([len(vectors), len(vectors)]).to(args.device)

    for i, v_i in enumerate(vectors):
        for j, v_j in enumerate(vectors[i:]):
            distance[i][j + i] = distance[j + i][i] = euclid(v_i, v_j)

    return distance


def fedavgg(w_locals):
    w_avg = copy.deepcopy(w_locals[0])

    with torch.no_grad():
        for k in w_avg.keys():
            for i in range(1, len(w_locals)):
                w_avg[k] += w_locals[i][k]
            w_avg[k] = torch.true_divide(w_avg[k], len(w_locals))

    return w_avg




def trimmed_mean(w_locals, c, args):
    n = len(w_locals) - 2 * c

    distance = pairwise_distance(w_locals, args)

    distance = distance.sum(dim=1)
    med = distance.median()
    _, chosen = torch.sort(abs(distance - med))
    chosen = chosen[: n]
        
    return fedavgg([copy.deepcopy(w_locals[int(i)]) for i in chosen])