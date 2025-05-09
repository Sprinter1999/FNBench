import copy
import numpy as np
import random
import time
from datetime import datetime
import os

import torchvision
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import load_dataset
from utils.options import args_parser
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
from utils.utils import noisify_label

from fl_components.fed import LocalModelWeights
from fl_components.nets import get_model
from fl_components.test import test_img
from fl_components.update import get_local_update_objects

from model_arch.build_model import build_model

if __name__ == '__main__':

    start = time.time()
    args = args_parser()
    args.device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else 'cpu',
    )

    args.all_clients = False
    args.schedule = [int(x) for x in args.schedule]
    args.send_2_models = args.method in [
        'coteaching', 'coteaching+', 'dividemix', ]


    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)



    args.selected_total_clients_num = args.num_users * args.frac





    ##############################
    # Load dataset and split users
    ##############################
    dataset_train, dataset_test, args.num_classes, collate_fn = load_dataset(args.dataset)
    labels = np.array(dataset_train.train_labels)
    
    
    args.collate_fn = None
    if collate_fn is not None:
        args.collate_fn = collate_fn

    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=args.test_bs,
        shuffle=False,
        collate_fn=collate_fn if collate_fn else None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )



    #TODO: Arbitrary gaussian noise, used for FedRN， not applicable for NLP tasks
    gaussian_noise = torch.randn(1, 3, 32, 32)


    #TODO: varibles for the design of the fedELC
    num_total_samples = len(np.array(dataset_train.train_labels))
    # global Soft_labels
    # forms a N * 10 classes all-zero matrix and moves it to GPU
    Soft_labels = torch.zeros([num_total_samples, args.num_classes], dtype=torch.float)
    # global Soft_labels_flag
    Soft_labels_flag = torch.zeros([num_total_samples], dtype=torch.int)
    # Soft_labels.cuda()

    True_Labels = copy.deepcopy(torch.tensor(np.array(dataset_train.train_labels)))
    args.True_Labels = True_Labels

    args.Soft_labels = Soft_labels

    #TODO: for methods need warmup training
    args.warmup_epochs = int(0.2 * args.epochs)





    if args.partition == 'shard':  # non-iid
        if(args.dataset == 'cifar10'):
            # 5 classes for a client at most (total clients=100)
            args.num_shards = 500
        elif(args.dataset == 'cifar100'):
            # 20 classes for a client at most (total clients=100)
            args.num_shards = 2000

        print("[Partitioning Via Sharding....]")
        dict_users = sample_noniid_shard(
            labels=np.array(dataset_train.train_labels),
            num_users=args.num_users,
            num_shards=args.num_shards,
        )

    elif args.partition == 'dirichlet':
        print("[Partitioning Via Dir....]")
        dict_users = sample_dirichlet(
            labels=np.array(dataset_train.train_labels),
            num_clients=args.num_users,
            alpha=args.dd_alpha,
            num_classes=args.num_classes,
        )
    else:
        print("[Partitioning Via IID....]")
        dict_users = sample_iid(
            labels=np.array(dataset_train.train_labels),
            num_users=args.num_users,
        )



    print("#############   Print  all  args param. ##########")
    for x in vars(args).items():
        print(x)

    if not torch.cuda.is_available():
        exit('ERROR: Cuda is not available!')
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)




    print("###########################################################")

    client_noise_map= {}

    ##############################
    # Add label noise to data
    ##############################

    noise_file = torch.load('./data/CIFAR-100_human.pt')
    clean_labels = noise_file['clean_label']
    worst_labels = noise_file['noisy_label']
    # aggre_labels = noise_file['aggre_label']
    # random_labels1 = noise_file['random_label1']
    # random_labels2 = noise_file['random_label2']
    # random_labels3 = noise_file['random_label3']

    print("$$$$$$$$$$$$ COMPARISON with labels $$$$$$$$$$$$")

    comparison_of_clean_labels = (labels == clean_labels).sum()
    print(comparison_of_clean_labels) 

    comparison_of_worst_labels = (labels == worst_labels).sum()
    print(comparison_of_worst_labels)
    
    dataset_train.targets = worst_labels




    logging_args = dict(
        batch_size=args.test_bs,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn if collate_fn else None,
    )

    acc_list_glob1 = []
    test_loss_list_glob1 = []
    precision_list_glob1 = []
    recall_list_glob1 = []
    f1_list_glob1 = []

    percentage_correct_glob1 = []
    before_percentage_correct_glob1, after_percentage_correct_glob1, merged_percentage_correct_glob1 = [], [], []



    # log_train_data_loader = torch.utils.data.DataLoader(
    #     dataset_train, **logging_args)

    log_test_data_loader = torch.utils.data.DataLoader(
        dataset_test, **logging_args)







    ##############################
    # Build model
    ##############################
    if args.dataset == 'AGNews':
        from model_arch.FastText import FastText
        net_glob = FastText(
            hidden_dim=args.hidden_dim,
            padding_idx=dataset_train.get_pad_idx(),
            vocab_size=dataset_train.get_vocab_size(),
            num_classes=args.num_classes,
        )
    else:
        net_glob = build_model(args)

    net_glob.to(args.device)






    if args.send_2_models:
        net_glob2 = build_model(args)
        # net_glob2 = net_glob2.to(args.device)
        acc_list_glob2 = []
        test_loss_list_glob2 = []
        precision_list_glob2 = []
        recall_list_glob2 = []
        f1_list_glob2 = []

    if args.model == 'resnet50':
        args.feature_dim = 2048
    elif args.model == 'resnet34':
        args.feature_dim = 512
    elif args.model == 'resnet18':
        args.feature_dim = 512
    elif args.model == 'fasttext':
        # Token embedding
        args.feature_dim = 256
    else:
        args.feature_dim = 128





    ##############################
    # Training
    ##############################
    CosineSimilarity = torch.nn.CosineSimilarity()
    # base_optim = torch.optim.SGD
    # sam_optimizer = SAM(net_glob.parameters(), base_optim, rho=args.fedsam_rho, adaptive=False,
    #                     lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ##############################
    # Class centroids for robust FL
    ##############################
    # glob_centroid = {i: None for i in range(args.num_classes)}
    f_G = torch.randn(args.num_classes, args.feature_dim, device=args.device)
    

    forget_rate_schedule = []

    if args.method in ['coteaching', 'coteaching+']:
        num_gradual = args.warmup_epochs
        forget_rate = args.forget_rate
        exponent = 1
        forget_rate_schedule = np.ones(args.epochs) * forget_rate
        forget_rate_schedule[:num_gradual] = np.linspace(
            0, forget_rate ** exponent, num_gradual)

    elif args.method == 'robustfl':
        forget_rate_schedule, schedule = [], []
        num_gradual = args.warmup_epochs
        exponent = 1
        forget_rate_schedule = np.ones(args.epochs) * args.forget_rate
        forget_rate_schedule[:num_gradual] = np.linspace(
            0, args.forget_rate ** exponent, num_gradual)
    else:
        pass

    pred_user_noise_rates = [args.forget_rate] * args.num_users

    # Initialize local model weights
    fed_args = dict(
        all_clients=args.all_clients,
        num_users=args.num_users,
        method=args.method,
        dict_users=dict_users,
        args=args,
    )

    local_weights = LocalModelWeights(net_glob=net_glob, **fed_args)
    if args.send_2_models:
        local_weights2 = LocalModelWeights(net_glob=net_glob2, **fed_args)

    # Initialize local update objects
    local_update_objects = get_local_update_objects(
        args=args,
        dataset_train=dataset_train,
        dict_users=dict_users,
        noise_rates=pred_user_noise_rates,
        gaussian_noise=gaussian_noise,
        glob_centroid=f_G,
    )

    for i in range(args.num_users):
        # local = local_update_objects[i]
        # local.weight = copy.deepcopy(net_glob.state_dict())
        local_update_objects[i].weight = copy.deepcopy(net_glob.state_dict())









    ####################Start Training#####################
    for epoch in tqdm(range(args.epochs), desc="Training Progress"):
        print("\n####################### Global Epoch {} Starts...".format(epoch))
        if (epoch + 1) in args.schedule:
            print("Learning Rate Decay Epoch {}".format(epoch + 1))
            print("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay

        if len(forget_rate_schedule) > 0:
            args.forget_rate = forget_rate_schedule[epoch]

        local_losses = []
        local_losses2 = []
        args.g_epoch = epoch
        feature_locals = []
        local_percentages = []
        before_local_percentages, after_local_percentages, merged_local_percentages = [], [], []

        print_flag=False
        

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Local Update
        # counter = 0
        for client_num, idx in enumerate(idxs_users):
            local = local_update_objects[idx]
            local.args = args
            # percentage_correct = 0
            percentage_correct, before_percentage_correct, after_percentage_correct, merged_percentage_correct = 0, 0, 0, 0



            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"...Select Client {client_num} and actual client idx#{idx} and name of updater {local.update_name} at the time: {current_time}...")

            if args.method == "fedrn":

                if epoch < args.warmup_epochs:
                    w, loss = local.train_phase1(
                        copy.deepcopy(net_glob).to(args.device))
                else:
                    # Get similarity, expertise values
                    sim_list = []
                    exp_list = []
                    for user in range(args.num_users):
                        sim = CosineSimilarity(
                            local.arbitrary_output.to(args.device),
                            local_update_objects[user].arbitrary_output.to(
                                args.device),
                        ).item()
                        exp = local_update_objects[user].expertise
                        sim_list.append(sim)
                        exp_list.append(exp)

                    # Normalize similarity & expertise values
                    sim_list = [
                        (sim - min(sim_list)) / (max(sim_list) - min(sim_list)) for sim in sim_list]
                    exp_list = [
                        (exp - min(exp_list)) / (max(exp_list) - min(exp_list)) for exp in exp_list]

                    # Compute & sort scores
                    prev_score = args.w_alpha * \
                        exp_list[idx] + (1 - args.w_alpha)

                    score_list = []
                    for neighbor_idx, (exp, sim) in enumerate(zip(exp_list, sim_list)):
                        if neighbor_idx != idx:
                            score = args.w_alpha * exp + \
                                (1 - args.w_alpha) * sim
                            score_list.append([score, neighbor_idx])
                    score_list.sort(key=lambda x: x[0], reverse=True)

                    # Get top-k neighbors
                    neighbor_list = []
                    neighbor_score_list = []
                    for k in range(args.num_neighbors):
                        neighbor_score, neighbor_idx = score_list[k]
                        neighbor_net = copy.deepcopy(
                            local_update_objects[neighbor_idx].net1)
                        neighbor_list.append(neighbor_net)
                        neighbor_score_list.append(neighbor_score)

                    w, loss = local.train_phase2(copy.deepcopy(net_glob).to(args.device),
                                                 prev_score,
                                                 neighbor_list,
                                                 neighbor_score_list)

            elif args.method == "fedlsr":
                w, loss = local.train(net=copy.deepcopy(
                    net_glob).to(args.device), cur_round=epoch)


            elif args.method == "robustfl":
                if len(forget_rate_schedule) > 0:
                    forget_rateee = forget_rate_schedule[epoch]

                w, loss, f_k = local.train(model=copy.deepcopy(
                    net_glob), current_round=epoch, f_G=copy.deepcopy(f_G).to(args.device), forget_ratee=forget_rateee)
                
                feature_locals.append(f_k)
            
            
            
            ############################## added by nannan, please check it ! ##############################
            elif args.method == "fednoro" or args.method == "fedELC":
                #TODO: Warm-up epochs
                epoch_of_stage1 = args.epoch_of_stage1 # just set it as the initial paper in the ICH dataset

                if epoch < epoch_of_stage1:
                    local_weights.noisy_clients = 0
                elif epoch == epoch_of_stage1 and client_num==0: # client selection by GMM
                    loader = DataLoader(dataset=dataset_train, batch_size=32,shuffle=False, collate_fn=collate_fn if collate_fn else None, num_workers=4)
                    criterion = torch.nn.CrossEntropyLoss(reduction='none')
                    from utils.utils import get_output
                    local_output, loss = get_output(loader, net_glob.to(args.device), args, False, criterion)
                    metrics = np.zeros((args.num_users, args.num_classes)).astype("float")
                    num = np.zeros((args.num_users, args.num_classes)).astype("float")
                    for id in range(args.num_users):
                        idxs = dict_users[id]
                        for idxx in idxs:
                            c = dataset_train.train_labels[idxx]
                            num[id, c] += 1
                            metrics[id, c] += loss[idxx]

                    metrics = metrics / num 

                    for i in range(metrics.shape[0]):
                        for j in range(metrics.shape[1]):
                            if np.isnan(metrics[i, j]):
                                metrics[i, j] = np.nanmin(metrics[:, j])
                    for j in range(metrics.shape[1]):
                        metrics[:, j] = (metrics[:, j]-metrics[:, j].min()) / \
                            (metrics[:, j].max()-metrics[:, j].min())
                        
                    from sklearn.mixture import GaussianMixture
                    vote = []
                    for i in range(9):
                        gmm = GaussianMixture(n_components=2, random_state=i).fit(metrics)
                        gmm_pred = gmm.predict(metrics)
                        noisy_clients = np.where(gmm_pred == np.argmax(gmm.means_.sum(1)))[0]
                        noisy_clients = set(list(noisy_clients))
                        vote.append(noisy_clients)
                   
                    cnt = []
                    for i in vote:
                        cnt.append(vote.count(i))
                    noisy_clients = list(vote[cnt.index(max(cnt))])
                    user_id = list(range(args.num_users))
                    clean_clients = list(set(user_id) - set(noisy_clients))
                    
                    local_weights.noisy_clients = noisy_clients
                    local_weights.clean_clients = clean_clients
                    local_weights.client_tag = [] # to indicate if this client is clean (1)

                    # print(f"###############\n Len of NOISY/CLEAN clients: {len(noisy_clients)} VS {len(clean_clients)} \n ###############")


                    noisyclient_rate_list, cleanclient_rate_list = [], []
                    # FOr noisy client, we retreive the noisy rate from client_noise_map
                    for each in noisy_clients:
                        noisyclient_rate_list.append(client_noise_map[each])
                    for each in clean_clients:
                        cleanclient_rate_list.append(client_noise_map[each])

                    print(f"#######[Division]########\n Len of NOISY/CLEAN clients: {len(noisy_clients)} VS {len(clean_clients)} \n ###############")
                    print(f"#######[Division]########\n Sum of noisy clients' rates total quantity: {sum(noisyclient_rate_list)}")
                    print(f"#######[Division]########\n Sum of clean clients' rates total quantity: {sum(cleanclient_rate_list)}")
                
                else:
                    pass

                # local training
                if epoch < epoch_of_stage1: # stage 1, 
                    w, loss = local.train_stage1(net=copy.deepcopy(net_glob).to(args.device))
                else: # stage 2, 
                    def sigmoid_rampup(current, begin, end):
                        current = np.clip(current, begin, end)
                        phase = 1.0 - (current-begin) / (end-begin)
                        return float(np.exp(-5.0 * phase * phase))

                    def get_current_consistency_weight(rnd, begin, end):
                        return sigmoid_rampup(rnd, begin, end)
                    
                    weight_kd = get_current_consistency_weight(epoch, epoch_of_stage1, args.epochs) * 0.8


                    print_flag = True

                    if idx in local_weights.clean_clients:
                        w, loss = local.train_stage1(net=copy.deepcopy(net_glob).to(args.device))
                        local_weights.client_tag.append(1)
                    else:
                        w, loss = local.train_stage2(net=copy.deepcopy(net_glob).to(args.device), global_net=copy.deepcopy(net_glob).to(args.device), weight_kd=weight_kd)
                        local_weights.client_tag.append(0)




            elif args.send_2_models:  # co-teaching, co-teaching+, divide-mix
                w, loss, w2, loss2 = local.train(
                    copy.deepcopy(net_glob).to(args.device),
                    copy.deepcopy(net_glob2).to(args.device),
                    epoch
                )
                local_weights2.update(idx, w2)
                local_losses2.append(copy.deepcopy(loss2))

            else:
                w, loss = local.train(copy.deepcopy(net_glob).to(args.device))




            local_weights.update(idx, w)
            local_losses.append(copy.deepcopy(loss))
            local_percentages.append(percentage_correct)



        ##############################  
        print(f"Global Epoch {epoch} Local Training is done!.....")





        # show local loss mean
        print(f"[LOSS] Average local loss mean is {sum(local_losses) / len(local_losses)}")

        #TODO: update the global model weights
        w_glob = local_weights.average()  # update global weights

        print(f"[WEIGHT] Average global weights, DONE!")
        
        if args.method == "robustfl":
            sim = torch.nn.CosineSimilarity(dim=1)
            tmp = 0
            w_sum = 0
            for i in feature_locals:
                sim_weight = sim(f_G, i).reshape(args.num_classes, 1)
                w_sum += sim_weight
                tmp += sim_weight * i
            f_G = torch.div(tmp, w_sum)


        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f'\n#### Testing for Method {args.method} in Round {epoch} and current time is {current_time} ####')
        net_glob.to(args.device)

        # update global weights
        local_weights.global_w_init = copy.deepcopy(net_glob.state_dict())

        local_weights.init()  # clear temp local weights for the next round aggregation

        print("####################### Called")
        # train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
        accuracy, test_loss, precision, recall, f1 = test_img(
            net_glob, test_loader, args)


        acc_list_glob1.append(accuracy)
        test_loss_list_glob1.append(test_loss)
        precision_list_glob1.append(precision)
        recall_list_glob1.append(recall)
        f1_list_glob1.append(f1)

        if args.send_2_models:
            w_glob2 = local_weights2.average()
            net_glob2.load_state_dict(w_glob2)

            net_glob2.to(args.device)

            local_weights2.init()

            accuracy_2, test_loss_2, precision_2, recall_2, f1_2 = test_img(
                net_glob2, test_loader, args)

            acc_list_glob2.append(accuracy_2)
            test_loss_list_glob2.append(test_loss_2)
            precision_list_glob2.append(precision_2)
            recall_list_glob2.append(recall_2)
            f1_list_glob2.append(f1_2)



    print("####################### Global Model 1")
    print(f"Total test acc list for global model 1 is shown: {acc_list_glob1}")
    print("#############################################")
    print(
        f"Total test loss list for global model 1 is shown: {test_loss_list_glob1}")
    print("#############################################")
    print(
        f"Total precision list for global model 1 is shown: {precision_list_glob1}")
    print("#############################################")
    print(
        f"Total recall list for global model 1 is shown: {recall_list_glob1}")
    print("#############################################")
    print(f"Total f1 list for global model 1 is shown: {f1_list_glob1}")

    print("@@@@@@@@ Print Average Metrics of last 10 rounds for global model 1 @@@@@@@@@")
    print(
        f"Mean Accuracy of last 10 rounds for global model 1 is shown: {np.mean(acc_list_glob1[-10:])}")
    print(
        f"Mean F1 Score of last 10 rounds for global model 1 is shown: {np.mean(f1_list_glob1[-10:])}")
    print(
        f"Mean Precision of last 10 rounds for global model 1 is shown: {np.mean(precision_list_glob1[-10:])}")
    print(
        f"Mean Recall of last 10 rounds for global model 1 is shown: {np.mean(recall_list_glob1[-10:])}")
    print(
        f"Mean Test Loss of last 10 rounds for global model 1 is shown: {np.mean(test_loss_list_glob1[-10:])}")



    if args.send_2_models:

        print("####################### Global Model 2")
        print(
            f"Total test acc list for global model 2 is shown: {acc_list_glob2}")
        print("#############################################")
        print(
            f"Total test loss list for global model 2 is shown: {test_loss_list_glob2}")
        print("#############################################")
        print(
            f"Total precision list for global model 2 is shown: {precision_list_glob2}")
        print("#############################################")
        print(
            f"Total recall list for global model 2 is shown: {recall_list_glob2}")
        print("#############################################")
        print(f"Total f1 list for global model 2 is shown: {f1_list_glob2}")

        print(
            "@@@@@@@@ Print Average Metrics of last 10 rounds for global model 2 @@@@@@@@@")
        print(
            f"Mean Accuracy of last 10 rounds for global model 2 is shown: {np.mean(acc_list_glob2[-10:])}")
        print(
            f"Mean F1 Score of last 10 rounds for global model 2 is shown: {np.mean(f1_list_glob2[-10:])}")
        print(
            f"Mean Precision of last 10 rounds for global model 2 is shown: {np.mean(precision_list_glob2[-10:])}")
        print(
            f"Mean Recall of last 10 rounds for global model 2 is shown: {np.mean(recall_list_glob2[-10:])}")
        print(
            f"Mean Test Loss of last 10 rounds for global model 2 is shown: {np.mean(test_loss_list_glob2[-10:])}")