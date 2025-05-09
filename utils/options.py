import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # label noise method
    parser.add_argument('--method', type=str, default='fedavg',
                        choices=['fedavg', 'fedprox', 'fedexp', 'fedprox', 
                                 'krum', 'median', 'trimmedMean','RFA','clipping',
                                 'selfie', 'jointoptim', 'symmetricce', 'coteaching', 'coteaching+', 'dividemix', 
                                 'fedrn', 'fedlsr', 'robustfl','fednoro','fedELC'],
                        help='method name')


    # federated arguments
    parser.add_argument('--epochs', type=int, default=120,
                        help="rounds of training")
    parser.add_argument('--num_users', type=int,
                        default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    # parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--test_bs', type=int, default=128,
                        help="test batch size")

    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="SGD momentum (default: 0.9)")
    parser.add_argument('--split', type=str, default='user',
                        help="train-test split type, user or sample")
    parser.add_argument('--schedule', nargs='+', default=[],
                        help='decrease learning rate at these epochs.')
    parser.add_argument('--lr_decay', type=float,
                        default=0.1, help="learning rate decay")
    parser.add_argument('--weight_decay', type=float,
                        default=0.0005, help="sgd weight decay")
    parser.add_argument('--partition', type=str,
                        choices=['shard', 'dirichlet','IID'], default='IID')
    parser.add_argument('--dd_alpha', type=float, default=0.5,
                        help="dirichlet distribution alpha, you can select 1.0 or 0.5")
    parser.add_argument('--num_shards', type=int,
                        default=500, help="number of shards,eg: cifar-10:200 and cifar-100:2000 ")




    # model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet20', 'fasttext'], help='model name')
    parser.add_argument('--pretrained', action='store_false',
                        help='if use pretrained model')



    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset",
                        choices=['cifar10', 'cifar100', 'AGNews'])
    
    parser.add_argument('--iid', action='store_true',
                        help='whether i.i.d or not')
    
    parser.add_argument('--num_classes', type=int,
                        default=10, help="number of classes")
    
    parser.add_argument('--num_channels', type=int, default=3,
                        help="number of channels of imges")
    
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID, -1 for CPU")
    
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    
    parser.add_argument('--all_clients', action='store_false',
                        help='aggregation over all clients')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num_workers to load data')

    # noise label arguments
    parser.add_argument('--noise_type_lst', nargs='+',
                        default=['symmetric'], help='[pairflip, symmetric]')
    
    parser.add_argument('--noise_group_num', nargs='+',
                        default=[100], type=int)
    
    parser.add_argument('--group_noise_rate', nargs='+', default=[0.2], type=float,
                        help='Should be 2 noise rates for each group: min_group_noise_rate max_group_noise_rate but '
                             'if there is only 1 group and 1 noise rate, same noise rate will be applied to all users')
    
    parser.add_argument('--warmup_epochs', type=int,
                        default=30, help='number of warmup epochs for fedrn...')

    parser.add_argument('--vocab_size', type=int,
                        default=95814, help='vocab size for fasttext')

    parser.add_argument('--hidden_dim', type=int,
                        default=256, help='embedding dim for fasttext')

    parser.add_argument('--n_gram_vocab', type=int,
                        default=100000, help='n_gram vocab size for fasttext')





    # SELFIE / Joint optimization arguments
    parser.add_argument('--queue_size', type=int,
                        default=15, help='size of history queue')
    # SELFIE / Co-teaching arguments
    parser.add_argument('--forget_rate', type=float,
                        default=0.2, help="forget rate for co-teaching")
    # SELFIE arguments
    parser.add_argument('--uncertainty_threshold', type=float,
                        default=0.05, help='uncertainty threshold')
    # Joint optimization arguments
    parser.add_argument('--alpha', type=float, default=1.2,
                        help="alpha for joint optimization")
    parser.add_argument('--beta', type=float, default=0.8,
                        help="beta for joint optimization")
    parser.add_argument('--labeling', type=str,
                        default='soft', help='[soft, hard]')

    # MixMatch arguments
    parser.add_argument('--mm_alpha', default=4, type=float)
    parser.add_argument('--lambda_u', default=25, type=float,
                        help='weight for unsupervised loss')
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--p_threshold', default=0.5, type=float)

    # FedProx
    parser.add_argument('--mu', type=float, default=0.1, help="mu for FedProx")

    # FedRN
    parser.add_argument('--num_neighbors', type=int,
                        default=1, help="number of neighbors")
    parser.add_argument('--w_alpha', type=float,
                        help='weight alpha for our method', default=0.6)

    # Robust Aggregation based methods
    parser.add_argument('--compromised_rate', type=float,
                        help='compromised rate for robust aggregation', default=0.3)

    # FedSAM
    parser.add_argument('--fedsam_rho', type=int, default=0.05,
                        help="fedsam's rho hyper-param")

    # FedLSR,gamma_e 0.3
    parser.add_argument('--gamma_e', type=float, default=0.3,
                        help='weight for entropy loss named gamma_e for FedLSR')
    parser.add_argument('--gamma', type=float, default=0.4,
                        help='weight for self-distillation named gamma for FedLSR')
    parser.add_argument('--distill_t', type=float, default=3.0,
                        help='temperature(reverse) for self-distillation')
    parser.add_argument("--warm_up_ratio_lsr", type=int, default=0.2,
                        help="warm up epochs ratio (compared to total training epochs) for FedLSR")

    # gradient clipping which requires max_grad_norm=10.0 as default
    parser.add_argument('--max_grad_norm', type=float,
                        default=5.0, help='max grad norm for gradient clipping')

    # RobustFL
    parser.add_argument('--T_pl', type=int, help='T_pl', default=10)
    parser.add_argument('--feature_dim', type=int,
                        help='feature dimension', default=256)
    parser.add_argument('--lambda_cen', type=float,
                        help='lambda_cen', default=1.0)
    parser.add_argument('--lambda_e', type=float, help='lambda_e', default=0.8)
    parser.add_argument('--feature_return',
                        action='store_true', help='feature extraction')
    
    # whether to exploit representation regularization
    parser.add_argument('--is_svd_loss', action='store_true',
                        help='whether using representation regularization or not')
    # DECORR_COEF=0.05 by default
    parser.add_argument('--decorr_coef', type=float,
                        help='decorrelation coefficient', default=0.1)
    



    parser.add_argument('--K_pencil', type=int, default=10,
                        help='number of pencils')
    parser.add_argument('--lamda_pencil', type=float, default=1000,
                        help='lamda for pencil loss')
    parser.add_argument('--alpha_pencil', type=float, default=0.5,
                        help='alpha for pencil loss')
    parser.add_argument('--beta_pencil', type=float, default=0.2,
                        help='beta for pencil loss')
    # parser.add_argument('--threshold_pl', default=0.5, type=float,help='pseudo label threshold for fixmatch')

    # epoch_of_stage1
    parser.add_argument('--epoch_of_stage1', type=int, default=20,
                        help='number of epochs for stage 1')


    args = parser.parse_args()
    return args
