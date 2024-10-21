import copy
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions.kl import kl_divergence
from datasets import build_dataset, CustomDataset
import torch.nn as nn
import torch.nn.functional as F
from model import GVMVC
from evaluate import evaluate
import numpy as np
import random
import argparse
from sklearn.cluster import KMeans
from torch.distributions import Normal, Bernoulli, Categorical, Dirichlet
from sklearn.mixture import GaussianMixture
import math

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def test(model, mv_loaders, args):
    print('Test......')
    eval_data = copy.deepcopy(mv_loaders.dataset.data)
    for v in range(args.num_views):
        eval_data[v] = eval_data[v].clone().detach().to(dtype=torch.float32, device=args.device)
    eval_labels = mv_loaders.dataset.labels

    with torch.no_grad():
        z_sample, z_sample_list, adj_list_tensor, z_mus, z_vars, stack_mu, stack_var, xr_list = model(eval_data, args)

        kmeans = KMeans(n_clusters=args.class_num, n_init=10).fit(z_sample.cpu().numpy())
        predict_kmeans = kmeans.labels_
        acc_kmeans, nmi_kmeans, ari_kmeans, pur_kmeans = evaluate(eval_labels, predict_kmeans)

    print(f'Testing Results:  ACC:{acc_kmeans:.4f}  NMI:{nmi_kmeans:.4f}  ARI:{ari_kmeans:.4f}  PUR:{pur_kmeans:.4f}')
    return acc_kmeans, nmi_kmeans, ari_kmeans, pur_kmeans

def main(args):
    for t in range(1, args.test_times + 1):
        print(f'Test {t}')
        np.random.seed(t)
        random.seed(t)
        cmv_data, labels = build_dataset(args)
        setup_seed(args.seed)
        cmv_dataset = CustomDataset(cmv_data, labels)
        cmv_loader = DataLoader(cmv_dataset, batch_size=args.batch_size, shuffle=True)
        model = GVMVC(args).to(args.device)

        best_model_path = f'ckpt/best_model_{args.dataset_name}.pt'
        model = torch.load(best_model_path)
        model.eval()
        acc1, nmi1, ari1, pur1 = test(model, cmv_loader, args)
        test_record["ACC"].append(acc1)
        test_record["NMI"].append(nmi1)
        test_record["ARI"].append(ari1)
        test_record["PUR"].append(pur1)
    print('Average ACC {:.4f} Average NMI {:.4f} Average ARI {:.4f} Average PUR {:.4f}'.format(np.mean(test_record["ACC"]),
                                                                                               np.mean(test_record["NMI"]),
                                                                                               np.mean(test_record["ARI"]),
                                                                                               np.mean(test_record["PUR"])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--z_dim', type=int, default=256, help='latent dimensions')

    parser.add_argument('--dataset', type=int, default=5, choices=range(4), help='0:Caltech7-5v, 1:Scene-15, 2:BDGP, 3:NoisyMNIST, 4:RGBD, 5:Reuters_dim10, 6:MNIST_USPS,7: Caltech101-20,8:LandUse_21,9:ccv')
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--test_times', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=5)
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.dataset_dir_base = "./data/"
    args.k = 5

    if args.dataset == 0:
        args.dataset_name = 'Scene-15'
        args.alpha = 20
        args.seed = 19
        args.likelihood = 'Gaussian'
        args.lambda1 = 1
        args.lambda2 = 0.1
    elif args.dataset == 1:
        args.dataset_name = 'RGBD'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'
        args.batch_size = 512
        args.lambda1 = 0.1
        args.lambda2 = 0.1
    elif args.dataset == 2:
        args.dataset_name = 'Reuters_dim10'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'
        args.batch_size = 512
        args.lambda1 = 1
        args.lambda2 = 0.1
    elif args.dataset == 3:
        args.dataset_name = 'LandUse_21'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'
        args.batch_size = 512
        args.lambda1 = 10
        args.lambda2 = 0.1
    elif args.dataset == 4:
        args.dataset_name = 'Caltech_2V'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'
        args.batch_size = 512
        args.lambda1 = 1
        args.lambda2 = 0.1
    elif args.dataset == 5:
        args.dataset_name = 'Caltech_3V'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'
        args.batch_size = 512
        args.lambda1 = 1
        args.lambda2 = 0.1
    elif args.dataset == 6:
        args.dataset_name = 'Caltech_4V'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'
        args.batch_size = 512
        args.lambda1 = 1
        args.lambda2 = 0.1
    elif args.dataset == 7:
        args.dataset_name = 'Caltech_5V'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'
        args.batch_size = 512
        args.lambda1 = 10
        args.lambda2 = 0.01

    for iter in range(1):
        print(f"Running iteration {iter + 1}")
        test_record = {"ACC": [], "NMI": [], "PUR": [], "ARI": []}
        main(args)
