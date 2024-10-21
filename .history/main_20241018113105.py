import copy
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions.kl import kl_divergence
from datasets import build_dataset, CustomDataset
import torch.nn as nn
import torch.nn.functional as F
from model import DVIMC
from evaluate import evaluate
import numpy as np
import random
import argparse
from sklearn.cluster import KMeans
from torch.distributions import Normal, Bernoulli, Categorical, Dirichlet
from sklearn.mixture import GaussianMixture
import math


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def MMD_loss(x, y, sigma=1.0):
    Kxx = compute_rbf_kernel(x, x, sigma)
    Kxy = compute_rbf_kernel(x, y, sigma)
    Kyy = compute_rbf_kernel(y, y, sigma)
    loss = torch.sum(Kxx) + torch.sum(Kyy) - 2 * torch.sum(Kxy)

    return loss/(args.batch_size*args.batch_size)

def gaussian_pdfs_log(x,mus,log_sigma2s):
    G=[]
    for c in range(args.class_num):
        G.append(gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
    return torch.cat(G,1)

def graph_contrastive_loss(z_sample, adj):
    recon_adj = torch.sigmoid(torch.mm(z_sample, z_sample.T))
    recon_adj = torch.clamp(recon_adj, min=1e-3, max=1 - 1e-3)
    return F.binary_cross_entropy(recon_adj, adj)

def gaussian_pdf_log(x,mu,log_sigma2):
    return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))

def gaussian_dskl_divergence(mu1, logvar1, mu2, logvar2):

    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)

    # Calculate the KL divergence
    kl_divergence = 0.5 * (torch.sum(var1 / var2, dim=-1)
                           + torch.sum((mu2 - mu1).pow(2) / var2, dim=-1)
                           + torch.sum(logvar2, dim=-1)
                           - torch.sum(logvar1, dim=-1)
                           - mu1.shape[-1])

    return torch.sum(kl_divergence)/(mu1.shape[0]*mu1.shape[1])


def compute_rbf_kernel(x, y, sigma=1.0):
    dist = torch.sum(x ** 2, dim=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
    kernel = torch.exp(-dist / (2 * sigma ** 2))
    return kernel


def initialization(model, mv_loaders, cmv_data, args):
    print('Initializing......')
    criterion = nn.MSELoss()
    optimizers = []
    
    for v in range(args.num_views):
        optimizer = optim.Adam([{"params": model.encoders[f'view_{v}'].parameters(), 'lr': 0.001},
                                {"params": model.decoders[f'view_{v}'].parameters(), 'lr': 0.001}])
        optimizers.append(optimizer)

    for e in range(1, args.initial_epochs + 1):
        for v in range(len(mv_loaders.dataset.data)):  # 迭代每个视角
            for batch_idx, (xv_batch_list, labels_batch) in enumerate(mv_loaders):
                optimizers[v].zero_grad()
                xv = xv_batch_list[v]
                batch_size = xv.shape[0]
                _, xvr = model.sv_encode(xv, v, args)
                view_rec_loss = criterion(xvr, xv)
                view_rec_loss.backward()
                
                optimizers[v].step()
    
    with torch.no_grad():
        initial_data_list = [csv_data.clone().detach().to(args.device).float() for csv_data in cmv_data]
        latent_representation_list = model.mv_encode(initial_data_list, args)
        assert len(latent_representation_list) == args.num_views
        fused_latent_representations = sum(latent_representation_list) / len(latent_representation_list)
        fused_latent_representations = fused_latent_representations.detach().cpu().numpy()

        gmm = GaussianMixture(n_components=args.class_num, covariance_type='diag')
        pre = gmm.fit_predict(fused_latent_representations)
        model.pi.data = torch.from_numpy(gmm.weights_).cuda().float()
        model.mu.data = torch.from_numpy(gmm.means_).cuda().float()
        model.logvar.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())



def train(model, optimizer, scheduler, mv_loaders, best_model_path, args):
    print('Training......')
    eval_data = [sv_d.clone().detach().to(dtype=torch.float32, device=args.device) for sv_d in mv_loaders.dataset.data]
    eval_labels = mv_loaders.dataset.labels

    if args.likelihood == 'Bernoulli':
        likelihood_fn = nn.BCEWithLogitsLoss(reduction='none')
    else:
        likelihood_fn = nn.MSELoss(reduction='none')

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = []
        model.train()
        for batch_idx, (xv_batch_list, labels_batch) in enumerate(mv_loaders):

            optimizer.zero_grad()
            batch_data = [sv_d.to(args.device) for sv_d in xv_batch_list]
            z_sample, z_sample_list, adj_list_tensor, z_mus, z_vars, stack_mu, stack_var, xr_list = model(batch_data, args)
            # z_sample: [batch_size, d_z]
            # model.pi: [d_c]   model.mu: [d_c, d_z] model.logvar:  [d_c, d_z]

            # 1. From vade-pytorch
            total_loss = 0
            for v in range(args.num_views):
                z = model.sampling_fn(z_mus[v], z_vars[v])
                yita_c = torch.exp(torch.log(model.pi.unsqueeze(0)) + gaussian_pdfs_log(z, z_mus[v], model.logvar)) + 1e-10
                loss = 0.5 * torch.mean(torch.sum(yita_c * torch.sum(
                                        model.logvar.unsqueeze(0) +
                                        torch.exp(z_vars[v].unsqueeze(1) - model.logvar.unsqueeze(0)) +
                                        (z_mus[v].unsqueeze(1) - model.mu.unsqueeze(0)).pow(2) / torch.exp(model.logvar.unsqueeze(0)), 2), 1))
                loss -= torch.mean(torch.sum(yita_c * torch.log(model.pi.unsqueeze(0) / (yita_c + 1e-10)), 1))
                loss += 0.5 * torch.mean(torch.sum(1 + z_vars[v], 1))
                
                total_loss += loss
            total_loss = total_loss / args.num_views

            dskl_loss = 0
            for v1 in range(args.num_views):
                for v2 in range(v1 + 1, args.num_views):
                    dskl_loss += gaussian_dskl_divergence(z_mus[v1], z_vars[v1], z_mus[v2], z_vars[v2])
            dskl_loss = dskl_loss / (args.num_views * (args.num_views - 1) / 2)

            rec_term = []
            for v in range(args.num_views):
                rec_X = torch.sum(likelihood_fn(xr_list[v], batch_data[v]), dim=1)  # ( Batch size * Dv )
                rec_A = Bernoulli(torch.sigmoid(z_sample@z_sample.T)).log_prob(adj_list_tensor[v]).mean()
                rec_A1 = graph_contrastive_loss(z_sample, adj_list_tensor[v])
                mmd_Z = MMD_loss(z_sample, z_sample_list[v])
                view_rec_loss = torch.mean(rec_X) + mmd_Z * args.lambda1+ rec_A1
                rec_term.append(view_rec_loss)
            rec_loss = sum(rec_term) / args.num_views

            batch_loss = rec_loss + total_loss + dskl_loss * args.lambda2
            epoch_loss.append(batch_loss.item())
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            model.pi.data = model.pi.data / model.pi.data.sum()

        scheduler.step()
        overall_loss = sum(epoch_loss) / len(epoch_loss)
        
        if epoch % args.interval == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                z_sample, z_sample_list, adj_list_tensor, z_mus, z_vars, stack_mu, stack_var, xr_list = model(eval_data, args)

                # 使用KMeans计算聚类分配
                kmeans = KMeans(n_clusters=args.class_num, n_init=10).fit(z_sample.detach().cpu().numpy())
                predict_kmeans = kmeans.labels_
                acc_kmeans, nmi_kmeans, ari_kmeans, pur_kmeans = evaluate(eval_labels, predict_kmeans)

                print(f'Epoch {epoch:>3}/{args.epochs}  Loss:{overall_loss:.2f}')
                print(f'KMeans:  ACC:{acc_kmeans:.4f}  NMI:{nmi_kmeans:.4f}  ARI:{ari_kmeans:.4f}  PUR:{pur_kmeans:.4f}')

                if acc_kmeans > best_acc:
                    best_acc = acc_kmeans
                    best_ari = ari_kmeans
                    best_nmi = nmi_kmeans
                    best_pur = pur_kmeans
                    # torch.save(model, best_model_path)
                    print('New best model saved at epoch {}'.format(epoch))
                
    print('Finish training')
    print(f'KMeans:  ACC:{best_acc:.4f}  NMI:{best_nmi:.4f}  ARI:{best_ari:.4f}  PUR:{best_pur:.4f}')
    
    return acc_kmeans, nmi_kmeans, ari_kmeans, pur_kmeans

def test(model, mv_loaders, args):
    print('Test......')
    eval_data = [sv_d.clone().detach().to(dtype=torch.float32, device=args.device) for sv_d in mv_loaders.dataset.data]
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
        model = DVIMC(args).to(args.device)

        optimizer = optim.Adam(
            [{"params": model.encoders.parameters(), 'lr': args.learning_rate},
             {"params": model.decoders.parameters(), 'lr': args.learning_rate},
             {"params": model.pi, 'lr': args.prior_learning_rate},
             {"params": model.mu, 'lr': args.prior_learning_rate},
             {"params": model.logvar, 'lr': args.prior_learning_rate},
             ])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_factor)
        initialization(model, cmv_loader, cmv_data, args)
        best_model_path = f'ckpt/best_model_{args.dataset_name}.pt'
        acc, nmi, ari, pur = train(model, optimizer, scheduler, cmv_loader, best_model_path, args)
        # model = torch.load(best_model_path)
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
    parser.add_argument('--epochs', type=int, default=200, help='training epochs')
    parser.add_argument('--initial_epochs', type=int, default=100, help='initialization epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='initial learning rate')
    parser.add_argument('--prior_learning_rate', type=float, default=0.05, help='initial mixture-of-gaussian learning rate')
    parser.add_argument('--z_dim', type=int, default=64, help='latent dimensions')
    parser.add_argument('--lr_decay_step', type=float, default=10, help='StepLr_Step_size')
    parser.add_argument('--lr_decay_factor', type=float, default=0.9, help='StepLr_Gamma')

    parser.add_argument('--dataset', type=int, default=5, choices=range(4), help='0:Caltech7-5v, 1:Scene-15, 2:BDGP, 3:NoisyMNIST, 4:RGBD, 5:Reuters_dim10, 6:MNIST_USPS,7:Caltech101-20,8:LandUse_21,9:Fashion')
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--test_times', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=5)
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.dataset_dir_base = "./data/"
    args.k = 5

    if args.dataset == 0:
        args.dataset_name = 'Caltech7-5V'
        args.alpha = 5
        args.seed = 5
        args.likelihood = 'Gaussian'
    elif args.dataset == 1:
        args.dataset_name = 'Scene-15'
        args.alpha = 20
        args.seed = 19
        args.likelihood = 'Gaussian'
        args.lambda1 = 1
        args.lambda2 = 0.1
    elif args.dataset == 2:
        args.dataset_name = 'BDGP'
        args.alpha = 10
        args.seed = 15
        args.likelihood = 'Gaussian'
    elif args.dataset == 3:
        args.dataset_name = 'NoisyMNIST'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Bernoulli'
        args.batch_size = 512
    elif args.dataset == 4:
        args.dataset_name = 'RGBD'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'
        args.batch_size = 512
        args.lambda1 = 0.1
        args.lambda2 = 0.1
    elif args.dataset == 5:
        args.dataset_name = 'Reuters_dim10'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'
        args.batch_size = 512
        args.lambda1 = 1
        args.lambda2 = 0.1
    elif args.dataset == 6:
        args.dataset_name = 'MNIST-USPS'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'
        args.batch_size = 512
    elif args.dataset == 7:
        args.dataset_name = 'Caltech101-20'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'
        args.batch_size = 512
    elif args.dataset == 8:
        args.dataset_name = 'LandUse_21'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'
        args.batch_size = 512
        args.lambda1 = 10
        args.lambda2 = 0.1
    elif args.dataset == 9:
        args.dataset_name = 'Caltech_2V'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'
        args.batch_size = 512
        args.lambda1 = 1
        args.lambda2 = 0.1
    elif args.dataset == 10:
        args.dataset_name = 'Caltech_3V'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'
        args.batch_size = 512
        args.lambda1 = 1
        args.lambda2 = 0.1
    elif args.dataset == 11:
        args.dataset_name = 'Caltech_4V'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Gaussian'
        args.batch_size = 512
        args.lambda1 = 1
        args.lambda2 = 0.1
    elif args.dataset == 12:
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
