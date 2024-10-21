import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from torch.distributions import Normal, Bernoulli, Categorical, Dirichlet



class Gaussian_sampling(nn.Module):
    def forward(self, mu, var):
        std = torch.sqrt(var)
        epi = std.data.new(std.size()).normal_()
        return epi * std + mu
    
def compute_Laplacian(data, n_neighbors=5):
    '''data: (n_samples, n_features)'''
    adj = kneighbors_graph(data, n_neighbors, include_self=True, metric='cosine').toarray()
    adj = adj + adj.T
    adj[adj != 0] = 1
    Lap = adj.sum(1, keepdims=True)**(-.5) * adj * adj.sum(0, keepdims=True)**(-.5)
    return adj, Lap


class GConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1))**(0.5)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, rowLap, colLap=None):
        support = torch.mm(input, self.weight)
        output = rowLap @ support
        if colLap is not None:
            output = output @ colLap

        if self.bias is not None:
            return output + self.bias
        else:
            return output
    

class view_specific_encoder_Gaussian(nn.Module):
    def __init__(self, view_dim, latent_dim):
        super(view_specific_encoder_Gaussian, self).__init__()
        self.x_dim = view_dim
        self.z_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.x_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU()
        )
        self.GConv = GConv(2000, self.z_dim)
        self.encoder_mu = nn.Linear(self.z_dim, self.z_dim)
        self.encoder_logvar = nn.Linear(self.z_dim, self.z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x, adj):
        hidden_feature = self.encoder(x)
        h = F.relu(self.GConv(hidden_feature, adj))  # 添加ReLU激活函数
        mu = self.encoder_mu(h)
        logvar = self.softplus(self.encoder_logvar(h))
        return mu, logvar


class view_specific_decoder(nn.Module):
    def __init__(self, view_dim, latent_dim):
        super(view_specific_decoder, self).__init__()
        self.x_dim = view_dim
        self.z_dim = latent_dim
        self.decoder = nn.Sequential(nn.Linear(self.z_dim, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, self.x_dim),
                                     )

    def forward(self, z):
        xr = self.decoder(z)
        return xr


class DVIMC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.x_dim_list = args.multiview_dims
        self.k = args.class_num  # 7
        self.z_dim = args.z_dim  # 10
        self.num_views = args.num_views  # 5

        self.pi = nn.Parameter(torch.full((self.k,), 1 / self.k), requires_grad=True)  # torch.Size([7])
        self.mu = nn.Parameter(torch.full((self.k, self.z_dim), 0.0), requires_grad=True)  # torch.Size([7, 10])
        self.logvar = nn.Parameter(torch.full((self.k, self.z_dim), 1.0), requires_grad=True)  # torch.Size([7, 10])

        self.encoders = nn.ModuleDict({f'view_{v}': view_specific_encoder_Gaussian(self.x_dim_list[v], self.z_dim) for v in range(self.num_views)})
        self.decoders = nn.ModuleDict({f'view_{v}': view_specific_decoder(self.x_dim_list[v], self.z_dim) for v in range(self.num_views)})
        self.qx2cNet = nn.ModuleDict({f'view_{v}': GConv(self.x_dim_list[v], self.k) for v in range(self.num_views)})

        self.sampling_fn = Gaussian_sampling()
        self.softplus = nn.Softplus()

    def mv_encode(self, x_list, args):
        adj_list, Lap_list = zip(*[compute_Laplacian(np.array(x.detach().cpu()), args.k) for x in x_list])
        adj_list_np = np.array(adj_list)
        Lap_list_np = np.array(Lap_list)
        adj_list_tensor = torch.FloatTensor(adj_list_np).to(args.device)
        Lap_list_tensor = torch.FloatTensor(Lap_list_np).to(args.device)

        latent_representation_list = []
        for v in range(self.num_views):
            vs_mu, vs_var = self.encoders[f'view_{v}'](x_list[v], Lap_list_tensor[v])
            latent_representation = self.sampling_fn(vs_mu, vs_var)
            latent_representation_list.append(latent_representation)
        return latent_representation_list

    def sv_encode(self, x, view_idx, args):
        adj, Lap = compute_Laplacian(np.array(x.detach().cpu()), args.k)

        adj, Lap = torch.FloatTensor(adj), torch.FloatTensor(Lap).to(args.device)

        vs_mu, vs_var = self.encoders[f'view_{view_idx}'](x, Lap)
        latent_representation = self.sampling_fn(vs_mu, vs_var)
        xr = self.decoders[f'view_{view_idx}'](vs_mu)
        return latent_representation, xr

    def inference_z(self, x_list, Lap_list):
        z_mus, z_vars = [], []
        for v in range(self.num_views):
            vs_mu, vs_var = self.encoders[f'view_{v}'](x_list[v], Lap_list[v])
            z_mus.append(vs_mu)
            z_vars.append(vs_var)
        mu = torch.stack(z_mus)  # torch.Size([5, 256, 10])
        var = torch.stack(z_vars)  # torch.Size([5, 256, 10])
        return z_mus, z_vars, mu, var

    def generation_x(self, z_list):
        xr_list = []
        for view_idx in range(len(z_list)):
            xr_view = self.decoders[f'view_{view_idx}'](z_list[view_idx])
            xr_list.append(xr_view)
        return xr_list

    def forward(self, x_list, args):
        adj_list, Lap_list = zip(*[compute_Laplacian(np.array(x.detach().cpu()), args.k) for x in x_list])
        adj_list_np = np.array(adj_list)
        Lap_list_np = np.array(Lap_list)
        adj_list_tensor = torch.FloatTensor(adj_list_np).to(args.device)
        Lap_list_tensor = torch.FloatTensor(Lap_list_np).to(args.device)
        z_mus, z_vars, stack_mu, stack_var = self.inference_z(x_list, Lap_list_tensor)
        
        # 采样 z_sample 列表
        z_sample_list = [self.sampling_fn(mu, var) for mu, var in zip(z_mus, z_vars)]
        z_sample_mean = torch.mean(torch.stack(z_sample_list), dim=0)

        xr_list = self.generation_x(z_sample_list)
        return z_sample_mean, z_sample_list, adj_list_tensor, z_mus, z_vars, stack_mu, stack_var, xr_list