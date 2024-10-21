import math
import random
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
from scipy import sparse


class SingleviewDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [d[idx] for d in self.data], self.labels[idx]
    
def normalize(x):
    x = (x-np.min(x)) / (np.max(x)-np.min(x))
    return x


def load_multiview_data(args):
    data_path = args.dataset_dir_base + args.dataset_name + '.npz'
    data = np.load(data_path)
    num_views = int(data['n_views'])
    data_list = [data[f'view_{v}'].astype(np.float32) for v in range(num_views)]
    labels = data['labels']
    dims = [dv.shape[1] for dv in data_list]
    data_size = labels.shape[0]
    class_num = len(np.unique(labels))
    if np.max(labels) == class_num:
        labels = labels - 1
    args.multiview_dims = dims
    args.num_views = num_views
    args.class_num = class_num
    args.data_size = data_size
    return data_list, labels

def load_scene15_data(path, args):
    mat = sio.loadmat(path)
    data_list = [mat['X'][0][i] for i in range(len(mat['X'][0]))]
    # data_list = [normalize(dv).astype('float32') for dv in data_list]
    labels = np.squeeze(mat['Y'])
    dims = [dv.shape[1] for dv in data_list]
    data_size = labels.shape[0]
    class_num = len(np.unique(labels))
    if np.max(labels) == class_num:
        labels = labels - 1
    args.multiview_dims = dims
    args.num_views = len(data_list)
    args.class_num = class_num
    args.data_size = data_size
    return data_list, labels

def load_RGBD_data(path, args):
    mat = sio.loadmat(path)
    data_list = [mat['X1'].astype('float32'), mat['X2'].astype('float32')]
    data_list = [normalize(dv).astype('float32') for dv in data_list]
    labels = np.squeeze(mat['Y'])
    dims = [dv.shape[1] for dv in data_list]
    data_size = labels.shape[0]
    class_num = len(np.unique(labels))
    if np.max(labels) == class_num:
        labels = labels - 1
    args.multiview_dims = dims
    args.num_views = len(data_list)
    args.class_num = class_num
    args.data_size = data_size
    return data_list, labels


def load_Reuters_data(path, args):
    mat = sio.loadmat(path)
    x1 = np.vstack((mat['x_train'][0], mat['x_test'][0])).astype('float32')
    x2 = np.vstack((mat['x_train'][1], mat['x_test'][1])).astype('float32')
    data_list = [x1, x2]
    data_list = [normalize(dv).astype('float32') for dv in data_list]
    labels = np.hstack((mat['y_train'], mat['y_test'])).astype('int').reshape(18758, )
    dims = [dv.shape[1] for dv in data_list]
    data_size = labels.shape[0]
    class_num = len(np.unique(labels))
    if np.max(labels) == class_num:
        labels = labels - 1
    args.multiview_dims = dims
    args.num_views = len(data_list)
    args.class_num = class_num
    args.data_size = data_size
    return data_list, labels

def load_LandUse_21_data(path, args):
    mat = sio.loadmat(path)
    x1 = sparse.csr_matrix(mat['X'][0, 0]).A
    x2 = sparse.csr_matrix(mat['X'][0, 1]).A
    x3 = sparse.csr_matrix(mat['X'][0, 2]).A
    data_list = [x1, x2, x3]
    data_list = [normalize(dv).astype('float32') for dv in data_list]
    labels = np.squeeze(mat['Y']).astype('int').reshape(2100, )
    dims = [dv.shape[1] for dv in data_list]
    data_size = labels.shape[0]
    class_num = len(np.unique(labels))
    if np.max(labels) == class_num:
        labels = labels - 1
    args.multiview_dims = dims
    args.num_views = len(data_list)
    args.class_num = class_num
    args.data_size = data_size
    return data_list, labels


def load_Caltech2v_data(path, args):
    mat = sio.loadmat(path)
    scaler = MinMaxScaler()
    x1 = scaler.fit_transform(mat['X1'].astype(np.float32))
    x2 = scaler.fit_transform(mat['X2'].astype(np.float32))

    data_list = [x1, x2]
    # data_list = [normalize(dv).astype('float32') for dv in data_list]
    labels = np.squeeze(mat['Y']).astype('int').reshape(1400, )
    dims = [dv.shape[1] for dv in data_list]
    data_size = labels.shape[0]
    class_num = len(np.unique(labels))
    if np.max(labels) == class_num:
        labels = labels - 1
    args.multiview_dims = dims
    args.num_views = len(data_list)
    args.class_num = class_num
    args.data_size = data_size
    return data_list, labels

def load_Caltech3v_data(path, args):
    mat = sio.loadmat(path)
    scaler = MinMaxScaler()
    x1 = scaler.fit_transform(mat['X1'].astype(np.float32))
    x2 = scaler.fit_transform(mat['X2'].astype(np.float32))
    x3 = scaler.fit_transform(mat['X5'].astype(np.float32))

    data_list = [x1, x2, x3]
    # data_list = [normalize(dv).astype('float32') for dv in data_list]
    labels = np.squeeze(mat['Y']).astype('int').reshape(1400, )
    dims = [dv.shape[1] for dv in data_list]
    data_size = labels.shape[0]
    class_num = len(np.unique(labels))
    if np.max(labels) == class_num:
        labels = labels - 1
    args.multiview_dims = dims
    args.num_views = len(data_list)
    args.class_num = class_num
    args.data_size = data_size
    return data_list, labels

def load_Caltech4v_data(path, args):
    mat = sio.loadmat(path)
    scaler = MinMaxScaler()
    x1 = scaler.fit_transform(mat['X1'].astype(np.float32))
    x2 = scaler.fit_transform(mat['X2'].astype(np.float32))
    x3 = scaler.fit_transform(mat['X4'].astype(np.float32))
    x4 = scaler.fit_transform(mat['X5'].astype(np.float32))

    data_list = [x1, x2, x3, x4]
    labels = np.squeeze(mat['Y']).astype('int').reshape(1400, )
    dims = [dv.shape[1] for dv in data_list]
    data_size = labels.shape[0]
    class_num = len(np.unique(labels))
    if np.max(labels) == class_num:
        labels = labels - 1
    args.multiview_dims = dims
    args.num_views = len(data_list)
    args.class_num = class_num
    args.data_size = data_size
    return data_list, labels

def load_Caltech5v_data(path, args):
    mat = sio.loadmat(path)
    scaler = MinMaxScaler()
    x1 = scaler.fit_transform(mat['X1'].astype(np.float32))
    x2 = scaler.fit_transform(mat['X2'].astype(np.float32))
    x3 = scaler.fit_transform(mat['X3'].astype(np.float32))
    x4 = scaler.fit_transform(mat['X4'].astype(np.float32))
    x5 = scaler.fit_transform(mat['X5'].astype(np.float32))
    
    data_list = [x1, x2, x3, x4, x5]
    data_list = [normalize(dv).astype('float32') for dv in data_list]
    labels = np.squeeze(mat['Y']).astype('int').reshape(1400, )
    dims = [dv.shape[1] for dv in data_list]
    data_size = labels.shape[0]
    class_num = len(np.unique(labels))
    if np.max(labels) == class_num:
        labels = labels - 1
    args.multiview_dims = dims
    args.num_views = len(data_list)
    args.class_num = class_num
    args.data_size = data_size
    return data_list, labels

def pixel_normalize(data):
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)


def build_dataset(args):
    
    if args.dataset_name == 'Scene-15':
        data_list, labels = load_scene15_data('data/Scene15.mat', args)
        data_list = [StandardScaler().fit_transform(dv) for dv in data_list]
    elif args.dataset_name == 'Reuters_dim10':
        data_list, labels = load_Reuters_data('data/Reuters_dim10.mat', args)
        data_list = [StandardScaler().fit_transform(dv) for dv in data_list]
    elif args.dataset_name == 'RGBD':
        data_list, labels = load_RGBD_data('data/RGB-D.mat', args)
        data_list = [StandardScaler().fit_transform(dv) for dv in data_list]
    elif args.dataset_name == 'LandUse_21':
        data_list, labels = load_LandUse_21_data('data/LandUse-21.mat', args)
        data_list = [StandardScaler().fit_transform(dv) for dv in data_list]
    elif args.dataset_name == 'Caltech_2V':
        data_list, labels = load_Caltech2v_data('data/Caltech-5V.mat', args)
        data_list = [StandardScaler().fit_transform(dv) for dv in data_list]
    elif args.dataset_name == 'Caltech_3V':
        data_list, labels = load_Caltech3v_data('/home/yanwenbiao/CODE/MFLVC-main/data/Caltech-5V.mat', args)
        data_list = [StandardScaler().fit_transform(dv) for dv in data_list]
    elif args.dataset_name == 'Caltech_4V':
        data_list, labels = load_Caltech4v_data('/home/yanwenbiao/CODE/MFLVC-main/data/Caltech-5V.mat', args)
        data_list = [StandardScaler().fit_transform(dv) for dv in data_list]
    elif args.dataset_name == 'Caltech_5V':
        data_list, labels = load_Caltech5v_data('/home/yanwenbiao/CODE/MFLVC-main/data/Caltech-5V.mat', args)
        data_list = [StandardScaler().fit_transform(dv) for dv in data_list]
    else:
        pass

    complete_multiview_data = [torch.tensor(dv, dtype=torch.float32).to(args.device) for dv in data_list]

    return complete_multiview_data, labels
