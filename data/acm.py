import os.path as osp
import numpy as np
import torch
from torch_geometric.data import HeteroData


def load_data(root: str = './data/ACM'):
    # 根目录定义
    path = osp.abspath(root)

    # 数据加载
    data = HeteroData()
    data['author'].x = np.load(osp.join(path, 'a_feat.npz'))['arr_0']
    data['paper'].x = np.load(osp.join(path, 'p_feat.npz'))['arr_0']
    data['author', 'writes', 'paper'].edge_index = torch.tensor(
        np.load(osp.join(path, 'pap.npz'))['indices'], dtype=torch.long)
    data['paper', 'written_by', 'author'].edge_index = data['author', 'writes', 'paper'].edge_index[[1, 0]]

    # 加载训练、验证、测试集
    data['paper'].train_mask = torch.tensor(np.load(osp.join(path, 'train_60.npy')), dtype=torch.bool)
    data['paper'].val_mask = torch.tensor(np.load(osp.join(path, 'val_60.npy')), dtype=torch.bool)
    data['paper'].test_mask = torch.tensor(np.load(osp.join(path, 'test_60.npy')), dtype=torch.bool)
    data['paper'].y = torch.tensor(np.load(osp.join(path, 'labels.npy')), dtype=torch.long)

    return data, "acm"
