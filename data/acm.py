import os.path as osp
import numpy as np
import torch
from torch_geometric.data import HeteroData
from scipy.sparse import coo_matrix


def load_data(root: str = './data/acm'):
    # 根目录定义
    path = osp.abspath(root)
    # 验证路径调试
    print(f"Loading dataset from: {path}")
    print(osp.join(path, 'a_feat.npz'))  # 检查具体文件路径是否正确

    # 加载具体文件
    a_feat_file = np.load(osp.join(path, 'a_feat.npz'))  # 加载具体文件，而不是目录
    print(f"Keys in a_feat.npz: {a_feat_file.keys()}")  # 检查键名
    print(f"Keys in a_feat.npz: {list(a_feat_file.keys())}")  # 打印所有键名

    # 从键构建稀疏矩阵
    if all(key in a_feat_file for key in ['row', 'col', 'data', 'shape']):
        row = a_feat_file['row']
        col = a_feat_file['col']
        data = a_feat_file['data']
        shape = tuple(a_feat_file['shape'])
        author_features = coo_matrix((data, (row, col)), shape=shape).toarray()
    else:
        raise KeyError("Expected keys ['row', 'col', 'data', 'shape'] not found in a_feat.npz")

        # 数据加载
    data = HeteroData()
    data['author'].x = torch.tensor(author_features, dtype=torch.float)
    data['paper'].x = torch.tensor(np.load(osp.join(path, 'p_feat.npz'))['arr_0'], dtype=torch.float)
    data['author', 'writes', 'paper'].edge_index = torch.tensor(
        np.load(osp.join(path, 'pap.npz'))['indices'], dtype=torch.long)
    data['paper', 'written_by', 'author'].edge_index = data['author', 'writes', 'paper'].edge_index[[1, 0]]

    # 加载训练、验证、测试集
    data['paper'].train_mask = torch.tensor(np.load(osp.join(path, 'train_60.npy')), dtype=torch.bool)
    data['paper'].val_mask = torch.tensor(np.load(osp.join(path, 'val_60.npy')), dtype=torch.bool)
    data['paper'].test_mask = torch.tensor(np.load(osp.join(path, 'test_60.npy')), dtype=torch.bool)
    data['paper'].y = torch.tensor(np.load(osp.join(path, 'labels.npy')), dtype=torch.long)

    return data, "acm"
