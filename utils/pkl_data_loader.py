import os
import pickle
import torch
from torch_geometric.data import HeteroData


class PKLDataLoader:
    def __init__(self, root: str, dataset: str):
        """
        Args:
            root: 数据集根目录 (如 'data/')
            dataset: 数据集名称 ('acm', 'dblp', 'imdb')
        """
        self.path = os.path.join(root, dataset.upper())
        self.dataset = dataset.lower()

    def load(self) -> HeteroData:
        data = HeteroData()

        # 加载节点特征
        with open(os.path.join(self.path, 'node_features.pkl'), 'rb') as f:
            node_features = pickle.load(f)
            for node_type, feat in node_features.items():
                data[node_type].x = torch.tensor(feat).float()

        # 加载边数据
        with open(os.path.join(self.path, 'edges.pkl'), 'rb') as f:
            edges = pickle.load(f)
            for (src_type, rel_type, dst_type), edge_index in edges.items():
                data[(src_type, rel_type, dst_type)].edge_index = edge_index.long().t().contiguous()

        # 加载标签（需根据数据集指定目标节点类型）
        with open(os.path.join(self.path, 'labels.pkl'), 'rb') as f:
            labels = pickle.load(f)
            target_type = self._get_target_type()
            data[target_type].y = torch.tensor(labels).long()

        return data

    def _get_target_type(self) -> str:
        """获取每个数据集的标签对应节点类型"""
        return {
            'acm': 'paper',
            'dblp': 'author',
            'imdb': 'movie'
        }[self.dataset]