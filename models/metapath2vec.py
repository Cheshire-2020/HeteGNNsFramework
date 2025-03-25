import torch
from torch_geometric.nn import MetaPath2Vec


def build_model(edge_index_dict, embedding_dim=128, metapath=None, walk_length=50,
                context_size=7, walks_per_node=5, num_negative_samples=5):
    """
    构建 MetaPath2Vec 模型

    :param edge_index_dict: 异构图的边索引字典
    :param embedding_dim: 嵌入向量维度
    :param metapath: 元路径信息列表 (如 [("author", "writes", "paper"), ...])
    :param walk_length: 随机游走的长度
    :param context_size: 上下文窗口大小
    :param walks_per_node: 每个节点的随机游走次数
    :param num_negative_samples: 负采样数量
    :return: MetaPath2Vec 模型实例
    """
    if metapath is None:
        raise ValueError("Metapath cannot be None for MetaPath2Vec")

    return MetaPath2Vec(
        edge_index_dict=edge_index_dict, embedding_dim=embedding_dim,
        metapath=metapath, walk_length=walk_length, context_size=context_size,
        walks_per_node=walks_per_node, num_negative_samples=num_negative_samples,
        sparse=True  # 使用稀疏更新以提升效率
    )
