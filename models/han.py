import torch
import torch.nn as nn
from torch_geometric.nn import HANConv


class HAN(nn.Module):
    """
    Heterogeneous Attention Network (HAN) 实现
    """

    def __init__(self, in_channels, out_channels, hidden_channels=128, heads=8, metadata=None):
        """
        初始化 HAN 模型

        :param in_channels: 输入特征维度 (int or dict of int)
        :param out_channels: 输出特征维度 (int, 通常为类别数)
        :param hidden_channels: 隐藏层维度 (默认 128)
        :param heads: 多头注意力数量 (默认 8)
        :param metadata: 数据的元信息，用于异构图定义
        """
        super().__init__()
        self.han_conv = HANConv(
            in_channels, hidden_channels, heads=heads, dropout=0.6, metadata=metadata
        )  # HANConv 层
        self.lin = nn.Linear(hidden_channels, out_channels)  # 输出分类层

    def forward(self, x_dict, edge_index_dict):
        """
        前向传播过程

        :param x_dict: 节点特征字典
        :param edge_index_dict: 边索引字典
        :return: 分类结果
        """
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out['movie'])  # 使用 'movie' 节点的输出
        return out


def build_model(metadata, in_channels=-1, out_channels=3, hidden_channels=128, heads=8):
    """
    构建 HAN 模型的便捷方法

    :param metadata: 异构图的元数据
    :param in_channels: 输入特征维度
    :param out_channels: 输出类别数
    :param hidden_channels: 隐藏层维度
    :param heads: 注意力头数量
    :return: HAN 模型实例
    """
    return HAN(
        in_channels=in_channels, out_channels=out_channels,
        hidden_channels=hidden_channels, heads=heads, metadata=metadata
    )
