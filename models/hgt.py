import torch
from torch import nn
from torch_geometric.nn import HGTConv, Linear


class HGT(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) 模型实现
    """

    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        """
        初始化 HGT 模型

        :param hidden_channels: 隐藏层维度
        :param out_channels: 输出特征维度 (类别数)
        :param num_heads: Transformer 注意力头数量
        :param num_layers: HGTConv 层数
        :param metadata: 异构图的元信息
        """
        super().__init__()
        # 定义每种节点类型的线性变换，统一到 hidden_channels
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # 堆叠 HGTConv 层
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        # 最终的线性分类层
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        """
        前向传播过程

        :param x_dict: 节点特征字典
        :param edge_index_dict: 边索引字典
        :return: 分类结果
        """
        # 对每种节点类型的特征进行线性变换并 ReLU 激活
        x_dict = {key: self.lin_dict[key](x).relu_() for key, x in x_dict.items()}

        # 经过每一层 HGTConv 处理
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # 返回 'author' 节点的分类结果
        return self.lin(x_dict['author'])


def build_model(metadata, hidden_channels=64, out_channels=4, num_heads=2, num_layers=1):
    """
    构建 HGT 模型的便捷方法

    :param metadata: 异构图的元数据
    :param hidden_channels: 隐藏层维度
    :param out_channels: 输出分类类别数
    :param num_heads: 注意力头数量
    :param num_layers: HGTConv 层数
    :return: HGT 模型实例
    """
    return HGT(
        hidden_channels=hidden_channels, out_channels=out_channels,
        num_heads=num_heads, num_layers=num_layers, metadata=metadata
    )
