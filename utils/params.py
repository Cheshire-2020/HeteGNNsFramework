import argparse
import sys

# 从命令行参数中获取数据集名称（dataset）
argv = sys.argv
dataset = argv[1]


def acm_params():
    '''
    通用参数
    --save_emb：是否保存节点嵌入（用于后续分析）。
    --turn：当前运行的轮次（用于多次实验）。
    --ratio：训练集比例（如 20%、40%、60%）。
    --gpu：指定使用的 GPU ID。
    --seed：随机种子，确保实验结果可复现。
    --hidden_dim：隐藏层维度，控制嵌入的大小。
    --nb_epochs：最大训练轮数，防止过拟合。
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")  # 是否保存嵌入
    parser.add_argument('--turn', type=int, default=0)  # 当前运行的轮次
    parser.add_argument('--dataset', type=str, default="acm")  # 数据集名称
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])  # 训练集比例
    parser.add_argument('--gpu', type=int, default=0)  # GPU ID
    parser.add_argument('--seed', type=int, default=0)  # 随机种子
    parser.add_argument('--hidden_dim', type=int, default=64)  # 隐藏层维度
    parser.add_argument('--nb_epochs', type=int, default=10000)  # 最大训练轮数

    '''
    评估参数
    --eva_lr：分类器（如逻辑回归）的学习率。
    --eva_wd：分类器的权重衰减系数。
    '''
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)  # 评估阶段的学习率
    parser.add_argument('--eva_wd', type=float, default=0)  # 评估阶段的权重衰减

    '''
    训练参数
    --patience：用于提前停止的耐心值（如果验证集性能在 patience 轮内没有提升，则停止训练）。
    --lr：模型的学习率。
    --l2_coef：L2 正则化系数，用于防止过拟合。
    '''
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)

    '''
    模型特定参数
    --tau：对比学习中的温度参数，控制正负样本的分布。
    --feat_drop：特征丢弃率，用于防止过拟合。
    --attn_drop：注意力丢弃率，用于防止过拟合。
    --sample_rate：采样率，控制不同类型节点的采样数量。
    --lam：损失函数中的权重参数，用于平衡不同损失项。
    '''
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[7, 1])
    parser.add_argument('--lam', type=float, default=0.5)

    '''
    数据集特定参数
    type_num：每种节点类型的数量（如 ACM 数据集中有 4019 篇论文节点、7167 个作者节点、60 个领域节点）。
    nei_num：邻居类型的数量（如 ACM 数据集中每个节点有 2 种邻居类型）。
    '''
    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[6])
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # the number of every node type
    args.nei_num = 1  # the number of neighbors' types
    return args


def aminer_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="aminer")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--feat_drop', type=float, default=0.5)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[3, 8])
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()
    args.type_num = [6564, 13329, 35890]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args


def freebase_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="freebase")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--feat_drop', type=float, default=0.1)
    parser.add_argument('--attn_drop', type=float, default=0.3)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[1, 18, 2])
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()
    args.type_num = [3492, 2502, 33401, 4459]  # the number of every node type
    args.nei_num = 3  # the number of neighbors' types
    return args


'''
数据集参数设置
根据 dataset 的值调用对应的参数设置函数（如 acm_params, dblp_params 等）
'''


def set_params():
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == "aminer":
        args = aminer_params()
    elif dataset == "freebase":
        args = freebase_params()
    return args
