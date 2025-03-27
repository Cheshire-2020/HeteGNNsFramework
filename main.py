import argparse
import torch
from data import get_dataset
from models import get_model
from trainer import get_trainer
from utils import set_params, load_data, evaluate
import random
import numpy as np

# 1. 参数加载：从 utils 中加载所有超参数。
args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

def main():
    # 2. 数据加载：调用 utils/load_data.py 中的数据处理方法。
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.ratio, args.type_num)

    # nb_classes: 节点类别数。
    nb_classes = label.shape[-1]
    # feats_dim_list: 每种节点特征的维度。
    feats_dim_list = [i.shape[1] for i in feats]
    P = int(len(mps))

    # 2. 加载数据集
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", P)

    data, metadata = get_dataset(name=args.dataset)

    # 3. 构建模型
    print(f"Building model: {args.model}")

    # 啊啊啊明天看咋改，放main里还是params里啊啊啊啊感觉这里的data也得debug看看到底是不是符合的啊啊啊
    metadata = {
        'node_types': args.node_types,  # 节点类型
        'edge_types': args.edge_types  # 边类型
    }

    # 分配设备
    device = torch.device(args.device)
    data = data.to(device)

    # 初始化 Trainer
    model = get_model(
        name=args.model,  # 模型类型 (e.g., "han", "hgt", "metapath2vec")
        metadata=metadata,  # 异质图元数据
        in_channels=feats_dim_list,  # 输入特征维度
        out_channels=nb_classes,  # 输出类别数
        hidden_channels=args.hidden_dim,  # 隐藏层维度
        heads=args.num_heads,  # 注意力头数 (仅对 HAN 等适用)
        num_layers=args.num_layers  # 模型层数 (适用于 HGT)
    )

    # 4. 初始化 Trainer。
    trainer = get_trainer(
        model=model,
        data={
            'x_dict': feats,  # 特征字典
            'edge_index_dict': mps,  # 元路径邻接矩阵
            'label': label,  # 节点标签
            'train_mask': idx_train,  # 训练集索引
            'val_mask': idx_val,  # 验证集索引
            'test_mask': idx_test  # 测试集索引
        },
        device=device,
        task_type=args.task_type,  # 任务类型 ('node_classification', 'embedding_learning')
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 5. 模型训练与评估。
    print("Start training...")
    best_val_acc, best_test_acc = trainer.train(
        num_epochs=args.epochs,
        log_interval=args.log_interval
    )


if __name__ == "__main__":
    main()
