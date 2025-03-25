import argparse
import torch
from data import get_dataset
from models import get_model
from trainer import get_trainer


def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="Heterogeneous Graph Neural Network Training")
    parser.add_argument("--dataset", type=str, required=True, choices=["acm", "imdb", "dblp", "aminer"],
                        help="Dataset to use (e.g., acm, imdb, dblp, aminer)")
    parser.add_argument("--model", type=str, required=True, choices=["han", "hgt", "metapath2vec"],
                        help="Model to use (e.g., han, hgt, metapath2vec)")
    parser.add_argument("--task", type=str, default="node_classification",
                        choices=["node_classification", "embedding_learning"],
                        help="Task type (node_classification or embedding_learning)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run (e.g., cpu, cuda)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay")
    args = parser.parse_args()

    # 2. 加载数据集
    print(f"Loading dataset: {args.dataset}")
    data, metadata = get_dataset(name=args.dataset)

    # 3. 构建模型
    print(f"Building model: {args.model}")
    model_params = {}
    if args.model == "han" or args.model == "hgt":
        model_params["metadata"] = metadata
        model_params["in_channels"] = -1  # 自动检测输入维度
        model_params["out_channels"] = len(set(data['movie'].y.tolist()))  # 类别数
    elif args.model == "metapath2vec":
        model_params["edge_index_dict"] = data.edge_index_dict
        model_params["metapath"] = [["author", "writes", "paper"], ["paper", "cites", "paper"]]

    model = get_model(name=args.model, **model_params)

    # 4. 分配设备
    device = torch.device(args.device)
    data = data.to(device)

    # 5. 初始化 Trainer
    print(f"Initializing trainer for task: {args.task}")
    trainer = get_trainer(
        model=model,
        data=data,
        device=device,
        task_type=args.task,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 6. 训练并输出结果
    print("Training...")
    best_val_acc, best_test_acc = trainer.train(num_epochs=args.epochs)

    # 7. 打印结果
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Test Accuracy: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()
