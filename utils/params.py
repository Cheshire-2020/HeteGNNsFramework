import argparse

def set_params():
    parser = argparse.ArgumentParser(description="Training Heterogeneous Graph Models")
    parser.add_argument('--model', type=str, default='hgt',
                        choices=['hgt', 'han'],
                        help='Name of the model to use')
    parser.add_argument('--dataset', type=str, default='dblp',
                        choices=['dblp', 'imdb'],
                        help='Dataset to use')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Hidden layer dimensions for the model')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='Number of attention heads (for HGT)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in the model')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay for optimizer')
    # 直接解析参数为 args 并返回
    args = parser.parse_args()

    # 转换参数为统一格式（如转换为大写）
    args.model = args.model.upper()
    args.dataset = args.dataset.upper()

    return args