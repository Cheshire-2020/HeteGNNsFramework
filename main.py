import torch
from utils.params import set_params
from utils.load_data import load_data
from models.hgt import HGT
from train.trainer import train, test

def main():
    args = set_params()
    data = load_data(args.dataset)

    metadata = data.metadata()
    if args.model == 'HGT':
        model = HGT(args.hidden_channels, 4, args.num_heads, args.num_layers, metadata)
    else:
        raise ValueError("Invalid model name")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, model = data.to(device), model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

if __name__ == '__main__':
    main()