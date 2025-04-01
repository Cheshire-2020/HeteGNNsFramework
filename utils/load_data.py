import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP, IMDB
import os

def load_data(dataset_name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), f'../data/{dataset_name}')
    print("Dataset path:", path)
    print("Path exists:", os.path.exists(path))

    print(f'Loading {dataset_name}...')
    print(path)
    if dataset_name == 'DBLP':
        dataset = DBLP(path, transform=T.Constant(node_types='conference'))
    elif dataset_name == 'IMDB':
        dataset = IMDB(path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return dataset[0]