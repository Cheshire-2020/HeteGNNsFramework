from .acm import load_data as load_acm
from .imdb import load_data as load_imdb
from .dblp import load_data as load_dblp
from .aminer import load_data as load_aminer


def get_dataset(name: str, root: str = './data'):
    if name == "acm":
        return load_acm(root)
    elif name == "imdb":
        return load_imdb(root)
    elif name == "dblp":
        return load_dblp(root)
    elif name == "aminer":
        return load_aminer(root)
    else:
        raise ValueError(f"Unknown dataset: {name}")
