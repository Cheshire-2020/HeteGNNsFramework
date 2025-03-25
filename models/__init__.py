from .han import build_model as build_han
from .hgt import build_model as build_hgt
from .metapath2vec import build_model as build_metapath2vec


def get_model(name, metadata=None, **kwargs):
    """
    动态加载模型

    :param name: 模型名称 ("han", "hgt", "metapath2vec")
    :param metadata: 异构图的元数据 (用于 han 和 hgt)
    :param kwargs: 模型构建所需的额外参数
    :return: 指定模型的实例
    """
    if name == "han":
        return build_han(metadata, **kwargs)
    elif name == "hgt":
        return build_hgt(metadata, **kwargs)
    elif name == "metapath2vec":
        if 'edge_index_dict' not in kwargs or 'metapath' not in kwargs:
            raise ValueError("MetaPath2Vec requires edge_index_dict and metapath as inputs")
        return build_metapath2vec(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")
