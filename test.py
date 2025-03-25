import pickle
import os.path as osp

# 修改为实际路径
path_to_edges_pkl = "./data/IMDB/edges.pkl"

# 查看 edges.pkl 的内容
with open(path_to_edges_pkl, "rb") as f:
    edges_data = pickle.load(f)

print(type(edges_data))  # 查看类型
print(edges_data)        # 打印内容