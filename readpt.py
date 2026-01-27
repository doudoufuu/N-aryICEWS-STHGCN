import torch

# 加载文件
data = torch.load("/home/beihang/hsy/Spatio-Temporal-Hypergraph-Model/data/csv_events/preprocessed/entity_graph.pt")  

import torch
from torch_geometric.data import Data

print(data)  # 打印 Data 对象的基本信息

# 查看具体属性
print("节点特征 x 的形状:", data.x.shape)          # 查看节点特征的形状
print("边索引 edge_index 的形状:", data.edge_index.shape)  # 查看边索引的形状
print("边特征 edge_attr 的形状:", data.edge_attr.shape)   # 查看边特征的形状
print("超边数量:", data.num_hyperedges)           # 查看超边数量

print(data.x[:5])  # 如果 data.x 是张量，直接切片