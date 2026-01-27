

import numpy as np
from typing import List, Optional, NamedTuple
from scipy.sparse import coo_matrix
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_mean, scatter_max, scatter_add, scatter_min
from utils import haversine
import logging
from collections import defaultdict
import random
import torch.nn.functional as F

data = torch.load("data/csv_events/preprocessed_1/chain_graph.pt")
print("事件链数:", data.num_nodes)
print("edge_index max:", data.edge_index.max().item())
print("edge_index min:", data.edge_index.min().item())
print("num_nodes:", data.num_nodes)

max_node_id = int(data.edge_index.max().item())
if max_node_id >= data.num_nodes:
    print(f"[修复] num_nodes 从 {data.num_nodes} 改为 {max_node_id + 1}")
    data.num_nodes = max_node_id + 1

degree = torch.zeros(data.num_nodes)


max_node_id = int(data.edge_index.max().item())
if max_node_id >= data.num_nodes:
    print(f"[修复] num_nodes 从 {data.num_nodes} 改为 {max_node_id + 1}")
    data.num_nodes = max_node_id + 1

for edge in data.edge_index.t():
    degree[edge[0]] += 1
    degree[edge[1]] += 1
print("没有连接的事件链:", (degree == 0).nonzero())