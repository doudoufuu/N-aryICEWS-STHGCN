import torch
import os
from torch_geometric.data import Data

# 三层超图路径
base_path = "data/csv_events/preprocessed_1"
files = ["entity_graph.pt", "event_graph.pt", "chain_graph.pt"]

def inspect_graph(file_path):
    print(f"\n=== Inspecting: {file_path} ===")
    data = torch.load(file_path)
    
    if not isinstance(data, Data):
        print(f"⚠️ Warning: {file_path} 不是 torch_geometric.data.Data 对象，实际类型：{type(data)}")
        return
    
    print(f"节点数: {data.num_nodes}")
    if hasattr(data, 'x') and data.x is not None:
        print(f"节点特征: {data.x.shape}")
    else:
        print("节点特征: None")
    
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        print(f"边数量: {data.edge_index.size(1)}")
        print(f"边索引 shape: {data.edge_index.shape}")
    else:
        print("边索引: None")
    
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        print(f"边特征: {data.edge_attr.shape}")
    else:
        print("边特征: None")
    
    # 打印前几个节点和边
    if hasattr(data, 'x') and data.x is not None:
        print("前 3 个节点特征:\n", data.x[:3])
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        print("前 5 条边 (src, dst):\n", data.edge_index[:, :5])
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        print("前 5 条边特征:\n", data.edge_attr[:5])

for f in files:
    file_path = os.path.join(base_path, f)
    if os.path.exists(file_path):
        inspect_graph(file_path)
    else:
        print(f"❌ 文件不存在: {file_path}")
