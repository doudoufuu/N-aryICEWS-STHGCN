import os
import torch

root = r"data/csv_events/preprocessed_3"  # 改成你的预处理输出目录

files = {
    "entity_graph": "entity_graph.pt",
    "event_graph": "event_graph.pt",
    "event2chain_graph": "event2chain_graph.pt",
    "chain_graph": "chain_graph.pt",
}

def describe_graph(name, path):
    full = os.path.join(root, path)
    print(f"\n=== {name} ===")
    if not os.path.exists(full):
        print(f"  File not found: {full}")
        return
    g = torch.load(full, map_location="cpu")
    ei = getattr(g, "edge_index", None)
    if ei is None:
        print("  edge_index: None")
        return
    print(f"  edge_index.shape: {tuple(ei.shape)}")
    if ei.numel() == 0:
        print("  edge_index is empty")
        return
    # 行/列分别的最小最大
    row_min = int(ei[0].min().item()); row_max = int(ei[0].max().item())
    col_min = int(ei[1].min().item()); col_max = int(ei[1].max().item())
    gmin = int(ei.min().item()); gmax = int(ei.max().item())
    print(f"  row(min,max): ({row_min}, {row_max})")
    print(f"  col(min,max): ({col_min}, {col_max})")
    print(f"  all(min,max): ({gmin}, {gmax})")
    # 额外属性（若存在）
    for attr in ["checkin_offset", "chain_offset", "num_event", "num_nodes"]:
        if hasattr(g, attr):
            try:
                print(f"  {attr}: {int(getattr(g, attr))}")
            except Exception:
                print(f"  {attr}: {getattr(g, attr)}")
    # x 特征大小
    x = getattr(g, "x", None)
    if isinstance(x, torch.Tensor):
        print(f"  x.shape: {tuple(x.shape)}")

for name, fname in files.items():
    describe_graph(name, fname)
