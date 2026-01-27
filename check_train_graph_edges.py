import torch

try:
    # 如果 torch==2.6+ 默认 weights_only=True，需要显式允许 Data
    from torch_geometric.data import Data
    torch.serialization.add_safe_globals([Data])
except Exception:
    pass

PATH = r"data/csv_events/preprocessed_5/chain_graph.pt"

def main():
    print(f"[ChainGraph] loading {PATH}")
    data = torch.load(PATH, map_location="cpu", weights_only=False)
    edge_index = getattr(data, "edge_index", data["edge_index"])
    print("[ChainGraph] edge_index shape:", tuple(edge_index.shape))
    print("[ChainGraph] total edges:", edge_index.shape[1])

    if hasattr(data, "edge_attr"):
        edge_attr = data.edge_attr
        print("[ChainGraph] edge_attr shape:", tuple(edge_attr.shape))

if __name__ == "__main__":
    main()
