import os
import os.path as osp
import sys
import torch


def _ensure_torch_geometric_stubs() -> None:
    """Ensure classes needed by pickled PyG objects exist at import paths.

    Two modes:
    1) torch_geometric is installed but missing storage classes (e.g., GlobalStorage):
       define lightweight classes in torch_geometric.data.storage with correct __module__.
    2) torch_geometric not installed: create minimal stub modules/classes.
    """
    import types
    import sys as _sys

    try:
        import importlib
        tg_data = importlib.import_module("torch_geometric.data")
        try:
            tg_storage = importlib.import_module("torch_geometric.data.storage")
        except Exception:
            tg_storage = types.ModuleType("torch_geometric.data.storage")
            _sys.modules["torch_geometric.data.storage"] = tg_storage

        # Define missing storage classes with correct module path
        names = ("BaseStorage", "NodeStorage", "EdgeStorage", "GlobalStorage")
        missing = [n for n in names if not hasattr(tg_storage, n)]
        if missing:
            def _define(module, name):
                cls = type(name, (object,), {})
                cls.__module__ = module.__name__
                setattr(module, name, cls)
                return cls

            defined = [
                _define(tg_storage, nm)
                for nm in names if nm in missing
            ]
            if defined:
                torch.serialization.add_safe_globals(defined)

        # Ensure Data exists; sometimes pickles reference torch_geometric.data.data.Data
        if not hasattr(tg_data, "Data"):
            module_data_data = types.ModuleType("torch_geometric.data.data")

            class _Data(dict):
                __module__ = module_data_data.__name__
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.__dict__ = self
                def __getattr__(self, item):
                    return self.__dict__.get(item)
                def __setattr__(self, key, value):
                    self.__dict__[key] = value

            module_data_data.Data = _Data
            _sys.modules[module_data_data.__name__] = module_data_data
            torch.serialization.add_safe_globals([_Data])
        return

    except Exception:
        # torch_geometric not installed; create stubs
        import types
        module_root = types.ModuleType("torch_geometric")
        module_data = types.ModuleType("torch_geometric.data")
        module_data_data = types.ModuleType("torch_geometric.data.data")
        module_storage = types.ModuleType("torch_geometric.data.storage")

        class _Stub(dict):
            __module__ = module_data_data.__name__
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.__dict__ = self
            def __getattr__(self, item):
                return self.__dict__.get(item)
            def __setattr__(self, key, value):
                self.__dict__[key] = value

        class Data(_Stub):
            pass
        Data.__module__ = module_data_data.__name__

        class DataEdgeAttr(_Stub):
            pass
        DataEdgeAttr.__module__ = module_data_data.__name__

        class DataTensorAttr(_Stub):
            pass
        DataTensorAttr.__module__ = module_data_data.__name__

        # Storage hierarchy
        class BaseStorage(object):
            pass
        class NodeStorage(BaseStorage):
            pass
        class EdgeStorage(BaseStorage):
            pass
        class GlobalStorage(BaseStorage):
            pass
        for cls in (BaseStorage, NodeStorage, EdgeStorage, GlobalStorage):
            cls.__module__ = module_storage.__name__

        module_data.Data = Data
        module_data_data.Data = Data
        module_data_data.DataEdgeAttr = DataEdgeAttr
        module_data_data.DataTensorAttr = DataTensorAttr
        module_storage.BaseStorage = BaseStorage
        module_storage.NodeStorage = NodeStorage
        module_storage.EdgeStorage = EdgeStorage
        module_storage.GlobalStorage = GlobalStorage

        _sys.modules["torch_geometric"] = module_root
        _sys.modules["torch_geometric.data"] = module_data
        _sys.modules["torch_geometric.data.data"] = module_data_data
        _sys.modules["torch_geometric.data.storage"] = module_storage

        torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, BaseStorage, NodeStorage, EdgeStorage, GlobalStorage])


def tload(path: str):
    _ensure_torch_geometric_stubs()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except AttributeError as e:
        # Retry after ensuring stubs; final fallback to weights_only
        if "GlobalStorage" in str(e) or "storage" in str(e):
            _ensure_torch_geometric_stubs()
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                return torch.load(path, map_location="cpu", weights_only=True)
        raise


def describe_edges(name: str, g) -> None:
    if not hasattr(g, "edge_index") or g.edge_index is None:
        print(f"{name}: no edge_index")
        return
    ei = g.edge_index
    if not torch.is_tensor(ei):
        try:
            ei = torch.as_tensor(ei)
        except Exception:
            print(f"{name}: edge_index not a tensor and cannot convert")
            return
    mn = int(ei.min().item()) if ei.numel() else -1
    mx = int(ei.max().item()) if ei.numel() else -1
    print(f"{name}: edges shape={tuple(ei.shape)} min={mn} max={mx}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_graph_offsets_v2.py <preprocessed_dir>")
        sys.exit(1)
    root = sys.argv[1]

    ent_p = osp.join(root, "entity_graph.pt")
    evt_p = osp.join(root, "event_graph.pt")
    e2c_p = osp.join(root, "event2chain_graph.pt")
    chn_p = osp.join(root, "chain_graph.pt")

    for p in (ent_p, evt_p, e2c_p, chn_p):
        if not osp.exists(p):
            print(f"Missing file: {p}")
            sys.exit(2)

    entity = tload(ent_p)
    event = tload(evt_p)
    e2c = tload(e2c_p)
    chain = tload(chn_p)

    # Report features
    for name, g in (("entity_graph", entity), ("event_graph", event), ("event2chain_graph", e2c), ("chain_graph", chain)):
        x_shape = tuple(getattr(g, "x", torch.empty(0, 0)).shape) if hasattr(g, "x") and g.x is not None else None
        extra = []
        for attr in ("checkin_offset", "chain_offset", "num_entity", "num_event"):
            if hasattr(g, attr):
                try:
                    extra.append(f"{attr}={int(getattr(g, attr))}")
                except Exception:
                    pass
        print(f"{name}: x={x_shape} {' '.join(extra)}")
        describe_edges(name, g)

    # Offsets
    try:
        checkin_offset = int(getattr(entity, "checkin_offset", getattr(event, "checkin_offset", 0)))
    except Exception:
        checkin_offset = 0
    try:
        chain_base = int(getattr(e2c, "chain_offset", getattr(chain, "chain_offset", checkin_offset * 2)))
    except Exception:
        chain_base = checkin_offset * 2

    num_events = event.x.size(0) if hasattr(event, "x") and torch.is_tensor(event.x) else None
    num_chains = chain.x.size(0) if hasattr(chain, "x") and torch.is_tensor(chain.x) else None

    print("\n[Expect] event_base=", checkin_offset, "chain_base=", chain_base, "#events=", num_events, "#chains=", num_chains)

    # Validate event2chain splits by base
    ei = e2c.edge_index if hasattr(e2c, "edge_index") else None
    if torch.is_tensor(ei) and ei.numel() > 0:
        left = ei[0]
        right = ei[1]
        bad_l = int((left < checkin_offset).sum().item())
        bad_r = int((right < chain_base).sum().item())
        print(f"[Check] event2chain left>=event_base violations: {bad_l}")
        print(f"[Check] event2chain right>=chain_base violations: {bad_r}")
        if bad_l == 0 and bad_r == 0:
            print("[OK] event2chain edges fall into expected segments.")
        else:
            print("[WARN] Found out-of-segment indices in event2chain.")


if __name__ == "__main__":
    main()

