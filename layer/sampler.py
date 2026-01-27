import logging  # [ADDED-DEBUG]
import torch  # [ADDED]
from torch import Tensor  # [ADDED]
from torch_sparse import SparseTensor  # [ADDED]
from typing import List, NamedTuple, Optional, Tuple  # [ADDED]


class NeighborSampler(torch.utils.data.DataLoader):  # [ADDED]
    """Four-layer neighbor sampler aligned with Entity→Event→Chain→ChainNetwork."""  # [ADDED]

    def __init__(  # [ADDED]
        self,  # [ADDED]
        x: Tensor,  # [MODIFIED]
        edge_index: List[Tensor],  # [ADDED]
        edge_attr: List[Optional[Tensor]],  # [ADDED]
        edge_t: List[Optional[Tensor]],  # [ADDED]
        edge_delta_t: List[Optional[Tensor]],  # [ADDED]
        edge_delta_s: List[Optional[Tensor]],  # [ADDED]
        edge_type: List[Optional[Tensor]],  # [ADDED]
        sizes: List[int],  # [ADDED]
        sample_idx: Tensor,  # [ADDED]
        node_idx: Tensor,  # [ADDED]
        label: Tensor,  # [ADDED]
        candidates: Tensor,  # [ADDED]
        max_time: Optional[Tensor] = None,  # [ADDED]
        intra_jaccard_threshold: float = 0.0,  # [MODIFIED-SAMPLER]
        inter_jaccard_threshold: float = 0.0,  # [MODIFIED-SAMPLER]
        total_nodes: Optional[int] = None,  # [ADDED]
        **kwargs,  # [ADDED]
    ):  # [ADDED]
        # sampler initialization
        if len(edge_index) != 4:  # [ADDED]
            raise ValueError("NeighborSampler expects four adjacency tensors.")  # [ADDED]
        self.sizes = sizes  # [ADDED]
        self.node_idx = node_idx.cpu()  # [ADDED]
        self.labels = label.cpu()  # [ADDED]
        self.candidates = candidates.cpu()  # [ADDED]
        self.max_time = max_time.cpu() if max_time is not None else None  # [ADDED]
        self.intra_jaccard_threshold = intra_jaccard_threshold  # [MODIFIED-SAMPLER]
        self.inter_jaccard_threshold = inter_jaccard_threshold  # [MODIFIED-SAMPLER]
        self.x = x.cpu()  # [MODIFIED]
        self.total_nodes = int(total_nodes) if total_nodes is not None else self.x.size(0)  # [ADDED]

        self.entity_adj = self._build_sparse(edge_index[0], edge_attr[0])  # [ADDED]
        self.event_adj = self._build_sparse(edge_index[1], edge_attr[1])  # [ADDED]
        self.event_chain_adj = self._build_sparse(edge_index[2], edge_attr[2])  # [ADDED]
        self.chain_adj = self._build_sparse(edge_index[3], edge_attr[3])  # [ADDED]

        self.entity_attr = edge_attr[0]  # [ADDED]
        self.event_attr = edge_attr[1]  # [ADDED]
        self.event_chain_attr = edge_attr[2]  # [ADDED]
        self.chain_attr = edge_attr[3]  # [ADDED]

        self.entity_type = edge_type[0]  # [ADDED]
        self.event_type = edge_type[1]  # [ADDED]
        self.event_chain_type = edge_type[2]  # [ADDED]
        self.chain_type = edge_type[3]  # [ADDED]

        self.entity_t = edge_t[0]  # [ADDED]
        self.event_t = edge_t[1]  # [ADDED]
        self.event_chain_t = edge_t[2]  # [ADDED]
        self.chain_t = edge_t[3]  # [ADDED]

        self.entity_delta_t = edge_delta_t[0]  # [ADDED]
        self.event_delta_t = edge_delta_t[1]  # [ADDED]
        self.event_chain_delta_t = edge_delta_t[2]  # [ADDED]
        self.chain_delta_t = edge_delta_t[3]  # [ADDED]

        self.entity_delta_s = edge_delta_s[0]  # [ADDED]
        self.event_delta_s = edge_delta_s[1]  # [ADDED]
        self.event_chain_delta_s = edge_delta_s[2]  # [ADDED]
        self.chain_delta_s = edge_delta_s[3]  # [ADDED]

        self.levels = [  # [ADDED]
            ("entity2event", self.entity_adj, self.entity_attr, self.entity_type, self.entity_t, self.entity_delta_t, self.entity_delta_s),  # [ADDED]
            ("event2event", self.event_adj, self.event_attr, self.event_type, self.event_t, self.event_delta_t, self.event_delta_s),  # [ADDED]
            ("event2chain", self.event_chain_adj, self.event_chain_attr, self.event_chain_type, self.event_chain_t, self.event_chain_delta_t, self.event_chain_delta_s),  # [ADDED]
            ("chain2chain", self.chain_adj, self.chain_attr, self.chain_type, self.chain_t, self.chain_delta_t, self.chain_delta_s),  # [ADDED]
        ]  # [ADDED]

        kwargs.pop("collate_fn", None)  # [ADDED]
        super().__init__(sample_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)  # [ADDED]

    def _build_sparse(self, edge_index: Tensor, edge_attr: Optional[Tensor]) -> Optional[SparseTensor]:  # [MODIFIED]
        if edge_index is None:  # [ADDED]
            return None  # [ADDED]
        value = torch.arange(edge_index.size(1))  # [MODIFIED]
        max_row = int(edge_index[0].max().item()) if edge_index.size(1) > 0 else -1  # [ADDED]
        max_col = int(edge_index[1].max().item()) if edge_index.size(1) > 0 else -1  # [ADDED]
        num_rows = max(self.total_nodes, max_row + 1)  # [ADDED]
        num_cols = max(self.total_nodes, max_col + 1)  # [ADDED]
        return SparseTensor(  # [MODIFIED]
            row=edge_index[0],  # [ADDED]
            col=edge_index[1],  # [ADDED]
            value=value,  # [ADDED]
            sparse_sizes=(num_rows, num_cols),  # [ADDED]
        ).t()  # [MODIFIED]

    def sample(self, batch: List[int]) -> "Batch":  # [ADDED]
        if not isinstance(batch, Tensor):  # [ADDED]
            batch = torch.tensor(batch, dtype=torch.long)  # [ADDED]
        sample_idx = batch  # [ADDED]
        n_id = self.node_idx[sample_idx]  # [ADDED]
        adjs_collected: List[dict] = []  # [MODIFIED-SAMPLER]

        for size, (name, adj, attr, typ, time, delta_t, delta_s) in zip(self.sizes[::-1], self.levels[::-1]):  # [ADDED]
            logging.info(
                "[NeighborSampler] Start sampling level=%s, target_nodes=%d, request_size=%d",
                name,
                n_id.size(0),
                size,
            )  # [ADDED-DEBUG]
            if adj is None:  # [ADDED]
                logging.info("[NeighborSampler] Level %s has no adjacency tensor; skip.", name)  # [ADDED-DEBUG]
                adjs_collected.append(
                    {
                        "name": name,
                        "adj": None,
                        "attr": None,
                        "type": None,
                        "time": None,
                        "delta_t": None,
                        "delta_s": None,
                    }
                )  # [MODIFIED-SAMPLER]
                continue  # [ADDED]
            sample_res = adj.sample_adj(n_id, size, replace=False)
            if len(sample_res) == 3:
                adj_t, n_id, edge_id = sample_res
            else:
                adj_t, n_id = sample_res
                edge_id = adj_t.storage.value() if adj_t is not None else None
            if adj_t is None or edge_id is None:  # [MODIFIED-SAMPLER]
                logging.info(
                    "[NeighborSampler] Level %s sampling returned empty subgraph (adj_t or edge_id None).",
                    name,
                )  # [ADDED-DEBUG]
                adjs_collected.append(
                    {
                        "name": name,
                        "adj": None,
                        "attr": None,
                        "type": None,
                        "time": None,
                        "delta_t": None,
                        "delta_s": None,
                    }
                )  # [MODIFIED-SAMPLER]
                continue  # [MODIFIED-SAMPLER]

            edge_attr = attr[edge_id] if attr is not None else None  # [MODIFIED-SAMPLER]
            edge_type = typ[edge_id] if typ is not None else None  # [MODIFIED-SAMPLER]
            edge_time = time[edge_id] if time is not None else None  # [MODIFIED-SAMPLER]
            edge_dt = delta_t[edge_id] if delta_t is not None else None  # [MODIFIED-SAMPLER]
            edge_ds = delta_s[edge_id] if delta_s is not None else None  # [MODIFIED-SAMPLER]
            edge_count = edge_id.numel() if isinstance(edge_id, Tensor) else len(edge_id)  # [ADDED-DEBUG]
            logging.info(
                "[NeighborSampler] Level %s sampled edges=%d, nodes=%s",
                name,
                edge_count,
                adj_t.sparse_sizes(),
            )  # [ADDED-DEBUG]
            adjs_collected.append(
                {
                    "name": name,
                    "adj": adj_t,
                    "attr": edge_attr,
                    "type": edge_type,
                    "time": edge_time,
                    "delta_t": edge_dt,
                    "delta_s": edge_ds,
                }
            )  # [MODIFIED-SAMPLER]

        adjs_collected.reverse()  # [ADDED]
        level_map = {item["name"]: idx for idx, item in enumerate(adjs_collected)}  # [MODIFIED-SAMPLER]

        adjs_t = [item["adj"] for item in adjs_collected]  # [MODIFIED-SAMPLER]
        edge_attrs = [item["attr"] for item in adjs_collected]  # [MODIFIED-SAMPLER]
        edge_ts = [item["time"] for item in adjs_collected]  # [MODIFIED-SAMPLER]
        edge_types = [item["type"] for item in adjs_collected]  # [MODIFIED-SAMPLER]
        edge_delta_ts = [item["delta_t"] for item in adjs_collected]  # [MODIFIED-SAMPLER]
        edge_delta_ss = [item["delta_s"] for item in adjs_collected]  # [MODIFIED-SAMPLER]

        batch_size = sample_idx.size(0)
        max_time = self.max_time[sample_idx] if self.max_time is not None else None

        def _filter_optional(seq, idx, mask):  # [MODIFIED-SAMPLER]
            if seq[idx] is not None:  # [MODIFIED-SAMPLER]
                seq[idx] = seq[idx][mask]  # [MODIFIED-SAMPLER]

        chain_event_map = {}  # [MODIFIED-SAMPLER]
        if "event2chain" in level_map:  # [MODIFIED-SAMPLER]
            idx_ec = level_map["event2chain"]  # [MODIFIED-SAMPLER]
            adj_ec = adjs_t[idx_ec]
        else:
            idx_ec = None  # [MODIFIED-SAMPLER]
            adj_ec = None  # [MODIFIED-SAMPLER]

        if adj_ec is not None and adj_ec.nnz() > 0:  # [MODIFIED-SAMPLER]
            logging.info(
                "[NeighborSampler] event2chain edges before time filter=%d",
                adj_ec.nnz(),
            )  # [ADDED-DEBUG]
            row_ec, col_ec, _ = adj_ec.coo()  # [MODIFIED-SAMPLER]
            keep_mask = torch.ones_like(row_ec, dtype=torch.bool)  # [MODIFIED-SAMPLER]
            if max_time is not None and edge_ts[idx_ec] is not None:  # [MODIFIED-SAMPLER]
                target_mask = row_ec < batch_size  # [MODIFIED-SAMPLER]
                if target_mask.any():  # [MODIFIED-SAMPLER]
                    allowed_time = max_time[row_ec[target_mask]]  # [MODIFIED-SAMPLER]
                    current_time = edge_ts[idx_ec][target_mask]  # [MODIFIED-SAMPLER]
                    keep_mask[target_mask] = current_time < allowed_time  # [MODIFIED-SAMPLER]
            if keep_mask.sum().item() == 0:  # [MODIFIED-SAMPLER]
                raise ValueError("[NeighborSampler] No event→chain edges remain after time filtering.")  # [MODIFIED-SAMPLER]
            if keep_mask.sum().item() != row_ec.numel():  # [MODIFIED-SAMPLER]
                adjs_t[idx_ec] = SparseTensor(  # [MODIFIED-SAMPLER]
                    row=row_ec[keep_mask],  # [MODIFIED-SAMPLER]
                    col=col_ec[keep_mask],  # [MODIFIED-SAMPLER]
                    sparse_sizes=adj_ec.sparse_sizes(),
                )  # [MODIFIED-SAMPLER]
                row_ec, col_ec, _ = adjs_t[idx_ec].coo()  # [MODIFIED-SAMPLER]
                _filter_optional(edge_attrs, idx_ec, keep_mask)  # [MODIFIED-SAMPLER]
                _filter_optional(edge_ts, idx_ec, keep_mask)  # [MODIFIED-SAMPLER]
                _filter_optional(edge_types, idx_ec, keep_mask)  # [MODIFIED-SAMPLER]
                _filter_optional(edge_delta_ts, idx_ec, keep_mask)  # [MODIFIED-SAMPLER]
                _filter_optional(edge_delta_ss, idx_ec, keep_mask)  # [MODIFIED-SAMPLER]
            for r, c in zip(row_ec.tolist(), col_ec.tolist()):  # [MODIFIED-SAMPLER]
                chain_event_map.setdefault(int(r), set()).add(int(c))  # [MODIFIED-SAMPLER]

        max_event_size = max((len(v) for v in chain_event_map.values()), default=1)  # [MODIFIED-SAMPLER]
        if idx_ec is not None and adjs_t[idx_ec] is not None:  # [ADDED-DEBUG]
            logging.info(
                "[NeighborSampler] event2chain edges after time filter=%d",
                adjs_t[idx_ec].nnz(),
            )  # [ADDED-DEBUG]

        # Time filter for event2event layer based on per-sample allowed_time  # [ADDED-TIME-FILTER]
        if "event2event" in level_map:  # [ADDED-TIME-FILTER]
            idx_ee = level_map["event2event"]  # [ADDED-TIME-FILTER]
            adj_ee = adjs_t[idx_ee]  # [ADDED-TIME-FILTER]
        else:
            idx_ee = None  # [ADDED-TIME-FILTER]
            adj_ee = None  # [ADDED-TIME-FILTER]

        if (
            adj_ee is not None
            and adj_ee.nnz() > 0
            and max_time is not None
            and edge_ts[idx_ee] is not None
        ):  # [ADDED-TIME-FILTER]
            logging.info("[NeighborSampler] event2event edges before time filter=%d", adj_ee.nnz())
            # Map target events (rows in event2event) that belong to original batch chains  # [ADDED-TIME-FILTER]
            # Build event->allowed_time via chain_event_map (rows < batch_size correspond to original chains)  # [ADDED-TIME-FILTER]
            event_allowed = {}  # type: dict  # [ADDED-TIME-FILTER]
            for r_chain, events in chain_event_map.items():  # [ADDED-TIME-FILTER]
                if r_chain < batch_size:  # only original batch chains  # [ADDED-TIME-FILTER]
                    allow = max_time[r_chain]  # [ADDED-TIME-FILTER]
                    for e_idx in events:  # [ADDED-TIME-FILTER]
                        # keep the earliest constraint if duplicated  # [ADDED-TIME-FILTER]
                        if e_idx not in event_allowed:
                            event_allowed[e_idx] = allow  # [ADDED-TIME-FILTER]
                        else:
                            event_allowed[e_idx] = torch.minimum(event_allowed[e_idx], allow)  # [ADDED-TIME-FILTER]

            row_ee, col_ee, _ = adj_ee.coo()  # [ADDED-TIME-FILTER]
            keep_mask_ee = torch.ones_like(row_ee, dtype=torch.bool)  # [ADDED-TIME-FILTER]
            if len(event_allowed) > 0:  # [ADDED-TIME-FILTER]
                # Build mask for edges whose target-event row has an allowed_time  # [ADDED-TIME-FILTER]
                apply_mask = torch.tensor(
                    [int(i.item()) in event_allowed for i in row_ee], dtype=torch.bool
                )  # [ADDED-TIME-FILTER]
                if apply_mask.any():  # [ADDED-TIME-FILTER]
                    # Gather per-edge allowed_time  # [ADDED-TIME-FILTER]
                    allowed_list = [
                        event_allowed[int(i.item())] for i in row_ee[apply_mask]
                    ]  # [ADDED-TIME-FILTER]
                    allowed_t = torch.stack(allowed_list, dim=0)  # [ADDED-TIME-FILTER]
                    current_t = edge_ts[idx_ee][apply_mask]  # [ADDED-TIME-FILTER]
                    keep_mask_ee[apply_mask] = current_t < allowed_t  # [ADDED-TIME-FILTER]

            if keep_mask_ee.sum().item() == 0:  # [ADDED-TIME-FILTER]
                raise ValueError(
                    "[NeighborSampler] No event→event edges remain after time filtering."
                )  # [ADDED-TIME-FILTER]
            if keep_mask_ee.sum().item() != row_ee.numel():  # [ADDED-TIME-FILTER]
                adjs_t[idx_ee] = SparseTensor(  # [ADDED-TIME-FILTER]
                    row=row_ee[keep_mask_ee],  # [ADDED-TIME-FILTER]
                    col=col_ee[keep_mask_ee],  # [ADDED-TIME-FILTER]
                    sparse_sizes=adj_ee.sparse_sizes(),  # [ADDED-TIME-FILTER]
                )  # [ADDED-TIME-FILTER]
                _filter_optional(edge_attrs, idx_ee, keep_mask_ee)  # [ADDED-TIME-FILTER]
                _filter_optional(edge_ts, idx_ee, keep_mask_ee)  # [ADDED-TIME-FILTER]
                _filter_optional(edge_types, idx_ee, keep_mask_ee)  # [ADDED-TIME-FILTER]
                _filter_optional(edge_delta_ts, idx_ee, keep_mask_ee)  # [ADDED-TIME-FILTER]
                _filter_optional(edge_delta_ss, idx_ee, keep_mask_ee)  # [ADDED-TIME-FILTER]
            logging.info("[NeighborSampler] event2event edges after time filter=%d", adjs_t[idx_ee].nnz())

        if "chain2chain" in level_map:  # [MODIFIED-SAMPLER]
            idx_cc = level_map["chain2chain"]  # [MODIFIED-SAMPLER]
            adj_cc = adjs_t[idx_cc]  # [MODIFIED-SAMPLER]
        else:
            idx_cc = None  # [MODIFIED-SAMPLER]
            adj_cc = None  # [MODIFIED-SAMPLER]

        if adj_cc is not None and adj_cc.nnz() > 0:  # [MODIFIED-SAMPLER]
            logging.debug(
                "[NeighborSampler] chain2chain edges before jaccard filter=%d",
                adj_cc.nnz(),
            )  # [ADDED-DEBUG]
            row_cc, col_cc, _ = adj_cc.coo()  # [MODIFIED-SAMPLER]
            # 基于时间的方向过滤：仅保留过去->现在（edge_delta_t > 0）的链间边
            try:
                dt_vals = edge_delta_ts[idx_cc]
                if dt_vals is not None and dt_vals.numel() == row_cc.numel():
                    if dt_vals.dim() == 2 and dt_vals.size(1) == 1:
                        dt_vals = dt_vals.view(-1)
                    keep_mask_cc = dt_vals > 0
                    if keep_mask_cc.any() and keep_mask_cc.sum().item() != row_cc.numel():
                        adjs_t[idx_cc] = SparseTensor(
                            row=row_cc[keep_mask_cc],
                            col=col_cc[keep_mask_cc],
                            sparse_sizes=adj_cc.sparse_sizes(),
                        )
                        _filter_optional(edge_attrs, idx_cc, keep_mask_cc)
                        _filter_optional(edge_ts, idx_cc, keep_mask_cc)
                        _filter_optional(edge_types, idx_cc, keep_mask_cc)
                        _filter_optional(edge_delta_ts, idx_cc, keep_mask_cc)
                        _filter_optional(edge_delta_ss, idx_cc, keep_mask_cc)
                        row_cc, col_cc, _ = adjs_t[idx_cc].coo()
            except Exception:
                pass
            keep_indices: List[int] = []  # [MODIFIED-SAMPLER]
            new_attrs: List[List[float]] = []  # [MODIFIED-SAMPLER]
            edge_types_cc = edge_types[idx_cc]  # [MODIFIED-SAMPLER]
            for idx_edge, (r_val, c_val) in enumerate(zip(row_cc.tolist(), col_cc.tolist())):  # [MODIFIED-SAMPLER]
                events_r = chain_event_map.get(int(r_val), set())  # [MODIFIED-SAMPLER]
                events_c = chain_event_map.get(int(c_val), set())  # [MODIFIED-SAMPLER]
                if not events_r and not events_c:  # [MODIFIED-SAMPLER]
                    jacc = 0.0  # [MODIFIED-SAMPLER]
                else:  # [MODIFIED-SAMPLER]
                    inter = len(events_r & events_c)  # [MODIFIED-SAMPLER]
                    union = len(events_r | events_c)  # [MODIFIED-SAMPLER]
                    jacc = inter / union if union > 0 else 0.0  # [MODIFIED-SAMPLER]
                src_size = len(events_r) / max_event_size if max_event_size else 0.0  # [MODIFIED-SAMPLER]
                tgt_size = len(events_c) / max_event_size if max_event_size else 0.0  # [MODIFIED-SAMPLER]
                # [DISABLED-JACCARD] 仍计算并记录 jacc，但不再根据阈值剔除边。  # [DISABLED-JACCARD]
                keep_indices.append(idx_edge)  # [DISABLED-JACCARD]
                new_attrs.append([src_size, tgt_size, jacc])  # [MODIFIED-SAMPLER]
            if not keep_indices:  # [MODIFIED-SAMPLER]
                raise ValueError("[NeighborSampler] All chain→chain edges were filtered by jaccard thresholds.")  # [MODIFIED-SAMPLER]
            keep_idx_tensor = torch.tensor(keep_indices, dtype=torch.long)  # [MODIFIED-SAMPLER]
            adjs_t[idx_cc] = SparseTensor(  # [MODIFIED-SAMPLER]
                row=row_cc[keep_idx_tensor],  # [MODIFIED-SAMPLER]
                col=col_cc[keep_idx_tensor],  # [MODIFIED-SAMPLER]
                sparse_sizes=adj_cc.sparse_sizes(),  # [MODIFIED-SAMPLER]
            )  # [MODIFIED-SAMPLER]
            edge_attrs[idx_cc] = torch.tensor(new_attrs, dtype=torch.float32)  # [MODIFIED-SAMPLER]
            _filter_optional(edge_ts, idx_cc, keep_idx_tensor)  # [MODIFIED-SAMPLER]
            _filter_optional(edge_types, idx_cc, keep_idx_tensor)  # [MODIFIED-SAMPLER]
            _filter_optional(edge_delta_ts, idx_cc, keep_idx_tensor)  # [MODIFIED-SAMPLER]
            _filter_optional(edge_delta_ss, idx_cc, keep_idx_tensor)  # [MODIFIED-SAMPLER]
            logging.info(
                "[NeighborSampler] chain2chain edges after jaccard filter=%d",
                adjs_t[idx_cc].nnz(),
            )  # [ADDED-DEBUG]

        return Batch(  # [ADDED]
            sample_idx=sample_idx,  # [ADDED]
            n_id=n_id,  # [ADDED]
            x=self.x[n_id],  # [ADDED]
            y=self.labels[sample_idx],  # [ADDED]
            candidates=self.candidates[sample_idx],  # [ADDED]
            adjs_t=adjs_t,  # [ADDED]
            edge_attrs=edge_attrs,  # [ADDED]
            edge_ts=edge_ts,  # [ADDED]
            edge_types=edge_types,  # [ADDED]
            edge_delta_ts=edge_delta_ts,  # [ADDED]
            edge_delta_ss=edge_delta_ss,  # [ADDED]
        )  # [ADDED]


class Batch(NamedTuple):  # [ADDED]
    sample_idx: Tensor  # [ADDED]
    n_id: Tensor  # [ADDED]
    x: Tensor  # [ADDED]
    y: Tensor  # [ADDED]
    candidates: Tensor  # [ADDED]
    adjs_t: List[Optional[SparseTensor]]  # [ADDED]
    edge_attrs: List[Optional[Tensor]]  # [ADDED]
    edge_ts: List[Optional[Tensor]]  # [ADDED]
    edge_types: List[Optional[Tensor]]  # [ADDED]
    edge_delta_ts: List[Optional[Tensor]]  # [ADDED]
    edge_delta_ss: List[Optional[Tensor]]  # [ADDED]

    def to(self, *args, **kwargs) -> "Batch":  # [ADDED]
        return Batch(  # [ADDED]
            sample_idx=self.sample_idx.to(*args, **kwargs),  # [ADDED]
            n_id=self.n_id.to(*args, **kwargs),  # [ADDED]
            x=self.x.to(*args, **kwargs),  # [ADDED]
            y=self.y.to(*args, **kwargs),  # [ADDED]
            candidates=self.candidates.to(*args, **kwargs),  # [ADDED]
            adjs_t=[adj.to(*args, **kwargs) if adj is not None else None for adj in self.adjs_t],  # [ADDED]
            edge_attrs=[attr.to(*args, **kwargs) if attr is not None else None for attr in self.edge_attrs],  # [ADDED]
            edge_ts=[ts.to(*args, **kwargs) if ts is not None else None for ts in self.edge_ts],  # [ADDED]
            edge_types=[typ.to(*args, **kwargs) if typ is not None else None for typ in self.edge_types],  # [ADDED]
            edge_delta_ts=[dt.to(*args, **kwargs) if dt is not None else None for dt in self.edge_delta_ts],  # [ADDED]
            edge_delta_ss=[ds.to(*args, **kwargs) if ds is not None else None for ds in self.edge_delta_ss],  # [ADDED]
        )  # [ADDED]
