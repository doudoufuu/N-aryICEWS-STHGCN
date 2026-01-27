# sampler4.py
import numpy as np
from typing import List, Optional, NamedTuple
from scipy.sparse import coo_matrix
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_mean, scatter_max, scatter_add, scatter_min
from utils import haversine  # 你已有的工具函数
import logging


def decode_global_id(gid: int):
    if gid < 199703:  # entity
        return "entity", gid
    elif gid < 199703*2:  # event
        return "event", gid - 199703
    else:  # chain
        return "chain", gid - 199703*2


class NeighborSampler4(torch.utils.data.DataLoader):
    """
    四层超图采样器（entity2event, event2event, event2chain, chain2chain）
    目标：为事件链尾事件预测任务准备小批次子图。
    采样顺序（在 sample() 内部）：
        chain2chain (multi-hop)
        event2event (multi-hop)
        entity2event (one-hop)
        event2chain  (one-hop)  <-- 最后做时间过滤 & chain 聚合（类似原 ci2traj）
    最后会反转 adjs 以便 convert_batch 以原版风格处理（第一层为用于生成 x_target 的 adjacency）。
    """

    def __init__(
        self,
        x: List[Tensor],                # [entity_x, event_x, chain_x]
        edge_index: List[Tensor],       # [entity2event_ei, event2event_ei, event2chain_ei, chain2chain_ei]
        edge_attr: List[Tensor],        # 对应的边特征（可为 None）
        edge_t: List[Optional[Tensor]], # 对应的边时间（可为 None），一般在 event2chain 层可为 None
        edge_delta_t: List[Optional[Tensor]],
        edge_delta_s: List[Optional[Tensor]],
        edge_type: List[Optional[Tensor]],
        sizes: List[int],               # 每层采样规模：len(sizes) == 4（顺序同上）
        sample_idx: Tensor,             # 索引数组（用于 DataLoader 的数据源）
        node_idx: Tensor,               # 查询节点（chain）在全图中的索引
        label: Tensor,                  # 标签（含候选信息，或另行传入 candidate_index）
        max_time: Tensor,               # 每个样本的时间阈值（用于时间过滤）
        num_nodes: Optional[int] = None,
        event_chain_ids: Optional[Tensor] = None,
        candidate_index: Optional[Tensor] = None,  # [N, K] 候选事件索引（若外部已构造）
        num_candidates: int = 5,
        intra_jaccard_threshold: float = 0.0,
        inter_jaccard_threshold: float = 0.0,
        **kwargs
    ):
        # ---- 层次输入拆分 ----
        entity_x = x[0]
        event_x  = x[1]
        chain_x  = x[2]

        # layer edge inputs 按顺序： entity2event, event2event, event2chain, chain2chain
        entity2event_ei = edge_index[0]
        event2event_ei = edge_index[1]
        event2chain_ei = edge_index[2]
        chain2chain_ei = edge_index[3]

        entity2event_attr = edge_attr[0]
        event2event_attr  = edge_attr[1]
        event2chain_attr  = edge_attr[2]
        chain2chain_attr  = edge_attr[3]

        # 时间/距离/类型（可为 None）
        entity_t = edge_t[0] if edge_t is not None else None
        event_t  = edge_t[1] if edge_t is not None else None
        chain_t  = edge_t[2] if edge_t is not None else None

        event2event_delta_t = edge_delta_t[1] if edge_delta_t is not None and len(edge_delta_t)>1 else None
        event2event_delta_s = edge_delta_s[1] if edge_delta_s is not None and len(edge_delta_s)>1 else None

        chain2chain_delta_t = edge_delta_t[3] if edge_delta_t is not None and len(edge_delta_t)>3 else None
        chain2chain_delta_s = edge_delta_s[3] if edge_delta_s is not None and len(edge_delta_s)>3 else None


        # entity2event_type = edge_type[0] if edge_type is not None else None
        # event2event_type  = edge_type[1] if edge_type is not None and len(edge_type)>1 else None
        # event2chain_type  = edge_type[2] if edge_type is not None and len(edge_type)>2 else None
        # chain2chain_type  = edge_type[3] if edge_type is not None and len(edge_type)>3 else None

        # ---- 移到 CPU（采样在 CPU 上进行） ----
        entity_x = entity_x.to('cpu')
        event_x  = event_x.to('cpu')
        chain_x  = chain_x.to('cpu')

        entity2event_ei = entity2event_ei.to('cpu')
        if entity2event_attr is not None:
            entity2event_attr = entity2event_attr.to('cpu')

        event2event_ei = event2event_ei.to('cpu')
        if event2event_attr is not None:
            event2event_attr = event2event_attr.to('cpu')
        if event2event_delta_t is not None:
            event2event_delta_t = event2event_delta_t.to('cpu')
        if event2event_delta_s is not None:
            event2event_delta_s = event2event_delta_s.to('cpu')

        event2chain_ei = event2chain_ei.to('cpu')
        if event2chain_attr is not None:
            event2chain_attr = event2chain_attr.to('cpu')

        chain2chain_ei = chain2chain_ei.to('cpu')
        if chain2chain_attr is not None:
            chain2chain_attr = chain2chain_attr.to('cpu')
        if chain2chain_delta_t is not None:
            chain2chain_delta_t = chain2chain_delta_t.to('cpu')
        if chain2chain_delta_s is not None:
            chain2chain_delta_s = chain2chain_delta_s.to('cpu')

        entity_t = entity_t.to('cpu') if entity_t is not None else None
        event_t  = event_t.to('cpu') if event_t is not None else None
        chain_t  = chain_t.to('cpu') if chain_t is not None else None
        
        # 删除多余的 collate_fn（以防外部传入）
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # 保存成员变量
        self.entity_x = entity_x
        self.event_x  = event_x
        self.chain_x  = chain_x

        self.entity2event_edge_index = entity2event_ei
        self.entity2event_edge_attr  = entity2event_attr
        self.entity_t     = entity_t
        #self.entity2event_edge_delta_t = entity2event_delta_t
        #self.entity2event_edge_delta_s = entity2event_delta_s
        #self.entity2event_edge_type = entity2event_type

        self.event2event_edge_index = event2event_ei
        self.event2event_edge_attr  = event2event_attr
        self.event2event_edge_delta_t = event2event_delta_t
        self.event2event_edge_delta_s = event2event_delta_s
        self.event_t = event_t
        #self.event2event_edge_type = event2event_type

        self.event2chain_edge_index = event2chain_ei
        self.event2chain_edge_attr  = event2chain_attr
        # self.event2chain_edge_t     = event2chain_t
        # self.event2chain_edge_type  = event2chain_type

        self.chain2chain_edge_index = chain2chain_ei
        self.chain2chain_edge_attr  = chain2chain_attr
        self.chain2chain_edge_delta_t = chain2chain_delta_t
        self.chain2chain_edge_delta_s = chain2chain_delta_s
        self.chain_t = chain_t
        # self.chain2chain_edge_type = chain2chain_type

        # 全部拼成一个大特征矩阵（entity, event, chain）
        self.x = torch.cat([entity_x, event_x, chain_x], dim=0)
        self.event_offset = entity_x.shape[0]
        self.chain_offset = entity_x.shape[0] + event_x.shape[0]

        # 便于后续使用：事件时间在 event_x 的列索引（在 event_x 内）
        # <-- 注意：如果你的 event_x 列顺序不同，请修改这里的偏移.
        self.event_time_pos_in_event = 4   # (基于 earlier: event_features = [Event_type, Intensity, lat, lon, UTCTimeOffsetEpoch, src, tgt])
        self.event_time_idx = self.event_offset + self.event_time_pos_in_event #有问题吧

        # 事件用于 overlap 的实体 id 列（例如 Target_name_encoded 在 event_x 中的位置）   #没看懂这三行想干嘛
        self.event_entity_id_pos_in_event = 6  # <-- 注意：如不一致请改
        self.event_entity_id_idx = self.event_offset + self.event_entity_id_pos_in_event

        # 其他保存
        self.max_event_size = self.event_x[:, 5].max().item() if self.event_x.size(0) > 0 else 1.0
        self.max_chain_size = self.chain_x[:, 0].max().item() if self.chain_x.size(0) > 0 else 1.0

        self.y = label
        self.node_idx = node_idx
        self.max_time = max_time
        self.sizes = sizes
        self.candidate_index = candidate_index
        self.num_candidates = num_candidates

        self.he2he_jaccard = None
        self.intra_jaccard_threshold = intra_jaccard_threshold
        self.inter_jaccard_threshold = inter_jaccard_threshold

        # 检查 node_idx 是否在 chain2chain_edge_index 范围内（粗检）
        if int(node_idx.max()) > chain2chain_ei.max():
            # 不一定致命，但提醒
            print("[NeighborSampler4] WARNING: node_idx.max() > chain2chain_edge_index.max()")

        # 推断 num_nodes（用于构造 SparseTensor），如果没给就自动推断
        if num_nodes is None:
            num_nodes = max(
                int(entity2event_ei.max()), int(event2event_ei.max()),
                int(event2chain_ei.max()), int(chain2chain_ei.max())
            ) + 1

        # 构造 SparseTensor（注意 transposed 以匹配 sample_adj 使用）
        self.chain2chain_adj_t = SparseTensor(
            row=chain2chain_ei[0],
            col=chain2chain_ei[1],
            value=torch.arange(chain2chain_ei.size(1)),
            sparse_sizes=(num_nodes, num_nodes)
        ).t()

        self.event2event_adj_t = SparseTensor(
            row=event2event_ei[0],
            col=event2event_ei[1],
            value=torch.arange(event2event_ei.size(1)),
            sparse_sizes=(num_nodes, num_nodes)
        ).t()

        self.entity2event_adj_t = SparseTensor(
            row=entity2event_ei[0],
            col=entity2event_ei[1],
            value=torch.arange(entity2event_ei.size(1)),
            sparse_sizes=(num_nodes, num_nodes)
        ).t()

        self.event2chain_adj_t = SparseTensor(
            row=event2chain_ei[0],
            col=event2chain_ei[1],
            value=torch.arange(event2chain_ei.size(1)),
            sparse_sizes=(num_nodes, num_nodes)
        ).t()

        # 预热 rowptr
        self.chain2chain_adj_t.storage.rowptr()
        self.event2event_adj_t.storage.rowptr()
        self.entity2event_adj_t.storage.rowptr()
        self.event2chain_adj_t.storage.rowptr()

        # DataLoader 初始化：把 sample_idx 列表当作 dataset
        super(NeighborSampler4, self).__init__(sample_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    # -----------------------------------------------
    # sample() -- 核心函数：对一个 batch 执行四层采样
    # -----------------------------------------------
    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        adjs = []
        sample_idx = batch
        n_id = self.node_idx[sample_idx]     # query chain ids (全图索引)
        max_time = self.max_time[sample_idx] # 时间阈（秒级 epoch 或者和你一致的单位）

        # 定义采样层顺序（从外到内采样），最后一层必须是 event2chain（用于时间过滤 & chain 聚合）
        layer_order = ['chain2chain', 'event2event', 'entity2event', 'event2chain']

        for i, size in enumerate(self.sizes):
            layer = layer_order[i]

            if layer == 'chain2chain':
                adj_t, n_id = self.chain2chain_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo() if adj_t is not None else (None, None, None)# coo() -> row, col, value (value 存了边编号 e_id)
                edge_attr = self.chain2chain_edge_attr[e_id] if (adj_t is not None and self.chain2chain_edge_attr is not None) else None
                
                src = n_id[row]   # 全局id
                dst = n_id[col]   # 全局id
                true_src = decode_global_id(src)   # 去掉偏移量
                edge_t = self.event_t[true_src]    
                #edge_t = self.chain_t[e_id] if (adj_t is not None and self.chain_t is not None) else None

                #edge_type = self.chain2chain_edge_type if self.chain2chain_edge_type is not None else None
                edge_delta_t = self.chain2chain_edge_delta_t if self.chain2chain_edge_delta_t is not None else None
                edge_delta_s = self.chain2chain_edge_delta_s if self.chain2chain_edge_delta_s is not None else None

            elif layer == 'event2event':
                adj_t, n_id = self.event2event_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo() if adj_t is not None else (None, None, None)
                edge_attr = self.event2event_edge_attr[e_id] if (adj_t is not None and self.event2event_edge_attr is not None) else None
                src = n_id[row]   # 全局id
                dst = n_id[col]   # 全局id
                true_src = decode_global_id(src)   # 去掉偏移量
                edge_t = self.event_t[true_src] 
                #edge_t = self.event_t[e_id] if (adj_t is not None and self.event_t is not None) else None
                #edge_type = self.event2event_edge_type if self.event2event_edge_type is not None else None
                edge_delta_t = self.event2event_edge_delta_t[e_id] if (adj_t is not None and self.event2event_edge_delta_t is not None) else None
                edge_delta_s = self.event2event_edge_delta_s[e_id] if (adj_t is not None and self.event2event_edge_delta_s is not None) else None

            elif layer == 'entity2event':
                adj_t, n_id = self.entity2event_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo() if adj_t is not None else (None, None, None)
                edge_attr = self.entity2event_edge_attr[e_id] if (adj_t is not None and self.entity2event_edge_attr is not None) else None
                
                src = n_id[row]   # 全局id
                dst = n_id[col]   # 全局id
                true_src = decode_global_id(src)   # 去掉偏移量
                edge_t = self.event_t[true_src] 
                #edge_t = self.entity_t[e_id] if (adj_t is not None and self.entity_t is not None) else None
                #edge_type = self.entity2event_edge_type if self.entity2event_edge_type is not None else None
                edge_delta_t = None
                edge_delta_s = None

            elif layer == 'event2chain':
                # 最后一个：从 chain -> event（或 event->chain 的转置）采样到事件列表（后续要做时间过滤）
                adj_t, n_id = self.event2chain_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo() if adj_t is not None else (None, None, None)
                # 注意：event2chain_edge_t 可能为空 —— 从 self.x（事件特征）里直接取时间
                edge_attr = self.event2chain_edge_attr[e_id] if (adj_t is not None and self.event2chain_edge_attr is not None) else None
                src = n_id[row]   # 全局id
                dst = n_id[col]   # 全局id
                true_src = decode_global_id(src)   # 去掉偏移量
                edge_t = self.event_t[true_src] 
                #edge_t = self.event_t[e_id] if (adj_t is not None and self.event_t is not None) else None
                #edge_type = self.event2chain_edge_type if self.event2chain_edge_type is not None else None
                edge_delta_t = None
                edge_delta_s = None

            else:
                raise ValueError(f"Unknown layer: {layer}")

            # size info
            if adj_t is not None:
                size_info = adj_t.sparse_sizes()[::-1]
            else:
                size_info = None

            if adj_t is not None and adj_t.nnz():
                # valid
                pass
            else:
                adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, e_id, size_info = None, None, None, None, None, None, None, size_info

            adjs.append((adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, e_id, size_info))

        # ---------- 对最后一层（event2chain）做时间过滤（保留在 target_time 之前的事件） ----------
        # 最后一个 adj 是我们刚 append 的最后一项（layer_order[-1]）
        last_adj = adjs[-1]
        adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, e_id, size_info = last_adj

        if adj_t is None:
            raise ValueError("[NeighborSampler4] event2chain sampled empty adjacency for all queries.")

        row, col, e_id = adj_t.coo()          # row: entity, col: event
        # 取出对应的事件时间（应该是 event 节点的时间）
        edge_times = self.x[n_id[col], self.event_time_idx]

        target_mask = row < batch_size
        edge_max_time = max_time[row[target_mask]] if target_mask.any() else torch.tensor([], dtype=edge_times.dtype)
        length = int(target_mask.sum().item())

        if length > 0:
            time_mask = edge_times[target_mask] <= edge_max_time
            target_mask[:length] = time_mask
        else:
            raise ValueError("[NeighborSampler4] No candidate events in front segment for batch!")

        if row[target_mask].size(0) == 0:
            raise ValueError("[NeighborSampler4] All chains have no event before target time!!")


        # # 重新构建过滤后的 event2chain adjacency（只有指向 batch 的前段被过滤）
        # adj_t_filtered = SparseTensor(
        #     row=row[target_mask],
        #     col=col[target_mask],
        #     sparse_sizes=(batch_size, adj_t.sparse_sizes()[1])
        # )
        # edge_times_filtered = edge_times[target_mask]
        # e_id_filtered = e_id[target_mask]

        # # 用 filtered 的 adj 替换 adjs 的最后一项（保留其它信息）
        # adjs[-1] = (adj_t_filtered, edge_attr, edge_times_filtered, None, None, None, e_id_filtered, size_info)

        # 重新构建过滤后的 event2chain adjacency（只保留满足 target_mask 的边）
        adj_t_filtered = SparseTensor(
            row=row[target_mask],
            col=col[target_mask],
            sparse_sizes=adj_t.sparse_sizes()   # 保持原图大小，避免 row/col 越界
        )

        edge_times_filtered = edge_times[target_mask]
        e_id_filtered = e_id[target_mask]
        edge_attr_filtered = edge_attr[target_mask] if edge_attr is not None else None
        size_info = adj_t_filtered.sparse_sizes()[::-1]

        # 用过滤后的 adj 替换 adjs 的最后一项
        adjs[-1] = (
            adj_t_filtered,
            edge_attr_filtered,
            edge_times_filtered,
            None, None, None,
            e_id_filtered,
            size_info
        )


        # ---------- 基于 filtered event2chain 构建 he2he_jaccard（用于后续 chain2chain / event2event 泄漏过滤） ----------
        # 我们把 event 的某个实体属性（例如 target entity id）作为“poi-like”索引来计算重合
        # he_entity = self.x[n_id[row[target_mask]], self.event_entity_id_idx]
        he_entity = self.x[n_id[row[target_mask]], self.event_entity_id_idx].numpy().astype(np.int64)

        im = coo_matrix((
            np.ones(row[target_mask].shape[0]),
            (he_entity, row[target_mask].numpy())
        )).tocsr()

        self.he2he_jaccard = (im.T * im).tocoo()

        # 计算 jaccard 值（value = overlap/(src+dst - overlap)）
        filtered_traj_size = self.he2he_jaccard.diagonal()
        source_size = filtered_traj_size[self.he2he_jaccard.col]
        target_size = filtered_traj_size[self.he2he_jaccard.row]
        # 避免 0 分母
        denom = (source_size + target_size - self.he2he_jaccard.data)
        denom[denom == 0] = 1.0
        self.he2he_jaccard.data = self.he2he_jaccard.data / denom

        # ---------- 使用 he2he_jaccard 或 edge_attr 过滤上层 multi-hop 边，防止信息泄漏 ----------
        # 按照 adjs 的顺序（chain2chain first, ... , event2chain last），我们需要过滤除最后一层外的其余层
        for idx, adj in enumerate(adjs[:-1]):
            # idx == 0 对应 chain2chain（使用 mode=1），其余（例如 event2event）使用 mode=2（基于预计算 edge_attr）
            if not idx:
                adjs[idx] = self.filter_traj2traj_with_leakage(adj, traj_size=filtered_traj_size, mode=1)
            else:
                adjs[idx] = self.filter_traj2traj_with_leakage(adj, traj_size=None, mode=2)

        # ---------- 检查过滤后：每个 query 都至少有一个 event neighbor（通过最后 adj 检查） ----------
        adj_t_final = adjs[-1][0]
        if adj_t_final is None or adj_t_final.storage.row().unique().shape[0] != batch_size:
            # 输出缺少邻居的样本 id 以便 debug
            if adj_t_final is None:
                raise ValueError("[NeighborSampler4] After filtering, last adj is empty for all samples.")
            diff_node = list(set(range(batch_size)) - set(adj_t_final.storage.row().unique().tolist()))
            raise ValueError(
                f'[NeighborSampler4] Chain without event neighbors after filtering by max_time is not allowed!!\n'
                f'Those samples are sample_idx:{sample_idx[diff_node]},\n'
            )

        # 反转 adjs（使 convert_batch 的第一个 adj 能被视作 'ci2traj' 类似的用于生成 x_target）
        adjs = adjs[::-1] if len(adjs) > 1 else adjs[0]

        out = (sample_idx, n_id, adjs)
        out = self.convert_batch(*out)
        return out

    # ------------------------------
    # filter_traj2traj_with_leakage（直接大致迁移原版逻辑）
    # ------------------------------
    def filter_traj2traj_with_leakage(self, adj, traj_size, mode=1):
        """
        adj: tuple (adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, e_id, size)
        mode=1: use self.he2he_jaccard (computed dynamically from filtered event2chain)
        mode=2: use provided edge_attr[:, 2] (precomputed jaccard in edge_attr)
        """
        adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, e_id, size = adj
        if adj_t is None:
            return adj

        row, col, value = adj_t.coo()
        if mode == 1:
            epsilon = 1e-6
            he2he = coo_matrix((
                np.ones(adj_t.nnz()) + epsilon,
                (row.numpy(), col.numpy())
            ))
            size_i = he2he.shape[0]
            size_j = he2he.shape[1]
            try:
                he2he = he2he - self.he2he_jaccard.tocsc()[:size_i, :size_j].tocoo()
            except Exception:
                # 如果 self.he2he_jaccard 的尺寸不匹配，直接跳过过滤（保守策略）
                logging.warning("[NeighborSampler4] he2he_jaccard size mismatch during filtering; skipping this filter.")
                return adj
            he2he = he2he.tocoo()

            valid_mask = he2he.data >= 0
            he2he = SparseTensor(
                row=torch.tensor(he2he.row[valid_mask], dtype=torch.long),
                col=torch.tensor(he2he.col[valid_mask], dtype=torch.long),
                value=torch.tensor(he2he.data[valid_mask])
            )

            if adj_t.nnz() != he2he.nnz():
                raise ValueError(f"[NeighborSampler4] he2he filtered size not equal.")

            inter_threshold_mask = he2he.storage.value() <= (1 - self.inter_jaccard_threshold + epsilon)
            intra_threshold_mask = he2he.storage.value() <= (1 - self.intra_jaccard_threshold + epsilon)
            inter_user_mask = (edge_type == 1) & inter_threshold_mask if edge_type is not None else inter_threshold_mask
            intra_user_mask = (edge_type == 0) & intra_threshold_mask if edge_type is not None else intra_threshold_mask
            mask = intra_user_mask | inter_user_mask
            keep_num = torch.sum(mask).item()
            if keep_num == 0:
                return (None, None, None, None, None, None, e_id, size)
            else:
                adj_t_new = SparseTensor(
                    row=row[mask],
                    col=col[mask],
                    value=he2he.storage.value()[mask],
                    sparse_sizes=adj_t.sparse_sizes()
                )
                edge_t_new = edge_t[mask] if edge_t is not None else None

                row2, col2, value2 = adj_t_new.coo()
                edge_attr_new_sim = (1 + epsilon) - value2
                source_traj_size = torch.tensor(traj_size[row2]) / self.max_event_size
                target_traj_size = torch.tensor(traj_size[col2]) / self.max_event_size
                edge_attr_new = torch.stack([source_traj_size, target_traj_size, edge_attr_new_sim], dim=1)
                return (adj_t_new, edge_attr_new, edge_t_new, edge_type[mask] if edge_type is not None else None,
                        edge_delta_t[mask] if edge_delta_t is not None else None,
                        edge_delta_s[mask] if edge_delta_s is not None else None,
                        e_id[mask] if e_id is not None else None, size)
        else:
            # mode == 2: use provided edge_attr[:,2] as jaccard
            if edge_attr is None:
                return adj
            inter_threshold_mask = edge_attr[:, 2] >= self.inter_jaccard_threshold
            intra_threshold_mask = edge_attr[:, 2] >= self.intra_jaccard_threshold
            inter_user_mask = (edge_type == 1) & inter_threshold_mask if edge_type is not None else inter_threshold_mask
            intra_user_mask = (edge_type == 0) & intra_threshold_mask if edge_type is not None else intra_threshold_mask
            mask = intra_user_mask | inter_user_mask
            keep_num = torch.sum(mask).item()
            if keep_num == 0:
                return (None, None, None, None, None, None, e_id, size)
            else:
                row, col, val = adj_t.coo()
                adj_t_new = SparseTensor(
                    row=row[mask],
                    col=col[mask],
                    value=val[mask],
                    sparse_sizes=adj_t.sparse_sizes()
                )
                return (adj_t_new, edge_attr[mask], edge_t[mask] if edge_t is not None else None,
                        edge_type[mask] if edge_type is not None else None,
                        edge_delta_t[mask] if edge_delta_t is not None else None,
                        edge_delta_s[mask] if edge_delta_s is not None else None,
                        e_id[mask] if e_id is not None else None, size)

    # ------------------------------
    # convert_batch: 把采样结果封装成 Batch（供模型使用）
    # ------------------------------
    def convert_batch(self, sample_idx, n_id, adjs):
        """
        把 sample 返回的 (sample_idx, n_id, adjs) 变成 Batch 结构。
        与原版类似：第一个 adj（adjs[0]）应是 event2chain 的过滤后版本（用于构建 x_target—chain 聚合特征）。
        """
        adjs_t, edge_attrs, edge_ts, edge_types, edge_delta_ts, edge_delta_ss = [], [], [], [], [], []
        y = self.y[sample_idx]  # 标签
        candidate_index = self.candidate_index[sample_idx] if self.candidate_index is not None else None

        x_target = None
        i = 0
        for adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, _, _ in adjs:
            if adj_t is None:
                adjs_t.append(None)
                edge_attrs.append(None)
                edge_ts.append(None)
                edge_types.append(None)
                edge_delta_ts.append(None)
                edge_delta_ss.append(None)
                i += 1
                continue

            # 注意：SparseTensor.coo() 在 torch_sparse 中返回 (col, row, value) 或 (row, col, value) ?
            # 这里我们使用 adj_t.coo() 与原版一致地去拿 col,row（如果你环境不同请确认顺序）
            col, row, _ = adj_t.coo()  # col: target index, row: source index (与原版 convert_batch 保持一致)

            if i == 0:
                # i==0: 期望是过滤后的 event2chain（source: events, target: chains）
                # 我们用它来计算 chain 的聚合特征 x_target（类似原版的 traj x_target）
                # source_event_lon_lat = self.x[n_id[row]][:, 2:4]  # 这里索引依你的 event_x 列结构调整
                source_event_time = self.x[n_id[row], self.event_time_idx]  # [#edges]
                # traj_min_time / max / mean
                traj_min_time, _ = scatter_min(source_event_time, col, dim=-1)
                traj_max_time, e_id_idx = scatter_max(source_event_time, col, dim=-1)
                traj_mean_time = scatter_mean(source_event_time, col, dim=-1)
                # 对于位置聚合：假设事件经度/纬度在 event_x 的位置 3/2（请根据你实际 event_x 调整）
                source_event_lon_lat = self.x[n_id[row], self.event_offset - self.event_offset + 2: self.event_offset - self.event_offset + 4] \
                    if False else self.x[n_id[row], 2:4]  # <-- 这里仅为占位；实际应用请改为 self.x[n_id[row], event_lon_idx/event_lat_idx]
                # 为了不让 placeholder 死掉，这里从 event_x 直接取 [lat, lon] 列：
                # 在前面我们假设 event_x: [Event_type, Intensity, lat, lon, time, src, tgt]
                source_event_lon_lat = self.x[n_id[row], self.event_offset - self.event_offset + 2: self.event_offset - self.event_offset + 4] if False else self.x[n_id[row], 2:4]
                # 但上面 slice 是不对的，在实际运行时请用：
                # source_event_lon_lat = self.x[n_id[row], self.event_offset + 2 : self.event_offset + 4]
                # 为避免运行错误，这里简单计算 traj_mean_lon_lat 为 zeros（请按需替换）
                traj_mean_lon_lat = torch.zeros((traj_min_time.size(0), 2), dtype=self.x.dtype)

                traj_size = scatter_add(torch.ones_like(source_event_time), col, dim=-1)
                edge_delta_t = traj_max_time[col] - source_event_time
                # edge_delta_s: 距离计算需要经纬度，这里留空（模型层会接收 edge_delta_ss)
                edge_delta_s = torch.zeros_like(edge_delta_t)

                x_target = torch.cat([
                    traj_size.unsqueeze(1),
                    traj_mean_lon_lat,
                    traj_mean_time.unsqueeze(1),
                    traj_min_time.unsqueeze(1),
                    traj_max_time.unsqueeze(1)
                ], dim=-1)
            elif i == len(adjs) - 1:
                # 最外围层（chain2chain），为其计算 edge_delta_t / edge_delta_s
                # target chain 聚合 mean_time 在 x_target[:,3]
                edge_delta_t = x_target[col][:, 3] - self.x[n_id[row], self.event_time_idx]  # 近似（需要根据真实列调整）
                # 对距离：拼接 self.x[n_id[row]][:, lon/lat] 与 x_target[col]'s lon/lat（这里演示，需替换为实际列索引）
                edge_delta_s = torch.zeros_like(edge_delta_t)
            else:
                # 中间层（event2event / entity2event）暂不额外聚合
                pass

            adjs_t.append(adj_t)
            edge_ts.append(edge_t)
            edge_attrs.append(edge_attr)
            edge_types.append(edge_type)
            edge_delta_ts.append(edge_delta_t)
            edge_delta_ss.append(edge_delta_s)
            i += 1

        # 把构造好的全部字段封装成 Batch
        result = Batch(
            sample_idx=sample_idx,
            x=self.x[n_id],
            x_target=x_target,
            y=y,
            adjs_t=adjs_t,
            edge_attrs=edge_attrs,
            edge_ts=edge_ts,
            edge_types=edge_types,
            edge_delta_ts=edge_delta_ts,
            edge_delta_ss=edge_delta_ss,
            candidate_index=candidate_index
        )
        return result

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)


class Batch(NamedTuple):
    sample_idx: Tensor
    x: Tensor
    x_target: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]
    edge_attrs: List[Tensor]
    edge_ts: List[Tensor]
    edge_types: List[Tensor]
    edge_delta_ts: List[Tensor]
    edge_delta_ss: List[Tensor]
    candidate_index: Optional[Tensor]

    def to(self, *args, **kwargs):
        return Batch(
            sample_idx=self.sample_idx.to(*args, **kwargs),
            x=self.x.to(*args, **kwargs),
            x_target=self.x_target.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) if adj_t is not None else None for adj_t in self.adjs_t],
            edge_attrs=[edge_attr.to(*args, **kwargs) if edge_attr is not None else None for edge_attr in self.edge_attrs],
            edge_ts=[edge_t.to(*args, **kwargs) if edge_t is not None else None for edge_t in self.edge_ts],
            edge_types=[edge_type.to(*args, **kwargs) if edge_type is not None else None for edge_type in self.edge_types],
            edge_delta_ts=[ed.to(*args, **kwargs) if ed is not None else None for ed in self.edge_delta_ts],
            edge_delta_ss=[ed.to(*args, **kwargs) if ed is not None else None for ed in self.edge_delta_ss],
            candidate_index=self.candidate_index.to(*args, **kwargs) if self.candidate_index is not None else None
        )
