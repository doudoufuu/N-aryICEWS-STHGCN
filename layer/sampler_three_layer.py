import torch
from torch import Tensor
from typing import List, Optional, NamedTuple
from collections import defaultdict
import random
import logging

class NeighborSampler(torch.utils.data.DataLoader):
    """
    Three-layer hypergraph sampler (Entity -> Event -> EventChain)
    with dynamic candidate set for tail event prediction.

    Args:
        x: feature matrices [entity_x, event_x, chain_x]  (保持与数据集一致，作为三层特征返回给模型)
        edge_index: list of adjacency edges [event2event, entity2event, (optional) chain2event]
                    —— 本版本只用于兼容签名，不在采样中使用
        edge_attr:  list of edge attributes (same order as edge_index)
                    —— 本版本只用于兼容签名，不在采样中使用
        edge_t:     list of edge timestamps [entity_time, event_time, chain_time]
                    —— 用到 event_time 来确定"最后的真实尾事件"
        sizes:      兼容参数（不使用）
        sample_idx: DataLoader 的样本 id（外部传入）
        node_idx:   查询的"事件链"ID（注意：可能是稠密索引 0..num_chain-1，也可能是稀疏原始ID）
        num_event:  事件总数（用于负采样池）
        event_chain_ids: Tensor[num_event]，每个事件所属"事件链"的 ID（可能是稠密或稀疏）
        num_candidates: 动态候选集大小（默认 5）
        **kwargs:   传给 DataLoader 的其他参数
    """

    def __init__(
        self,
        x: List[Tensor],
        edge_index: List[Tensor],
        edge_attr: List[Tensor],
        edge_t: List[Tensor],
        sizes: List[int],
        sample_idx: Tensor,
        node_idx: Tensor,
        num_event: int,
        event_chain_ids: Tensor,
        num_candidates: int = 5,  # [MODIFIED] 新增：候选集大小
        **kwargs,
    ):
        # ---- 三层特征（直接作为输出的一部分传回模型） ----
        self.entity_x = x[0].to("cpu")
        self.event_x = x[1].to("cpu")
        self.chain_x = x[2].to("cpu")
        # ---- 边相关字段 ----
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.edge_t = edge_t

        self.delta_ss = [torch.zeros_like(ea) for ea in self.edge_attr]
        self.edge_type = [torch.zeros(ea.size(0), dtype=torch.long) for ea in self.edge_attr]

        # ---- 时间：仅事件时间需要用于定位真实尾事件 ----
        self.entity_time = edge_t[0].to("cpu") if edge_t and len(edge_t) > 0 else None
        self.event_time = edge_t[1].to("cpu") if edge_t and len(edge_t) > 1 else None
        self.chain_time = edge_t[2].to("cpu") if edge_t and len(edge_t) > 2 else None

        # ---- 基本属性 ----
        self.sizes = sizes                      # 兼容参数，不使用
        self.sample_idx = sample_idx            # DataLoader 内部会迭代这个 index 列表
        self.node_idx = node_idx.to("cpu")      # 可能是稠密索引，也可能是稀疏原始ID
        self.num_event = int(num_event)
        self.num_candidates = int(num_candidates)

        # ---- 事件 -> 链 ID（可能是稀疏/原始 ID，或稠密索引） ----
        event_chain_ids = event_chain_ids.to("cpu").view(-1)

        # [MODIFIED] 构建"链 ID 稠密化"映射（把任意稀疏/原始 ID 映射成 0..C-1）
        unique_chain_keys = torch.unique(event_chain_ids)
        # 排序以保证可复现
        unique_chain_keys, _ = torch.sort(unique_chain_keys)
        self.chain_key_to_dense = {int(k): i for i, k in enumerate(unique_chain_keys.tolist())}
        self.chain_dense_to_key = unique_chain_keys  # Tensor[C]，dense -> raw key

        # 把每个事件的"链 ID"转换为稠密索引（0..C-1）
        self.event_chain_ids_dense = torch.tensor(
            [self.chain_key_to_dense[int(k)] for k in event_chain_ids.tolist()],
            dtype=torch.long,
        )

        # [MODIFIED] 判断 node_idx 当前是"稠密链索引"还是"稀疏原始链 ID"
        # 条件1：全部落在 [0, C-1] 范围内，视作稠密
        C = len(unique_chain_keys)
        node_vals = self.node_idx.unique().tolist()
        all_in_dense_range = all((0 <= int(v) < C) for v in node_vals)
        # 条件2：全部存在于 raw key 集合中，视作稀疏
        chain_key_set = set(k.item() for k in unique_chain_keys)
        all_in_raw_keys = all(int(v) in chain_key_set for v in node_vals)

        if all_in_dense_range and not all_in_raw_keys:
            self.node_idx_space = "dense"  # 0..C-1
        elif all_in_raw_keys and not all_in_dense_range:
            self.node_idx_space = "raw"    # 原始/稀疏 ID
        elif all_in_dense_range and all_in_raw_keys:
            # 既在范围内，也在原始集合内 —— 二者重合的少数情况，默认按 dense 处理
            self.node_idx_space = "dense"
        else:
            # 两边都不满足（说明 node_idx 里混了别的值），直接报错更安全
            raise ValueError(
                "[NeighborSampler] node_idx 既不是稠密链索引，也不在原始链 ID 集合中，请检查数据构建！"
            )

        # [MODIFIED] 构建"链(稠密) -> 事件列表"的映射（后续采样候选依赖它）
        self.chain2event_map_dense = defaultdict(list)
        for eid, c_dense in enumerate(self.event_chain_ids_dense.tolist()):
            self.chain2event_map_dense[c_dense].append(eid)

        # 构建实体-事件关系映射，用于三层采样
        self.entity2event_map = defaultdict(list)
        if self.edge_index and len(self.edge_index) > 1:  # 确保有entity2event边
            entity2event_edges = self.edge_index[1]
            for i in range(entity2event_edges.size(1)):
                entity = entity2event_edges[0, i].item()
                event = entity2event_edges[1, i].item()
                self.entity2event_map[entity].append(event)

        # 构建 event2event 关系映射，用于三层采样
        self.event2event_map = defaultdict(list)
        if self.edge_index and len(self.edge_index) > 0:  # 确保有event2event边
            event2event_edges = self.edge_index[0]
            for i in range(event2event_edges.size(1)):
                source_event = event2event_edges[0, i].item()
                target_event = event2event_edges[1, i].item()
                self.event2event_map[source_event].append(target_event)
                self.event2event_map[target_event].append(source_event)  # 无向图

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        super(NeighborSampler, self).__init__(
            sample_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs
        )

    # [MODIFIED] 工具：把输入的"链 ID（可能是 raw 或 dense）"统一成 dense 索引
    def _to_dense_chain_id(self, chain_id_val: int) -> Optional[int]:
        if self.node_idx_space == "dense":
            return int(chain_id_val)
        else:
            # raw -> dense
            return self.chain_key_to_dense.get(int(chain_id_val), None)

    # [MODIFIED] 三层图邻居采样逻辑
    def _sample_candidates_dense(self, chain_dense_id: int, k: int = 5):
        """采样候选事件，基于三层图结构

        Args:
            chain_dense_id: 稠密化的事件链ID
            k: 候选集大小

        Returns:
            candidates: 候选事件ID列表
            label: 真实尾事件在候选集中的位置
        """
        # 第一层采样：从事件链中获取所有事件
        events = self.chain2event_map_dense.get(chain_dense_id, [])
        if len(events) == 0:
            # 这里提示 raw key，便于你回溯源数据
            raw_key = int(self.chain_dense_to_key[chain_dense_id].item()) if (
                0 <= chain_dense_id < len(self.chain_dense_to_key)
            ) else chain_dense_id
            raise ValueError(f"[NeighborSampler] 事件链(稠密:{chain_dense_id}, 原始ID:{raw_key}) 没有事件！")

        # 第二层采样：获取与这些事件相关的实体
        entities = set()
        if self.edge_index and len(self.edge_index) > 1:  # 确保有entity2event边
            # 找到与这些事件相连的所有实体
            for event in events:
                if event in self.entity2event_map:
                    entities.update(self.entity2event_map[event])

        # 第三层采样：获取与这些实体相关的其他事件（扩展候选集）
        extended_events = set(events)  # 初始候选集包含链内所有事件

        # 通过实体关系扩展候选集
        if entities:
            for entity in entities:
                if entity in self.entity2event_map:
                    related_events = self.entity2event_map[entity]
                    extended_events.update(related_events)

        # 通过事件关系进一步扩展候选集
        for event in events:
            if event in self.event2event_map:
                related_events = self.event2event_map[event]
                extended_events.update(related_events)

        # 确保候选集不超过总事件数
        extended_events = list(extended_events)
        if len(extended_events) > self.num_event:
            extended_events = extended_events[:self.num_event]

        # 从扩展候选集中选择最终候选
        if len(extended_events) < k:
            # 如果扩展候选集不够，从剩余事件中随机补充
            remaining_events = list(set(range(self.num_event)) - set(extended_events))
            needed = k - len(extended_events)
            if needed > len(remaining_events):
                needed = len(remaining_events)
            if needed > 0:
                extended_events.extend(random.sample(remaining_events, needed))

        # 从最终候选集中随机选择k个事件
        final_candidates = random.sample(extended_events, min(k, len(extended_events)))

        # 确定真实尾事件：取时间最晚的事件
        times = self.event_time[final_candidates]
        true_tail_idx = torch.argmax(times).item()
        true_tail = final_candidates[true_tail_idx]

        # 打乱候选顺序，并记录真实尾事件的位置
        random.shuffle(final_candidates)
        label = final_candidates.index(true_tail)

        return torch.tensor(final_candidates, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def _sample_neighbors_three_layer(self, chain_dense_id: int, entity_sample_size: int = 10, event_sample_size: int = 10):
        """三层图邻居采样

        Args:
            chain_dense_id: 稠密化的事件链ID
            entity_sample_size: 每层采样的实体数量
            event_sample_size: 每层采样的事件数量

        Returns:
            entity_neighbors: 实体邻居ID列表
            event_neighbors: 事件邻居ID列表
            chain_neighbors: 链邻居ID列表
        """
        # 第一层：采样事件链内的实体
        events = self.chain2event_map_dense.get(chain_dense_id, [])
        entity_neighbors = set()

        if events and self.edge_index and len(self.edge_index) > 1:
            # 获取与这些事件相连的所有实体
            for event in events:
                if event in self.entity2event_map:
                    entities = self.entity2event_map[event]
                    entity_neighbors.update(entities)

            # 限制采样数量
            if len(entity_neighbors) > entity_sample_size:
                entity_neighbors = set(random.sample(list(entity_neighbors), entity_sample_size))

        # 第二层：采样与这些实体相关的事件
        event_neighbors = set(events)  # 初始包含链内所有事件

        if entity_neighbors:
            for entity in entity_neighbors:
                if entity in self.entity2event_map:
                    related_events = self.entity2event_map[entity]
                    event_neighbors.update(related_events)

            # 限制采样数量
            if len(event_neighbors) > event_sample_size:
                event_neighbors = set(random.sample(list(event_neighbors), event_sample_size))

        # 第三层：采样与这些事件相关的链
        chain_neighbors = set([chain_dense_id])  # 初始包含当前链

        if event_neighbors:
            for event_id in event_neighbors:
                # 找到该事件所属的链
                if 0 <= event_id < len(self.event_chain_ids_dense):
                    chain_id = self.event_chain_ids_dense[event_id].item()
                    chain_neighbors.add(chain_id)

            # 限制采样数量
            if len(chain_neighbors) > event_sample_size:
                chain_neighbors = set(random.sample(list(chain_neighbors), event_sample_size))

        return list(entity_neighbors), list(event_neighbors), list(chain_neighbors)

    def sample(self, batch):
        """
        DataLoader 的 collate_fn：
        输入：batch 中是 sample_idx 的一组整数
        输出：Batch，其中包含：
            - x: (entity_x, event_x, chain_x)  三层特征（原样返回）
            - candidates: [B, K]  每个样本的候选事件ID
            - labels:     [B]     候选中真实尾事件的下标（0..K-1）
        """
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        sample_idx = batch
        chain_ids_in = self.node_idx[sample_idx]  # 可能是 raw，也可能是 dense

        candidate_ids, labels = [], []
        for v in chain_ids_in.tolist():
            c_dense = self._to_dense_chain_id(v)
            if c_dense is None:
                raise ValueError(
                    f"[NeighborSampler] 在事件链映射中找不到链 ID {v}（当前 node_idx_space={self.node_idx_space}）。"
                )
            cand, lab = self._sample_candidates_dense(c_dense, k=self.num_candidates)
            candidate_ids.append(cand)
            labels.append(lab)

        candidate_ids = torch.stack(candidate_ids, dim=0)  # [B, K]
        labels = torch.stack(labels, dim=0)               # [B]

        result = Batch(
            sample_idx=sample_idx,
            x=(self.entity_x, self.event_x, self.chain_x),
            candidates=candidate_ids,
            labels=labels,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            delta_ts=self.edge_t,
            delta_ss=self.delta_ss,
            edge_type=self.edge_type,
        )

        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(num_candidates={self.num_candidates})"


class Batch(NamedTuple):
    sample_idx: Tensor
    x: tuple
    candidates: Tensor
    labels: Tensor
    edge_index: list = None
    edge_attr: list = None
    delta_ts: list = None
    delta_ss: list = None
    edge_type: list = None

    def to(self, *args, **kwargs):
        return Batch(
            sample_idx = self.sample_idx.to(*args, **kwargs),
            # ✅ 保持特征原 dtype，不要转 long
            x = tuple(xi.to(*args, **kwargs) for xi in self.x),
            # ✅ 索引/标签才转 long
            candidates = self.candidates.long().to(*args, **kwargs),
            labels = self.labels.long().to(*args, **kwargs),

            edge_index = [ei.long().to(*args, **kwargs) for ei in self.edge_index] if self.edge_index is not None else None,
            edge_attr  = [ea.to(*args, **kwargs) for ea in self.edge_attr] if self.edge_attr is not None else None,
            delta_ts   = [dt.to(*args, **kwargs) for dt in self.delta_ts] if self.delta_ts is not None else None,
            delta_ss   = [ds.to(*args, **kwargs) for ds in self.delta_ss] if self.delta_ss is not None else None,
            edge_type  = [et.long().to(*args, **kwargs) for et in self.edge_type] if self.edge_type is not None else None
        )
