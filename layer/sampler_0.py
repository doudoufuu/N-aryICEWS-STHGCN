import numpy as np
from typing import List, Optional, NamedTuple
from scipy.sparse import coo_matrix
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_mean, scatter_max, scatter_add, scatter_min
from utils import haversine
import logging

# ---------------------------
# NeighborSampler 类（核心）
# ---------------------------
class NeighborSampler(torch.utils.data.DataLoader):
    """
    这是一个基于 PyG/torch_sparse 的邻居采样器（继承自 DataLoader），
    专为“两个层次的超图”（traj2traj + ci2traj）做多跳动态采样并做泄漏过滤（time/jaccard）。
    核心方法是 __init__()（构建索引/预处理）和 sample()（collate_fn，用于 DataLoader 批量采样）。
    """

    # 构造函数：把所有输入（特征、边索引、边属性、时间、距离等）保存并构建稀疏邻接
    def __init__(
            self,
            x: List[Tensor],
            edge_index: List[Tensor],
            edge_attr: List[Tensor],

            edge_t: List[Tensor],
            edge_delta_t: List[Tensor],
            edge_type: List[Tensor],

            sizes: List[int],
            sample_idx: Tensor,
            node_idx: Tensor,
            label: Tensor,
            
            edge_delta_s: List[Tensor] = None,
            max_time: Optional[Tensor] = None,
            num_nodes: Optional[int] = None,
            intra_jaccard_threshold: float = 0.0,
            inter_jaccard_threshold: float = 0.0,
            **kwargs
    ):
        # ---------- 输入拆包（把传进来的 list/tuple 拆成 traj / ci 两部分） ----------
        traj_x = x[0]  # 轨迹级节点特征矩阵（每行是一个轨迹节点的特征）
        ci_x = x[1]    # 签到（checkin）节点特征矩阵（每行是一个签到点的特征）

        # traj2traj（轨迹-轨迹）相关的输入：边属性、索引、类型、时间差、空间差
        traj2traj_edge_attr = edge_attr[0]
        traj2traj_edge_index = edge_index[0]
        traj2traj_edge_type = edge_type[0]
        traj2traj_edge_delta_t = edge_delta_t[0]
        traj2traj_edge_delta_s = edge_delta_s[0]

        # ci2traj（签到->轨迹）相关的输入：边索引、签到时间、相应的时间差与空间差
        ci2traj_edge_index = edge_index[1]
        ci2traj_edge_t = edge_t[1]
        ci2traj_edge_delta_t = edge_delta_t[1]
        ci2traj_edge_delta_s = edge_delta_s[1]

        # ---------- 把数据搬到 CPU（采样在 CPU 上进行更方便/节省显存） ----------
        traj_x = traj_x.to('cpu')
        ci_x = ci_x.to('cpu')
        traj2traj_edge_attr = traj2traj_edge_attr.to('cpu')
        traj2traj_edge_index = traj2traj_edge_index.to('cpu')
        traj2traj_edge_type = traj2traj_edge_type.to('cpu')
        traj2traj_edge_delta_t = traj2traj_edge_delta_t.to('cpu')
        traj2traj_edge_delta_s = traj2traj_edge_delta_s.to('cpu')
        ci2traj_edge_index = ci2traj_edge_index.to('cpu')
        ci2traj_edge_t = ci2traj_edge_t.to('cpu')
        ci2traj_edge_delta_t = ci2traj_edge_delta_t.to('cpu')
        ci2traj_edge_delta_s = ci2traj_edge_delta_s.to('cpu')

        # 如果 kwargs 中有人传入 collate_fn（比如外层尝试传入），这里删除它：我们用 self.sample 做 collate_fn
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # ---------- 保存成员变量，后续 sample() 会使用这些数据 ----------
        self.traj_x = traj_x
        self.ci_x = ci_x
        # max_traj_size: 假设 traj_x 的第一列是轨迹长度（代码里用到这列），取最大做归一化基准
        self.max_traj_size = self.traj_x[:, 0].max().item()
        # 把 ci_x 放前面，traj_x 放后面，合并成一个大特征矩阵 self.x（用于 col/row 索引）
        self.x = torch.cat([ci_x, traj_x], dim=0)
        # ci_offset = ci_x 的行数，用来把“合并矩阵编码”的索引映射回原来的轨迹/签到空间
        self.ci_offset = ci_x.shape[0]

        # 保存边与时间/距离等信息
        self.traj2traj_edge_attr = traj2traj_edge_attr  
        self.traj2traj_edge_type = traj2traj_edge_type
        self.traj2traj_edge_delta_t = traj2traj_edge_delta_t
        self.traj2traj_edge_delta_s = traj2traj_edge_delta_s
        self.ci2traj_edge_t = ci2traj_edge_t
        self.ci2traj_edge_delta_t = ci2traj_edge_delta_t
        self.ci2traj_edge_delta_s = ci2traj_edge_delta_s

        # 标签、query 索引与采样配置
        self.y = label                 # 原始标签表（可按 sample_idx 索引）
        self.node_idx = node_idx       # query 节点在大图中的原始索引（例如轨迹 id）
        self.max_time = max_time       # 每个 query 的时间界（过滤签到用）
        self.sizes = sizes             # 每层采样的规模列表（最后一个是 ci2traj 的采样数）
        self.he2he_jaccard = None     # 稍后会保存轨迹->轨迹基于 poi 重叠的 jaccard 矩阵（scipy.sparse）
        self.intra_jaccard_threshold = intra_jaccard_threshold
        self.inter_jaccard_threshold = inter_jaccard_threshold

        # ---------- 构建 SparseTensor 索引（并转置以便用 sample_adj） ----------
        # 检查 node_idx 最大值是否在 traj2traj_edge_index 的范围内
        if int(node_idx.max()) > traj2traj_edge_index.max():
            raise ValueError('Query node index is not in graph.')
        # 如果外面没给 num_nodes，自动推断图中节点数（取边索引中的最大值 + 1）
        if num_nodes is None:
            num_nodes = max(int(traj2traj_edge_index.max()), int(ci2traj_edge_index.max())) + 1

        # 使用 torch_sparse.SparseTensor 构造稀疏邻接索引（值位置放边 id）
        self.traj2traj_adj_t = SparseTensor(
            row=traj2traj_edge_index[0],  # source
            col=traj2traj_edge_index[1],  # target
            value=torch.arange(traj2traj_edge_index.size(1)),  # 每条边的唯一编号
            sparse_sizes=(num_nodes, num_nodes)
        ).t()  # 注意转置：构造为转置形式，方便按行采样（sample_adj 使用转置存储）

        self.ci2traj_adj_t = SparseTensor(
            row=ci2traj_edge_index[0],
            col=ci2traj_edge_index[1],
            value=torch.arange(ci2traj_edge_index.size(1)),
            sparse_sizes=(num_nodes, num_nodes)
        ).t()

        # 预先构建 rowptr（内部数据结构），以便 sample_adj 的性能更好
        self.traj2traj_adj_t.storage.rowptr()
        self.ci2traj_adj_t.storage.rowptr()

        # 最后调用 DataLoader 的构造：把 sample_idx 转成 list 作为数据源，collate_fn 指定为 self.sample
        super(NeighborSampler, self).__init__(sample_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    # ---------- collate_fn（每次 DataLoader 组 batch 时会调用 sample） ----------
    def sample(self, batch):
        """
        这个函数是 DataLoader 的 collate_fn：输入是一个 batch（即 sample_idx 的一组 id），
        输出是一个封装好的 Batch 对象供模型使用。
        返回的 adjs 包含 traj2traj 多跳采样信息和一个 ci2traj 一跳信息（有时间/距离/edge_attr）。
        """
        # 保证 batch 是 Tensor（DataLoader 传进来的是 list）
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)  # 当前批大小
        adjs = []                     # 用来保存每一层采样得到的邻接信息
        sample_idx = batch            # 本批的样本索引（索引用于 y 等）
        n_id = self.node_idx[sample_idx]  # 根据 sample_idx 得到 query 的真实节点 id（在 self.x 的 index 上）
        max_time = self.max_time[sample_idx]  # 每个 query 的过滤时间阈值

        # ---------- 逐层采样（traj2traj 的多跳 + ci2traj 的一跳） ----------
        for i, size in enumerate(self.sizes):
            # sizes 最后一个对应 ci2traj（签到->轨迹），其余为 traj2traj 的多跳
            if i == len(self.sizes) - 1:
                # 对 ci2traj 做一跳采样： sample_adj 返回 (adj_t, n_id)
                adj_t, n_id = self.ci2traj_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()      # coo() -> row, col, value (value 存了边编号 e_id)
                edge_attr = None                  # ci2traj 没有 traj2traj 形式的 edge_attr
                edge_t = self.ci2traj_edge_t[e_id]            # 每条边对应的 checkin 时间
                edge_type = None
                edge_delta_t = self.ci2traj_edge_delta_t[e_id]
                edge_delta_s = self.ci2traj_edge_delta_s[e_id]
            else:
                # 对 traj2traj 做多跳邻居采样
                adj_t, n_id = self.traj2traj_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                # 这里取到的是与边编号 e_id 对应的预先计算好的边属性（如 jaccard, src_size, tgt_size）
                edge_attr = self.traj2traj_edge_attr[e_id]
                edge_t = None
                edge_type = self.traj2traj_edge_type[e_id]
                edge_delta_t = self.traj2traj_edge_delta_t[e_id]
                edge_delta_s = self.traj2traj_edge_delta_s[e_id]

            # adj_t.sparse_sizes()[::-1]：得到 (num_src, num_dst) 之类的尺寸信息并反转
            size = adj_t.sparse_sizes()[::-1]

            # 如果子图非空，校验索引合法性（source 的最大 idx 不超过 sparse size）
            if adj_t.nnz():
                assert size[0] >= col.max() + 1, '[NeighborSampler] adj_t source max index exceed sparse_sizes[1]'
            else:
                # 若子图为空，则把所有与之有关的属性设为 None
                adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s = None, None, None, None, None, None

            # 将这一层的采样结果 append（包含 adj_t, edge_attr, 时间/距离/类型, e_id, size）
            adjs.append((adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, e_id, size))

        # ---------- 对 ci2traj 部分做时间过滤（仅保留在目标时间之前的签到） ----------
        # 'row' here refers to the source indices of the last sampled adj_t (ci2traj)
        target_mask = row < batch_size               # 只保留指向“当前 batch 的目标轨迹”的边
        edge_max_time = max_time[row[target_mask]]   # 对应这些边的目标时间阈
        length = torch.sum(target_mask)              # 有效边的数量（位于前段的）
        time_mask = edge_t[target_mask] <= edge_max_time  # 只保留签到时间在目标时间之前的边
        target_mask[:length] = time_mask             # 把前 length 条边的 mask 替换为 time_mask

        # 若被过滤后某条轨迹前面没有任何满足条件的 checkin，抛错（不能没有候选签到）
        if row[target_mask].size(0) == 0:
            raise ValueError(
                f'[NeighborSampler] All trajectories have no checkin before target time!!'
            )

        # 重建一个过滤后的 SparseTensor（只包含满足 target_mask 的边）
        adj_t = SparseTensor(
           row=row[target_mask],
           col=col[target_mask],
           sparse_sizes=(batch_size, adj_t.sparse_sizes()[1])
        )
        edge_t = edge_t[target_mask]
        edge_type = None
        edge_delta_t = edge_delta_t[target_mask]
        edge_delta_s = edge_delta_s[target_mask]
        e_id = e_id[target_mask]
        # 把过滤后的 ci2traj 邻接信息追加到 adjs
        adjs.append((adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, e_id, size))

        # ---------- 用过滤后的 ci2traj 构建轨迹-轨迹上的 Jaccard（用于泄漏过滤） ----------
        # target_mask[length:] = True 将 traj2traj 部分全部设置为 True（因为只对 ci2traj 的前部分做 mask）
        target_mask[length:] = True
        # 取出被选中的源 checkin 的 poi id（self.ci_x 的第2列是 poi id，注意索引）
        he_poi = self.ci_x[col[target_mask]][:, 1]
        # 构建稀疏矩阵 im：行 = poi_id, 列 = edge(row)（将 checkin -> traj 的 incidence 转为 poi vs traj）
        im = coo_matrix((
            np.ones(row[target_mask].shape[0]),
            (he_poi.numpy().astype(np.long), row[target_mask].numpy())
        )).tocsr()
        # he2he_jaccard = im.T * im -> 得到轨迹 x 轨迹 的 POI 重合计数（COO 格式）
        self.he2he_jaccard = im.T * im
        self.he2he_jaccard = self.he2he_jaccard.tocoo()

        # ---------- 把重合计数转成 Jaccard 相似度（value/(src+dst - overlap)） ----------
        filtered_traj_size = self.he2he_jaccard.diagonal()  # 每个轨迹包含的 POI 数（对角线）
        source_size = filtered_traj_size[self.he2he_jaccard.col]
        target_size = filtered_traj_size[self.he2he_jaccard.row]
        # 更新 he2he_jaccard.data 为实际的 jaccard 值
        self.he2he_jaccard.data = self.he2he_jaccard.data / (source_size + target_size - self.he2he_jaccard.data)

        # ---------- 使用 he2he_jaccard 或 edge_attr 过滤 traj2traj 邻边（防止信息泄漏） ----------
        # 遍历 adjs[:-2]：注意 adjs 包含了多层（如 [traj2traj_hopK..., ci2traj, filtered_ci2traj]）
        for i, adj in enumerate(adjs[:-2]):
            if not i:
                # 第一层（i==0）使用 mode=1（基于 he2he_jaccard）
                adjs[i] = self.filter_traj2traj_with_leakage(adj, traj_size=filtered_traj_size, mode=1)
            else:
                # 其他层（i>0）使用 mode=2（基于原始 edge_attr 的第三列）
                adjs[i] = self.filter_traj2traj_with_leakage(adj, traj_size=None, mode=2)

        # ---------- 检查每个 target trajectory 是否至少有一个 checkin neighbor（过滤后） ----------
        if adj_t.storage.row().unique().shape[0] != batch_size:
            # 找出没有邻居的样本并报错（这种样本不允许出现）
            diff_node = list(set(range(batch_size)) - set(adj_t.storage.row().unique().tolist()))
            raise ValueError(
                f'[NeighborSampler] Trajectory without checkin neighbors after filtering by max_time is not allowed!!\n'
                f'Those samples are sample_idx:{sample_idx[diff_node]},\n'
                f'and the corresponding query trajectories are: {n_id[diff_node]},\n'
                f'the original query trajectories are: {n_id[diff_node] - self.ci_offset}.'
            )

        # 如果只有一层采样（不常见），直接返回 adjs[0]，否则反转 adjs（把最后一层放最前面，便于模型逐层消费）
        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (sample_idx, n_id, adjs)
        # 把元组转换成 Batch 对象（结构化），便于上层模型直接取字段
        out = self.convert_batch(*out)
        return out

    # ---------- 辅助函数：基于 jaccard 或 edge_attr 做 traj2traj 的泄漏过滤 ----------
    def filter_traj2traj_with_leakage(self, adj, traj_size, mode=1):
        """The original traj2traj topology is in adj_t, we set the value to all ones, and
        then we substitute it with traj2traj_jaccard, and keep the data within [0, 1].

        mode=1: use self.he2he_jaccard (computed from ci2traj filtered)
        mode=2: use provided edge_attr[:, 2] (precomputed jaccard in edge_attr)
        """
        adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, e_id, size = adj

        # 如果子图为空，直接返回
        if adj_t is None:
            return adj

        row, col, value = adj_t.coo()  # 取得行列和值（value 最初是边的索引或 id）
        if mode == 1:
            # mode 1：利用 he2he_jaccard 来替换 adj 的 value（并做一些校正）
            epsilon = 1e-6  # 防止完全重叠的轨迹被负值过滤掉
            # he2he 是当前 adj 上原始重叠计数加上 epsilon（以免除 0）
            he2he = coo_matrix((
                np.ones(adj_t.nnz()) + epsilon,
                (row.numpy(), col.numpy())
            ))
            size_i = he2he.shape[0]
            size_j = he2he.shape[1]
            # 从 he2he 中减去全局计算得到的 he2he_jaccard（只取对应尺寸子块）
            he2he = he2he - self.he2he_jaccard.tocsc()[:size_i, :size_j].tocoo()
            he2he = he2he.tocoo()

            # 只保留非负的数据（valid within [0,1]）
            valid_mask = he2he.data >= 0
            he2he = SparseTensor(
                row=torch.tensor(he2he.row[valid_mask], dtype=torch.long),
                col=torch.tensor(he2he.col[valid_mask], dtype=torch.long),
                value=torch.tensor(he2he.data[valid_mask])
            )

            # 校验点：过滤前后非零元数量应该一致，否则抛错
            if adj_t.nnz() != he2he.nnz():
                raise ValueError(f"[NeighborSampler] he2he filtered size not equal.")

            # 基于 jaccard 与 edge_type (intra/inter user) 做阈值过滤：
            inter_threshold_mask = he2he.storage.value() <= (1 - self.inter_jaccard_threshold + epsilon)
            intra_threshold_mask = he2he.storage.value() <= (1 - self.intra_jaccard_threshold + epsilon)
            inter_user_mask = (edge_type == 1) & inter_threshold_mask
            intra_user_mask = (edge_type == 0) & intra_threshold_mask
            mask = intra_user_mask | inter_user_mask  # 保留满足阈值的边
            keep_num = torch.sum(mask).item()
            if keep_num == 0:
                # 若全部被过滤，则返回一个空的 adj（但保留 e_id/size 供上游检查）
                adj = (None, None, None, None, None, None, e_id, size)
                return adj
            else:
                # 构建新的 adj_t（只保留 mask 为 True 的边），并把 value 替换为 he2he 的值
                adj_t = SparseTensor(
                    row=row[mask],
                    col=col[mask],
                    value=he2he.storage.value()[mask],
                    sparse_sizes=adj_t.sparse_sizes()
                )
                edge_t = edge_t[mask] if edge_t is not None else None

                # 恢复 edge_attr（把 similarity 变换为 edge_attr 的第三项）
                row, col, value = adj_t.coo()
                edge_attr = (1 + epsilon) - value  # 恢复成某种相似性度量
                # 计算 source/target 的归一化轨迹长度作为 edge_attr 的前两项
                source_traj_size = torch.tensor(traj_size[row]) / self.max_traj_size
                target_traj_size = torch.tensor(traj_size[col]) / self.max_traj_size
                edge_attr = torch.stack([source_traj_size, target_traj_size, edge_attr], dim=1)
        else:
            # mode == 2：直接在原始 edge_attr 上用阈值过滤（使用 edge_attr[:,2] 作为 jaccard）
            inter_threshold_mask = edge_attr[:, 2] >= self.inter_jaccard_threshold
            intra_threshold_mask = edge_attr[:, 2] >= self.intra_jaccard_threshold
            inter_user_mask = (edge_type == 1) & inter_threshold_mask
            intra_user_mask = (edge_type == 0) & intra_threshold_mask
            mask = intra_user_mask | inter_user_mask
            keep_num = torch.sum(mask).item()
            if keep_num == 0:
                adj = (None, None, None, None, None, None, e_id, size)
                return adj
            else:
                # 直接筛选 edge_attr & adj_t（value 保持原有）
                edge_attr = edge_attr[mask]
                adj_t = SparseTensor(
                    row=row[mask],
                    col=col[mask],
                    value=value[mask],
                    sparse_sizes=adj_t.sparse_sizes()
                )

        # 最终根据 mask 截取 edge_type / edge_delta_t / edge_delta_s / e_id
        edge_type = edge_type[mask]
        edge_delta_t = edge_delta_t[mask]
        edge_delta_s = edge_delta_s[mask]
        e_id = e_id[mask]
        adj = (adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, e_id, size)
        return adj

    # ---------- 把采样结果结构化为 Batch（供模型直接使用） ----------
    def convert_batch(self, sample_idx, n_id, adjs):
        """
        这个函数把 sample() 返回的元组转成一个 Batch NamedTuple，计算并补全
        target trajectory 的统计特征（如 size, mean_lon, mean_lat, mean_time, min/max time 等）。
        """
        adjs_t, edge_attrs, edge_ts, edge_types, edge_delta_ts, edge_delta_ss = [], [], [], [], [], []
        y = self.y[sample_idx]  # 根据 sample_idx 取标签表中的对应行

        x_target = None  # 用于存放目标轨迹的特征（会被构造）

        # 说明：self.x 的列分别对应检查点特征与轨迹特征——文档里指出：
        # checkin_feature 'user_id', 'poi_id', 'poi_cat', 'time', 'poi_lon', 'poi_lat'
        # trajectory_feature 'size', 'mean_lon', 'mean_lat', 'mean_time', 'start_time', 'end_time'
        i = 0
        for adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, _, _ in adjs:

            if adj_t is None:
                pass
            else:
                # adj_t.coo() 返回 (col, row, value) — 注意 torch_sparse 的 coo 顺序与 scipy 可能有差异
                col, row, _ = adj_t.coo()
                if not i:
                    # i == 0 表示第一个元素，通常是 ci2traj（已被过滤），我们用它来构建 x_target（目标轨迹特征）
                    # source_checkin_lon_lat: 取 self.x 中对应 row 的 checkin 的经纬度（字段在列 4:6）
                    source_checkin_lon_lat = self.x[n_id[row]][:, 4:6]  # shape [#edge, 2]
                    traj_min_time, _ = scatter_min(edge_t, col, dim=-1)   # 每个轨迹的最早签到时间
                    traj_max_time, e_id = scatter_max(edge_t, col, dim=-1)  # 每个轨迹的最晚签到时间及其对应 edge 下标
                    traj_mean_time = scatter_mean(edge_t, col, dim=-1)    # 每个轨迹的平均签到时间
                    traj_last_lon_lat = source_checkin_lon_lat[e_id]     # 最后一次签到的经纬度 [N, 2]
                    traj_mean_lon_lat = scatter_mean(source_checkin_lon_lat, col, dim=0)  # 平均经纬度 [N,2]
                    traj_size = scatter_add(torch.ones_like(edge_t), col, dim=-1)  # 每个轨迹的签到数（长度）

                    # 计算每条边相对于所属轨迹最后一次签到的时间差
                    edge_delta_t = traj_max_time[col] - edge_t
                    # 计算每条边与所属轨迹最后签到点之间的地理距离（haversine）
                    edge_delta_s = torch.cat([traj_last_lon_lat[col], source_checkin_lon_lat], dim=-1)
                    edge_delta_s = haversine(
                        edge_delta_s[:, 0],
                        edge_delta_s[:, 1],
                        edge_delta_s[:, 2],
                        edge_delta_s[:, 3]
                    )
                    # x_target 为目标轨迹的汇总特征向量： size, mean_lon, mean_lat, mean_time, min_time, max_time
                    x_target = torch.cat([
                        traj_size.unsqueeze(1),
                        traj_mean_lon_lat,
                        traj_mean_time.unsqueeze(1),
                        traj_min_time.unsqueeze(1),
                        traj_max_time.unsqueeze(1)
                    ], dim=-1)
                elif i == len(adjs) - 1:
                    # 当遍历到最后一个 adj（通常是 traj2traj 的最外层），需要计算 traj2traj 的 edge_delta_t / edge_delta_s
                    # edge_delta_t: 目标轨迹的 mean_time - 邻居轨迹的 time
                    edge_delta_t = x_target[col][:, 3] - self.x[n_id[row]][:, 3]
                    # edge_delta_s: 把两个轨迹的 lon/lat 拼起来算距离
                    edge_delta_s = torch.cat([self.x[n_id[row]][:, 1:3], x_target[col][:, 1:3]], dim=-1)
                    edge_delta_s = haversine(
                        edge_delta_s[:, 0],
                        edge_delta_s[:, 1],
                        edge_delta_s[:, 2],
                        edge_delta_s[:, 3]
                    )
                else:
                    # 中间层暂时没有额外的聚合逻辑（保留空白以备扩展）
                    pass

            # 把这一层的 adj_t、edge_t、edge_attr、edge_delta 等 append 到列表
            adjs_t.append(adj_t)
            edge_ts.append(edge_t)
            edge_attrs.append(edge_attr)
            edge_types.append(edge_type)
            edge_delta_ts.append(edge_delta_t)
            edge_delta_ss.append(edge_delta_s)
            i += 1

        # 最后把构造好的所有字段封装到 Batch NamedTuple 中并返回
        result = Batch(
            sample_idx=sample_idx,
            x=self.x[n_id],       # 把被采样到的节点特征（按 n_id 顺序）作为 x
            x_target=x_target,    # 目标轨迹特征矩阵
            y=y,
            adjs_t=adjs_t,
            edge_attrs=edge_attrs,
            edge_ts=edge_ts,
            edge_types=edge_types,
            edge_delta_ts=edge_delta_ts,
            edge_delta_ss=edge_delta_ss
        )
        return result

    # 简洁的字符串表示
    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)

# ---------- Batch NamedTuple：定义采样器返回的结构（并带 .to 方法便于搬到 GPU） ----------
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

    # to() 方法把内部所有 Tensor / SparseTensor 转移到指定 device（例如 .to(device)）
    def to(self, *args, **kwargs):
        return Batch(
            sample_idx=self.sample_idx.to(*args, **kwargs),
            x=self.x.to(*args, **kwargs),
            x_target=self.x_target.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) if adj_t is not None else None for adj_t in self.adjs_t],
            edge_attrs=[
                edge_attr.to(*args, **kwargs)
                if edge_attr is not None
                else None
                for edge_attr in self.edge_attrs
            ],
            edge_ts=[
                edge_t.to(*args, **kwargs)
                if edge_t is not None
                else None
                for edge_t in self.edge_ts
            ],
            edge_types=[
                edge_type.to(*args, **kwargs)
                if edge_type is not None
                else None
                for edge_type in self.edge_types
            ],
            edge_delta_ts=[
                edge_delta_t.to(*args, **kwargs)
                if edge_delta_t is not None
                else None
                for edge_delta_t in self.edge_delta_ts
            ],
            edge_delta_ss=[
                edge_delta_s.to(*args, **kwargs)
                if edge_delta_s is not None
                else None
                for edge_delta_s in self.edge_delta_ss
            ]
        )
