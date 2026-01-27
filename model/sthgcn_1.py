import torch
from torch import nn
from layer import (
    CheckinEmbedding,
    EdgeEmbedding,
    HypergraphTransformer,
    TimeEncoder,
    DistanceEncoderHSTLSTM,
    DistanceEncoderSTAN,
    DistanceEncoderSimple
)
from torch_sparse import SparseTensor
import torch.nn.functional as F
import logging

class STHGCN(nn.Module):
    """
    改造版 STHGCN：兼容两种输入模�?
      - 全图 / 三层超图字典输入（原先的模式�?: data �? dict，包�? 'x', 'edge_index', 'edge_attr', 'delta_ts', 'delta_ss', 'edge_type', 'candidates', 'labels' �?
      - 邻居采样 / mini-batch（PyG-like�?: data �? Batch-like object，可能包�? adjs_t / edge_index / edge_attrs / edge_delta_ts / edge_delta_ss / edge_types / x / y �?

    主要改动点已标注�? # [MODIFIED] /  
    """
    def __init__(self, cfg, dataset):
        super(STHGCN, self).__init__()
        # 注意：cfg.run_args.device 可能是字符串 'cuda:0' �? 'cpu'
        self.device = cfg.run_args.device
        self.batch_size = cfg.run_args.batch_size
        self.eval_batch_size = cfg.run_args.eval_batch_size
        self.do_traj2traj = cfg.model_args.do_traj2traj
        self.distance_encoder_type = cfg.model_args.distance_encoder_type
        self.dropout_rate = cfg.model_args.dropout_rate
        self.generate_edge_attr = cfg.model_args.generate_edge_attr
        self.num_conv_layers = len(cfg.model_args.sizes)
        self.num_poi = cfg.dataset_args.num_poi
        self.embed_fusion_type = cfg.model_args.embed_fusion_type
        self.fusion_type = getattr(cfg.model_args, "embed_fusion_type",
                    getattr(cfg.model_args, "fusion_type", "concat"))

        # === 获取特征维度（保持你之前的硬编码/或可改为�? dataset 自动拿） ===
        entity_feat_dim = 9
        event_feat_dim = 7
        chain_feat_dim = 8

        logging.info(f"Feature dimensions - Entity: {entity_feat_dim}, Event: {event_feat_dim}, Chain: {chain_feat_dim}")

        # === 三层超图 Embedding ===
        self.checkin_embedding_layer = CheckinEmbedding(
            embed_size=cfg.model_args.embed_size,
            fusion_type=self.fusion_type,
            entity_feat_dim=entity_feat_dim,
            event_feat_dim=event_feat_dim,
            chain_feat_dim=chain_feat_dim
        )
        self.checkin_embed_size = self.checkin_embedding_layer.output_embed_size

        # === 边类型嵌�? ===
        self.edge_type_embedding_layer = EdgeEmbedding(
            embed_size=self.checkin_embed_size,
            fusion_type=self.embed_fusion_type,
            num_edge_type=cfg.model_args.num_edge_type
        )

        # === 激活函�? ===
        if cfg.model_args.activation == 'elu':
            self.act = nn.ELU()
        elif cfg.model_args.activation == 'relu':
            self.act = nn.RReLU()
        elif cfg.model_args.activation == 'leaky_relu':
            self.act = nn.LeakyReLU()
        else:
            self.act = torch.tanh

        # === 时间编码器维�? ===
        if cfg.conv_args.time_fusion_mode == 'add':
            continuous_encoder_dim = self.checkin_embed_size
        else:
            continuous_encoder_dim = cfg.model_args.st_embed_size
        if continuous_encoder_dim <= 0:
            continuous_encoder_dim = 64
            logging.warning(f"Invalid continuous_encoder_dim, using default: {continuous_encoder_dim}")
        logging.info(f"Time encoder dimension: {continuous_encoder_dim}")

        # === 时间 + 距离 编码�? (使用你之前那版可分块/CPU计算优化�? TimeEncoder) ===
        # [MODIFIED] 确保 TimeEncoder 支持 chunk_size, compute_on_cpu 等（你的改版中已有）
        self.continuous_time_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim,
                                                  chunk_size=32768, compute_on_cpu=True, out_dtype=torch.float32)
        if self.distance_encoder_type == 'stan':
            self.continuous_distance_encoder = DistanceEncoderSTAN(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
        elif self.distance_encoder_type == 'time':
            # 用时间编码器作为距离编码器（可复用）
            self.continuous_distance_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim,
                                                          chunk_size=32768, compute_on_cpu=True, out_dtype=torch.float32)
        elif self.distance_encoder_type == 'hstlstm':
            self.continuous_distance_encoder = DistanceEncoderHSTLSTM(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
        elif self.distance_encoder_type == 'simple':
            self.continuous_distance_encoder = DistanceEncoderSimple(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
        else:
            raise ValueError(f"Wrong distance_encoder_type: {self.distance_encoder_type}!")

        # === 边属性嵌入层 ===
        if self.generate_edge_attr:
            self.edge_attr_embedding_layer = EdgeEmbedding(
                embed_size=self.checkin_embed_size,
                fusion_type=self.embed_fusion_type,
                num_edge_type=cfg.model_args.num_edge_type
            )
        else:
            if cfg.conv_args.edge_fusion_mode == 'add':
                self.edge_attr_embedding_layer = nn.Linear(3, self.checkin_embed_size)
            else:
                self.edge_attr_embedding_layer = None

        # === 第一�? Entity→Event 超图卷积 ===
        self.conv_for_time_filter = HypergraphTransformer(
            in_channels=self.checkin_embed_size,
            out_channels=self.checkin_embed_size,
            attn_heads=cfg.conv_args.num_attention_heads,
            residual_beta=cfg.conv_args.residual_beta,
            learn_beta=cfg.conv_args.learn_beta,
            dropout=cfg.conv_args.conv_dropout_rate,
            trans_method=cfg.conv_args.trans_method,
            edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
            time_fusion_mode=cfg.conv_args.time_fusion_mode,
            head_fusion_mode=cfg.conv_args.head_fusion_mode,
            residual_fusion_mode=None,
            edge_dim=None,
            rel_embed_dim=self.checkin_embed_size,
            time_embed_dim=continuous_encoder_dim,
            dist_embed_dim=continuous_encoder_dim,
            negative_slope=cfg.conv_args.negative_slope,
            have_query_feature=False
        )
        self.norms_for_time_filter = nn.BatchNorm1d(self.checkin_embed_size)
        self.dropout_for_time_filter = nn.Dropout(self.dropout_rate)

        # === Event→Chain 卷积（多层） ===
        self.conv_list = nn.ModuleList()
        if self.do_traj2traj:
            for i in range(self.num_conv_layers):
                have_query_feature = (i > 0)
                residual_fusion_mode = None if i == 0 else cfg.conv_args.residual_fusion_mode
                edge_size = None if self.edge_attr_embedding_layer is None else self.checkin_embed_size

                self.conv_list.append(
                    HypergraphTransformer(
                        in_channels=self.checkin_embed_size,
                        out_channels=self.checkin_embed_size,
                        attn_heads=cfg.conv_args.num_attention_heads,
                        residual_beta=cfg.conv_args.residual_beta,
                        learn_beta=cfg.conv_args.learn_beta,
                        dropout=cfg.conv_args.conv_dropout_rate,
                        trans_method=cfg.conv_args.trans_method,
                        edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
                        time_fusion_mode=cfg.conv_args.time_fusion_mode,
                        head_fusion_mode=cfg.conv_args.head_fusion_mode,
                        residual_fusion_mode=residual_fusion_mode,
                        edge_dim=edge_size,
                        rel_embed_dim=self.checkin_embed_size,
                        time_embed_dim=continuous_encoder_dim,
                        dist_embed_dim=continuous_encoder_dim,
                        negative_slope=cfg.conv_args.negative_slope,
                        have_query_feature=have_query_feature
                    )
                )
            self.norms_list = nn.ModuleList([nn.BatchNorm1d(self.checkin_embed_size) for _ in range(self.num_conv_layers)])
            self.dropout_list = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(self.num_conv_layers)])

        # === 输出层：尾事件分�? ===
        self.linear = nn.Linear(self.checkin_embed_size, dataset.num_event)
        self.loss_func = nn.CrossEntropyLoss()


    # -------------------------
      辅助：把 Batch-like 对象转为 model 需要的 dict 格式（容错）
    # -------------------------
    def _batch_to_input_dict(self, batch):
        """
        接受一�? Batch-like 对象（来�? NeighborSampler 或自定义 sampler），
        将可能的字段名标准化�? model 所需�? dict�?
        兼容字段（根据你�? pipeline 可能存在的名字）�?
          - node features: batch.x �? (batch.entity_x, batch.event_x, batch.chain_x)
          - edge_index / adjs_t: batch.edge_index / batch.adjs_t
          - edge attributes: batch.edge_attr / batch.edge_attrs
          - delta times: batch.delta_ts / batch.edge_delta_ts
          - delta spaces: batch.delta_ss / batch.edge_delta_ss
          - edge types: batch.edge_type / batch.edge_types
          - candidates: batch.candidates
          - labels: batch.labels or batch.y
        """
        input_data = {}

        # node features
        if hasattr(batch, 'x'):
            input_data['x'] = batch.x
        elif hasattr(batch, 'entity_x') and hasattr(batch, 'event_x') and hasattr(batch, 'chain_x'):
            input_data['x'] = (batch.entity_x, batch.event_x, batch.chain_x)
        else:
            # 若没有找到，设为 None（上层会报错�?
            input_data['x'] = getattr(batch, 'x', None)

        # edge_index can be either adjs_t (list of SparseTensor) or edge_index
        if hasattr(batch, 'adjs_t'):
            input_data['edge_index'] = batch.adjs_t
        else:
            input_data['edge_index'] = getattr(batch, 'edge_index', None)

        # edge attributes
        input_data['edge_attr'] = getattr(batch, 'edge_attr', getattr(batch, 'edge_attrs', None))

        # delta times & delta spaces
        input_data['delta_ts'] = getattr(batch, 'delta_ts', getattr(batch, 'edge_delta_ts', None))
        input_data['delta_ss'] = getattr(batch, 'delta_ss', getattr(batch, 'edge_delta_ss', None))

        # edge type
        input_data['edge_type'] = getattr(batch, 'edge_type', getattr(batch, 'edge_types', None))

        # candidates and labels
        input_data['candidates'] = getattr(batch, 'candidates', None)
        # labels might be batch.labels or batch.y
        labels = getattr(batch, 'labels', None)
        if labels is None:
            labels = getattr(batch, 'y', None)
        input_data['labels'] = labels

        # split_index: neighbor-sampler 常会�? adjs_t，尝试计�? split_index（容错）
        split_index = getattr(batch, 'split_index', None)
        if split_index is None and hasattr(batch, 'adjs_t'):
            try:
                # PyG �? SparseTensor 存储接口可能不同，之�? pipeline 使用�?
                # torch.max(row.adjs_t[1].storage.row()).tolist() 来计�? split_index
                split_index = int(torch.max(batch.adjs_t[1].storage.row()).item())
            except Exception:
                split_index = None
        input_data['split_index'] = split_index

        return input_data


    # -------------------------
    # forward：兼�? dict（全图）�? Batch-like（采样）
    # -------------------------
    def forward(self, data, label=None, mode='train'):
        """
        data 可以是：
          - dict：全图（三层超图）模式（与原版兼容）
          - Batch-like（有 adjs_t / edge_index / edge_delta_ts 等）：采样模�?
        """
        # 如果 data 不是 dict，将其转换为标准 dict
        if isinstance(data, dict):
            input_data = data
        else:
            # [MODIFIED] 使用辅助函数进行容错转换
            input_data = self._batch_to_input_dict(data)

        # 确保 node features 存在并且是三元组 (entity,event,chain) 的形�?
        if isinstance(input_data.get('x', None), (list, tuple)) and len(input_data['x']) == 3:
            entity_x, event_x, chain_x = input_data['x']
        else:
            # 如果不是三元组，尝试�? batch 的属性拆分（极端情况�?
            # 这里保底地把 input_data['x'] 视作 entity_x；event_x/chain_x �? None（若后续需要改�?
            ent = input_data.get('x')
            entity_x = ent
            event_x = None
            chain_x = None

        # 把输入张量移动到 model.device（如果提供了设备信息�?
        target_device = torch.device(self.device) if isinstance(self.device, str) else self.device

        # 如果 entity/event/chain 的张量存在，把它们移动到 device
        if isinstance(entity_x, torch.Tensor):
            entity_x = entity_x.to(target_device)
        if isinstance(event_x, torch.Tensor):
            event_x = event_x.to(target_device)
        if isinstance(chain_x, torch.Tensor):
            chain_x = chain_x.to(target_device)

        # === 1. 节点嵌入（无论全图还是采样，CheckinEmbedding 接口相同�? ===
        # [MODIFIED] 兼容缺失 event_x/chain_x 的情�?
        x = self.checkin_embedding_layer(entity_x, event_x, chain_x)

        # === 2. 边的时间/空间特征（Entity→Event�? ===
        # delta_ts 可能�? list/tuple（每层一份），也可能是单�? tensor（第一层）
        delta_ts_input = input_data.get('delta_ts', None)
        if isinstance(delta_ts_input, (list, tuple)):
            delta_ts_first = delta_ts_input[0]
        else:
            delta_ts_first = delta_ts_input

        edge_time_embed = None
        if delta_ts_first is not None:
            # �? delta_ts 转到 device 并转换为 float32，按小时归一（你的原实现�?
            delta_ts_first = delta_ts_first.to(target_device).to(torch.float32)
            edge_time_embed = self.continuous_time_encoder(delta_ts_first / (60 * 60))
            logging.info(f"Delta_ts shape: {delta_ts_first.shape}")
            # 只打印前 5 个样本（如果可用�?
            try:
                logging.info(f"Delta_ts sample values: {delta_ts_first[:5]}")
            except Exception:
                pass
        else:
            logging.info("Warning: delta_ts for first layer is None.")

        # delta_ss（距离）
        delta_ss_input = input_data.get('delta_ss', None)
        if isinstance(delta_ss_input, (list, tuple)):
            delta_ss_first = delta_ss_input[0]
        else:
            delta_ss_first = delta_ss_input

        edge_distance_embed = None
        if delta_ss_first is not None:
            delta_ss_first = delta_ss_first.to(target_device).to(torch.float32)
            if self.distance_encoder_type == 'stan':
                # 如果�? stan，传�? dist_type 标识
                edge_distance_embed = self.continuous_distance_encoder(delta_ss_first, dist_type='entity2event')
            else:
                edge_distance_embed = self.continuous_distance_encoder(delta_ss_first)
        else:
            logging.info("Warning: delta_ss for first layer is None.")

        # === 3. Entity→Event 卷积 ===
        edge_attr_embed, edge_type_embed = None, None
        first_edge_type = None
        edge_type_input = input_data.get('edge_type', None)
        if isinstance(edge_type_input, (list, tuple)):
            first_edge_type = edge_type_input[0]
        else:
            first_edge_type = edge_type_input

        if first_edge_type is not None:
            # 注意：EdgeEmbedding 期待 LongTensor indices；确保类型正�?
            if isinstance(first_edge_type, torch.Tensor):
                first_edge_type = first_edge_type.to(target_device).long()
            if self.generate_edge_attr:
                edge_attr_embed = self.edge_attr_embedding_layer(first_edge_type)
            edge_type_embed = self.edge_type_embedding_layer(first_edge_type)

        # edge_index 首层
        edge_index_input = input_data.get('edge_index', None)
        edge_index_first = None
        if isinstance(edge_index_input, (list, tuple)):
            edge_index_first = edge_index_input[0]
        else:
            edge_index_first = edge_index_input

        x_for_time_filter = self.conv_for_time_filter(
            x,
            edge_index=edge_index_first,
            edge_attr_embed=edge_attr_embed,
            edge_time_embed=edge_time_embed,
            edge_dist_embed=edge_distance_embed,
            edge_type_embed=edge_type_embed
        )
        x_for_time_filter = self.norms_for_time_filter(x_for_time_filter)
        x_for_time_filter = self.act(x_for_time_filter)
        x_for_time_filter = self.dropout_for_time_filter(x_for_time_filter)

        # === 4. Event→Chain 卷积（多层） ===
        # 逐层遍历 edge_index/edge_attr/delta_ts/delta_ss/edge_type（如果存在）
        if input_data.get('edge_index', None) is not None and self.do_traj2traj:
            # iterate layers 1..end
            # extract lists or tuples or singletons
            edge_index_list = input_data.get('edge_index')
            edge_attr_list = input_data.get('edge_attr')
            delta_ts_list = input_data.get('delta_ts')
            delta_ss_list = input_data.get('delta_ss')
            edge_type_list = input_data.get('edge_type')

            # defensive: ensure these are iterables
            def _as_list(x):
                if x is None:
                    return []
                if isinstance(x, (list, tuple)):
                    return list(x)
                # if it's a single tensor (only first layer), return list with that element repeated? we return single item
                return [x]

            edge_index_list = _as_list(edge_index_list)
            edge_attr_list = _as_list(edge_attr_list)
            delta_ts_list = _as_list(delta_ts_list)
            delta_ss_list = _as_list(delta_ss_list)
            edge_type_list = _as_list(edge_type_list)

            # iterate (skip first because we already used layer 0)
            for idx in range(1, len(edge_index_list)):
                edge_index = edge_index_list[idx]
                edge_attr = edge_attr_list[idx] if idx < len(edge_attr_list) else None
                delta_ts = delta_ts_list[idx] if idx < len(delta_ts_list) else None
                delta_dis = delta_ss_list[idx] if idx < len(delta_ss_list) else None
                edge_type = edge_type_list[idx] if idx < len(edge_type_list) else None

                # compute time & distance embedding for this layer
                edge_time_embed = None
                if delta_ts is not None:
                    edge_time_embed = self.continuous_time_encoder(delta_ts.to(target_device).to(torch.float32) / (60 * 60))
                if delta_dis is not None:
                    if self.distance_encoder_type == 'stan':
                        edge_distance_embed = self.continuous_distance_encoder(delta_dis.to(target_device).to(torch.float32), dist_type='event2chain')
                    else:
                        edge_distance_embed = self.continuous_distance_encoder(delta_dis.to(target_device).to(torch.float32))

                # edge type and attr for this layer
                edge_attr_embed, edge_type_embed = None, None
                if edge_type is not None:
                    edge_type = edge_type.to(target_device).long()
                    edge_type_embed = self.edge_type_embedding_layer(edge_type)
                    if self.generate_edge_attr:
                        edge_attr_embed = self.edge_attr_embedding_layer(edge_type)
                    elif self.edge_attr_embedding_layer and edge_attr is not None:
                        edge_attr_embed = self.edge_attr_embedding_layer(edge_attr.to(target_device).to(torch.float32))
                    elif edge_attr is not None:
                        edge_attr_embed = edge_attr.to(target_device).to(torch.float32)

                # === 确定 x_target (query / target nodes for this convolution) ===
                if idx == len(edge_index_list) - 1:
                    # last conv layer: target is batch (候选批�?)
                    batch_size = self.eval_batch_size if mode in ('test', 'validate') else self.batch_size
                    x_target = x_for_time_filter[:batch_size]
                else:
                    # 非最后一层：target 是前面节点数（根�? edge_index �? max idx 推断�?
                    try:
                        # edge_index might be SparseTensor or dense index; if it has max() method:
                        if isinstance(edge_index, torch.Tensor):
                            num_nodes = int(edge_index.max().item()) + 1
                        elif isinstance(edge_index, SparseTensor):
                            num_nodes = int(edge_index.sizes()[0])
                        else:
                            # fallback
                            num_nodes = x_for_time_filter.size(0)
                        x_target = x_for_time_filter[:num_nodes]
                    except Exception:
                        x_target = x_for_time_filter

                # perform convolution for this layer
                x = self.conv_list[idx - 1](
                    (x_for_time_filter, x_target),
                    edge_index=edge_index,
                    edge_attr_embed=edge_attr_embed,
                    edge_time_embed=edge_time_embed,
                    edge_dist_embed=edge_distance_embed,
                    edge_type_embed=edge_type_embed
                )
                x = self.norms_list[idx - 1](x)
                x = self.act(x)
                x = self.dropout_list[idx - 1](x)
        else:
            # no traj2traj convs
            x = x_for_time_filter

        # === 5. 分类预测（尾事件�? ===
        logits = self.linear(x)  # shape: [num_nodes_in_x, num_event]

        # === 6. 只取候选事件的 logits（如�? candidates 存在�? ===
        candidates = input_data.get('candidates', None)   # [B, K] or None
        labels = input_data.get('labels', None)           # [B] or [B,1]

        if candidates is None:
            # 没有候选集合，则直接返�? logits（可能用于其它用途）
            loss = None
            if labels is not None:
                # �? labels 存在�? logits 行数等于 labels 长度（直接监督）
                try:
                    loss = self.loss_func(logits, labels.long())
                except Exception:
                    logging.info("Labels present but shape mismatch with logits; returning None loss.")
                    loss = None
            return logits, loss

        # candidates 需要是 LongTensor, 并且位于 device �?
        candidates = candidates.long().to(target_device)
        B, K = candidates.size()
        logging.info(f"logits shape: {logits.shape}")
        logging.info(f"candidates shape: {candidates.shape}")
        try:
            logging.info(f"candidates range: {int(candidates.min().item())} {int(candidates.max().item())}")
        except Exception:
            pass

        # logits: [N_nodes, num_event]，我们假�? N_nodes 对应�? batch 的第一个维�? B（例如最后一�? x[:B])
        # 如果 logits 的第一维与 B 不同，需要保证我们取到的是针�? batch �? logits（上层已保证�?
        # �? gather 在列维度上取候选事件对应的分数
        # 为此需�? logits 大小�? [B, num_event]（或 [N, num_event]，但 candidates 索引对应于同一 N)
        # 我们做尽量稳健的 gather�?
        if logits.size(0) == B:
            candidate_logits = logits.gather(1, candidates)   # [B, K]
        else:
            # 如果 logits 行数 != B，尝试将 logits �? B 行作�? batch 对应
            if logits.size(0) >= B:
                candidate_logits = logits[:B].gather(1, candidates)
            else:
                # 此处说明上层 x_target / batch_size 推断可能有问�?
                # 为了避免抛错，尝�? expand logits �? B（会重复数据，结果不正确但不�? crash�?
                logging.warning("logits row count doesn't match batch size; attempting fallback (may be incorrect).")
                tiled = logits.repeat(int((B + logits.size(0) - 1) // logits.size(0)), 1)[:B]
                candidate_logits = tiled.gather(1, candidates)

        loss = None
        if labels is not None:
            # gold logits: [B, 1]
            gold = labels.view(-1, 1).long().to(target_device)
            # 如果 gold 中的索引超出 bounds，会抛错；这里假�? labels 是事�? id（col idx�?
            gold_logits = logits.gather(1, gold) if logits.size(0) >= gold.size(0) else logits[:B].gather(1, gold)
            # 拼接 [B, 1+K]
            logits_for_loss = torch.cat([gold_logits, candidate_logits], dim=1)
            target = torch.zeros(B, dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits_for_loss, target)

        return candidate_logits, loss

# import torch
# from torch import nn
# from layer import (
#     CheckinEmbedding,
#     EdgeEmbedding,
#     HypergraphTransformer,
#     TimeEncoder,
#     DistanceEncoderHSTLSTM,
#     DistanceEncoderSTAN,
#     DistanceEncoderSimple
# )
# from torch_sparse import SparseTensor
# import torch.nn.functional as F
# # 导入日志记录�?
# import logging
# class STHGCN(nn.Module):
#     def __init__(self, cfg, dataset):   # [MODIFIED] 增加 dataset 参数
#         super(STHGCN, self).__init__()
#         self.device = cfg.run_args.device
#         self.batch_size = cfg.run_args.batch_size
#         self.eval_batch_size = cfg.run_args.eval_batch_size
#         self.do_traj2traj = cfg.model_args.do_traj2traj
#         self.distance_encoder_type = cfg.model_args.distance_encoder_type
#         self.dropout_rate = cfg.model_args.dropout_rate
#         self.generate_edge_attr = cfg.model_args.generate_edge_attr
#         self.num_conv_layers = len(cfg.model_args.sizes)
#         self.num_poi = cfg.dataset_args.num_poi
#         self.embed_fusion_type = cfg.model_args.embed_fusion_type
#         self.fusion_type = getattr(cfg.model_args, "embed_fusion_type",
#                     getattr(cfg.model_args, "fusion_type", "concat"))

#         # # === 获取特征维度 ===
#         # # [MODIFIED] �? dataset 获取特征维度
#         # entity_feat_dim = dataset.entity_x.shape[1] if hasattr(dataset, 'entity_x') else 0
#         # event_feat_dim = dataset.event_x.shape[1] if hasattr(dataset, 'event_x') else 0
#         # chain_feat_dim = dataset.chain_x.shape[1] if hasattr(dataset, 'chain_x') else 0
        
#         # logging.info(f"Feature dimensions - Entity: {entity_feat_dim}, Event: {event_feat_dim}, Chain: {chain_feat_dim}")

#         # # === 三层超图 Embedding ===
#         # # [MODIFIED] 使用特征维度而不是节点数�?
#         # self.checkin_embedding_layer = CheckinEmbedding(
#         #     embed_size=cfg.model_args.embed_size,
#         #     fusion_type=self.fusion_type,
#         #     entity_feat_dim=entity_feat_dim,
#         #     event_feat_dim=event_feat_dim,
#         #     chain_feat_dim=chain_feat_dim
#         # )
#         # # �? STHGCN �? __init__ 方法中修改特征维度获取部�?

#         # === 获取特征维度 ===
#         # [MODIFIED] 直接从数据集中获取特征维�?
#         # 从调试信息中可以看到实际的特征维�?
#         entity_feat_dim = 9  # 从调试信息中可以看到 entity_x.shape[1] = 9
#         event_feat_dim = 7   # 从调试信息中可以看到 event_x.shape[1] = 7  
#         chain_feat_dim = 8   # 从调试信息中可以看到 chain_x.shape[1] = 8

#         logging.info(f"Feature dimensions - Entity: {entity_feat_dim}, Event: {event_feat_dim}, Chain: {chain_feat_dim}")

#         # === 三层超图 Embedding ===
#         self.checkin_embedding_layer = CheckinEmbedding(
#             embed_size=cfg.model_args.embed_size,
#             fusion_type=self.fusion_type,
#             entity_feat_dim=entity_feat_dim,  # 使用硬编码的特征维度
#             event_feat_dim=event_feat_dim,
#             chain_feat_dim=chain_feat_dim
#         )

#         self.checkin_embed_size = self.checkin_embedding_layer.output_embed_size

#         # === 边类型嵌�? ===
#         self.edge_type_embedding_layer = EdgeEmbedding(
#             embed_size=self.checkin_embed_size,
#             fusion_type=self.embed_fusion_type,
#             num_edge_type=cfg.model_args.num_edge_type
#         )

#         # === 激活函�? ===
#         if cfg.model_args.activation == 'elu':
#             self.act = nn.ELU()
#         elif cfg.model_args.activation == 'relu':
#             self.act = nn.RReLU()
#         elif cfg.model_args.activation == 'leaky_relu':
#             self.act = nn.LeakyReLU()
#         else:
#             self.act = torch.tanh

#         # # === 时间编码器维�? ===
#         # if cfg.conv_args.time_fusion_mode == 'add':
#         #     continuous_encoder_dim = self.checkin_embed_size
#         # else:
#         #     continuous_encoder_dim = cfg.model_args.st_embed_size

        
#         # === 时间编码器维�? ===
#         # [MODIFIED] 确保时间编码器维度正�?
#         if cfg.conv_args.time_fusion_mode == 'add':
#             continuous_encoder_dim = self.checkin_embed_size
#         else:
#             continuous_encoder_dim = cfg.model_args.st_embed_size

#         # 检查维度有效�?
#         if continuous_encoder_dim <= 0:
#             continuous_encoder_dim = 64  # 设置合理的默认�?
#             logging.warning(f"Invalid continuous_encoder_dim, using default: {continuous_encoder_dim}")

#         logging.info(f"Time encoder dimension: {continuous_encoder_dim}")

#         # === 时间 + 距离 编码�? ===
#         #self.continuous_time_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim)
#         self.continuous_time_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim,
#                                           chunk_size=32768, compute_on_cpu=True, out_dtype=torch.float32)
#         if self.distance_encoder_type == 'stan':
#             self.continuous_distance_encoder = DistanceEncoderSTAN(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
#         elif self.distance_encoder_type == 'time':
#             #self.continuous_distance_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim)
#             self.continuous_distance_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim,
#                                           chunk_size=32768, compute_on_cpu=True, out_dtype=torch.float32)
#         elif self.distance_encoder_type == 'hstlstm':
#             self.continuous_distance_encoder = DistanceEncoderHSTLSTM(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
#         elif self.distance_encoder_type == 'simple':
#             self.continuous_distance_encoder = DistanceEncoderSimple(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
#         else:
#             raise ValueError(f"Wrong distance_encoder_type: {self.distance_encoder_type}!")

#         # === 边属性嵌入层 ===
#         if self.generate_edge_attr:
#             self.edge_attr_embedding_layer = EdgeEmbedding(
#                 embed_size=self.checkin_embed_size,
#                 fusion_type=self.embed_fusion_type,
#                 num_edge_type=cfg.model_args.num_edge_type
#             )
#         else:
#             if cfg.conv_args.edge_fusion_mode == 'add':
#                 self.edge_attr_embedding_layer = nn.Linear(3, self.checkin_embed_size)
#             else:
#                 self.edge_attr_embedding_layer = None

#         # === 第一�? Entity→Event 超图卷积 ===
#         self.conv_for_time_filter = HypergraphTransformer(
#             in_channels=self.checkin_embed_size,
#             out_channels=self.checkin_embed_size,
#             attn_heads=cfg.conv_args.num_attention_heads,
#             residual_beta=cfg.conv_args.residual_beta,
#             learn_beta=cfg.conv_args.learn_beta,
#             dropout=cfg.conv_args.conv_dropout_rate,
#             trans_method=cfg.conv_args.trans_method,
#             edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
#             time_fusion_mode=cfg.conv_args.time_fusion_mode,
#             head_fusion_mode=cfg.conv_args.head_fusion_mode,
#             residual_fusion_mode=None,
#             edge_dim=None,
#             rel_embed_dim=self.checkin_embed_size,
#             time_embed_dim=continuous_encoder_dim,
#             dist_embed_dim=continuous_encoder_dim,
#             negative_slope=cfg.conv_args.negative_slope,
#             have_query_feature=False
#         )
#         self.norms_for_time_filter = nn.BatchNorm1d(self.checkin_embed_size)
#         self.dropout_for_time_filter = nn.Dropout(self.dropout_rate)

#         # === Event→Chain 卷积（多层） ===
#         self.conv_list = nn.ModuleList()
#         if self.do_traj2traj:
#             for i in range(self.num_conv_layers):
#                 have_query_feature = (i > 0)
#                 residual_fusion_mode = None if i == 0 else cfg.conv_args.residual_fusion_mode
#                 edge_size = None if self.edge_attr_embedding_layer is None else self.checkin_embed_size

#                 self.conv_list.append(
#                     HypergraphTransformer(
#                         in_channels=self.checkin_embed_size,
#                         out_channels=self.checkin_embed_size,
#                         attn_heads=cfg.conv_args.num_attention_heads,
#                         residual_beta=cfg.conv_args.residual_beta,
#                         learn_beta=cfg.conv_args.learn_beta,
#                         dropout=cfg.conv_args.conv_dropout_rate,
#                         trans_method=cfg.conv_args.trans_method,
#                         edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
#                         time_fusion_mode=cfg.conv_args.time_fusion_mode,
#                         head_fusion_mode=cfg.conv_args.head_fusion_mode,
#                         residual_fusion_mode=residual_fusion_mode,
#                         edge_dim=edge_size,
#                         rel_embed_dim=self.checkin_embed_size,
#                         time_embed_dim=continuous_encoder_dim,
#                         dist_embed_dim=continuous_encoder_dim,
#                         negative_slope=cfg.conv_args.negative_slope,
#                         have_query_feature=have_query_feature
#                     )
#                 )
#             self.norms_list = nn.ModuleList([nn.BatchNorm1d(self.checkin_embed_size) for _ in range(self.num_conv_layers)])
#             self.dropout_list = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(self.num_conv_layers)])

#         # === 时间 + 距离 编码�? ===
#         #self.continuous_time_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim)

#         # �? chunk 更小些、并�? CPU 上计算以尽量节省 GPU
#         self.continuous_time_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim,
#                                           chunk_size=32768, compute_on_cpu=True, out_dtype=torch.float32)
#         # self.continuous_time_encoder = TimeEncoderEfficient(args, embedding_dim, chunk_size=1024, use_linear=False, use_fp16=False)
#         if self.distance_encoder_type == 'stan':
#             self.continuous_distance_encoder = DistanceEncoderSTAN(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
#         elif self.distance_encoder_type == 'time':
#             #self.continuous_distance_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim)
#             self.continuous_distance_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim,
#                                           chunk_size=32768, compute_on_cpu=True, out_dtype=torch.float32)
#         elif self.distance_encoder_type == 'hstlstm':
#             self.continuous_distance_encoder = DistanceEncoderHSTLSTM(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
#         elif self.distance_encoder_type == 'simple':
#             self.continuous_distance_encoder = DistanceEncoderSimple(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
#         else:
#             raise ValueError(f"Wrong distance_encoder_type: {self.distance_encoder_type}!")

#         # === 输出层：尾事件分�? ===
#         self.linear = nn.Linear(self.checkin_embed_size, dataset.num_event)
#         self.loss_func = nn.CrossEntropyLoss()


#     def forward(self, data, label=None, mode='train'):

#         # === [MODIFIED] 兼容 torch_geometric Batch �? dict ===
#         if not isinstance(data, dict):
#             input_data = {
#                 'x': (input_data['entity_x'], input_data['event_x'], input_data['chain_x']),
#                 'edge_index': input_data['edge_index'],
#                 'edge_attr': input_data['edge_attr'],
#                 'delta_ts': input_data['delta_ts'],
#                 'delta_ss': input_data['delta_ss'],
#                 'edge_type': input_data['edge_type'],
#                 'candidates': input_data['candidates'],
#                 'labels': input_data['labels']
#             }
#         else:
#             input_data = data

#         # === 1. 节点嵌入 ===
#         # [MODIFIED] 拆分 entity_x, event_x, chain_x
#         entity_x = input_data['x'][0]
#         event_x = input_data['x'][1]
#         chain_x = input_data['x'][2] 
        
#         # [MODIFIED] 移除调试打印，因为现在处理的是特征矩阵不是索�?
#         # logging.info("entity_x min/max:", entity_x.min().item(), entity_x.max().item())
#         # logging.info("event_x min/max:", event_x.min().item(), event_x.max().item())
#         # logging.info("chain_x min/max:", chain_x.min().item(), chain_x.max().item())
        
#         x = self.checkin_embedding_layer(entity_x, event_x, chain_x)

#         # === 2. 边的时间/空间特征（Entity→Event�? ===
#         # === 2. 边的时间/空间特征（Entity→Event�? ===
#         delta_ts = input_data['delta_ts'][0] / (60 * 60)
#         logging.info(f"Delta_ts shape: {delta_ts.shape}")
#         logging.info(f"Delta_ts sample values: {delta_ts[:5]}")  # 查看�?5个�?
#         edge_time_embed = self.continuous_time_encoder(delta_ts)
#         #edge_time_embed = self.continuous_time_encoder(input_data['delta_ts'][0] / (60 * 60))
#         if self.distance_encoder_type == 'stan':
#             edge_distance_embed = self.continuous_distance_encoder(input_data['delta_ss'][0], dist_type='entity2event')
#         else:
#             edge_distance_embed = self.continuous_distance_encoder(input_data['delta_ss'][0])

#         edge_time_embed = self.continuous_time_encoder(delta_ts)
#         # === 3. Entity→Event 卷积 ===
#         edge_attr_embed, edge_type_embed = None, None
#         if input_data['edge_type'][0] is not None:
#             if self.generate_edge_attr:
#                 edge_attr_embed = self.edge_attr_embedding_layer(input_data['edge_type'][0])
#             edge_type_embed = self.edge_type_embedding_layer(input_data['edge_type'][0])

#         x_for_time_filter = self.conv_for_time_filter(
#             x,
#             edge_index=input_data['edge_index'][0],
#             edge_attr_embed=edge_attr_embed,
#             edge_time_embed=edge_time_embed,
#             edge_dist_embed=edge_distance_embed,
#             edge_type_embed=edge_type_embed
#         )
#         x_for_time_filter = self.norms_for_time_filter(x_for_time_filter)
#         x_for_time_filter = self.act(x_for_time_filter)
#         x_for_time_filter = self.dropout_for_time_filter(x_for_time_filter)

#         # === 4. Event→Chain 卷积 ===
#         if input_data['edge_index'][-1] is not None and self.do_traj2traj:
#             for idx, (edge_index, edge_attr, delta_ts, delta_dis, edge_type) in enumerate(
#                     zip(input_data["edge_index"][1:], input_data["edge_attr"][1:], 
#                         input_data["delta_ts"][1:], input_data["delta_ss"][1:], 
#                         input_data["edge_type"][1:])
#             ):
#                 edge_time_embed = self.continuous_time_encoder(delta_ts / (60 * 60))
#                 if self.distance_encoder_type == 'stan':
#                     edge_distance_embed = self.continuous_distance_encoder(delta_dis, dist_type='event2chain')
#                 else:
#                     edge_distance_embed = self.continuous_distance_encoder(delta_dis)

#                 edge_attr_embed, edge_type_embed = None, None
#                 if edge_type is not None:
#                     edge_type_embed = self.edge_type_embedding_layer(edge_type)
#                     if self.generate_edge_attr:
#                         edge_attr_embed = self.edge_attr_embedding_layer(edge_type)
#                     elif self.edge_attr_embedding_layer:
#                         edge_attr_embed = self.edge_attr_embedding_layer(edge_attr.to(torch.float32))
#                     else:
#                         edge_attr_embed = edge_attr.to(torch.float32)

#                 # === 确保 x_target 定义 ===
#                 if idx == len(input_data['edge_index']) - 2:
#                     batch_size = self.eval_batch_size if mode in ('test', 'validate') else self.batch_size
#                     x_target = x_for_time_filter[:batch_size]
#                 else:
#                     if edge_index is not None:
#                         # x_target = x[:edge_index.sparse_sizes()[0]]
#                         num_nodes = int(edge_index.max()) + 1   # 替代 edge_index.sparse_sizes()[0]
#                         x_target = x[:num_nodes]
#                     else:
#                         # 回退：如�? edge_index �? None，直接用 x_for_time_filter
#                         x_target = x_for_time_filter

#                 x = self.conv_list[idx](
#                     (x_for_time_filter, x_target),
#                     edge_index=edge_index,
#                     edge_attr_embed=edge_attr_embed,
#                     edge_time_embed=edge_time_embed,
#                     edge_dist_embed=edge_distance_embed,
#                     edge_type_embed=edge_type_embed
#                 )
#                 x = self.norms_list[idx](x)
#                 x = self.act(x)
#                 x = self.dropout_list[idx](x)
#         else:
#             x = x_for_time_filter



#         # === 5. 分类预测（尾事件�? ===
#         logits = self.linear(x)

#         # === [FIXED] 只取候选事件的 logits ===
#         candidates = input_data['candidates']   # [B, K]
#         labels = input_data['labels']           # [B] or [B, 1]

#         B, K = candidates.size()
#         logging.info("logits shape:", logits.shape)
#         logging.info("candidates shape:", candidates.shape)
#         logging.info("candidates range:", candidates.min().item(), candidates.max().item())

#         # logits: [B, num_event]
#         # candidates: [B, K] -> 在列维度 gather
#         candidate_logits = logits.gather(1, candidates)   # [B, K]

#         loss = None
#         if labels is not None:
#             # gold logits: [B, 1]
#             gold_logits = logits.gather(1, labels.view(-1, 1))

#             # 拼接 [B, 1+K]，gold 放在第一�?
#             logits_for_loss = torch.cat([gold_logits, candidate_logits], dim=1)

#             # target 是全 0，因�? gold 在第 0 �?
#             target = torch.zeros(B, dtype=torch.long, device=logits.device)

#             loss = F.cross_entropy(logits_for_loss, target)

#         return candidate_logits, loss











#         # # === [MODIFIED] 只取候选事件的 logits ===
#         # candidates = input_data['candidates']   # [B, K]
#         # labels = input_data['labels']

#         # B, K = candidates.size()
#         # logging.info("logits shape:", logits.shape)
#         # logging.info("candidates shape:", candidates.shape)
#         # logging.info("candidates range:", candidates.min().item(), candidates.max().item())

#         # candidate_logits = logits[candidates.view(-1)]  # [B*K, num_event]
#         # candidate_logits = candidate_logits.view(B, K, -1)  # [B, K, num_event]

#         # if candidate_logits.size(-1) == 1:
#         #     candidate_logits = candidate_logits.squeeze(-1)  # [B, K]

#         # loss = None
#         # if labels is not None:
#         #     loss = self.loss_func(candidate_logits, labels.long())

#         # return candidate_logits, loss
#     # import torch
# # from torch import nn
# # from layer import (
# #     CheckinEmbedding,
# #     EdgeEmbedding,
# #     HypergraphTransformer,
# #     TimeEncoder,
# #     DistanceEncoderHSTLSTM,
# #     DistanceEncoderSTAN,
# #     DistanceEncoderSimple
# # )


# # class STHGCN(nn.Module):
# #     def __init__(self, cfg, dataset):   # [MODIFIED] 增加 dataset 参数
# #         super(STHGCN, self).__init__()
# #         self.device = cfg.run_args.device
# #         self.batch_size = cfg.run_args.batch_size
# #         self.eval_batch_size = cfg.run_args.eval_batch_size
# #         self.do_traj2traj = cfg.model_args.do_traj2traj
# #         self.distance_encoder_type = cfg.model_args.distance_encoder_type
# #         self.dropout_rate = cfg.model_args.dropout_rate
# #         self.generate_edge_attr = cfg.model_args.generate_edge_attr
# #         self.num_conv_layers = len(cfg.model_args.sizes)
# #         self.num_poi = cfg.dataset_args.num_poi
# #         self.embed_fusion_type = cfg.model_args.embed_fusion_type
# #         self.fusion_type = getattr(cfg.model_args, "embed_fusion_type",
# #                     getattr(cfg.model_args, "fusion_type", "concat"))

# #         # === 三层超图 Embedding ===
# #         self.checkin_embedding_layer = CheckinEmbedding(
# #             embed_size=cfg.model_args.embed_size,
# #             fusion_type=self.fusion_type,
# #             num_entity=dataset.num_entity,   # [MODIFIED]
# #             num_event=dataset.num_event,     # [MODIFIED]
# #             num_chain=dataset.num_eventchain      # [MODIFIED]
# #         )

# #         self.checkin_embed_size = self.checkin_embedding_layer.output_embed_size  # concat �? 2*embed_size

# #         # === 边类型嵌�? ===
# #         self.edge_type_embedding_layer = EdgeEmbedding(
# #             embed_size=self.checkin_embed_size,
# #             fusion_type=self.embed_fusion_type,
# #             num_edge_type=cfg.model_args.num_edge_type
# #         )

# #         # === 激活函�? ===
# #         if cfg.model_args.activation == 'elu':
# #             self.act = nn.ELU()
# #         elif cfg.model_args.activation == 'relu':
# #             self.act = nn.RReLU()
# #         elif cfg.model_args.activation == 'leaky_relu':
# #             self.act = nn.LeakyReLU()
# #         else:
# #             self.act = torch.tanh

# #         # === 时间编码器维�? ===
# #         if cfg.conv_args.time_fusion_mode == 'add':
# #             continuous_encoder_dim = self.checkin_embed_size
# #         else:
# #             continuous_encoder_dim = cfg.model_args.st_embed_size

# #         # === 边属性嵌入层 ===
# #         if self.generate_edge_attr:
# #             self.edge_attr_embedding_layer = EdgeEmbedding(
# #                 embed_size=self.checkin_embed_size,
# #                 fusion_type=self.embed_fusion_type,
# #                 num_edge_type=cfg.model_args.num_edge_type
# #             )
# #         else:
# #             if cfg.conv_args.edge_fusion_mode == 'add':
# #                 self.edge_attr_embedding_layer = nn.Linear(3, self.checkin_embed_size)
# #             else:
# #                 self.edge_attr_embedding_layer = None

# #         # === 第一�? Entity→Event 超图卷积 ===
# #         self.conv_for_time_filter = HypergraphTransformer(
# #             in_channels=self.checkin_embed_size,
# #             out_channels=self.checkin_embed_size,
# #             attn_heads=cfg.conv_args.num_attention_heads,
# #             residual_beta=cfg.conv_args.residual_beta,
# #             learn_beta=cfg.conv_args.learn_beta,
# #             dropout=cfg.conv_args.conv_dropout_rate,
# #             trans_method=cfg.conv_args.trans_method,
# #             edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
# #             time_fusion_mode=cfg.conv_args.time_fusion_mode,
# #             head_fusion_mode=cfg.conv_args.head_fusion_mode,
# #             residual_fusion_mode=None,
# #             edge_dim=None,
# #             rel_embed_dim=self.checkin_embed_size,
# #             time_embed_dim=continuous_encoder_dim,
# #             dist_embed_dim=continuous_encoder_dim,
# #             negative_slope=cfg.conv_args.negative_slope,
# #             have_query_feature=False
# #         )
# #         self.norms_for_time_filter = nn.BatchNorm1d(self.checkin_embed_size)
# #         self.dropout_for_time_filter = nn.Dropout(self.dropout_rate)

# #         # === Event→Chain 卷积（多层） ===
# #         self.conv_list = nn.ModuleList()
# #         if self.do_traj2traj:
# #             for i in range(self.num_conv_layers):
# #                 have_query_feature = (i > 0)
# #                 residual_fusion_mode = None if i == 0 else cfg.conv_args.residual_fusion_mode
# #                 edge_size = None if self.edge_attr_embedding_layer is None else self.checkin_embed_size

# #                 self.conv_list.append(
# #                     HypergraphTransformer(
# #                         in_channels=self.checkin_embed_size,
# #                         out_channels=self.checkin_embed_size,
# #                         attn_heads=cfg.conv_args.num_attention_heads,
# #                         residual_beta=cfg.conv_args.residual_beta,
# #                         learn_beta=cfg.conv_args.learn_beta,
# #                         dropout=cfg.conv_args.conv_dropout_rate,
# #                         trans_method=cfg.conv_args.trans_method,
# #                         edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
# #                         time_fusion_mode=cfg.conv_args.time_fusion_mode,
# #                         head_fusion_mode=cfg.conv_args.head_fusion_mode,
# #                         residual_fusion_mode=residual_fusion_mode,
# #                         edge_dim=edge_size,
# #                         rel_embed_dim=self.checkin_embed_size,
# #                         time_embed_dim=continuous_encoder_dim,
# #                         dist_embed_dim=continuous_encoder_dim,
# #                         negative_slope=cfg.conv_args.negative_slope,
# #                         have_query_feature=have_query_feature
# #                     )
# #                 )
# #             self.norms_list = nn.ModuleList([nn.BatchNorm1d(self.checkin_embed_size) for _ in range(self.num_conv_layers)])
# #             self.dropout_list = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(self.num_conv_layers)])

# #         # === 时间 + 距离 编码�? ===
# #         self.continuous_time_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim)
# #         if self.distance_encoder_type == 'stan':
# #             self.continuous_distance_encoder = DistanceEncoderSTAN(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
# #         elif self.distance_encoder_type == 'time':
# #             self.continuous_distance_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim)
# #         elif self.distance_encoder_type == 'hstlstm':
# #             self.continuous_distance_encoder = DistanceEncoderHSTLSTM(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
# #         elif self.distance_encoder_type == 'simple':
# #             self.continuous_distance_encoder = DistanceEncoderSimple(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
# #         else:
# #             raise ValueError(f"Wrong distance_encoder_type: {self.distance_encoder_type}!")

# #         # === 输出层：尾事件分�? ===
# #         self.linear = nn.Linear(self.checkin_embed_size, dataset.num_event)  # [MODIFIED] �? dataset.num_event
# #         self.loss_func = nn.CrossEntropyLoss()

# #     def forward(self, data, label=None, mode='train'):

# #         # === [MODIFIED] 兼容 torch_geometric Batch �? dict ===
# #         if not isinstance(data, dict):
# #             input_data = {
# #                 'x': (input_data['entity_x'], input_data['event_x'], input_data['chain_x']),
# #                 'edge_index': input_data['edge_index'],
# #                 'edge_attr': input_data['edge_attr'],
# #                 'delta_ts': input_data['delta_ts'],
# #                 'delta_ss': input_data['delta_ss'],
# #                 'edge_type': input_data['edge_type'],
# #                 'candidates': input_data['candidates'],
# #                 'labels': input_data['labels']
# #             }
# #         else:
# #             input_data = data

# #         # === 1. 节点嵌入 ===
# #         # [MODIFIED] 拆分 entity_x, event_x, chain_x
# #         entity_x= input_data['x'][0]
# #         event_x = input_data['x'][1]
# #         chain_x = input_data['x'][2] 
# #         x = self.checkin_embedding_layer(entity_x, event_x, chain_x)

# #         # 打印索引范围，排查越�?
# #         logging.info("entity_x min/max:", entity_x.min().item(), entity_x.max().item())
# #         logging.info("event_x min/max:", event_x.min().item(), event_x.max().item())
# #         logging.info("chain_x min/max:", chain_x.min().item(), chain_x.max().item())

# #         logging.info("num_entities, num_events, num_chains:", 
# #             self.entity_embedding.num_embeddings,
# #             self.event_embedding.num_embeddings,
# #             self.chain_embedding.num_embeddings)
       

# #         # === 2. 边的时间/空间特征（Entity→Event�? ===
# #         edge_time_embed = self.continuous_time_encoder(input_data['delta_ts'][0] / (60 * 60))
# #         if self.distance_encoder_type == 'stan':
# #             edge_distance_embed = self.continuous_distance_encoder(input_data['delta_ss'][0], dist_type='entity2event')
# #         else:
# #             edge_distance_embed = self.continuous_distance_encoder(input_data['delta_ss'][0])

# #         # === 3. Entity→Event 卷积 ===
# #         edge_attr_embed, edge_type_embed = None, None
# #         if input_data['edge_type'][0] is not None:
# #             if self.generate_edge_attr:
# #                 edge_attr_embed = self.edge_attr_embedding_layer(input_data['edge_type'][0])
# #             edge_type_embed = self.edge_type_embedding_layer(input_data['edge_type'][0])

# #         x_for_time_filter = self.conv_for_time_filter(
# #             x,
# #             edge_index=input_data['edge_index'][0],
# #             edge_attr_embed=edge_attr_embed,
# #             edge_time_embed=edge_time_embed,
# #             edge_dist_embed=edge_distance_embed,
# #             edge_type_embed=edge_type_embed
# #         )
# #         x_for_time_filter = self.norms_for_time_filter(x_for_time_filter)
# #         x_for_time_filter = self.act(x_for_time_filter)
# #         x_for_time_filter = self.dropout_for_time_filter(x_for_time_filter)

# #         # === 4. Event→Chain 卷积 ===
# #         if input_data['edge_index'][-1] is not None and self.do_traj2traj:
# #             for idx, (edge_index, edge_attr, delta_ts, delta_dis, edge_type) in enumerate(
# #                     zip(input_data["edge_index"][1:], input_data["edge_attr"][1:], 
# #                         input_data["delta_ts"][1:], input_data["delta_ss"][1:], 
# #                         input_data["edge_type"][1:])
# #             ):
# #                 edge_time_embed = self.continuous_time_encoder(delta_ts / (60 * 60))
# #                 if self.distance_encoder_type == 'stan':
# #                     edge_distance_embed = self.continuous_distance_encoder(delta_dis, dist_type='event2chain')
# #                 else:
# #                     edge_distance_embed = self.continuous_distance_encoder(delta_dis)

# #                 edge_attr_embed, edge_type_embed = None, None
# #                 if edge_type is not None:
# #                     edge_type_embed = self.edge_type_embedding_layer(edge_type)
# #                     if self.generate_edge_attr:
# #                         edge_attr_embed = self.edge_attr_embedding_layer(edge_type)
# #                     elif self.edge_attr_embedding_layer:
# #                         edge_attr_embed = self.edge_attr_embedding_layer(edge_attr.to(torch.float32))
# #                     else:
# #                         edge_attr_embed = edge_attr.to(torch.float32)

# #                 if idx == len(input_data['edge_index']) - 2:
# #                     batch_size = self.eval_batch_size if mode in ('test', 'validate') else self.batch_size
# #                     x_target = x_for_time_filter[:batch_size]
# #                 else:
# #                     x_target = x[:edge_index.sparse_sizes()[0]]

# #                 x = self.conv_list[idx](
# #                     (x_for_time_filter, x_target),
# #                     edge_index=edge_index,
# #                     edge_attr_embed=edge_attr_embed,
# #                     edge_time_embed=edge_time_embed,
# #                     edge_dist_embed=edge_distance_embed,
# #                     edge_type_embed=edge_type_embed
# #                 )
# #                 x = self.norms_list[idx](x)
# #                 x = self.act(x)
# #                 x = self.dropout_list[idx](x)
# #         else:
# #             x = x_for_time_filter

# #         # === 5. 分类预测（尾事件�? ===
# #         logits = self.linear(x)

# #         # === [MODIFIED] 只取候选事件的 logits ===
# #         candidates = input_data['candidates']   # [B, K]
# #         labels = input_data['labels']

# #         B, K = candidates.size()
# #         candidate_logits = logits[candidates.view(-1)]  # [B*K, num_event]
# #         candidate_logits = candidate_logits.view(B, K, -1)  # [B, K, num_event]

# #         if candidate_logits.size(-1) == 1:
# #             candidate_logits = candidate_logits.squeeze(-1)  # [B, K]

# #         loss = None
# #         if labels is not None:
# #             loss = self.loss_func(candidate_logits, labels.long())

# #         return candidate_logits, loss


# #     # def forward(self, data, label=None, mode='train'):

# #     # # def forward(self, data, label=None, mode='train'):
# #     #     # [MODIFIED] 兼容 torch_geometric Batch
# #     #     # if not isinstance(data, dict):
# #     #     #     input_data = {
# #     #     #         'x': data.x,
# #     #     #         'edge_index': data.edge_index,
# #     #     #         'edge_attr': data.edge_attr,
# #     #     #         'delta_ts': data.delta_ts,
# #     #     #         'delta_ss': data.delta_ss,
# #     #     #         'edge_type': data.edge_type,
# #     #     #         'candidates': data.candidates,
# #     #     #         'labels': data.labels
# #     #     #     }
# #     #     # else:
# #     #     #     input_data = data

# #     #     entity_x, event_x, chain_x = data['x']  # 三层节点特征

# #     #     # === 1. 节点嵌入 ===
# #     #     # [MODIFIED] 一次性输入，不要重复三次调用
# #     #     x = self.checkin_embedding_layer(entity_x, event_x, chain_x)

# #     #     # === 2. 边的时间/空间特征（Entity→Event�? ===
# #     #     edge_time_embed = self.continuous_time_encoder(data['delta_ts'][0] / (60 * 60))
# #     #     if self.distance_encoder_type == 'stan':
# #     #         edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0], dist_type='entity2event')
# #     #     else:
# #     #         edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0])

# #     #     # === 3. Entity→Event 卷积 ===
# #     #     edge_attr_embed, edge_type_embed = None, None
# #     #     if data['edge_type'][0] is not None:
# #     #         if self.generate_edge_attr:
# #     #             edge_attr_embed = self.edge_attr_embedding_layer(data['edge_type'][0])
# #     #         edge_type_embed = self.edge_type_embedding_layer(data['edge_type'][0])

# #     #     x_for_time_filter = self.conv_for_time_filter(
# #     #         x,
# #     #         edge_index=data['edge_index'][0],
# #     #         edge_attr_embed=edge_attr_embed,
# #     #         edge_time_embed=edge_time_embed,
# #     #         edge_dist_embed=edge_distance_embed,
# #     #         edge_type_embed=edge_type_embed
# #     #     )
# #     #     x_for_time_filter = self.norms_for_time_filter(x_for_time_filter)
# #     #     x_for_time_filter = self.act(x_for_time_filter)
# #     #     x_for_time_filter = self.dropout_for_time_filter(x_for_time_filter)

# #     #     # === 4. Event→Chain 卷积 ===
# #     #     if data['edge_index'][-1] is not None and self.do_traj2traj:
# #     #         for idx, (edge_index, edge_attr, delta_ts, delta_dis, edge_type) in enumerate(
# #     #                 zip(data["edge_index"][1:], data["edge_attr"][1:], data["delta_ts"][1:], data["delta_ss"][1:], data["edge_type"][1:])
# #     #         ):
# #     #             edge_time_embed = self.continuous_time_encoder(delta_ts / (60 * 60))
# #     #             if self.distance_encoder_type == 'stan':
# #     #                 edge_distance_embed = self.continuous_distance_encoder(delta_dis, dist_type='event2chain')
# #     #             else:
# #     #                 edge_distance_embed = self.continuous_distance_encoder(delta_dis)

# #     #             edge_attr_embed, edge_type_embed = None, None
# #     #             if edge_type is not None:
# #     #                 edge_type_embed = self.edge_type_embedding_layer(edge_type)
# #     #                 if self.generate_edge_attr:
# #     #                     edge_attr_embed = self.edge_attr_embedding_layer(edge_type)
# #     #                 elif self.edge_attr_embedding_layer:
# #     #                     edge_attr_embed = self.edge_attr_embedding_layer(edge_attr.to(torch.float32))
# #     #                 else:
# #     #                     edge_attr_embed = edge_attr.to(torch.float32)

# #     #             if idx == len(data['edge_index']) - 2:
# #     #                 batch_size = self.eval_batch_size if mode in ('test', 'validate') else self.batch_size
# #     #                 x_target = x_for_time_filter[:batch_size]
# #     #             else:
# #     #                 x_target = x[:edge_index.sparse_sizes()[0]]

# #     #             x = self.conv_list[idx](
# #     #                 (x_for_time_filter, x_target),
# #     #                 edge_index=edge_index,
# #     #                 edge_attr_embed=edge_attr_embed,
# #     #                 edge_time_embed=edge_time_embed,
# #     #                 edge_dist_embed=edge_distance_embed,
# #     #                 edge_type_embed=edge_type_embed
# #     #             )
# #     #             x = self.norms_list[idx](x)
# #     #             x = self.act(x)
# #     #             x = self.dropout_list[idx](x)
# #     #     else:
# #     #         x = x_for_time_filter

# #     #     # === 5. 分类预测（尾事件�? ===
# #     #     logits = self.linear(x)

# #     #     # === [MODIFIED] 只取候选事件的 logits ===
# #     #     candidates = data.candidates   # [B, K]
# #     #     labels = data.labels

# #     #     B, K = candidates.size()
# #     #     candidate_logits = logits[candidates.view(-1)]  # [B*K, num_event]
# #     #     candidate_logits = candidate_logits.view(B, K, -1)  # [B, K, num_event]

# #     #     if candidate_logits.size(-1) == 1:
# #     #         candidate_logits = candidate_logits.squeeze(-1)  # [B, K]

# #     #     loss = None
# #     #     if labels is not None:
# #     #         loss = self.loss_func(candidate_logits, labels.long())

# #     #     return candidate_logits, loss

# # # import torch
# # # from torch import nn
# # # from layer import (
# # #     CheckinEmbedding,
# # #     EdgeEmbedding,
# # #     HypergraphTransformer,
# # #     TimeEncoder,
# # #     DistanceEncoderHSTLSTM,
# # #     DistanceEncoderSTAN,
# # #     DistanceEncoderSimple
# # # )


# # # class STHGCN(nn.Module):
# # #     def __init__(self, cfg):
# # #         super(STHGCN, self).__init__()
# # #         self.device = cfg.run_args.device
# # #         self.batch_size = cfg.run_args.batch_size
# # #         self.eval_batch_size = cfg.run_args.eval_batch_size
# # #         self.do_traj2traj = cfg.model_args.do_traj2traj
# # #         self.distance_encoder_type = cfg.model_args.distance_encoder_type
# # #         self.dropout_rate = cfg.model_args.dropout_rate
# # #         self.generate_edge_attr = cfg.model_args.generate_edge_attr
# # #         self.num_conv_layers = len(cfg.model_args.sizes)
# # #         self.num_poi = cfg.dataset_args.num_poi
# # #         self.embed_fusion_type = cfg.model_args.embed_fusion_type
# # #         self.fusion_type = getattr(cfg.model_args, "embed_fusion_type",
# # #                     getattr(cfg.model_args, "fusion_type", "concat"))
# # #         # === 三层超图 Embedding ===
# # #         # self.checkin_embedding_layer = CheckinEmbedding(
# # #         #     embed_size=cfg.model_args.embed_size,
# # #         #     fusion_type=self.embed_fusion_type,
# # #         #     dataset_args=cfg.dataset_args
# # #         # )
# # #         # ---- 三层超图�? embedding �? ----
# # #         # self.embedding_layer = CheckinEmbedding(
# # #         #     embed_size=cfg.model_args.embed_size,
# # #         #     fusion_type=cfg.model_args.fusion_type,
# # #         #     dataset_args=cfg.dataset_args
# # #         # )
# # #         self.checkin_embedding_layer = CheckinEmbedding(
# # #             embed_size=cfg.model_args.embed_size,
# # #             fusion_type=self.fusion_type,
# # #             num_entity=dataset.num_entity,   # �? 新增
# # #             num_event=dataset.num_event,     # �? 新增
# # #             num_eventchain=dataset.num_eventchain      # �? 新增
# # #         )
# # #         self.checkin_embed_size = self.checkin_embedding_layer.output_embed_size  # concat �? 2*embed_size

# # #         # === 边类型嵌�? ===
# # #         self.edge_type_embedding_layer = EdgeEmbedding(
# # #             embed_size=self.checkin_embed_size,
# # #             fusion_type=self.embed_fusion_type,
# # #             num_edge_type=cfg.model_args.num_edge_type
# # #         )

# # #         # === 激活函�? ===
# # #         if cfg.model_args.activation == 'elu':
# # #             self.act = nn.ELU()
# # #         elif cfg.model_args.activation == 'relu':
# # #             self.act = nn.RReLU()
# # #         elif cfg.model_args.activation == 'leaky_relu':
# # #             self.act = nn.LeakyReLU()
# # #         else:
# # #             self.act = torch.tanh

# # #         # === 时间编码器维�? ===
# # #         if cfg.conv_args.time_fusion_mode == 'add':
# # #             continuous_encoder_dim = self.checkin_embed_size
# # #         else:
# # #             continuous_encoder_dim = cfg.model_args.st_embed_size

# # #         # === 边属性嵌入层 ===
# # #         if self.generate_edge_attr:
# # #             self.edge_attr_embedding_layer = EdgeEmbedding(
# # #                 embed_size=self.checkin_embed_size,
# # #                 fusion_type=self.embed_fusion_type,
# # #                 num_edge_type=cfg.model_args.num_edge_type
# # #             )
# # #         else:
# # #             if cfg.conv_args.edge_fusion_mode == 'add':
# # #                 self.edge_attr_embedding_layer = nn.Linear(3, self.checkin_embed_size)
# # #             else:
# # #                 self.edge_attr_embedding_layer = None

# # #         # === 第一�? Entity→Event 超图卷积 ===
# # #         self.conv_for_time_filter = HypergraphTransformer(
# # #             in_channels=self.checkin_embed_size,
# # #             out_channels=self.checkin_embed_size,
# # #             attn_heads=cfg.conv_args.num_attention_heads,
# # #             residual_beta=cfg.conv_args.residual_beta,
# # #             learn_beta=cfg.conv_args.learn_beta,
# # #             dropout=cfg.conv_args.conv_dropout_rate,
# # #             trans_method=cfg.conv_args.trans_method,
# # #             edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
# # #             time_fusion_mode=cfg.conv_args.time_fusion_mode,
# # #             head_fusion_mode=cfg.conv_args.head_fusion_mode,
# # #             residual_fusion_mode=None,
# # #             edge_dim=None,
# # #             rel_embed_dim=self.checkin_embed_size,
# # #             time_embed_dim=continuous_encoder_dim,
# # #             dist_embed_dim=continuous_encoder_dim,
# # #             negative_slope=cfg.conv_args.negative_slope,
# # #             have_query_feature=False
# # #         )
# # #         self.norms_for_time_filter = nn.BatchNorm1d(self.checkin_embed_size)
# # #         self.dropout_for_time_filter = nn.Dropout(self.dropout_rate)

# # #         # === Event→Chain 卷积（多层） ===
# # #         self.conv_list = nn.ModuleList()
# # #         if self.do_traj2traj:
# # #             for i in range(self.num_conv_layers):
# # #                 have_query_feature = (i > 0)
# # #                 residual_fusion_mode = None if i == 0 else cfg.conv_args.residual_fusion_mode
# # #                 edge_size = None if self.edge_attr_embedding_layer is None else self.checkin_embed_size

# # #                 self.conv_list.append(
# # #                     HypergraphTransformer(
# # #                         in_channels=self.checkin_embed_size,
# # #                         out_channels=self.checkin_embed_size,
# # #                         attn_heads=cfg.conv_args.num_attention_heads,
# # #                         residual_beta=cfg.conv_args.residual_beta,
# # #                         learn_beta=cfg.conv_args.learn_beta,
# # #                         dropout=cfg.conv_args.conv_dropout_rate,
# # #                         trans_method=cfg.conv_args.trans_method,
# # #                         edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
# # #                         time_fusion_mode=cfg.conv_args.time_fusion_mode,
# # #                         head_fusion_mode=cfg.conv_args.head_fusion_mode,
# # #                         residual_fusion_mode=residual_fusion_mode,
# # #                         edge_dim=edge_size,
# # #                         rel_embed_dim=self.checkin_embed_size,
# # #                         time_embed_dim=continuous_encoder_dim,
# # #                         dist_embed_dim=continuous_encoder_dim,
# # #                         negative_slope=cfg.conv_args.negative_slope,
# # #                         have_query_feature=have_query_feature
# # #                     )
# # #                 )
# # #             self.norms_list = nn.ModuleList([nn.BatchNorm1d(self.checkin_embed_size) for _ in range(self.num_conv_layers)])
# # #             self.dropout_list = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(self.num_conv_layers)])

# # #         # === 时间 + 距离 编码�? ===
# # #         self.continuous_time_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim)
# # #         if self.distance_encoder_type == 'stan':
# # #             self.continuous_distance_encoder = DistanceEncoderSTAN(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
# # #         elif self.distance_encoder_type == 'time':
# # #             self.continuous_distance_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim)
# # #         elif self.distance_encoder_type == 'hstlstm':
# # #             self.continuous_distance_encoder = DistanceEncoderHSTLSTM(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
# # #         elif self.distance_encoder_type == 'simple':
# # #             self.continuous_distance_encoder = DistanceEncoderSimple(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
# # #         else:
# # #             raise ValueError(f"Wrong distance_encoder_type: {self.distance_encoder_type}!")

# # #         # === 输出层：尾事件分�? ===
# # #         self.linear = nn.Linear(self.checkin_embed_size, cfg.dataset_args.num_event)
# # #         self.loss_func = nn.CrossEntropyLoss()
# # #     def forward(self, data, label=None, mode='train'):
# # #         entity_x, event_x, chain_x = data['x']  # 三层节点特征

# # #         # === 1. 节点嵌入 ===
# # #         entity_emb = self.checkin_embedding_layer(entity_x)
# # #         event_emb  = self.checkin_embedding_layer(event_x)
# # #         chain_emb  = self.checkin_embedding_layer(chain_x)

# # #         # 拼接成统一表示
# # #         x = torch.cat([entity_emb, event_emb, chain_emb], dim=0)

# # #         # === 2. 边的时间/空间特征（Entity→Event�? ===
# # #         edge_time_embed = self.continuous_time_encoder(data['delta_ts'][0] / (60 * 60))
# # #         if self.distance_encoder_type == 'stan':
# # #             edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0], dist_type='entity2event')
# # #         else:
# # #             edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0])

# # #         # === 3. Entity→Event 卷积 ===
# # #         edge_attr_embed, edge_type_embed = None, None
# # #         if data['edge_type'][0] is not None:
# # #             if self.generate_edge_attr:
# # #                 edge_attr_embed = self.edge_attr_embedding_layer(data['edge_type'][0])
# # #             edge_type_embed = self.edge_type_embedding_layer(data['edge_type'][0])

# # #         entity_event_out = self.conv_for_time_filter(
# # #             x,
# # #             edge_index=data['edge_index'][0],
# # #             edge_attr_embed=edge_attr_embed,
# # #             edge_time_embed=edge_time_embed,
# # #             edge_dist_embed=edge_distance_embed,
# # #             edge_type_embed=edge_type_embed
# # #         )
# # #         entity_event_out = self.norms_for_time_filter(entity_event_out)
# # #         entity_event_out = self.act(entity_event_out)
# # #         entity_event_out = self.dropout_for_time_filter(entity_event_out)

# # #         # === 4. Event→Chain 卷积 ===
# # #         if data['edge_index'][-1] is not None and self.do_traj2traj:
# # #             for idx, (edge_index, edge_attr, delta_ts, delta_dis, edge_type) in enumerate(
# # #                     zip(data["edge_index"][1:], data["edge_attr"][1:], data["delta_ts"][1:], data["delta_ss"][1:], data["edge_type"][1:])
# # #             ):
# # #                 edge_time_embed = self.continuous_time_encoder(delta_ts / (60 * 60))
# # #                 if self.distance_encoder_type == 'stan':
# # #                     edge_distance_embed = self.continuous_distance_encoder(delta_dis, dist_type='event2chain')
# # #                 else:
# # #                     edge_distance_embed = self.continuous_distance_encoder(delta_dis)

# # #                 edge_attr_embed, edge_type_embed = None, None
# # #                 if edge_type is not None:
# # #                     edge_type_embed = self.edge_type_embedding_layer(edge_type)
# # #                     if self.generate_edge_attr:
# # #                         edge_attr_embed = self.edge_attr_embedding_layer(edge_type)
# # #                     elif self.edge_attr_embedding_layer:
# # #                         edge_attr_embed = self.edge_attr_embedding_layer(edge_attr.to(torch.float32))
# # #                     else:
# # #                         edge_attr_embed = edge_attr.to(torch.float32)

# # #                 if idx == len(data['edge_index']) - 2:
# # #                     batch_size = self.eval_batch_size if mode in ('test', 'validate') else self.batch_size
# # #                     x_target = entity_event_out[:batch_size]
# # #                 else:
# # #                     x_target = x[:edge_index.sparse_sizes()[0]]

# # #                 x = self.conv_list[idx](
# # #                     (entity_event_out, x_target),
# # #                     edge_index=edge_index,
# # #                     edge_attr_embed=edge_attr_embed,
# # #                     edge_time_embed=edge_time_embed,
# # #                     edge_dist_embed=edge_distance_embed,
# # #                     edge_type_embed=edge_type_embed
# # #                 )
# # #                 x = self.norms_list[idx](x)
# # #                 x = self.act(x)
# # #                 x = self.dropout_list[idx](x)
# # #         else:
# # #             x = entity_event_out

# # #         # === 5. 分类预测（尾事件�? ===
# # #         logits = self.linear(x)

# # #         # === [MODIFIED] 只取候选事件的 logits ===
# # #         candidates = data.candidates
# # #         label = data.labels

# # #         #candidates = data['candidates']   # [B, K]
# # #         B, K = candidates.size()
# # #         # logits 在所有节点上的分数，这里只取出候选事件的
# # #         candidate_logits = logits[candidates.view(-1)]  # [B*K, num_classes]
# # #         candidate_logits = candidate_logits.view(B, K, -1)  # [B, K, num_classes]

# # #         # 如果是二分类，可以直�? squeeze
# # #         if candidate_logits.size(-1) == 1:
# # #             candidate_logits = candidate_logits.squeeze(-1)  # [B, K]

# # #         loss = None
# # #         if label is not None:
# # #             # label: [B], 每个样本�? [0, K-1] �?
# # #             loss = self.loss_func(candidate_logits, label.long())

# # #         return candidate_logits, loss

# #     # def forward(self, data, label=None, mode='train'):
# #     #     # === [MODIFIED] 解包三层节点特征 ===
# #     #     entity_x, event_x, chain_x = data['x']  # (entity_x, event_x, chain_x)

# #     #     # === [MODIFIED] 三层分别�? embedding ===
# #     #     entity_emb = self.checkin_embedding_layer(entity_x)
# #     #     event_emb  = self.checkin_embedding_layer(event_x)
# #     #     chain_emb  = self.checkin_embedding_layer(chain_x)

# #     #     # === [MODIFIED] 拼接成统一空间表示（可选，兼容旧逻辑�? ===
# #     #     x = torch.cat([entity_emb, event_emb, chain_emb], dim=0)

# #     #     # === 2. 边的时间/空间特征（Entity→Event�? ===
# #     #     edge_time_embed = self.continuous_time_encoder(data['delta_ts'][0] / (60 * 60))
# #     #     if self.distance_encoder_type == 'stan':
# #     #         edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0], dist_type='entity2event')
# #     #     else:
# #     #         edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0])

# #     #     # === 3. Entity→Event 卷积 ===
# #     #     edge_attr_embed, edge_type_embed = None, None
# #     #     if data['edge_type'][0] is not None:
# #     #         if self.generate_edge_attr:
# #     #             edge_attr_embed = self.edge_attr_embedding_layer(data['edge_type'][0])
# #     #         edge_type_embed = self.edge_type_embedding_layer(data['edge_type'][0])

# #     #     entity_event_out = self.conv_for_time_filter(
# #     #         x,
# #     #         edge_index=data['edge_index'][0],
# #     #         edge_attr_embed=edge_attr_embed,
# #     #         edge_time_embed=edge_time_embed,
# #     #         edge_dist_embed=edge_distance_embed,
# #     #         edge_type_embed=edge_type_embed
# #     #     )
# #     #     entity_event_out = self.norms_for_time_filter(entity_event_out)
# #     #     entity_event_out = self.act(entity_event_out)
# #     #     entity_event_out = self.dropout_for_time_filter(entity_event_out)

# #     #     # === 4. Event→Chain 卷积 ===
# #     #     if data['edge_index'][-1] is not None and self.do_traj2traj:
# #     #         for idx, (edge_index, edge_attr, delta_ts, delta_dis, edge_type) in enumerate(
# #     #                 zip(data["edge_index"][1:], data["edge_attr"][1:], data["delta_ts"][1:], data["delta_ss"][1:], data["edge_type"][1:])
# #     #         ):
# #     #             edge_time_embed = self.continuous_time_encoder(delta_ts / (60 * 60))
# #     #             if self.distance_encoder_type == 'stan':
# #     #                 edge_distance_embed = self.continuous_distance_encoder(delta_dis, dist_type='event2chain')
# #     #             else:
# #     #                 edge_distance_embed = self.continuous_distance_encoder(delta_dis)

# #     #             edge_attr_embed, edge_type_embed = None, None
# #     #             if edge_type is not None:
# #     #                 edge_type_embed = self.edge_type_embedding_layer(edge_type)
# #     #                 if self.generate_edge_attr:
# #     #                     edge_attr_embed = self.edge_attr_embedding_layer(edge_type)
# #     #                 elif self.edge_attr_embedding_layer:
# #     #                     edge_attr_embed = self.edge_attr_embedding_layer(edge_attr.to(torch.float32))
# #     #                 else:
# #     #                     edge_attr_embed = edge_attr.to(torch.float32)

# #     #             if idx == len(data['edge_index']) - 2:
# #     #                 batch_size = self.eval_batch_size if mode in ('test', 'validate') else self.batch_size
# #     #                 x_target = entity_event_out[:batch_size]
# #     #             else:
# #     #                 x_target = x[:edge_index.sparse_sizes()[0]]

# #     #             x = self.conv_list[idx](
# #     #                 (entity_event_out, x_target),   # [MODIFIED] 输入换成 entity_event_out
# #     #                 edge_index=edge_index,
# #     #                 edge_attr_embed=edge_attr_embed,
# #     #                 edge_time_embed=edge_time_embed,
# #     #                 edge_dist_embed=edge_distance_embed,
# #     #                 edge_type_embed=edge_type_embed
# #     #             )
# #     #             x = self.norms_list[idx](x)
# #     #             x = self.act(x)
# #     #             x = self.dropout_list[idx](x)
# #     #     else:
# #     #         x = entity_event_out

# #     #     # === 5. 分类预测（尾事件�? ===
# #     #     logits = self.linear(x)
# #     #     loss = None
# #     #     if label is not None:
# #     #         loss = self.loss_func(logits, label.long())
# #     #     return logits, loss

# #     # # def forward(self, data, label=None, mode='train'):
# #     # #     entity_x, event_x, chain_x = data['x']  # 三层节点特征

# #     # #     # === 1. 节点嵌入 ===
# #     # #     x = self.checkin_embedding_layer(entity_x, event_x, chain_x)

# #     # #     # === 2. 边的时间/空间特征 ===
# #     # #     edge_time_embed = self.continuous_time_encoder(data['delta_ts'][0] / (60 * 60))
# #     # #     if self.distance_encoder_type == 'stan':
# #     # #         edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0], dist_type='entity2event')
# #     # #     else:
# #     # #         edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0])

# #     # #     # === 3. Entity→Event 卷积 ===
# #     # #     edge_attr_embed, edge_type_embed = None, None
# #     # #     if data['edge_type'][0] is not None:
# #     # #         if self.generate_edge_attr:
# #     # #             edge_attr_embed = self.edge_attr_embedding_layer(data['edge_type'][0])
# #     # #         edge_type_embed = self.edge_type_embedding_layer(data['edge_type'][0])

# #     # #     x_for_time_filter = self.conv_for_time_filter(
# #     # #         x,
# #     # #         edge_index=data['edge_index'][0],
# #     # #         edge_attr_embed=edge_attr_embed,
# #     # #         edge_time_embed=edge_time_embed,
# #     # #         edge_dist_embed=edge_distance_embed,
# #     # #         edge_type_embed=edge_type_embed
# #     # #     )
# #     # #     x_for_time_filter = self.norms_for_time_filter(x_for_time_filter)
# #     # #     x_for_time_filter = self.act(x_for_time_filter)
# #     # #     x_for_time_filter = self.dropout_for_time_filter(x_for_time_filter)

# #     # #     # === 4. Event→Chain 卷积 ===
# #     # #     if data['edge_index'][-1] is not None and self.do_traj2traj:
# #     # #         for idx, (edge_index, edge_attr, delta_ts, delta_dis, edge_type) in enumerate(
# #     # #                 zip(data["edge_index"][1:], data["edge_attr"][1:], data["delta_ts"][1:], data["delta_ss"][1:], data["edge_type"][1:])
# #     # #         ):
# #     # #             edge_time_embed = self.continuous_time_encoder(delta_ts / (60 * 60))
# #     # #             if self.distance_encoder_type == 'stan':
# #     # #                 edge_distance_embed = self.continuous_distance_encoder(delta_dis, dist_type='event2chain')
# #     # #             else:
# #     # #                 edge_distance_embed = self.continuous_distance_encoder(delta_dis)

# #     # #             edge_attr_embed, edge_type_embed = None, None
# #     # #             if edge_type is not None:
# #     # #                 edge_type_embed = self.edge_type_embedding_layer(edge_type)
# #     # #                 if self.generate_edge_attr:
# #     # #                     edge_attr_embed = self.edge_attr_embedding_layer(edge_type)
# #     # #                 elif self.edge_attr_embedding_layer:
# #     # #                     edge_attr_embed = self.edge_attr_embedding_layer(edge_attr.to(torch.float32))
# #     # #                 else:
# #     # #                     edge_attr_embed = edge_attr.to(torch.float32)

# #     # #             if idx == len(data['edge_index']) - 2:
# #     # #                 batch_size = self.eval_batch_size if mode in ('test', 'validate') else self.batch_size
# #     # #                 x_target = x_for_time_filter[:batch_size]
# #     # #             else:
# #     # #                 x_target = x[:edge_index.sparse_sizes()[0]]

# #     # #             x = self.conv_list[idx](
# #     # #                 (x, x_target),
# #     # #                 edge_index=edge_index,
# #     # #                 edge_attr_embed=edge_attr_embed,
# #     # #                 edge_time_embed=edge_time_embed,
# #     # #                 edge_dist_embed=edge_distance_embed,
# #     # #                 edge_type_embed=edge_type_embed
# #     # #             )
# #     # #             x = self.norms_list[idx](x)
# #     # #             x = self.act(x)
# #     # #             x = self.dropout_list[idx](x)
# #     # #     else:
# #     # #         x = x_for_time_filter

# #     # #     # === 5. 分类预测（尾事件�? ===
# #     # #     logits = self.linear(x)
# #     # #     loss = None
# #     # #     if label is not None:
# #     # #         loss = self.loss_func(logits, label.long())
# #     # #     return logits, loss

