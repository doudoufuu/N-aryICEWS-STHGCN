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

class STHGCN(nn.Module):
    """
    三层超图时空图卷积网络，用于事件链中的尾事件预测任务
    三层结构：
    1. 实体-事件超图（L1）：实体节点连接到事件节点
    2. 事件-事件图（L2）：事件节点之间的相似性连接
    3. 事件链-事件链图（L3）：事件链节点之间的相似性连接
    """
    def __init__(self, cfg):
        super(STHGCN, self).__init__()
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

        # 节点嵌入层
        self.checkin_embedding_layer = CheckinEmbedding(
            embed_size=cfg.model_args.embed_size,
            fusion_type=self.embed_fusion_type,
            dataset_args=cfg.dataset_args
        )
        self.checkin_embed_size = self.checkin_embedding_layer.output_embed_size

        # 边类型嵌入层
        self.edge_type_embedding_layer = EdgeEmbedding(
            embed_size=self.checkin_embed_size,
            fusion_type=self.embed_fusion_type,
            num_edge_type=cfg.model_args.num_edge_type
        )

        # 激活函数
        if cfg.model_args.activation == 'elu':
            self.act = nn.ELU()
        elif cfg.model_args.activation == 'relu':
            self.act = nn.RReLU()
        elif cfg.model_args.activation == 'leaky_relu':
            self.act = nn.LeakyReLU()
        else:
            self.act = torch.tanh

        # 连续编码器维度
        if cfg.conv_args.time_fusion_mode == 'add':
            continuous_encoder_dim = self.checkin_embed_size
        else:
            continuous_encoder_dim = cfg.model_args.st_embed_size

        # 边属性嵌入层
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

        # 第一层：实体->事件超图
        self.entity2event_conv = HypergraphTransformer(
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
        self.entity2event_norm = nn.BatchNorm1d(self.checkin_embed_size)
        self.entity2event_dropout = nn.Dropout(self.dropout_rate)

        # 第二层：事件->事件图
        self.event2event_conv = HypergraphTransformer(
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
            residual_fusion_mode=cfg.conv_args.residual_fusion_mode,
            edge_dim=3,
            rel_embed_dim=self.checkin_embed_size,
            time_embed_dim=continuous_encoder_dim,
            dist_embed_dim=continuous_encoder_dim,
            negative_slope=cfg.conv_args.negative_slope,
            have_query_feature=True
        )
        self.event2event_norm = nn.BatchNorm1d(self.checkin_embed_size)
        self.event2event_dropout = nn.Dropout(self.dropout_rate)

        # 第三层：事件链->事件链图
        self.chain2chain_conv = HypergraphTransformer(
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
            residual_fusion_mode=cfg.conv_args.residual_fusion_mode,
            edge_dim=3,
            rel_embed_dim=self.checkin_embed_size,
            time_embed_dim=continuous_encoder_dim,
            dist_embed_dim=continuous_encoder_dim,
            negative_slope=cfg.conv_args.negative_slope,
            have_query_feature=True
        )
        self.chain2chain_norm = nn.BatchNorm1d(self.checkin_embed_size)
        self.chain2chain_dropout = nn.Dropout(self.dropout_rate)

        # 时间编码器
        self.continuous_time_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim)
        
        # 距离编码器
        if self.distance_encoder_type == 'stan':
            self.continuous_distance_encoder = DistanceEncoderSTAN(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
        elif self.distance_encoder_type == 'hstlstm':
            self.continuous_distance_encoder = DistanceEncoderHSTLSTM(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
        elif self.distance_encoder_type == 'simple':
            self.continuous_distance_encoder = DistanceEncoderSimple(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
        else:
            raise ValueError("Unsupported distance_encoder_type")

        # 尾事件预测层：从5个候选事件中选择1个
        self.pred_linear = nn.Linear(self.checkin_embed_size, 1)

    def forward(self, data, label=None, mode='train'):
        """
        前向传播函数
        
        Args:
            data: 包含三层图数据的字典
                - x: 节点特征 [N, D]
                - edge_index: 三层图的边索引列表 [3]
                - edge_attr: 三层图的边属性列表 [3]
                - edge_type: 三层图的边类型列表 [3]
                - delta_ts: 三层图的时间差列表 [3]
                - delta_ss: 三层图的空间差列表 [3]
                - candidate_index: 候选事件索引 [B, 5]
            label: 真实标签 [B]
            mode: 训练模式
            
        Returns:
            logits: 预测logits [B, 5]
            loss: 损失值（如果提供标签）
        """
        # 初始节点嵌入
        x = self.checkin_embedding_layer(data['x'])  # [N, D]

        # 第一层：实体->事件超图
        x = self.entity2event_conv(
            x, edge_index=data['edge_index'][0],
            edge_time_embed=self.continuous_time_encoder(data['delta_ts'][0] / 3600),
            edge_dist_embed=self.continuous_distance_encoder(data['delta_ss'][0]),
            edge_type_embed=self.edge_type_embedding_layer(data['edge_type'][0]) if data['edge_type'][0] is not None else None
        )
        x = self.entity2event_norm(x)
        x = self.act(x)
        x = self.entity2event_dropout(x)

        # 第二层：事件->事件图
        x = self.event2event_conv(
            (x, x), edge_index=data['edge_index'][1],
            edge_attr_embed=data['edge_attr'][1].float() if data['edge_attr'][1] is not None else None,
            edge_time_embed=self.continuous_time_encoder(data['delta_ts'][1] / 3600),
            edge_dist_embed=self.continuous_distance_encoder(data['delta_ss'][1]),
            edge_type_embed=self.edge_type_embedding_layer(data['edge_type'][1]) if data['edge_type'][1] is not None else None
        )
        x = self.event2event_norm(x)
        x = self.act(x)
        x = self.event2event_dropout(x)

        # 第三层：事件链->事件链图
        x = self.chain2chain_conv(
            (x, x), edge_index=data['edge_index'][2],
            edge_attr_embed=data['edge_attr'][2].float() if data['edge_attr'][2] is not None else None,
            edge_time_embed=self.continuous_time_encoder(data['delta_ts'][2] / 3600),
            edge_dist_embed=self.continuous_distance_encoder(data['delta_ss'][2]),
            edge_type_embed=self.edge_type_embedding_layer(data['edge_type'][2]) if data['edge_type'][2] is not None else None
        )
        x = self.chain2chain_norm(x)
        x = self.act(x)
        x = self.chain2chain_dropout(x)

        # 尾事件预测：从5个候选事件中选择1个
        candidate_event_idx = data['candidate_index']  # [B, 5]
        candidate_embeds = x[candidate_event_idx]  # [B, 5, D]
        logits = self.pred_linear(candidate_embeds).squeeze(-1)  # [B, 5]

        # 计算损失
        loss = None
        if label is not None:
            loss = nn.CrossEntropyLoss()(logits, label.long())
        
        return logits, loss

    @staticmethod
    def prepare_data_for_model(batch_data):
        """
        将采样器返回的Batch数据转换为模型期望的格式
        
        Args:
            batch_data: NeighborSampler返回的Batch对象
            
        Returns:
            dict: 模型期望的数据格式
        """
        # 提取三层图的边索引
        edge_index = []
        edge_attr = []
        edge_type = []
        delta_ts = []
        delta_ss = []
        
        for i, adj_t in enumerate(batch_data.adjs_t):
            if adj_t is not None:
                row, col, _ = adj_t.coo()
                edge_index.append(torch.stack([row, col], dim=0))
                edge_attr.append(batch_data.edge_attrs[i])
                edge_type.append(batch_data.edge_types[i])
                delta_ts.append(batch_data.edge_delta_ts[i])
                delta_ss.append(batch_data.edge_delta_ss[i])
            else:
                # 空图，创建空的边索引
                edge_index.append(torch.empty((2, 0), dtype=torch.long, device=batch_data.x.device))
                edge_attr.append(None)
                edge_type.append(None)
                delta_ts.append(torch.empty((0,), dtype=torch.float, device=batch_data.x.device))
                delta_ss.append(torch.empty((0,), dtype=torch.float, device=batch_data.x.device))
        
        return {
            'x': batch_data.x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'edge_type': edge_type,
            'delta_ts': delta_ts,
            'delta_ss': delta_ss,
            'candidate_index': batch_data.candidate_index
        }
