from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_sparse import SparseTensor

from layer import (
    CheckinEmbedding,
    EdgeEmbedding,
    HypergraphTransformer,
    TimeEncoder,
    DistanceEncoderHSTLSTM,
    DistanceEncoderSTAN,
    DistanceEncoderSimple,
)


class STHGCN(nn.Module):
    """Four-layer STHGCN with HypergraphTransformer blocks for Entity閳墮vent閳墫hain閳墫hainNetwork."""

    def __init__(self, cfg, dataset) -> None:
        super().__init__()
        self.device = cfg.run_args.device
        self.embed_size = cfg.model_args.embed_size
        self.dropout_rate = cfg.model_args.dropout_rate
        self.batch_size = cfg.run_args.batch_size
        self.eval_batch_size = cfg.run_args.eval_batch_size
        self.num_candidates = getattr(cfg.model_args, "num_candidates", 5)
        self.distance_encoder_type = cfg.model_args.distance_encoder_type
        self.time_fusion_mode = cfg.conv_args.time_fusion_mode
        self.edge_fusion_mode = cfg.conv_args.edge_fusion_mode
        self.use_full_softmax = getattr(cfg.run_args, "use_full_softmax", False)  # [MODIFIED-MODEL]
        self.tie_event_classifier = bool(getattr(cfg.model_args, "tie_event_classifier", self.use_full_softmax))  # [ADDED-MODEL]
        self.normalize_event_classifier = bool(getattr(cfg.model_args, "normalize_event_classifier", True))  # [ADDED-MODEL]
        self.event_classifier_temperature = float(getattr(cfg.model_args, "event_classifier_temperature", 1.0))
        if self.event_classifier_temperature <= 0.0:
            raise ValueError("event_classifier_temperature must be positive.")
        self._initial_logit_scale = torch.log(torch.tensor([1.0 / self.event_classifier_temperature], dtype=torch.float32))
        # Margin loss (candidate mode) controls  # [ADDED-MODEL]\n        
        self.margin_loss_enable = bool(getattr(cfg.run_args, 'margin_loss_enable', True))        
        self.margin_value = float(getattr(cfg.run_args, 'margin', 0.2))       
        self.margin_weight = float(getattr(cfg.run_args, 'margin_weight', 0.5))
        self.time_input_scale = float(getattr(cfg.model_args, "time_input_scale", 24.0))

        entity_x, event_x, chain_x = dataset.x
        self.entity_count = entity_x.size(0)
        self.event_count = event_x.size(0)
        self.chain_count = chain_x.size(0)
        self.entity_offset = getattr(dataset, "entity_base", 0)
        self.event_offset = getattr(dataset, "event_base", self.entity_offset + self.entity_count)
        self.chain_offset = getattr(
            dataset,
            "chain_base",
            self.event_offset + self.event_count,
        )

        self.register_buffer("entity_features", entity_x.float())
        self.register_buffer("event_features", event_x.float())
        self.register_buffer("chain_features", chain_x.float())

        self.embedding = CheckinEmbedding(
            entity_dim=entity_x.size(1),
            event_dim=event_x.size(1),
            chain_dim=chain_x.size(1),
            embed_size=self.embed_size,
            dropout=self.dropout_rate,
        )

        self.edge_type_embedding = EdgeEmbedding(
            num_edge_type=cfg.model_args.num_edge_type,
            embed_size=self.embed_size,
        )

        if self.time_fusion_mode == "add":
            continuous_dim = self.embed_size
        else:
            continuous_dim = cfg.model_args.st_embed_size

        self.time_encoder = TimeEncoder(cfg.model_args, continuous_dim)

        if self.distance_encoder_type == "stan":
            self.distance_encoder = DistanceEncoderSTAN(cfg.model_args, continuous_dim, dataset.spatial_slots)
        elif self.distance_encoder_type == "hstlstm":
            self.distance_encoder = DistanceEncoderHSTLSTM(cfg.model_args, continuous_dim, dataset.spatial_slots)
        elif self.distance_encoder_type == "simple":
            self.distance_encoder = DistanceEncoderSimple(cfg.model_args, continuous_dim, dataset.spatial_slots)
        elif self.distance_encoder_type == "time":
            self.distance_encoder = TimeEncoder(cfg.model_args, continuous_dim)
        else:
            raise ValueError(f"Unsupported distance encoder type: {self.distance_encoder_type}")

        self.time_filter_index = 2  # [MODIFIED-MODEL]
        self.layer_specs = [
            {"name": "entity2event", "dist_type": "ch2tj", "have_query": True, "is_time_filter": False},  # [MODIFIED-MODEL]
            {"name": "event2event", "dist_type": "tj2tj", "have_query": True, "is_time_filter": False},  # [MODIFIED-MODEL]
            {"name": "event2chain", "dist_type": "ch2tj", "have_query": False, "is_time_filter": True},  # [MODIFIED-MODEL]
            {"name": "chain2chain", "dist_type": "tj2tj", "have_query": True, "is_time_filter": False},  # [MODIFIED-MODEL]
        ]

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.attr_mlps = nn.ModuleList()

        for idx, spec in enumerate(self.layer_specs):
            attr_tensor = dataset.edge_attr[idx]
            if attr_tensor is not None:
                in_dim = attr_tensor.size(1) if attr_tensor.dim() > 1 else 1
                self.attr_mlps.append(nn.Linear(in_dim, self.embed_size))
            else:
                self.attr_mlps.append(nn.Identity())

            if spec.get("is_time_filter"):
                self.convs.append(None)
                self.norms.append(nn.Identity())
                self.dropouts.append(nn.Identity())
                continue

            conv = HypergraphTransformer(
                in_channels=self.embed_size,
                out_channels=self.embed_size,
                attn_heads=cfg.conv_args.num_attention_heads,
                residual_beta=cfg.conv_args.residual_beta,
                learn_beta=cfg.conv_args.learn_beta,
                dropout=cfg.conv_args.conv_dropout_rate,
                trans_method=cfg.conv_args.trans_method,
                edge_fusion_mode=self.edge_fusion_mode,
                time_fusion_mode=self.time_fusion_mode,
                head_fusion_mode=cfg.conv_args.head_fusion_mode,
                residual_fusion_mode=cfg.conv_args.residual_fusion_mode,
                edge_dim=self.embed_size if self.edge_fusion_mode == "concat" else self.embed_size,
                rel_embed_dim=self.embed_size,
                time_embed_dim=continuous_dim,
                dist_embed_dim=continuous_dim,
                negative_slope=cfg.conv_args.negative_slope,
                have_query_feature=spec["have_query"],
            )
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(self.embed_size))
            self.dropouts.append(nn.Dropout(self.dropout_rate))

        self.activation = nn.ReLU()
        if self.tie_event_classifier:
            self.tail_classifier = None
            self.event_classifier_bias = nn.Parameter(torch.zeros(self.event_count))
            self.event_classifier_log_scale = nn.Parameter(self._initial_logit_scale.clone())
        else:
            self.tail_classifier = nn.Linear(self.embed_size, self.event_count)
            self.event_classifier_bias = None
            self.event_classifier_log_scale = nn.Parameter(self._initial_logit_scale.clone())
        self.loss_fn = nn.CrossEntropyLoss()
        self.time_filter_conv = HypergraphTransformer(  # [MODIFIED-MODEL]
            in_channels=self.embed_size,
            out_channels=self.embed_size,
            attn_heads=cfg.conv_args.num_attention_heads,
            residual_beta=cfg.conv_args.residual_beta,
            learn_beta=cfg.conv_args.learn_beta,
            dropout=cfg.conv_args.conv_dropout_rate,
            trans_method=cfg.conv_args.trans_method,
            edge_fusion_mode=self.edge_fusion_mode,
            time_fusion_mode=self.time_fusion_mode,
            head_fusion_mode=cfg.conv_args.head_fusion_mode,
            residual_fusion_mode=cfg.conv_args.residual_fusion_mode,
            edge_dim=self.embed_size if self.edge_fusion_mode == "concat" else self.embed_size,
            rel_embed_dim=self.embed_size,
            time_embed_dim=continuous_dim,
            dist_embed_dim=continuous_dim,
            negative_slope=cfg.conv_args.negative_slope,
            have_query_feature=False,
        )
        self.time_filter_norm = nn.BatchNorm1d(self.embed_size)  # [MODIFIED-MODEL]
        self.time_filter_dropout = nn.Dropout(self.dropout_rate)  # [MODIFIED-MODEL]

    def forward(self, batch) -> dict:
        n_id = batch.n_id.to(self.device)
        x = self._gather_embeddings(n_id)
        self._assert_finite(x, "gather_embeddings")

        adjs: List[Optional[SparseTensor]] = [
            adj.to(self.device) if adj is not None else None for adj in batch.adjs_t
        ]
        edge_attrs = [self._to_device(t) for t in batch.edge_attrs]
        edge_types = [self._to_device(t) for t in batch.edge_types]
        edge_delta_ts = [self._to_device(t) for t in batch.edge_delta_ts]
        edge_delta_ss = [self._to_device(t) for t in batch.edge_delta_ss]

        time_filtered = None
        total_layers = len(self.layer_specs)
        offset = max(total_layers - len(adjs), 0)

        for local_idx, adj_t in enumerate(adjs):
            layer_idx = offset + local_idx
            spec = self.layer_specs[layer_idx]
            if adj_t is None or adj_t.nnz() == 0:
                continue

            adj_t = adj_t.to(self.device)
            row, col, _ = adj_t.coo()
            edge_index = torch.stack([col, row], dim=0)
            num_edges = adj_t.nnz()

            edge_type_embed = None
            if edge_types[local_idx] is not None:
                et = edge_types[local_idx].long()
                num_types = int(self.edge_type_embedding.embedding.num_embeddings)
                if et.numel() > 0:
                    et = et.remainder(num_types)
                edge_type_embed = self.edge_type_embedding(et.to(self.device))

            edge_attr_embed = self._encode_attr(layer_idx, edge_attrs[local_idx])
            edge_time_embed = self._encode_time(edge_delta_ts[local_idx], num_edges)
            edge_dist_embed = self._encode_distance(edge_delta_ss[local_idx], num_edges, spec["dist_type"])
            self._assert_finite(edge_attr_embed, f"edge_attr_embed_layer_{layer_idx}")
            self._assert_finite(edge_time_embed, f"edge_time_embed_layer_{layer_idx}")
            self._assert_finite(edge_dist_embed, f"edge_dist_embed_layer_{layer_idx}")
            self._assert_finite(edge_type_embed, f"edge_type_embed_layer_{layer_idx}")

            if spec.get("is_time_filter"):
                num_target = adj_t.sparse_sizes()[0]
                x_target = x[:num_target]
                x = self.time_filter_conv(
                    (x, x_target),
                    edge_index=edge_index,
                    edge_attr_embed=edge_attr_embed,
                    edge_time_embed=edge_time_embed,
                    edge_dist_embed=edge_dist_embed,
                    edge_type_embed=edge_type_embed,
                )
                x = self._apply_norm(self.time_filter_norm, x)
                x = self.activation(x)
                x = self.time_filter_dropout(x)
                self._assert_finite(x, f"time_filter_layer_{layer_idx}")
                time_filtered = x
                continue

            conv_layer = self.convs[layer_idx]
            if conv_layer is None:
                continue

            if layer_idx == len(self.layer_specs) - 1:
                num_target = batch.candidates.size(0)
                if time_filtered is not None:
                    x_target = time_filtered[:num_target]
                else:
                    x_target = x[:num_target]
            else:
                num_target = adj_t.sparse_sizes()[0]
                x_target = x[:num_target]

            conv_input = (x, x_target) if spec["have_query"] else x
            x = conv_layer(
                conv_input,
                edge_index=edge_index,
                edge_attr_embed=edge_attr_embed,
                edge_time_embed=edge_time_embed,
                edge_dist_embed=edge_dist_embed,
                edge_type_embed=edge_type_embed,
            )
            x = self._apply_norm(self.norms[layer_idx], x)
            x = self.activation(x)
            x = self.dropouts[layer_idx](x)
            self._assert_finite(x, f"conv_layer_{layer_idx}")

        candidate_matrix = batch.candidates.to(self.device)
        batch_size = candidate_matrix.size(0)

        chain_embed = x[:batch_size]
        self._assert_finite(chain_embed, "chain_embed_before_logits")
        logits_full = self._compute_logits(chain_embed)
        candidate_scores = logits_full.gather(1, candidate_matrix)
        gold_scores = candidate_scores[:, :1]
        if candidate_matrix.size(1) > 1:
            negative_scores = candidate_scores[:, 1:]
        else:
            negative_scores = torch.zeros(batch_size, 0, device=self.device)

        margin_mean = torch.tensor(0.0, device=self.device)
        if self.use_full_softmax:
            targets = batch.y.to(self.device)
            base_ce = self.loss_fn(logits_full, targets)
            loss = base_ce
        else:
            targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            base_ce = self.loss_fn(candidate_scores, targets)
            loss = base_ce
            if not self.use_full_softmax and negative_scores.numel() > 0 and getattr(self, "margin_loss_enable", True):
                import torch.nn.functional as F
                hardest_neg, _ = negative_scores.max(dim=1, keepdim=True)
                margin = float(getattr(self, "margin_value", 0.2))
                margin_weight = float(getattr(self, "margin_weight", 0.5))
                margin_term = F.relu(margin - gold_scores + hardest_neg)
                margin_mean = margin_term.mean()
                loss = loss + margin_weight * margin_mean
                # 额外返回 hardest_gap（正样本 - 最强负样本 的均值）
                hardest_gap_mean = (gold_scores - hardest_neg).mean().detach()
            else:
                hardest_gap_mean = torch.tensor(0.0, device=self.device)
        
        
        return {
            "loss": loss,
            "ce_loss": base_ce.detach(),
            "logits": logits_full,
            "gold_scores": gold_scores,
            "negative_scores": negative_scores,
            "margin_mean": margin_mean.detach(),
            "hardest_gap": hardest_gap_mean,
        }

    def _gather_embeddings(self, n_id: Tensor) -> Tensor:
        entity_mask = n_id < self.event_offset
        event_mask = (n_id >= self.event_offset) & (n_id < self.chain_offset)
        chain_mask = n_id >= self.chain_offset

        entity_idx = (n_id[entity_mask] - self.entity_offset).long()
        event_idx = (n_id[event_mask] - self.event_offset).long()
        chain_idx = (n_id[chain_mask] - self.chain_offset).long()

        def safe_gather(feat: Tensor, idx: Tensor, name: str) -> Tensor:
            if idx.numel() == 0:
                return feat.new_zeros((0, feat.size(1)))
            valid = (idx >= 0) & (idx < feat.size(0))
            if not bool(valid.all()):
                try:
                    if not hasattr(self, f"_warned_{name}"):
                        setattr(self, f"_warned_{name}", True)
                        invalid_count = int((~valid).sum().item())
                        total = int(idx.numel())
                        print(f"[Warn] {name} 索引越界: {invalid_count}/{total} feat_shape={tuple(feat.shape)} idx_min={int(idx.min().item())} idx_max={int(idx.max().item())}")
                except Exception:
                    pass
            idx = idx.clamp_(0, feat.size(0) - 1)
            return feat[idx]

        entity_feat = safe_gather(self.entity_features, entity_idx, "entity")
        event_feat = safe_gather(self.event_features, event_idx, "event")
        chain_feat = safe_gather(self.chain_features, chain_idx, "chain")

        entity_emb, event_emb, chain_emb = self.embedding(
            entity_feat.to(self.device),
            event_feat.to(self.device),
            chain_feat.to(self.device),
        )

        out = torch.zeros(n_id.size(0), self.embed_size, device=self.device)
        out[entity_mask] = entity_emb
        out[event_mask] = event_emb
        out[chain_mask] = chain_emb
        return out

    def _compute_logits(self, chain_embed: Tensor) -> Tensor:  # [ADDED-MODEL]
        self._assert_finite(chain_embed, "compute_logits.chain_embed_input")
        if self.tie_event_classifier:
            event_embeddings = (self.embedding.event_proj[0](self.event_features) if hasattr(self.embedding, "event_proj") and isinstance(self.embedding.event_proj, nn.Sequential) and isinstance(self.embedding.event_proj[0], nn.Linear) else self.embedding.event_proj(self.event_features))  # [ADDED-MODEL-FIX]
            self._assert_finite(event_embeddings, "compute_logits.event_embeddings_raw")
            if self.normalize_event_classifier:
                event_embeddings = F.normalize(event_embeddings, dim=1, eps=1e-12)  # [ADDED-MODEL-FIX]
                chain_embed_proj = F.normalize(chain_embed, dim=1, eps=1e-12)  # [ADDED-MODEL-FIX]
            else:
                chain_embed_proj = chain_embed  # [ADDED-MODEL]
            self._assert_finite(event_embeddings, "compute_logits.event_embeddings_norm")
            self._assert_finite(chain_embed_proj, "compute_logits.chain_embed_proj")
            logits = chain_embed_proj @ event_embeddings.t()  # [ADDED-MODEL]
            scale = self.event_classifier_log_scale.exp()  # [ADDED-MODEL]
            logits = logits * scale  # [ADDED-MODEL]
            logits = logits + self.event_classifier_bias  # [ADDED-MODEL]
            self._assert_finite(logits, "compute_logits.logits_tied")
            return logits  # [ADDED-MODEL]
        logits = self.tail_classifier(chain_embed)  # [ADDED-MODEL]
        if self.event_classifier_log_scale is not None:
            logits = logits * self.event_classifier_log_scale.exp()
        self._assert_finite(logits, "compute_logits.logits_linear")
        return logits  # [ADDED-MODEL]

    def _encode_attr(self, layer_idx: int, attr: Optional[Tensor]) -> Optional[Tensor]:
        mlp = self.attr_mlps[layer_idx]
        if isinstance(mlp, nn.Identity) or attr is None:
            return None
        attr = attr.to(self.device).float()
        if attr.dim() == 1:
            attr = attr.unsqueeze(-1)
        if isinstance(mlp, nn.Linear) and attr.size(1) != mlp.in_features:
            new_layer = nn.Linear(attr.size(1), self.embed_size).to(self.device)
            nn.init.xavier_uniform_(new_layer.weight)
            if new_layer.bias is not None:
                nn.init.zeros_(new_layer.bias)
            self.attr_mlps[layer_idx] = new_layer
            mlp = new_layer
        return mlp(attr)

    def _encode_time(self, delta_t: Optional[Tensor], num_edges: int) -> Tensor:
        if num_edges == 0:
            return torch.zeros(0, self.time_encoder.expand_dim, device=self.device)
        if delta_t is None:
            delta_t = torch.zeros(num_edges, device=self.device)
        # delta_t 目前为“天”，为了保持 TimeEncoder 稳定，这里进一步缩放
        delta_t = delta_t.to(self.device).float() / max(self.time_input_scale, 1e-6)
        return self.time_encoder(delta_t)

    def _encode_distance(self, delta_s: Optional[Tensor], num_edges: int, dist_type: str) -> Tensor:
        if num_edges == 0:
            if self.distance_encoder_type in ("stan", "hstlstm", "simple"):
                dim = self.distance_encoder.dist_dim
            else:
                dim = self.distance_encoder.expand_dim
            return torch.zeros(0, dim, device=self.device)
        if delta_s is None:
            delta_s = torch.zeros(num_edges, device=self.device)
        delta_s = delta_s.to(self.device).float()
        if self.distance_encoder_type == "stan":
            return self.distance_encoder(delta_s, dist_type=dist_type if dist_type in ("ch2tj", "tj2tj") else "ch2tj")
        elif self.distance_encoder_type in ("hstlstm", "simple"):
            return self.distance_encoder(delta_s)
        elif self.distance_encoder_type == "time":
            return self.distance_encoder(delta_s)
        else:
            raise ValueError(f"Unsupported distance encoder type: {self.distance_encoder_type}")

    def _to_device(self, tensor: Optional[Tensor]) -> Optional[Tensor]:
        if tensor is None:
            return None
        return tensor.to(self.device)

    def _assert_finite(self, tensor: Optional[Tensor], context: str) -> None:
        if tensor is None:
            return
        mask = ~torch.isfinite(tensor)
        if mask.any().item():
            num_nan = torch.isnan(tensor).sum().item()
            num_inf = torch.isinf(tensor).sum().item()
            stats = {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "nan": num_nan,
                "inf": num_inf,
                "min": float(torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).min().item())
                if tensor.numel() > 0
                else 0.0,
                "max": float(torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).max().item())
                if tensor.numel() > 0
                else 0.0,
            }
            raise RuntimeError(f"[STHGCN] Non-finite values detected in {context}: {stats}")

    def _apply_norm(self, norm: nn.Module, x: Tensor) -> Tensor:
        """Safely apply normalization. If BatchNorm1d receives batch of size 1 during
        training, fall back to using running stats (training=False) to avoid errors.
        """
        if isinstance(norm, nn.BatchNorm1d) and self.training and x.dim() == 2 and x.size(0) < 2:
            return F.batch_norm(
                x,
                norm.running_mean,
                norm.running_var,
                norm.weight,
                norm.bias,
                training=False,
                momentum=norm.momentum,
                eps=norm.eps,
            )
        return norm(x)



