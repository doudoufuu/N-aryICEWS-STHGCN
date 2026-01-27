import math
import os
import torch
from torch import nn
import numpy as np
from utils import cal_slot_distance_batch
# 导入日志记录库
import logging
class PositionEncoder(nn.Module):
    def __init__(self, d_model, device, dropout=0.1, max_len=500):
        super(PositionEncoder, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (- math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


# -----------------------------------------
# ⚡ 改进后的 TimeEncoder（内存友好版本）
# -----------------------------------------
class TimeEncoder(nn.Module):
    """
    Memory-efficient TimeEncoder.

    优化点：
    - 把 basis * ts + phase 及 cos 计算在 compute_device（可为 CPU）上分块完成，避免一次性在 GPU 上展开 [N, time_dim]。
    - 支持 chunk_size 控制分块大小，支持 compute_on_cpu（在 CPU 完成大部分计算）。
    - 支持 out_dtype（如 torch.float16）降低显存。
    - use_linear_trans 仍会用 Linear，在需要时把 chunk 移到目标 device 上做投影（分块进行）。
    """
    def __init__(self, args, embedding_dim,
                 chunk_size: int = 65536,
                 compute_on_cpu: bool = False,
                 out_dtype=torch.float32):
        super(TimeEncoder, self).__init__()
        self.time_dim = int(embedding_dim)
        self.expand_dim = int(embedding_dim)
        self.factor = getattr(args, "phase_factor", None)
        self.use_linear_trans = getattr(args, "use_linear_trans", False)

        # ⚡ basis 和 phase 使用 float32（不训练或可训练，按需求修改 requires_grad）
        basis_np = (1.0 / 10 ** np.linspace(0, 9, self.time_dim)).astype(np.float32)
        self.basis_freq = nn.Parameter(torch.from_numpy(basis_np), requires_grad=False)
        self.phase = nn.Parameter(torch.zeros(self.time_dim, dtype=torch.float32), requires_grad=False)

        if self.use_linear_trans:
            self.dense = nn.Linear(self.time_dim, self.expand_dim, bias=False)
            nn.init.xavier_normal_(self.dense.weight)

        # chunk 与计算设备策略
        self.chunk_size = max(1, int(chunk_size))
        self.compute_on_cpu = bool(compute_on_cpu)
        self.out_dtype = out_dtype  # 输出 dtype（torch.float32 / torch.float16）

        # debug 开关（设 False 可减少日志）
        self.debug = False

    def forward(self, ts: torch.Tensor):
        """
        :param ts: 输入时间张量，可能是 [N], [N,1] 或 [N, K]（取第一列）
        :return: [N, expand_dim] 或 [N, time_dim]（若 use_linear_trans=False）
        """
        if self.debug:
            logging.info("=== TimeEncoder Debug ===")
            logging.info("Input ts shape:", ts.shape, "dtype:", ts.dtype, "device:", ts.device)

        # 规范 ts：若多维取第一列；然后展平成 [N]
        if ts.dim() > 1 and ts.size(1) > 1:
            ts = ts[:, 0]
        ts = ts.view(-1)  # [N]

        N = ts.size(0)
        target_device = ts.device  # 最终要与模型一致
        # 选择在何处做映射计算：CPU 或 原始 device
        compute_device = torch.device('cpu') if self.compute_on_cpu else target_device

        # ⚡ 把 basis/phase 移到 compute_device 的非持久副本（不修改 self.* 的 device）
        basis = self.basis_freq.detach().to(compute_device)
        phase = self.phase.detach().to(compute_device)

        # 如果使用线性投影，保证 dense 在 target_device（GPU）上，以便投影使用 GPU 权重
        dense = None
        if self.use_linear_trans:
            dense = self.dense.to(target_device)

        out_chunks_cpu = []  # 当 compute_on_cpu=True 时，这里存放 CPU 上的 chunk（或 compute_device 上的）
        # 分块计算，避免一次性产生巨型张量
        chunk = max(1, int(self.chunk_size))
        for i in range(0, N, chunk):
            j = min(N, i + chunk)
            # ts_chunk 在 compute_device 上
            ts_chunk = ts[i:j].to(compute_device).unsqueeze(-1).to(torch.float32)  # [chunk,1]
            # map_ts = ts_chunk * basis + phase  -> [chunk, time_dim]
            map_ts = ts_chunk * basis.view(1, -1)
            map_ts = map_ts + phase.view(1, -1)

            # harmonic 在 compute_device 上
            harmonic = torch.cos(map_ts)  # [chunk, time_dim]

            if self.use_linear_trans:
                # 投影需要 dense（在 target_device），所以把 harmonic 移到 target_device 后再投影
                harmonic_device = harmonic.to(target_device)
                harmonic_device = harmonic_device.type(dense.weight.dtype)
                projected = dense(harmonic_device)  # [chunk, expand_dim] (在 target_device)
                # 统一 dtype 并放回 out_chunks_cpu 的 device：我们把所有 chunk 保存在 compute_device 上或 target 上，后面再统一处理
                # 为避免额外 GPU 临时占用，这里把 projected 移回 CPU if compute_on_cpu True
                if compute_device.type == 'cpu':
                    out_chunks_cpu.append(projected.to(torch.float32).cpu())  # keep cpu
                    # 释放临时 GPU 内存
                    if target_device.type == 'cuda':
                        try:
                            torch.cuda.synchronize(target_device)
                        except Exception:
                            pass
                        torch.cuda.empty_cache()
                else:
                    # compute_device == target_device
                    out_chunks_cpu.append(projected.to(self.out_dtype))
            else:
                # 不投影，直接保留 harmonic（在 compute_device）
                if compute_device.type == 'cpu':
                    out_chunks_cpu.append(harmonic.cpu())
                else:
                    out_chunks_cpu.append(harmonic.to(self.out_dtype))

        # ⚡ 此处 out_chunks_cpu 中的张量位于 compute_device（通常为 CPU）或 target_device（若 compute_on_cpu=False）
        # 我们需要返回最终在 target_device 上的张量（因为后续 conv 期望 GPU 张量）。
        # 为避免一次性产生双倍峰值显存：按 chunk 将数据搬到 target_device 并拼接。
        if compute_device.type == 'cpu' and target_device.type == 'cuda':
            # 把 CPU chunks 分块搬到 GPU（每次只占 chunk 的显存），再拼接成最终张量
            out_device_chunks = []
            for c in out_chunks_cpu:
                # c 在 CPU，逐块搬到 GPU，并 cast 到 out_dtype
                out_device_chunks.append(c.to(target_device).to(self.out_dtype))
                # 尝试释放 CPU 侧缓存（python GC 由调用者控制）
                try:
                    torch.cuda.synchronize(target_device)
                except Exception:
                    pass
                torch.cuda.empty_cache()
            out = torch.cat(out_device_chunks, dim=0)
            # 手动 free 临时列表引用
            del out_device_chunks
            del out_chunks_cpu
        else:
            # compute_device == target_device 或两者都是 CPU：直接 cat
            out = torch.cat(out_chunks_cpu, dim=0).to(target_device).to(self.out_dtype)

        if self.debug:
            logging.info("Final output shape:", out.shape)
            logging.info("=== TimeEncoder Debug End ===\n")
        return out


# -----------------------------------------
# Distance Encoders（同样做分块 / 内存友好）
# -----------------------------------------
class DistanceEncoderHSTLSTM(nn.Module):
    def __init__(self, args, embedding_dim, spatial_slots, chunk_size: int = 2048):
        super(DistanceEncoderHSTLSTM, self).__init__()
        self.dist_dim = embedding_dim
        self.spatial_slots = spatial_slots
        self.embed_q = nn.Embedding(len(spatial_slots), self.dist_dim)
        self.chunk_size = max(1, int(chunk_size))

    def place_parameters(self, ld, hd, l, h, device):
        ld = torch.from_numpy(np.array(ld)).to(device=device, dtype=torch.float32)
        hd = torch.from_numpy(np.array(hd)).to(device=device, dtype=torch.float32)
        l = torch.from_numpy(np.array(l)).to(device=device, dtype=torch.long)
        h = torch.from_numpy(np.array(h)).to(device=device, dtype=torch.long)
        return ld, hd, l, h

    def cal_inter(self, ld, hd, l, h, embed):
        l_embed = embed(l)
        h_embed = embed(h)
        return torch.stack([hd], -1) * l_embed + torch.stack([ld], -1) * h_embed

    def forward(self, dist):
        device = self.embed_q.weight.device
        dist = dist.contiguous().view(-1).to(device).to(torch.float32)
        slots = sorted(self.spatial_slots)

        d_ld, d_hd, d_l, d_h = cal_slot_distance_batch(dist, slots)
        d_ld, d_hd, d_l, d_h = self.place_parameters(d_ld, d_hd, d_l, d_h, device)

        N = dist.size(0)
        out = torch.empty((N, self.dist_dim), dtype=torch.float32, device=device)

        start = 0
        while start < N:
            end = min(start + self.chunk_size, N)
            ld_chunk = d_ld[start:end]
            hd_chunk = d_hd[start:end]
            l_chunk = d_l[start:end]
            h_chunk = d_h[start:end]
            l_embed = self.embed_q(l_chunk)
            h_embed = self.embed_q(h_chunk)
            out[start:end] = torch.stack([hd_chunk], -1) * l_embed + torch.stack([ld_chunk], -1) * h_embed
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            start = end

        return out


class DistanceEncoderSTAN(nn.Module):
    def __init__(self, args, embedding_dim, spatial_slots, chunk_size: int = 2048):
        super(DistanceEncoderSTAN, self).__init__()
        self.dist_dim = embedding_dim
        self.min_d, self.max_d_ch2tj, self.max_d_tj2tj = spatial_slots
        self.embed_min = nn.Embedding(1, self.dist_dim)
        self.embed_max = nn.Embedding(1, self.dist_dim)
        self.embed_max_traj = nn.Embedding(1, self.dist_dim)
        self.quantile = getattr(args, "quantile", 0.99)
        self.chunk_size = max(1, int(chunk_size))

    def forward(self, dist, dist_type='ch2tj'):
        device = self.embed_min.weight.device
        dist = dist.contiguous().view(-1).to(device).to(torch.float32)
        N = dist.size(0)

        if dist_type == 'ch2tj':
            emb_low = self.embed_min.weight
            emb_high = self.embed_max.weight
            max_d = self.max_d_ch2tj
        else:
            emb_low = self.embed_min.weight
            emb_high = self.embed_max_traj.weight
            max_d = self.max_d_tj2tj

        out = torch.empty((N, self.dist_dim), dtype=torch.float32, device=device)
        start = 0
        while start < N:
            end = min(start + self.chunk_size, N)
            d_chunk = dist[start:end].clamp(0, max_d).unsqueeze(-1).expand(-1, self.dist_dim)
            vsl = (d_chunk - self.min_d)
            vsu = (max_d - d_chunk)
            emb_low_exp = emb_low.expand(end - start, -1)
            emb_high_exp = emb_high.expand(end - start, -1)
            space_interval = (emb_low_exp * vsu + emb_high_exp * vsl) / (max_d - self.min_d)
            out[start:end] = space_interval
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            start = end
        return out


class DistanceEncoderSimple(nn.Module):
    def __init__(self, args, embedding_dim, spatial_slots, chunk_size: int = 2048):
        super(DistanceEncoderSimple, self).__init__()
        self.args = args
        self.dist_dim = embedding_dim
        self.min_d, self.max_d, self.max_d_traj = spatial_slots
        self.embed_unit = nn.Embedding(1, self.dist_dim)
        self.chunk_size = max(1, int(chunk_size))

    def forward(self, dist):
        device = self.embed_unit.weight.device
        dist = dist.contiguous().view(-1).to(device).to(torch.float32)
        N = dist.size(0)
        out = torch.empty((N, self.dist_dim), dtype=torch.float32, device=device)

        start = 0
        while start < N:
            end = min(start + self.chunk_size, N)
            d_chunk = dist[start:end].unsqueeze(-1).expand(-1, self.dist_dim)
            out[start:end] = d_chunk * self.embed_unit.weight
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            start = end

        return out
