# 导入进度条库，用于显示处理进度
from tqdm import tqdm
# 导入数据处理库
import pandas as pd
# 导入数值计算库
import numpy as np
# 导入稀疏矩阵库
from scipy.sparse import coo_matrix
# 导入PyTorch深度学习框架
import torch
# 导入PyTorch稀疏张量库
from torch_sparse import SparseTensor
# 导入PyTorch Geometric图数据类
from torch_geometric.data import Data
# 导入工具函数中的地理距离计算函数
from utils import haversine, get_root_dir
# 导入操作系统接口
import os
# 导入路径处理工具
import os.path as osp
# 导入日志记录库
import logging
# 导入JSON库，用于保存进度信息
import json
# 导入pickle库，用于保存中间数据
import pickle
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Normalize time-based features to reduce magnitude differences
TIME_SCALE = 86400.0  # seconds per day


def load_checkpoint(checkpoint_file):
    """
    加载检查点信息（使用pickle格式）
    
    :param checkpoint_file: 检查点文件路径
    :return: 检查点信息字典，如果文件不存在或损坏返回None
    """
    if osp.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"[Checkpoint] 加载检查点: {checkpoint}")
            return checkpoint
        except Exception as e:
            print(f"[Checkpoint] 检查点文件损坏，删除并重新开始: {e}")
            try:
                os.remove(checkpoint_file)
                # 同时删除相关的数据文件
                data_file = checkpoint_file.replace('.pkl', '_data.pkl')
                if osp.exists(data_file):
                    os.remove(data_file)
            except Exception as del_e:
                print(f"[Checkpoint] 删除损坏文件失败: {del_e}")
            return None
    return None


def save_checkpoint(checkpoint_file, step, data=None, **kwargs):
    """
    保存检查点信息（使用pickle格式）
    
    :param checkpoint_file: 检查点文件路径
    :param step: 当前步骤名称
    :param data: 需要保存的数据（可选）
    :param kwargs: 其他需要保存的信息
    """
    # 将.json后缀改为.pkl
    if checkpoint_file.endswith('.json'):
        checkpoint_file = checkpoint_file.replace('.json', '.pkl')
    
    checkpoint = {
        'step': step,
        'timestamp': pd.Timestamp.now().isoformat(),
        **kwargs
    }
    
    try:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"[Checkpoint] 保存检查点: {step}")
        
        # 如果有数据需要保存，保存到单独的文件
        if data is not None:
            data_file = checkpoint_file.replace('.pkl', '_data.pkl')
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"[Checkpoint] 保存数据: {data_file}")
    except Exception as e:
        print(f"[Checkpoint] 保存检查点失败: {e}")
        # 打印更详细的错误信息
        import traceback
        print(f"[Checkpoint] 错误详情: {traceback.format_exc()}")


def generate_hypergraph_from_file(input_file, output_path, args):
    """
    从原始记录构建[签到点->轨迹]的关联矩阵和[轨迹->轨迹]的邻接表
    边索引格式如下：
        [[ -签到点- ]
         [ -轨迹(超边)]]
    和
        [[ -轨迹(超边)- ]
         [ -轨迹(超边)]]
    分别保存。

    使用txt文件中的列进行下一个POI任务：
        UserId, check_ins_id, PoiId, latitude, longitude, PoiCategoryId, UTCTimeOffsetEpoch,
        pseudo_session_trajectory_id, UTCTimeOffsetWeekday, UTCTimeOffsetHour.

    两部分将保存为两个.pt文件。

    :param input_file: 超图原始数据路径
    :param output_path: pyg数据.pt输出目录
    :param args: 解析的输入参数
    :return: None
    """
    print(f"[Hypergraph] 开始生成三层超图 | input: {input_file} | out: {output_path}")
    
    # 创建检查点文件路径（使用.pkl扩展名）
    checkpoint_file = osp.join(output_path, 'hypergraph_checkpoint.pkl')
    
    # 尝试加载检查点
    checkpoint = load_checkpoint(checkpoint_file)
    
    # 定义要使用的列名
    usecols = [
        'ID', 'EventChain_id', 'Source_name_encoded', 'Target_name_encoded',
        'Source_Country_encoded', 
        'UTC_time', 'Location_encoded', 'Event_type', 'Intensity','latitude', 'longitude','check_ins_id','UTCTimeOffsetEpoch'
    ]
    
    # 获取阈值参数
    threshold = args.threshold
    # 获取过滤模式参数
    filter_mode = args.filter_mode
    print(f"[Hypergraph] 参数 | threshold={threshold} | filter_mode={filter_mode}")
    
    # 步骤1: 读取数据
    if checkpoint is None or checkpoint.get('step') != 'data_loaded':
        print("[Hypergraph] 步骤1: 读取预处理样本CSV...")
        missing_cols = []
        try:
            header = pd.read_csv(input_file, nrows=0)
            available_cols = [c for c in usecols if c in header.columns]
            missing_cols = [c for c in usecols if c not in header.columns]
            data = pd.read_csv(input_file, usecols=available_cols if available_cols else None)
        except Exception as e:
            print(f"[Hypergraph] 按usecols读取失败，回退全量读取: {e}")
            data = pd.read_csv(input_file)
            missing_cols = [c for c in usecols if c not in data.columns]

        defaults = {
            'Source_Country_encoded': 'UNK_COUNTRY',
            'Target_Country_encoded': 'UNK_COUNTRY',
            'Location_encoded': 'UNK_LOC',
            'latitude': 0.0,
            'longitude': 0.0,
            'Intensity': 0.0,
        }
        for col, val in defaults.items():
            if col not in data.columns:
                data[col] = val

        if 'check_ins_id' not in data.columns:
            data['check_ins_id'] = np.arange(len(data), dtype=np.int64)

        if 'UTCTimeOffsetEpoch' not in data.columns:
            if 'UTCTimeOffset' in data.columns:
                ts = pd.to_datetime(data['UTCTimeOffset'], errors='coerce')
            elif 'UTC_time' in data.columns:
                ts = pd.to_datetime(data['UTC_time'], errors='coerce')
            else:
                ts = None
            if ts is not None:
                data['UTCTimeOffsetEpoch'] = ts.view('int64') // 10**9
            else:
                data['UTCTimeOffsetEpoch'] = 0
        data['UTCTimeOffsetEpoch'] = pd.to_numeric(data['UTCTimeOffsetEpoch'], errors='coerce').fillna(0).astype(np.int64)

        if 'Event_type' in data.columns:
            data['Event_type'] = pd.to_numeric(data['Event_type'], errors='coerce').fillna(0).astype(int)
        if 'Intensity' in data.columns:
            data['Intensity'] = pd.to_numeric(data['Intensity'], errors='coerce').fillna(0.0)

        if missing_cols:
            print(f"[Hypergraph] 缺失列已回填默认值: {missing_cols}")
        print(f"[Hypergraph] 读取完成 | 行数={len(data)} | 列={len(data.columns)}")
        save_checkpoint(checkpoint_file, 'data_loaded', data=data)
    else:
        print("[Hypergraph] 步骤1: 从检查点恢复数据...")
        data_file = checkpoint_file.replace('.pkl', '_data.pkl')
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            print(f"[Hypergraph] 数据恢复完成 | 行数={len(data)}")
        except Exception as e:
            print(f"[Hypergraph] 数据恢复失败: {e}")
            return

    # 修改by hsy设置事件列名、事件链列名
    traj_column = 'ID'
    eventchain_id = 'EventChain_id'

    # 如果为True，在保存到pyg数据之前，轨迹ID会偏移#check_ins_id，这意味着签到点的索引在
    # 范围[0, #checkin_id - 1]内，轨迹的索引在范围[#checkin, #trajectory+#checkin-1]内
    traj_offset = True

    if traj_offset:
        # 计算签到点偏移量（最大签到点ID + 1），这里写成了最大记录点+1，但是感觉没关系，只是会造成节点的不连续
        checkin_offset = torch.as_tensor([data.check_ins_id.max() + 1], dtype=torch.long)
    else:
        # 不偏移，偏移量为0
        checkin_offset = torch.as_tensor([0], dtype=torch.long)
    print(f"[Hypergraph] checkin_offset={int(checkin_offset.item())}")

    # 步骤2: 生成事件超边统计信息
    if checkpoint is None or checkpoint.get('step') not in ['traj_stat_generated', 'eventchain_stat_generated', 'l1_completed', 'l2_completed', 'l3_completed']:
        print("[Hypergraph] 步骤2: 统计事件(超边)信息...")
        try:
            traj_stat = generate_hyperedge_stat(data, traj_column)
            # 保存统计信息到检查点
            save_checkpoint(checkpoint_file, 'traj_stat_generated', traj_stat=traj_stat)
        except Exception as e:
            print(f"[Hypergraph] 生成事件统计失败: {e}")
            return
    else:
        print("[Hypergraph] 步骤2: 从检查点恢复事件统计...")
        try:
            traj_stat = checkpoint.get('traj_stat')
            if traj_stat is None:
                traj_stat = generate_hyperedge_stat(data, traj_column)
        except Exception as e:
            print(f"[Hypergraph] 恢复事件统计失败: {e}")
            return

    # 步骤3: 生成事件链超边的统计信息
    if checkpoint is None or checkpoint.get('step') not in ['eventchain_stat_generated', 'l1_completed', 'l2_completed', 'l3_completed']:
        print("[Hypergraph] 步骤3: 统计事件链(超边)信息...")
        try:
            eventchain_stat = generate_chainhyperedge_stat(data, eventchain_id)
            # 保存统计信息到检查点
            save_checkpoint(checkpoint_file, 'eventchain_stat_generated', eventchain_stat=eventchain_stat)
        except Exception as e:
            print(f"[Hypergraph] 生成事件链统计失败: {e}")
            return
    else:
        print("[Hypergraph] 步骤3: 从检查点恢复事件链统计...")
        try:
            eventchain_stat = checkpoint.get('eventchain_stat')
            if eventchain_stat is None:
                eventchain_stat = generate_chainhyperedge_stat(data, eventchain_id)
        except Exception as e:
            print(f"[Hypergraph] 恢复事件链统计失败: {e}")
            return

    # 步骤4: 生成第一层图（实体-事件超图）
    #主动赋值
    #checkpoint = None
    #ci2traj_pyg_data = None
    if not osp.exists(osp.join(output_path, 'entity_graph.pt')) and (checkpoint is None or checkpoint.get('step') not in ['l1_completed', 'l2_completed', 'l3_completed', 'l4_completed']):
        #如果已经有了entity_graph.pt，就跳过
        print("[Hypergraph] 步骤4: 构建第一层：实体-事件超图...")
        try:
            ci2traj_pyg_data = new_generate_ci2traj_pyg_data(data, traj_stat, traj_column, checkin_offset, output_path)
            print(f"[Hypergraph] L1完成 | x={tuple(ci2traj_pyg_data.x.shape)} | edges={ci2traj_pyg_data.edge_index.size(1)}")
            
            # 保存第一层图
            l1_file = osp.join(output_path, 'entity_graph.pt')
            torch.save(ci2traj_pyg_data, l1_file)
            print(f"[Hypergraph] 已保存: {l1_file}")
            
            save_checkpoint(checkpoint_file, 'l1_completed', l1_file=l1_file)
        except Exception as e:
            print(f"[Hypergraph] 生成第一层图失败: {e}")
            return
    else:
        print("[Hypergraph] 步骤4: 从检查点恢复第一层图...")
        try:
            l1_file = osp.join(output_path, 'entity_graph.pt')
            ci2traj_pyg_data = torch.load(l1_file)
            print(f"[Hypergraph] L1恢复完成 | x={tuple(ci2traj_pyg_data.x.shape)}")
        except Exception as e:
            print(f"[Hypergraph] 恢复第一层图失败: {e}")
            return
    #checkpoint = None
    # 步骤5: 生成第二层图（事件图）
    if checkpoint is None or checkpoint.get('step') not in ['l2_completed', 'l3_completed','', 'l4_completed']:
        if osp.exists(osp.join(output_path, 'event_graph.pt')):
            #跳过
            print("[Hypergraph] 步骤5: 事件图已存在，跳过生成第二层图。")
        else:
            print("[Hypergraph] 步骤5: 构建第二层：事件图...")
            try:
                event_graph = generate_event_graph(data, args, ci2traj_pyg_data, checkin_offset, output_path)
                print(f"[Hypergraph] L2完成 | x={tuple(event_graph.x.shape)} | edges={event_graph.edge_index.size(1)}")
                
                # 保存第二层图
                l2_file = osp.join(output_path, 'event_graph.pt')
                torch.save(event_graph, l2_file)
                print(f"[Hypergraph] 已保存: {l2_file}")
                
                save_checkpoint(checkpoint_file, 'l2_completed', l2_file=l2_file)
            except Exception as e:
                print(f"[Hypergraph] 生成第二层图失败: {e}")
                return
    else:
        print("[Hypergraph] 步骤5: 从检查点恢复第二层图...")
        try:
            l2_file = osp.join(output_path, 'event_graph.pt')
            event_graph = torch.load(l2_file)
            print(f"[Hypergraph] L2恢复完成 | x={tuple(event_graph.x.shape)}")
        except Exception as e:
            print(f"[Hypergraph] 恢复第二层图失败: {e}")
            return
    #生成事件-事件链图
    #第三层图
    if checkpoint is None or checkpoint.get('step') not in ['l3_completed','l4_completed']:
        print("[Hypergraph] 步骤6: 构建第三层：事件-事件链图...")
        try:
            chain_graph = generate_event2chain_graph(data, args, traj_stat, checkin_offset, output_path)
            print(f"[Hypergraph] L3完成 | x={tuple(chain_graph.x.shape)} | edges={chain_graph.edge_index.size(1)}")
            
            # 保存第三层图
            l3_file = osp.join(output_path, 'event2chain_graph.pt')
            torch.save(chain_graph, l3_file)
            print(f"[Hypergraph] 已保存: {l3_file}")
            save_checkpoint(checkpoint_file, 'l3_completed', l3_file=l3_file)
        except Exception as e:
            print(f"[Hypergraph] 生成第三层图失败: {e}")
            return
    else:
        print("[Hypergraph] 步骤6: 从检查点恢复第三层图...")
        try:
            l3_file = osp.join(output_path, 'event2chain_graph.pt')
            chain_graph = torch.load(l3_file)
            print(f"[Hypergraph] L3恢复完成 | x={tuple(chain_graph.x.shape)}")
        except Exception as e:
            print(f"[Hypergraph] 恢复第三层图失败: {e}")
            return
    #checkpoint = None
    # 步骤7: 生成第四层图（事件链图）
    if checkpoint is None or checkpoint.get('step') != 'l4_completed':
        print("[Hypergraph] 步骤7: 构建第四层：事件链图...")
        try:
            # 使用候选+软权重版本的链图构建（保留旧函数以兼容）
            try:
                chain_graph = generate_chain_graph_v2(data, args, eventchain_stat, checkin_offset)
            except Exception as _e:
                print(f"[Hypergraph] L4 v2 失败，回退到旧版: {_e}")
                chain_graph = generate_chain_graph(data, args, eventchain_stat, checkin_offset, alpha=(0.3, 0.2, 0.3, 0.2))
            print(f"[Hypergraph] L4完成 | x={tuple(chain_graph.x.shape)} | edges={chain_graph.edge_index.size(1)}")

            # 保存第四层图
            l4_file = osp.join(output_path, 'chain_graph.pt')
            torch.save(chain_graph, l4_file)
            print(f"[Hypergraph] 已保存: {l4_file}")

            save_checkpoint(checkpoint_file, 'l4_completed', l4_file=l4_file)
        except Exception as e:
            print(f"[Hypergraph] 生成第四层图失败: {e}")
            return
    else:
        print("[Hypergraph] 步骤7: 从检查点恢复第四层图...")
        try:
            l4_file = osp.join(output_path, 'chain_graph.pt')
            chain_graph = torch.load(l4_file)
            print(f"[Hypergraph] L4恢复完成 | x={tuple(chain_graph.x.shape)}")
        except Exception as e:
            print(f"[Hypergraph] 恢复第四层图失败: {e}")
            return

    # 清理检查点文件
    try:
        os.remove(checkpoint_file)
        data_file = checkpoint_file.replace('.pkl', '_data.pkl')
        if osp.exists(data_file):
            os.remove(data_file)
        print("[Hypergraph] 清理检查点文件完成")
    except Exception as e:
        print(f"[Hypergraph] 清理检查点文件失败: {e}")

    # 记录日志信息
    logging.info('[Hypergraph] Done generating three-layer hypergraph data.')
    print("[Hypergraph] 三层超图生成完成")
    return



def jaccard(set1, set2):
    """计算两个集合的 Jaccard 相似度"""
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


def calculate_chain_similarity(chain1, chain2, alpha=(0.3, 0.2, 0.3, 0.2)):
    """
    计算两个事件链之间的相似度，基于四个加权因素
    """
    #print(f"DEBUG: alpha type={type(alpha)}, alpha={alpha}")
    alpha_overlap, alpha_country, alpha_type, alpha_time = alpha

    # 提取事件集合
    ids1 = set(chain1['ID'].values)
    ids2 = set(chain2['ID'].values)
    overlap_score = jaccard(ids1, ids2)
    if overlap_score == 0.0:
        return None  # 无事件交集，忽略该边

    # 国家是否一致
    countries1 = set(chain1['Source_Country_encoded'].values)
    countries2 = set(chain2['Source_Country_encoded'].values)
    country_score = 1.0 if countries1 & countries2 else 0.0

    # 事件类型相似度
    types1 = set(chain1['Event_type'].values)
    types2 = set(chain2['Event_type'].values)
    type_score = jaccard(types1, types2)

    # 平均时间差
    time1 = pd.to_datetime(chain1['UTC_time']).astype(int).mean()  # nanosecond
    time2 = pd.to_datetime(chain2['UTC_time']).astype(int).mean()
    time_diff_hr = abs(time1 - time2) / 1e9 / 3600
    time_score = 1 / (1 + time_diff_hr)

    # 加权组合
    weight = (
        alpha_overlap * overlap_score +
        alpha_country * country_score +
        alpha_type * type_score +
        alpha_time * time_score
    )
    return weight

import os.path as osp
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import pickle

def generate_chain_graph(data, args, eventchain_stat, checkin_offset, alpha=(0.3, 0.2, 0.3, 0.2)):
    """
    构建事件链图（第四层图）
    """
    try:
        checkpoint_file = 'l3_chain_graph_checkpoint.pkl'
        checkpoint = load_checkpoint(checkpoint_file)

        # 对齐事件链统计索引：仅保留实际出现的链，并使索引连续
        chain_groups = data.groupby('EventChain_id', sort=True)
        chain_ids = list(chain_groups.groups.keys())
        if not chain_ids:
            print("[ChainGraph] 无事件链, 跳过构建")
            return Data()

        try:
            eventchain_stat = eventchain_stat.reindex(chain_ids).fillna(0)
            eventchain_stat.index = range(len(eventchain_stat))
        except Exception as _e:
            print(f"[ChainGraph] 处理 eventchain_stat 失败: {_e}")
            eventchain_stat = eventchain_stat.copy()

        chain_data_list = []
        chain_event_sets = []
        event_to_chains = defaultdict(set)
        for idx, chain_id in enumerate(chain_ids):
            chain_df = chain_groups.get_group(chain_id)
            chain_data_list.append(chain_df)
            event_ids = set(chain_df['ID'].dropna().tolist())
            chain_event_sets.append(event_ids)
            for event_id in event_ids:
                event_to_chains[event_id].add(idx)

        mean_time_arr = eventchain_stat['mean_time'].to_numpy(dtype=float)
        mean_lat_arr = eventchain_stat['mean_lat'].to_numpy(dtype=float)
        mean_lon_arr = eventchain_stat['mean_lon'].to_numpy(dtype=float)

        chain_features = eventchain_stat[['size', 'mean_lon', 'mean_lat',
                                        'last_lon','last_lat',
                                        'mean_time', 'start_time', 'end_time']].to_numpy(dtype=float)
        chain_features[:, 5:8] = chain_features[:, 5:8] / TIME_SCALE

        x = torch.tensor(chain_features, dtype=torch.float)

        # chain_features = eventchain_stat[['size', 'mean_lon', 'mean_lat', 'last_lon','last_lat','mean_time', 'start_time', 'end_time']].to_numpy()
        # x = torch.tensor(chain_features, dtype=torch.float)

        num_chains = len(chain_ids)
        print(f"[ChainGraph] 事件链数量: {num_chains}")

        edge_index_list = []
        edge_attr_list = []
        delta_t_list = []   # ★ 新增
        delta_s_list = []   # ★ 新增

        start_chunk = 0
        if checkpoint is not None and checkpoint.get('step') == 'chain_graph_partial':
            start_chunk = checkpoint.get('completed_chunk', 0)
            print(f"[ChainGraph] 从检查点恢复，已处理 {start_chunk} 个chunk")
            try:
                data_file = checkpoint_file.replace('.pkl', '_data.pkl')
                with open(data_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    edge_index_list = saved_data['edge_index_list']
                    edge_attr_list = saved_data['edge_attr_list']
                    delta_t_list = saved_data.get('delta_t_list', [])   # ★ 新增
                    delta_s_list = saved_data.get('delta_s_list', [])   # ★ 新增
            except Exception as e:
                print(f"[ChainGraph] 恢复进度失败: {e}")
                start_chunk = 0
                edge_index_list, edge_attr_list, delta_t_list, delta_s_list = [], [], [], []

        chunk_size = 100
        total_chunks = (num_chains + chunk_size - 1) // chunk_size

        for i in tqdm(range(start_chunk, total_chunks), desc='Generating chain graph'):
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, num_chains)

            for j in range(chunk_start, chunk_end):
                chain_j = chain_data_list[j]
                candidate_indices = set()
                events_in_chain = chain_event_sets[j]
                for event_id in events_in_chain:
                    for neighbor_idx in event_to_chains[event_id]:
                        if neighbor_idx > j:
                            candidate_indices.add(neighbor_idx)

                if not candidate_indices:
                    continue

                for k in sorted(candidate_indices):
                    chain_k = chain_data_list[k]
                    idx_src, idx_dst = j, k
                    chain_src, chain_dst = chain_j, chain_k
                    #先根据时间做校验
                    if chain_src['UTCTimeOffsetEpoch'].iloc[0] > chain_dst['UTCTimeOffsetEpoch'].iloc[0]:
                        idx_src, idx_dst = idx_dst, idx_src
                        chain_src, chain_dst = chain_dst, chain_src
                    sim = calculate_chain_similarity(chain_src, chain_dst, alpha=alpha)
                    if sim is not None and sim >= args.threshold:
                        edge_index_list.append( [idx_src+int(checkin_offset*2), idx_dst+int(checkin_offset*2)] )  # 节点类型偏移
                        edge_attr_list.append([sim])

                        # 记录平均时间差与空间差（haversine）
                        # 使用纬度经度的排列顺序一致，方便之后查表
                        if idx_src < len(eventchain_stat) and idx_dst < len(eventchain_stat):
                            mean_time_j = mean_time_arr[idx_src]
                            mean_lat_j  = mean_lat_arr[idx_src]
                            mean_lon_j  = mean_lon_arr[idx_src]

                            mean_time_k = mean_time_arr[idx_dst]
                            mean_lat_k  = mean_lat_arr[idx_dst]
                            mean_lon_k  = mean_lon_arr[idx_dst]
                        else:
                            continue

                        dt = abs(mean_time_j - mean_time_k)
                        ds = haversine(mean_lon_j, mean_lat_j, mean_lon_k, mean_lat_k)  # 用 大圆距离

                        delta_t_list.append([dt])

                        delta_s_list.append([ds])

            if (i + 1) % 10 == 0:
                save_checkpoint(
                    checkpoint_file, 
                    'chain_graph_partial', 
                    data={
                        'edge_index_list': edge_index_list, 
                        'edge_attr_list': edge_attr_list,
                        'delta_t_list': delta_t_list,   # ★ 新增
                        'delta_s_list': delta_s_list    # ★ 新增
                    },
                    completed_chunk=i+1
                )
                print(f"[ChainGraph] 已处理 {i+1}/{total_chunks} 个chunk，当前边数: {len(edge_index_list)}")

        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
            chain_edge_delta_t = torch.tensor(delta_t_list, dtype=torch.float) / TIME_SCALE  # ★ 新增
            chain_edge_delta_s = torch.tensor(delta_s_list, dtype=torch.float)  # ★ 新增
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
            chain_edge_delta_t = torch.empty((0,1), dtype=torch.float)  # ★ 新增
            chain_edge_delta_s = torch.empty((0,1), dtype=torch.float)  # ★ 新增

        print(f"[ChainGraph] 构图完成 | edges={edge_index.size(1)} | nodes={num_chains}")

        pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            chain_edge_delta_t=chain_edge_delta_t,   # ★ 新增
            edge_delta_t=chain_edge_delta_t,
            chain_edge_delta_s=chain_edge_delta_s,   # ★ 新增
            num_nodes=len(chain_features)
        )
        print(f"[ChainGraph] 特征矩阵: x={tuple(x.shape)}")

        try:
            if osp.exists(checkpoint_file):
                os.remove(checkpoint_file)
            data_file = checkpoint_file.replace('.pkl', '_data.pkl')
            if osp.exists(data_file):
                os.remove(data_file)
            print("[ChainGraph] 清理检查点文件完成")
        except Exception as e:
            print(f"[ChainGraph] 清理检查点文件失败: {e}")

        return pyg_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

    try:
        pyg_data.chain_offset = int(_chain_base)
    except Exception:
        pass
    return pyg_data


def generate_chain_graph_v2(data, args, eventchain_stat, checkin_offset):
    """
    候选+软权重版本：仅离线产生较宽松的候选边与多因子边特征；
    在线由模型注意力学习边权。方向约束：仅早->晚；可选 Top-K 控制密度。
    edge_attr: [src_jacc, tgt_jacc, type_jacc, same_country, time_score, space_score, interact_flag]
    """
    from collections import defaultdict

    chain_groups = data.groupby("EventChain_id", sort=True)
    chain_ids = list(chain_groups.groups.keys())
    if not chain_ids:
        return Data(
            x=torch.empty((0, 8), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )

    # 节点特征
    eventchain_stat = eventchain_stat.reindex(chain_ids).fillna(0)
    eventchain_stat.index = range(len(eventchain_stat))
    chain_features = eventchain_stat[
        ["size", "mean_lon", "mean_lat", "last_lon", "last_lat", "mean_time", "start_time", "end_time"]
    ].to_numpy(dtype=float)
    chain_features[:, 5:8] = chain_features[:, 5:8] / TIME_SCALE
    x = torch.tensor(chain_features, dtype=torch.float)

    num_chains = len(chain_ids)
    mean_time_arr = eventchain_stat["mean_time"].to_numpy(dtype=float)
    mean_lat_arr = eventchain_stat["mean_lat"].to_numpy(dtype=float)
    mean_lon_arr = eventchain_stat["mean_lon"].to_numpy(dtype=float)

    # 配置
    bucket_days = int(getattr(args, "chain_time_bucket_days", 30) or 30)
    grid_deg = float(getattr(args, "chain_geo_grid_deg", 2.0) or 2.0)
    topk_per_node = int(getattr(args, "chain_candidate_topk", 30) or 30)

    source_sets: List[Set] = []
    target_sets: List[Set] = []
    type_sets: List[Set] = []
    country_sets: List[Set] = []
    source_to_chains: Dict[object, Set[int]] = defaultdict(set)
    target_to_chains: Dict[object, Set[int]] = defaultdict(set)
    country_to_chains: Dict[object, Set[int]] = defaultdict(set)
    time_bucket_to_chains: Dict[int, Set[int]] = defaultdict(set)
    geo_cell_to_chains: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    chain_event_sets: List[Set[object]] = []
    chain_data_list: List[pd.DataFrame] = []
    event_to_chains: Dict[object, Set[int]] = defaultdict(set)

    for idx, cid in enumerate(chain_ids):
        cdf = chain_groups.get_group(cid)
        chain_data_list.append(cdf)
        event_ids = set(cdf["ID"].dropna().tolist())
        chain_event_sets.append(event_ids)
        for eid in event_ids:
            event_to_chains[eid].add(idx)

        srcs = set(cdf.get("Source_name_encoded", pd.Series(dtype=float)).dropna().tolist())
        tgts = set(cdf.get("Target_name_encoded", pd.Series(dtype=float)).dropna().tolist())
        types = set(cdf.get("Event_type", pd.Series(dtype=float)).dropna().tolist())
        countries = set(cdf.get("Source_Country_encoded", pd.Series(dtype=float)).dropna().tolist())
        countries |= set(cdf.get("Target_Country_encoded", pd.Series(dtype=float)).dropna().tolist())
        source_sets.append(srcs)
        target_sets.append(tgts)
        type_sets.append(types)
        country_sets.append(countries)
        for s in srcs:
            source_to_chains[s].add(idx)
        for t in tgts:
            target_to_chains[t].add(idx)
        for c in countries:
            country_to_chains[c].add(idx)

        m_time = mean_time_arr[idx]
        bucket = int(m_time // (TIME_SCALE * bucket_days))
        time_bucket_to_chains[bucket].add(idx)

        m_lat = mean_lat_arr[idx]
        m_lon = mean_lon_arr[idx]
        lat_bin = int((m_lat + 90.0) // grid_deg)
        lon_bin = int((m_lon + 180.0) // grid_deg)
        geo_cell_to_chains[(lat_bin, lon_bin)].add(idx)

    edge_index_list: List[List[int]] = []
    edge_attr_list: List[List[float]] = []
    delta_t_list: List[List[float]] = []
    delta_s_list: List[List[float]] = []

    def _jac(set_a: Set, set_b: Set) -> float:
        if not set_a and not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union > 0 else 0.0

    for j in range(num_chains):
        candidates: Set[int] = set()
        for eid in chain_event_sets[j]:
            candidates.update(event_to_chains.get(eid, set()))
        for s in source_sets[j]:
            candidates.update(source_to_chains.get(s, set()))
        for t in target_sets[j]:
            candidates.update(target_to_chains.get(t, set()))
        for c in country_sets[j]:
            candidates.update(country_to_chains.get(c, set()))

        mean_t_j = mean_time_arr[j]
        bucket_j = int(mean_t_j // (TIME_SCALE * bucket_days))
        for b in (bucket_j - 1, bucket_j, bucket_j + 1):
            candidates.update(time_bucket_to_chains.get(b, set()))

        m_lat_j = mean_lat_arr[j]
        m_lon_j = mean_lon_arr[j]
        lat_bin_j = int((m_lat_j + 90.0) // grid_deg)
        lon_bin_j = int((m_lon_j + 180.0) // grid_deg)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                candidates.update(geo_cell_to_chains.get((lat_bin_j + di, lon_bin_j + dj), set()))

        candidate_indices = [k for k in candidates if k != j and mean_time_arr[k] >= mean_t_j]
        if not candidate_indices:
            continue

        scored: List[Tuple[float, int, int, float, float, List[float]]] = []
        for k in candidate_indices:
            src_j = _jac(source_sets[j], source_sets[k])
            tgt_j = _jac(target_sets[j], target_sets[k])
            type_j = _jac(type_sets[j], type_sets[k])
            same_country = 1.0 if (country_sets[j] & country_sets[k]) else 0.0
            interact_flag = 1.0 if (source_sets[j] & target_sets[k]) or (target_sets[j] & source_sets[k]) else 0.0
            mean_t_k = mean_time_arr[k]
            dt = abs(mean_t_j - mean_t_k)
            ds = haversine(m_lon_j, m_lat_j, mean_lon_arr[k], mean_lat_arr[k])
            time_score = 1.0 / (1.0 + (dt / TIME_SCALE))
            space_score = 1.0 / (1.0 + (ds / 1000.0))
            score = (
                (src_j + tgt_j + type_j)
                + 0.5 * same_country
                + time_score
                + space_score
                + interact_flag
            )
            feat = [src_j, tgt_j, type_j, same_country, time_score, space_score, interact_flag]
            scored.append((score, j, k, dt, ds, feat))

        if topk_per_node > 0 and len(scored) > topk_per_node:
            scored.sort(key=lambda x: x[0], reverse=True)
            picked = scored[:topk_per_node]
        else:
            picked = scored

        for score, idx_src, idx_dst, dt, ds, feat in picked:
            edge_index_list.append([idx_src + int(checkin_offset * 2), idx_dst + int(checkin_offset * 2)])
            edge_attr_list.append(feat)
            delta_t_list.append([dt])
            delta_s_list.append([ds])

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        chain_edge_delta_t = torch.tensor(delta_t_list, dtype=torch.float) / TIME_SCALE
        chain_edge_delta_s = torch.tensor(delta_s_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float)
        chain_edge_delta_t = torch.empty((0, 1), dtype=torch.float)
        chain_edge_delta_s = torch.empty((0, 1), dtype=torch.float)

    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        chain_edge_delta_t=chain_edge_delta_t,
        edge_delta_t=chain_edge_delta_t,
        chain_edge_delta_s=chain_edge_delta_s,
        num_nodes=len(chain_features),
    )
    print(
        f"[ChainGraph-v2] edges={edge_index.size(1)} | attr_dim={edge_attr.size(1) if edge_attr.numel() else 0}"
    )
    return pyg_data



def generate_hyperedge_stat(data, traj_column):
    """
    生成轨迹超边统计信息数据（pd.DataFrame）

    :param data: 原始伪会话轨迹数据
    :param traj_column: 轨迹列名
    :return: 轨迹统计信息DataFrame
    """
    grouped = data.groupby(traj_column)
    traj_stat = pd.DataFrame({
        'size': grouped.size()
    })
    traj_stat['mean_lon'] = grouped['longitude'].first()
    traj_stat['mean_lat'] = grouped['latitude'].first()
    traj_stat['last_lon'] = grouped['longitude'].last()
    traj_stat['last_lat'] = grouped['latitude'].last()
    traj_stat['start_time'] = grouped['UTCTimeOffsetEpoch'].min()
    traj_stat['end_time'] = grouped['UTCTimeOffsetEpoch'].max()
    traj_stat['mean_time'] = grouped['UTCTimeOffsetEpoch'].mean()
    traj_stat['event_type'] = grouped['Event_type'].first()
    traj_stat['event_intensity'] = grouped['Intensity'].first()

    logging.info(f'[Preprocess - Generate Hypergraph] Number of hyperedges(trajectory): {traj_stat.shape[0]}.')
    logging.info(
        f'[Preprocess - Generate Hypergraph] The min, mean, max size of hyperedges are: '
        f'{traj_stat["size"].min()}, {traj_stat["size"].mean()}, {traj_stat["size"].max()}.'
    )
   
    return traj_stat

def generate_chainhyperedge_stat(data, traj_column):
    """
    生成事件链超边统计信息数据（pd.DataFrame）

    :param data: 原始伪会话轨迹数据
    :param traj_column: 轨迹列名
    :return: 轨迹统计信息DataFrame
    """
    # 创建空的DataFrame用于存储统计信息
    traj_stat = pd.DataFrame()
    data_unique = data.drop_duplicates(
    subset=['ID', 'UTC_time', 'Location_encoded', 'Event_type', 'Intensity', 'latitude', 'longitude']
)
    # 计算每个事件链的大小（包含的事件数量）
    #traj_stat['size'] = data.groupby(traj_column)['UTCTimeOffsetEpoch'].apply(len)
    traj_stat['size'] = data_unique.groupby(traj_column)['ID'].apply(len)
    # 计算每个事件链的平均经度
    traj_stat['mean_lon'] = data_unique.groupby(traj_column)['longitude'].apply(sum) / traj_stat['size']
    # 计算每个事件链的平均纬度
    traj_stat['mean_lat'] = data_unique.groupby(traj_column)['latitude'].apply(sum) / traj_stat['size']
    # 获取每个事件链的最后一个签到点的经纬度
    traj_stat[['last_lon', 'last_lat']] = \
        data_unique.sort_values([traj_column, 'UTCTimeOffsetEpoch']).groupby(traj_column).last()[['longitude', 'latitude']]

    #计算每个事件链的开始时间
    traj_stat['start_time'] = data_unique.groupby(traj_column)['UTCTimeOffsetEpoch'].apply(min)
    # 计算每个事件链的结束时间
    traj_stat['end_time'] = data_unique.groupby(traj_column)['UTCTimeOffsetEpoch'].apply(max)
    # 计算每个事件链的平均时间
    traj_stat['mean_time'] = data_unique.groupby(traj_column)['UTCTimeOffsetEpoch'].apply(sum) / traj_stat['size']
    # 计算每个事件链的时间窗口（天）
    traj_stat['time_window_in_day'] = (traj_stat.end_time - traj_stat.start_time) / (24*60*60)
    # 记录超边（事件链）数量
    logging.info(f'[Preprocess - Generate Hypergraph] Number of hyperedges(eventchain): {traj_stat.shape[0]}.')
    # 记录超边大小的统计信息
    logging.info(
        f'[Preprocess - chainGenerate Hypergraph] The min, mean, max size of hyperedges are: '
        f'{traj_stat["size"].min()}, {traj_stat["size"].mean()}, {traj_stat["size"].max()}.'
    )
    # 记录时间窗口的统计信息
    logging.info(
        f'[Preprocess - chainGenerate Hypergraph] The min, mean, max time window of hyperedges are:'
        f'{traj_stat.time_window_in_day.min()}, {traj_stat.time_window_in_day.mean()}, '
        f'{traj_stat.time_window_in_day.max()}.'
    )
    return traj_stat

def compute_relation_features(row, history_df):
    """
    计算关系特征：相同事件类型的历史数量、正向情感强度平均值、负向情感强度平均值
    
    :param row: 当前行数据
    :param history_df: 历史数据
    :return: 关系特征Series
    """
    try:
        tgt = row['Target_name_encoded']
        rel = row['Event_type']

        # 过滤出相同目标实体的历史记录
        matched_all = history_df[history_df['Target_name_encoded'] == tgt]

        # 相同事件类型的历史数量
        same_relation_count = matched_all[matched_all['Event_type'] == rel].shape[0]

        # 正向情感强度平均值
        pos_records = matched_all[matched_all['Intensity'] > 0]
        pos_avg_intensity = pos_records['Intensity'].mean() if not pos_records.empty else 0.0

        # 负向情感强度平均值
        neg_records = matched_all[matched_all['Intensity'] < 0]
        neg_avg_intensity = neg_records['Intensity'].mean() if not neg_records.empty else 0.0

        return pd.Series([same_relation_count, pos_avg_intensity, neg_avg_intensity],
                         index=['relation_count', 'pos_avg_intensity', 'neg_avg_intensity'])
    except Exception as e:
        print(f"[RelationFeatures] 计算关系特征时出错: {e}")
        # 返回默认值
        return pd.Series([0.0, 0.0, 0.0],
                         index=['relation_count', 'pos_avg_intensity', 'neg_avg_intensity'])

def new_generate_ci2traj_pyg_data(data, traj_stat, traj_column, checkin_offset, output_path):
    """
    生成签到点到轨迹的关联矩阵、签到点特征矩阵，以及关系特征。
    包含进度保存和错误处理。

    :param data: 原始轨迹数据
    :param traj_stat: 超边（轨迹）统计信息
    :param traj_column: 轨迹列名
    :param checkin_offset: 最大签到点索引加1
    :return: 包含关联矩阵和签到点特征矩阵的PyG数据
    """
    print("[L1] 准备L1输入特征与关联矩阵...")
    os.makedirs(output_path, exist_ok=True)

    # 创建检查点文件路径
    checkpoint_file = osp.join(output_path, 'l1_relation_features_checkpoint.pkl')

    # 尝试加载检查点
    checkpoint = load_checkpoint(checkpoint_file)
    
    # # 定义签到点特征列
    # checkin_feature_columns = [
    #     #'UserId', 'PoiId', 'PoiCategoryId', 'latitude', 'longitude', 'UTCTimeOffsetEpoch', 'UTCTimeOffsetWeekday',
    #     #'UTCTimeOffsetHour', 'check_ins_id', 'pseudo_session_trajectory_id'
    #     'ID', 'Source_name_encoded', 'Target_name_encoded',
    #     'Source_Country_encoded', 
    #      'Location_encoded', 'Event_type', 'Intensity','latitude', 'longitude','check_ins_id','UTCTimeOffsetEpoch'
    # ]
        # 构建 Name_encoded & objecttype
    data1 = data.copy()
    data1['Name_encoded'] = data['Target_name_encoded']
    data1['objecttype'] = 1

    # 拼接 Source_Country_encoded 行
    extra_rows = data.copy()
    extra_rows['Name_encoded'] = extra_rows['Source_name_encoded']
    extra_rows['objecttype'] = 0

    # 合并两部分
    merged_data = pd.concat([data1, extra_rows], ignore_index=True)

    # 按新列顺序
    checkin_feature_columns = [
        'ID', 'Name_encoded', 'objecttype',
        'Location_encoded', 'Event_type', 'Intensity',
        'latitude', 'longitude', 'UTCTimeOffsetEpoch',
        'EventChain_id'
    ]
    entity_feature = [
        'Name_encoded','objecttype','Source_Country_encoded']
    #后面再加上check_ins_id列

    # 去重（按全部特征列去重）
    data = merged_data.drop_duplicates(subset=checkin_feature_columns, keep='first')
    #data新增一列entity，值为Name_encoded列
    data.loc[:, "entity"] = data["Name_encoded"]
    # data['entity'] = data['Name_encoded']
    # 打印每列的数据类型（从原始DataFrame获取）
    print("Checkin Feature 各列数据类型:")
    for i, col in enumerate(checkin_feature_columns):
        if col in data.columns:
            print(f"{col}: {data[col].dtype}")
        else:
            print(f"{col}: 列不存在")
    #打印size
    print(f"[L1] checkin_feature size={data.shape}")


    # 按签到点ID排序并提取特征
    data_sorted = data.sort_values('UTCTimeOffsetEpoch')
    #data_sorted['check_ins_id'] = np.arange(len(data_sorted), dtype=np.int64)
    data_sorted['check_ins_id'] = data_sorted['UTCTimeOffsetEpoch'].rank(ascending=True, method='first') - 1
    entity_feature = [
        'UTCTimeOffsetEpoch','Name_encoded','objecttype','Source_Country_encoded','check_ins_id']

    #data_sorted = data.sort_values('UTCTimeOffsetEpoch')
    checkin_feature = data_sorted[entity_feature].to_numpy()
    
    print(f"[L1] checkin_feature shape={checkin_feature.shape}")
    # 断言：确保签到点ID在原始数据中是按时间顺序排列的
    assert data_sorted.check_ins_id.unique().shape[0] == data_sorted.check_ins_id.max() + 1, \
        'check_ins_id222 is not chronological order in raw data'
    
    data = data_sorted
    
    #把data按照‘ID’分组，统计每组数量作为新列entity2event_size
    data['entity2event_size'] = data.groupby('ID')['ID'].transform('size')
    #构造map（key为eventid（‘ID’），value为‘entity2event_size’）
    event2size_map = data.set_index('ID')['entity2event_size'].to_dict()

    # 保存到文件
    map_path = osp.join(output_path, "event2size_map.pkl")
    with open(map_path, "wb") as f:
        pickle.dump(event2size_map, f)
    print(f"[L1] 已保存 event2size_map -> {map_path}")
    print(f"[L1] data after adding entity2event_size shape={data.shape}")
    #把data保存成文件，文件名unclassify_sourcename_data.csv，判断是否保存成功并捕捉异常
    try:
        data.to_csv(osp.join(output_path, 'unclassify_sourcename_data.csv'), index=False)
        print("[L1] 保存成功")
    except Exception as e:
        print(f"[L1] 保存失败：{e}")
    #后续在这个文件上操作
    


    # 对每一行计算关系特征
    print("[L1] 计算关系特征(relation features)...")
    
    # 检查是否有保存的进度
    start_idx = 0
    relation_features = []
    
    if checkpoint is not None and checkpoint.get('step') == 'relation_features_partial':
        start_idx = checkpoint.get('completed_count', 0)
        print(f"[L1] 从检查点恢复，已计算 {start_idx} 个关系特征")
        try:
            data_file = checkpoint_file.replace('.pkl', '_data.pkl')
            with open(data_file, 'rb') as f:
                relation_features = pickle.load(f)
        except Exception as e:
            print(f"[L1] 恢复关系特征失败: {e}")
            start_idx = 0
            relation_features = []
    
    # 计算剩余的关系特征
    try:
        for i in tqdm(range(start_idx, len(data)), desc='Computing relation features'):
            try:
                row = data.iloc[i]
                history = data.iloc[:i]  # 当前行之前的历史
                #relation_feature = compute_relation_features(row, history)
                #不用上面这种了
                relation_feature = pd.Series([
                    row['Event_type'],
                    row['Intensity'],
                    row['objecttype']
                ])

                relation_features.append(relation_feature)
                
                # 每10000个保存一次进度
                if (i + 1) % 100000 == 0:
                    save_checkpoint(checkpoint_file, 'relation_features_partial', 
                                  data=relation_features, completed_count=i+1)
                    print(f"[L1] 已计算 {i+1}/{len(data)} 个关系特征")
            except Exception as e:
                print(f"[L1] 计算第 {i} 个关系特征时出错: {e}")
                # 使用默认值
                relation_features.append(pd.Series([0.0, 0.0, 0.0], 
                                                 index=['relation_count', 'pos_avg_intensity', 'neg_avg_intensity']))
    except Exception as e:
        print(f"[L1] 关系特征计算过程中出错: {e}")
        # 如果出错，使用简化的边特征
        print("[L1] 使用简化的边特征...")
        relation_features = None

    # 为实体->事件创建关联矩阵
    print("[L1] 构建SparseTensor与边索引...")
    ci2traj_adj_t = SparseTensor(
        row=torch.as_tensor(data[traj_column].tolist(), dtype=torch.long),
        col=torch.as_tensor(data.check_ins_id.tolist(), dtype=torch.long), #用的是行号作为实体id
        value=torch.as_tensor(range(0, data.shape[0]), dtype=torch.long)
    )
    # 获取排列索引
    perm = ci2traj_adj_t.storage.value()
    # 获取边的时间信息
    ci2traj_edge_t = torch.tensor(data.UTCTimeOffsetEpoch.tolist())[perm]
    # 计算时间差（轨迹结束时间 - 当前时间）
    try:
        traj_end_series = traj_stat['end_time']
        mapped_end = traj_end_series.reindex(data[traj_column].values).to_numpy(dtype=float)
        utc_array = data['UTCTimeOffsetEpoch'].to_numpy(dtype=float)
        mapped_end = np.where(np.isnan(mapped_end), utc_array, mapped_end)
        delta_arr = (mapped_end - utc_array) / TIME_SCALE
        ci2traj_edge_delta_tensor = torch.tensor(delta_arr, dtype=torch.float).unsqueeze(1)
    except Exception as e:
        print(f"[L1] 计算ci2traj时间差失败，使用零填充: {e}")
        ci2traj_edge_delta_tensor = torch.zeros((data.shape[0], 1), dtype=torch.float)
    ci2traj_edge_delta_t = ci2traj_edge_delta_tensor[perm]
    # 获取空间距离信息
    #ci2traj_edge_delta_s = torch.tensor(delta_s_in_traj.distance_km.tolist())[perm]

    # 创建边索引（签到点在前，轨迹在后，轨迹ID加上偏移量）
    ci2traj_edge_index = torch.stack([ci2traj_adj_t.storage.col(), ci2traj_adj_t.storage.row() + checkin_offset])
    
    # 创建边特征
    if relation_features is not None:
        try:
            # 使用计算的关系特征
            relation_df = pd.DataFrame(relation_features)
            edge_features = torch.tensor(relation_df.values, dtype=torch.float)
            edge_features = edge_features[perm]  # 保证顺序一致
            print(f"[L1] 使用关系特征作为边特征: {edge_features.shape}")
        except Exception as e:
            print(f"[L1] 处理关系特征时出错: {e}")
            # 回退到简化特征
            edge_features = torch.stack([
                torch.tensor(data['Event_type'].values, dtype=torch.float),
                torch.tensor(data['Intensity'].values, dtype=torch.float),
                #实体对于事件的类型‘objecttype’，0为source，1为target
                torch.tensor(data['objecttype'].values, dtype=torch.float)
            ], dim=1)
            edge_features = edge_features[perm]
            print(f"[L1] 使用简化边特征: {edge_features.shape}")
    else:
        # 使用简化的边特征
        edge_features = torch.stack([
            torch.tensor(data['Event_type'].values, dtype=torch.float),
            torch.tensor(data['Intensity'].values, dtype=torch.float),
            #实体对于事件的类型‘objecttype’，0为source，1为target
            torch.tensor(data['objecttype'].values, dtype=torch.float)
        ], dim=1)
        edge_features = edge_features[perm]
        print(f"[L1] 使用简化边特征: {edge_features.shape}")

    print("[Debug] checkin_feature dtype:", checkin_feature.dtype)
    print("[Debug] checkin_feature shape:", checkin_feature.shape)
    print("[Debug] sample:", checkin_feature[:5])

    # 创建PyG数据对象
    ci2traj_pyg_data = Data(
        edge_index=ci2traj_edge_index,#边
        x=torch.tensor(checkin_feature),#节点
        edge_attr=edge_features,#边特征
        edge_delta_t=ci2traj_edge_delta_t,
    )
    # 设置超边数量(等于事件数量)
    ci2traj_pyg_data.num_hyperedges = data_sorted['ID'].nunique()
    print(f"[L1] 完成 | x={tuple(ci2traj_pyg_data.x.shape)} | edges={ci2traj_pyg_data.edge_index.size(1)}")
    
    # 清理检查点文件
    try:
        if osp.exists(checkpoint_file):
            os.remove(checkpoint_file)
        data_file = checkpoint_file.replace('.pkl', '_data.pkl')
        if osp.exists(data_file):
            os.remove(data_file)
        print("[L1] 清理检查点文件完成")
    except Exception as e:
        print(f"[L1] 清理检查点文件失败: {e}")
    
    
    
    return ci2traj_pyg_data

###PATH:Spatio-Temporal-Hypergraph-Model/preprocess/generate_hypergraph.py
def generate_event2chain_graph(data, args, traj_stat, checkin_offset, output_path):  # 事件所包含的实体数量 size):
    """
    第三层图：事件-事件链二分图 (L3)
    构建事件-事件链二分图 (L3)
    节点：事件 + 事件链
    边：事件 <-> 其所属的事件链
    """
    print("[Event2Chain] 开始构建事件-事件链图...")

    map_path = osp.join(output_path, "event2size_map.pkl")
    try:
        with open(map_path, "rb") as f:
            event2size_map = pickle.load(f)
    except Exception as exc:
        print(f"[L2] 读取 event2size_map 失败，使用当前数据重建: {exc}")
        tmp = data.copy()
        if 'entity2event_size' in tmp.columns:
            event2size_map = tmp.set_index('ID')['entity2event_size'].to_dict()
        else:
            event2size_map = tmp.groupby('ID').size().to_dict()
    print(f"[L2] 已加载 event2size_map (size={len(event2size_map)})")
    # ========= 事件节点 =========
    # 按事件ID去重（避免 Source / Target 重复）
    event_df = data.drop_duplicates(subset=['ID']).reset_index(drop=True)
    event_ids = np.sort(event_df['ID'].astype(int).tolist())
    num_events = len(event_ids)
    _eid2idx = {int(e): int(i) for i, e in enumerate(event_ids)}
    _event_base = int(checkin_offset)



    # ========= 事件节点特征 =========
     # --- 构造事件节点特征矩阵 ---
    try:
        data_unclassify = pd.read_csv(osp.join(output_path, 'unclassify_sourcename_data.csv'))
    except Exception as exc:
        print(f"[Event2Chain] 未找到预存的 unclassify 数据，使用当前 data 代替: {exc}")
        data_unclassify = data.copy()
    
    print("[EventGraph] 构造事件节点特征矩阵...")
    event_features = []
    eventid2rowidx = data.groupby('ID').apply(lambda df: df.index[0]).to_dict()# 事件对应多条记录里的第一条记录的行号组成的映射表，事件id-行号
    #把event_ids按顺序排列，并打印检查前几个
    event_ids = np.sort(event_ids)
    print(f"[Event2Chain] event_ids sorted: {event_ids[:5]} ... {event_ids[-5:]}")
    for eid in event_ids:
        row = data.loc[eventid2rowidx[eid]]
        feat = [
            float(row['Event_type']),
            float(row['Intensity']),
            float(row['latitude']),
            float(row['longitude']),
            float(row['UTCTimeOffsetEpoch']),
            float(event2size_map[eid])  # 事件所包含的实体数 size
        ]
        event_features.append(feat)
    event_features = np.asarray(event_features, dtype=float)
    event_features[:, 4] = event_features[:, 4] / TIME_SCALE
    x = torch.tensor(event_features, dtype=torch.float)


    # ========= 构造边 =========
    edge_index_list, edge_attr_list = [], []
    #src, tgt, weights = [], [], []

    # Chain ID contiguous mapping and base after events
    chain_ids_sorted = np.sort(pd.unique(data['EventChain_id'].values))
    _cid2idx = {int(c): int(i) for i, c in enumerate(chain_ids_sorted)}
    _chain_base = int(checkin_offset) * 2

    chain_time_series = data.groupby('EventChain_id')['UTCTimeOffsetEpoch'].mean()
    delta_t_list = []

    for idx, row in event_df.iterrows():
        event_id = _event_base + _eid2idx[int(row['ID'])]
        chain_id = _chain_base + _cid2idx[int(row['EventChain_id'])]

        # src.append(event_id)
        # tgt.append(chain_id)
        # 事件 <-> 事件链
        edge_index_list.append([event_id, chain_id])
        #edge_index_list.append([chain_id, event_id])
        edge_attr_list.append([1.0])
        #edge_attr_list.append([1.0])
        chain_mean_time = chain_time_series.get(row['EventChain_id'], row['UTCTimeOffsetEpoch'])
        dt = abs(float(chain_mean_time) - float(row['UTCTimeOffsetEpoch'])) / TIME_SCALE
        delta_t_list.append([dt])

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        event_chain_edge_delta_t = torch.tensor(delta_t_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,1), dtype=torch.float)
        event_chain_edge_delta_t = torch.empty((0,1), dtype=torch.float)

    print(f"[Event2Chain] 构图完成 | nodes={x.size(0)} | edges={edge_index.size(1)}")

    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        event_chain_edge_delta_t=event_chain_edge_delta_t,
        edge_delta_t=event_chain_edge_delta_t,
        num_nodes=x.size(0)
    )

    print(f"[Event2Chain] 特征矩阵: {tuple(x.shape)}")
    try:
        pyg_data.chain_offset = int(_chain_base)
    except Exception:
        pass
    return pyg_data
    

def calculate_event_edge_weight(
    row_i, row_j,
    relation_i, relation_j,
    alpha_entity=0.25,
    alpha_type=0.15,
    alpha_time=0.15,
    alpha_geo=0.15,
    alpha_rel_count=0.1,
    alpha_pos_sent=0.1,
    alpha_neg_sent=0.1
):
    """
    计算两个事件之间的边权，前提是有实体交集。

    参数：
    - row_i, row_j: 两个事件的行（DataFrame的Series）
    - relation_i, relation_j: 两事件的关系特征张量（Tensor Row）
    - alpha_*: 各项特征的加权系数

    返回：边权（float），若无交集返回 None
    """
    ents_i = {row_i['Source_name_encoded'], row_i['Target_name_encoded']}
    ents_j = {row_j['Source_name_encoded'], row_j['Target_name_encoded']}
    shared = len(ents_i & ents_j)
    total = len(ents_i | ents_j)
    if shared == 0:
        return None

    entity_score = shared / total if total > 0 else 0.0
    type_score = 1.0 if row_i['Event_type'] == row_j['Event_type'] else 0.0

    t1 = pd.to_datetime(row_i['UTC_time'])
    t2 = pd.to_datetime(row_j['UTC_time'])
    time_gap = abs((t1 - t2).total_seconds()) / 3600
    time_score = 1 / (1 + time_gap)

    loc1 = row_i['Location_encoded']
    loc2 = row_j['Location_encoded']
    geo_score = 1 / (1 + abs(loc1 - loc2))

    # relation features
    rel_count_i, pos_avg_i, neg_avg_i = relation_i.tolist()
    rel_count_j, pos_avg_j, neg_avg_j = relation_j.tolist()
    rel_count_score = (rel_count_i + rel_count_j) / 2
    pos_score = (pos_avg_i + pos_avg_j) / 2
    neg_score = (neg_avg_i + neg_avg_j) / 2

    edge_weight = (
        alpha_entity * entity_score +
        alpha_type * type_score +
        alpha_time * time_score +
        alpha_geo * geo_score +
        alpha_rel_count * rel_count_score +
        alpha_pos_sent * pos_score +
        alpha_neg_sent * neg_score
    )
    return edge_weight

import pickle, os, os.path as osp, tempfile
import sys
import itertools
from tqdm import tqdm

def atomic_pickle_dump(obj, path):
    """将 obj 原子性写入 path（先写临时文件再替换），并 fsync 确保落盘。"""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=osp.dirname(path))
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # atomic replace
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

def safe_pickle_load(path):
    """安全加载，出现 EOFError 等异常时给出友好提示并返回 None"""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"[ERROR] 加载 {path} 失败: {e} （文件可能损坏或被截断）。")
        return None
    except Exception as e:
        print(f"[ERROR] 无法加载 {path}: {e}")
        return None
import os
import pickle
from tqdm import tqdm

def build_event_graph(edge_weights, checkin_offset, checkpoint_path=None, save_every=10_000_000):
    # 初始化
    src, tgt, weights = [], [], []
    processed = 0

    # 如果有 checkpoint，先加载
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        src, tgt, weights = ckpt["src"], ckpt["tgt"], ckpt["weights"]
        processed = ckpt["processed"]
        print(f"[Checkpoint] 已恢复进度: {processed}/{len(edge_weights)} 条边")

    # 遍历 edge_weights
    for i, ((e1, e2), w) in enumerate(tqdm(edge_weights.items())):
        if i < processed:
            continue  # 跳过已经处理过的部分

        if w >= 1:  # args.threshold
            src.append(e1 + checkin_offset)
            tgt.append(e2 + checkin_offset)
            weights.append(w)

        processed += 1

        # 定期存 checkpoint
        if checkpoint_path and processed % save_every == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "src": src,
                    "tgt": tgt,
                    "weights": weights,
                    "processed": processed
                }, f)
            print(f"[Checkpoint] 已保存 {processed}/{len(edge_weights)}")

    print(f"[EventGraph] 生成边数: {len(src)} (阈值: 1)")

    # 最终保存一次
    if checkpoint_path:
        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "src": src,
                "tgt": tgt,
                "weights": weights,
                "processed": processed
            }, f)
        print(f"[Checkpoint] 最终保存完成 {processed}/{len(edge_weights)}")

    return src, tgt, weights


def generate_event_graph(data1, args, ci2traj_pyg_data, checkin_offset, output_path=None):
    """
    生成第二层事件图（事件节点图），事件以唯一ID聚合，支持 checkpoint
    """
    import pickle, os
    from collections import defaultdict

    base_path = output_path if output_path else os.getcwd()
    os.makedirs(base_path, exist_ok=True)
    # checkpoint 文件路径
    checkpoint_path = osp.join(base_path, "event_edges_ckpt.pkl")

    map_path = osp.join(base_path, "event2size_map.pkl")
    try:
        with open(map_path, "rb") as f:
            event2size_map = pickle.load(f)
    except Exception:
        # Fallback: build from current data
        try:
            tmp = data1.copy()
            if 'entity2event_size' in tmp.columns:
                event2size_map = tmp.set_index('ID')['entity2event_size'].to_dict()
            else:
                event2size_map = tmp.groupby('ID').size().to_dict()
        except Exception:
            event2size_map = {}

    # 把 value 转成 int，兼容 tensor / numpy / 其他类型
    # event2size_map = {
    #     k: int(v.item()) if hasattr(v, "item") else int(v)
    #     for k, v in event2size_map.items()
    # }
    event2size_map = {
        k: int(v.item()) if hasattr(v, "item") and v.ndim == 0 else
        int(v[0].item()) if hasattr(v, "ndim") and v.ndim == 1 and len(v) == 1 else
        int(v)
        for k, v in event2size_map.items()
    }

    
    print("[DEBUG] event2size_map loaded, sample:", list(event2size_map.items())[:5])
    print(f"[L2] 已加载 event2size_map (size={len(event2size_map)})")
    #打印类型
    print(f"[L2] event2size_map type: {type(event2size_map)}")
    print(f"[L2] event2size_map sample: {list(event2size_map.items())[:5]}")


    #data来copydata1
    data = data1.copy()

    # 为每条数据加上偏移量
    #打印id列前5个
    print(f'[DEBUG] before data["ID"][:5]: {data["ID"][:5]}')  #打印data['ID'][:5])
    data['ID'] = data['ID'].astype(np.int64) + int(checkin_offset)
    print(f'[DEBUG] after data["ID"][:5]: {data["ID"][:5]}')  #打印data['ID'][:5])




    print("[DEBUG]111111111")
    
    times = data.groupby('ID')['UTCTimeOffsetEpoch'].first()
    times = times.to_dict()

    print("[EventGraph] 事件唯一ID聚合...")
    event_ids_unique = data['ID'].unique()
    num_events = len(event_ids_unique)
    print(f"[EventGraph] 事件数量（唯一ID）: {num_events}")

    # 事件ID -> 索引 映射
    eventid2idx = {eid: idx for idx, eid in enumerate(event_ids_unique)}

    # 为每条数据映射对应事件索引
    data['event_idx'] = data['ID'].map(eventid2idx) #存的索引

    # --- 聚合第一层边特征到事件级 ---
    event_indices_in_edges = ci2traj_pyg_data.edge_index[1]
    relation_tensor_edges = ci2traj_pyg_data.edge_attr

    event_relation_features = []
    for eid_idx in range(num_events):
        mask = (event_indices_in_edges == eid_idx)
        if mask.any():
            avg_feat = relation_tensor_edges[mask].mean(dim=0)
        else:
            avg_feat = torch.zeros(relation_tensor_edges.size(1))
        event_relation_features.append(avg_feat)
    relation_tensor = torch.stack(event_relation_features, dim=0)
    print(f"[EventGraph] 聚合完成: {relation_tensor.shape}")

    # --- 构建实体到事件的倒排索引 ---
    print("[EventGraph] 构建实体到事件倒排索引...")
    event2entities = defaultdict(set)
    for _, row in data.iterrows():
        eid_idx = row['ID']#event id已经加过偏移了
        if _ <= 5:
            print(f"[DEBUG] eid_idx: {eid_idx}")
            print(f"[DEBUG] row: {row[['ID','Source_name_encoded','Target_name_encoded']]}")
        event2entities[eid_idx].add(row['Source_name_encoded'])
        event2entities[eid_idx].add(row['Target_name_encoded'])#由于set的特性，不会重复

    entity2events = defaultdict(set)
    for eid_idx, entities in event2entities.items():
        for ent in entities:
            entity2events[ent].add(eid_idx)
    print(f"[EventGraph] 实体数量len(entity2events)：{len(entity2events)}")

    # --- 生成事件-事件边（基于实体共享），支持 checkpoint ---
    print("[EventGraph] 根据实体共享生成事件-事件边...")

    edge_weights = defaultdict(int)
    start_idx = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
            edge_weights = ckpt["edge_weights"]
            start_idx = ckpt["last_idx"]
        print(f"[Checkpoint] 恢复进度: 已处理 {start_idx}/{len(entity2events)} 实体, 当前边数: {len(edge_weights)}")

    for idx, events in enumerate(tqdm(entity2events.values(), total=len(entity2events))):
        if idx < start_idx:
            continue

        events = list(events)
        for i in range(len(events)):
            for j in range(i+1, len(events)):
                e1, e2 = events[i], events[j]
                # 判断时序关系，由小的指向大的
                # 取e1,e2的时间

                time1 = times.get(e1, None)
                time2 = times.get(e2, None)

                if time1 is None or time2 is None:
                    print(f"[WARN] time not found for e1={e1}, e2={e2}")
                    continue

                if time1 > time2:
                    e1, e2 = e2, e1

                edge_weights[(e1, e2)] += 1

        # 每 1000 个实体存一次
        if (idx + 1) % 1000 == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({"edge_weights": edge_weights, "last_idx": idx+1}, f)
            print(f"[Checkpoint] 已保存进度: {idx+1}/{len(entity2events)} 实体, 边数: {len(edge_weights)}")

    # 保存最终版本
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"edge_weights": edge_weights, "last_idx": len(entity2events)}, f)
    print(f"[Checkpoint] 最终结果已保存: 边数 {len(edge_weights)}")

    print(f"[EventGraph] 初始生成边数: {len(edge_weights)}")


    # # --- 按阈值筛选边 ---
    # print("[EventGraph] 按阈值筛选边...")
    # src, tgt, weights = [], [], []
    # for (e1, e2), w in tqdm(edge_weights.items()):
    #     if w >= 1:  # args.threshold
    #         src.append(e1 + checkin_offset)
    #         tgt.append(e2 + checkin_offset)
    #         weights.append(w)
    # print(f"[EventGraph] 生成边数: {len(src)} (阈值: 1)")

    # Use configured preprocessed directory to store optional caches
    save_dir = getattr(args, 'data_path', None)
    if save_dir is None:
        save_dir = output_path if 'output_path' in locals() and output_path else os.getcwd()
    if not osp.isabs(str(save_dir)):
        try:
            save_dir = osp.join(get_root_dir(), str(save_dir))
        except Exception:
            save_dir = osp.abspath(str(save_dir))
    os.makedirs(save_dir, exist_ok=True)

    # 文件名里带上阈值，避免混淆
    edge_file = os.path.join(save_dir, f"event_edges_threshold{1}.pt")  # 可以改成 args.threshold

    # --- 按阈值筛选边 ---
    if os.path.exists(edge_file):
        print(f"[EventGraph] 检测到缓存文件 {edge_file}，直接加载...")
        data_edge = torch.load(edge_file)
        src, tgt, weights = data_edge["src"], data_edge["tgt"], data_edge["weights"]
    else:
        print("[EventGraph] 按阈值筛选边...")
        src, tgt, weights = [], [], []
        for (e1, e2), w in tqdm(edge_weights.items()):
            if w >= 1:  # args.threshold
                src.append(e1)
                tgt.append(e2)
                weights.append(w)

        print(f"[EventGraph] 生成边数len(src): {len(src)} (阈值: 1)")

        # 保存到文件，下次直接恢复
        torch.save({"src": src, "tgt": tgt, "weights": weights}, edge_file)
        print(f"[EventGraph] 已保存边到 {edge_file}")


    if len(src) == 0:
        print("[EventGraph] 无符合阈值的事件边，生成空图。")
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,1), dtype=torch.float)
        event_edge_delta_t = torch.empty((0,1), dtype=torch.float)
        event_edge_delta_s = torch.empty((0,1), dtype=torch.float)
    else:
        try:
            _eid_sorted = np.sort(data['ID'].unique())
            _id2idx = {int(e): int(i) for i, e in enumerate(_eid_sorted)}
            _base = int(checkin_offset)
            src_idx = [ _base + _id2idx[int(e)] for e in src ]
            tgt_idx = [ _base + _id2idx[int(e)] for e in tgt ]
            edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)
        except Exception:
            edge_index = torch.tensor([src, tgt], dtype=torch.long)
        print("[EventGraph] 计算事件边的权重...")
        edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
        print(f"[EventGraph] 边权范围: {edge_attr.min().item()} - {edge_attr.max().item()}")
        
        # 构建基于 event_idx 的属性 Series
        times1 = data.groupby('ID')['UTCTimeOffsetEpoch'].first()
        lats  = data.groupby('ID')['latitude'].first()
        lons  = data.groupby('ID')['longitude'].first()

        print("[EventGraph] 计算事件边的时间和空间差异...")
        delta_t, delta_s = [], []
        # for x1, x2 in tqdm(zip(src, tgt)):
        #     e1 = x1
        #     e2 = x2

        #     t1, t2 = times[e1], times[e2]
        #     lat1, lon1 = lats[e1], lons[e1]
        #     lat2, lon2 = lats[e2], lons[e2]

        #     delta_t.append(abs(t1 - t2))
        #     delta_s.append(((lat1 - lat2)**2 + (lon1 - lon2)**2) ** 0.5)
        
        delta_file = osp.join(save_dir, "event_edge_deltas.pt")

        # --- 如果已有缓存，直接加载 ---
    if os.path.exists(delta_file):
        print(f"[EventGraph] 检测到缓存文件 {delta_file}，直接加载...")
        checkpoint = torch.load(delta_file)
        event_edge_delta_t = checkpoint["event_edge_delta_t"]
        cache_scale = checkpoint.get("time_scale", None)
        if cache_scale is None or abs(float(cache_scale) - TIME_SCALE) > 1e-6:
            event_edge_delta_t = event_edge_delta_t / TIME_SCALE
        event_edge_delta_s = checkpoint["event_edge_delta_s"]
        print(f"[EventGraph] 恢复完成，delta_t.shape={event_edge_delta_t.shape}, delta_s.shape={event_edge_delta_s.shape}")
    else:
            # --- 正常计算 ---
            delta_t, delta_s = [], []

            # 建立字典查表，加速 & 避免 KeyError
            times_dict = times1.to_dict()
            lats_dict  = lats.to_dict()
            lons_dict  = lons.to_dict()

            for e1, e2 in tqdm(zip(src, tgt), total=len(src), desc="[EventGraph] 计算边特征"):
                try:
                    t1, t2 = times_dict[e1], times_dict[e2]
                    lat1, lon1 = lats_dict[e1], lons_dict[e1]
                    lat2, lon2 = lats_dict[e2], lons_dict[e2]
                except KeyError:
                    print("[WARN] event time/coord not found for:", e1, e2)
                    continue

                delta_t.append(abs(t1 - t2) / TIME_SCALE)
                delta_s.append(((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5)

            print(f"[DEBUG] delta_t[:10] = {delta_t[:10]}")
            print(f"[DEBUG] 时间差范围: {min(delta_t)} - {max(delta_t)} 秒")

            # 转为 Tensor
            event_edge_delta_t = torch.tensor(delta_t, dtype=torch.float).unsqueeze(1)
            event_edge_delta_s = torch.tensor(delta_s, dtype=torch.float).unsqueeze(1)

            # --- 保存结果，下次直接用 ---
            torch.save(
                {
                    "event_edge_delta_t": event_edge_delta_t,
                    "event_edge_delta_s": event_edge_delta_s,
                    "time_scale": TIME_SCALE,
                },
                delta_file
            )
            print(f"[EventGraph] 已保存缓存到 {delta_file}")

    # --- 构造事件节点特征矩阵 ---
    # print("[EventGraph] 构造事件节点特征矩阵...")
    # event_features = []
    # eventid2rowidx = data.groupby('ID').apply(lambda df: df.index[0]).to_dict()
    # for eid in event_ids_unique:
    #     row = data.loc[eventid2rowidx[eid]]
    #     feat = [
    #         float(row['Event_type']),
    #         float(row['Intensity']),
    #         float(row['latitude']),
    #         float(row['longitude']),
    #         float(row['UTCTimeOffsetEpoch']),
    #         float(event2size_map[eid-checkin_offset])
    #     ]
    #     event_features.append(feat)
    # x = torch.tensor(event_features, dtype=torch.float)
    print("[EventGraph] 构造事件节点特征矩阵...")

    event_features = []
    eventid2rowidx = data.groupby('ID').apply(lambda df: df.index[0]).to_dict()

    print(f"[DEBUG] 事件总数: {len(event_ids_unique)}")
    print(f"[DEBUG] eventid2rowidx 样例: {list(eventid2rowidx.items())[:5]}")

    #把event_ids_unique按顺序排列
    event_ids_unique = np.sort(event_ids_unique)
    #检查event_ids_unique是否顺序排列
    print(f"[DEBUG] 排序后 event_ids_unique[:10]: {event_ids_unique[:10]}")

    checkin_offset = int(checkin_offset)
    for idx, eid in enumerate(tqdm(event_ids_unique, desc="构造事件特征")):
        if eid not in eventid2rowidx:
            print(f"[WARN] eid {eid} 不在 eventid2rowidx 中，跳过")
            continue
        
        row = data.loc[eventid2rowidx[eid]]

        try:
            feat = [
                float(row['Event_type']),
                float(row['Intensity']),
                float(row['latitude']),
                float(row['longitude']),
                float(row['UTCTimeOffsetEpoch']) / TIME_SCALE,
                float(event2size_map[eid-checkin_offset])
            ]
        except Exception as e:
            #打印异常信息
            print("异常信息为:", e)
            print(f"[ERROR] eid={eid}, row={row.to_dict()}, 错误={e}")
            print(f"[DEBUG] sample eid={eid},  checkin_offset={checkin_offset},eid-checkin_offset: {eid-checkin_offset}, in event2size_map? {eid-checkin_offset in event2size_map}")
            #直接中断程序
            raise e


        event_features.append(feat)

        # 每隔一段打印一次
        if idx % 10000 == 0 and idx > 0:
            print(f"[DEBUG] 已处理 {idx} / {len(event_ids_unique)} 个事件")

    x = torch.tensor(event_features, dtype=torch.float)
    print(f"[EventGraph] 事件特征矩阵完成, 形状: {x.shape}")

    print(f"[EventGraph] 特征矩阵形状: {x.shape}, 边数: {edge_index.size(1)}")

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        event_edge_delta_t=event_edge_delta_t,
        event_edge_delta_s=event_edge_delta_s,
        edge_delta_t=event_edge_delta_t,
    )

# def generate_event_graph(data, args, ci2traj_pyg_data, checkin_offset): 
#     """
#     生成第二层事件图（事件节点图），事件以唯一ID聚合
#     """

#     with open(map_path, "rb") as f:
#         event2size_map = pickle.load(f)
#     print(f"[L2] 已加载 event2size_map (size={len(event2size_map)})")

#     print("DEBUG event2size_map type:", type(event2size_map))
#     print(">>> inside generate_event_graph, event2size_map type:", type(event2size_map))
#     print(">>> sample keys:", list(event2size_map.keys())[:10])

#     print("[EventGraph] 事件唯一ID聚合...")
#     event_ids_unique = data['ID'].unique()
#     num_events = len(event_ids_unique)
#     print(f"[EventGraph] 事件数量（唯一ID）: {num_events}")

#     # 事件ID -> 索引 映射
#     eventid2idx = {eid: idx for idx, eid in enumerate(event_ids_unique)}

#     # 为每条数据映射对应事件索引
#     data['event_idx'] = data['ID'].map(eventid2idx)

#     # --- 聚合第一层边特征到事件级 ---
#     print("[EventGraph] 聚合第一层边特征到事件级别...") #不用聚合
#     event_indices_in_edges = ci2traj_pyg_data.edge_index[1]
#     relation_tensor_edges = ci2traj_pyg_data.edge_attr

#     event_relation_features = []
#     for eid_idx in range(num_events):
#         mask = (event_indices_in_edges == eid_idx)
#         if mask.any():
#             avg_feat = relation_tensor_edges[mask].mean(dim=0)
#         else:
#             avg_feat = torch.zeros(relation_tensor_edges.size(1))
#         event_relation_features.append(avg_feat)
#     relation_tensor = torch.stack(event_relation_features, dim=0)
#     print(f"[EventGraph] 聚合完成: {relation_tensor.shape}")

#     # --- 构建实体到事件的倒排索引 ---
#     print("[EventGraph] 构建实体到事件倒排索引...")
#     event2entities = defaultdict(set)
#     for _, row in data.iterrows():
#         eid_idx = row['event_idx']  #还没加偏移量 后面再加 和图一保持一致
#         event2entities[eid_idx].add(row['Source_name_encoded'])
#         event2entities[eid_idx].add(row['Target_name_encoded'])

#     entity2events = defaultdict(set)
#     for eid_idx, entities in event2entities.items():
#         for ent in entities:
#             entity2events[ent].add(eid_idx)
#     #打印实体数量
#     print(f"[EventGraph] 实体数量：{len(entity2events)}")

#     # --- 生成事件-事件边（基于实体共享） ---
#     print("[EventGraph] 根据实体共享生成事件-事件边...")
#     edge_weights = defaultdict(int)
#     for events in tqdm(entity2events.values()):
#         events = list(events)
#         for i in range(len(events)):
#             for j in range(i+1, len(events)):
#                 e1, e2 = events[i], events[j]
#                 if e1 > e2:
#                     e1, e2 = e2, e1
#                 edge_weights[(e1, e2)] += 1  # 每有一个共同实体就+1
#     print(f"[EventGraph] 初始生成边数: {len(edge_weights)}")

#     # 按阈值筛选边
#     src, tgt, weights = [], [], []
#     for (e1, e2), w in tqdm(edge_weights.items()):
#         if w >= 1:  # args.threshold:
#             src.append(e1+checkin_offset)
#             tgt.append(e2+checkin_offset)
#             weights.append(w)
#     print(f"[EventGraph] 生成边数: {len(src)} (阈值: 1)")


#     if len(src) == 0:
#         print("[EventGraph] 无符合阈值的事件边，生成空图。")
#         edge_index = torch.empty((2,0), dtype=torch.long)
#         edge_attr = torch.empty((0,1), dtype=torch.float)
#         event_edge_delta_t = torch.empty((0,1), dtype=torch.float)   # ★ 新增：空 tensor
#         event_edge_delta_s = torch.empty((0,1), dtype=torch.float)   # ★ 新增：空 tensor
#     else:
#         edge_index = torch.tensor([src, tgt], dtype=torch.long)
#         edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)

#         # ★ 新增：计算事件边的时间差 & 空间差
#         times = data.groupby('event_idx')['UTCTimeOffsetEpoch'].first()
#         lats  = data.groupby('event_idx')['latitude'].first()
#         lons  = data.groupby('event_idx')['longitude'].first()

#         print("[EventGraph] 计算事件边的时间差 & 空间差...")

#         delta_t = []
#         delta_s = []
#         for e1, e2 in zip(src, tgt):
#             e1 = e1 - checkin_offset
#             e2 = e2 - checkin_offset
#             t1, t2 = times[e1], times[e2]
#             lat1, lon1 = lats[e1], lons[e1]
#             lat2, lon2 = lats[e2], lons[e2]

#             print(f"Debug Event Edge: e1={e1}, e2={e2}, t1={t1}, t2={t2}, lat1={lat1}, lon1={lon1}, lat2={lat2}, lon2={lon2}")
#             delta_t.append(abs(t1 - t2))
#             # 欧式距离近似（可换成 haversine 距离）
#             delta_s.append(((lat1 - lat2)**2 + (lon1 - lon2)**2) ** 0.5)

#         event_edge_delta_t = torch.tensor(delta_t, dtype=torch.float).unsqueeze(1)
#         event_edge_delta_s = torch.tensor(delta_s, dtype=torch.float).unsqueeze(1)

#     print(f"[EventGraph] 事件边时间差 & 空间差计算完成: {event_edge_delta_t.shape}, {event_edge_delta_s.shape}")
#     # --- 构造事件节点特征矩阵 ---
#     print("[EventGraph] 构造事件节点特征矩阵...")
#     event_features = []
#     eventid2rowidx = data.groupby('ID').apply(lambda df: df.index[0]).to_dict()
#     for eid in event_ids_unique:
#         row = data.loc[eventid2rowidx[eid]]
#         print(">>> before using event2size_map, still type:", type(event2size_map))
#         feat = [
#             float(row['Event_type']),
#             float(row['Intensity']),
#             float(row['latitude']),
#             float(row['longitude']),
#             float(row['UTCTimeOffsetEpoch']),
#             float(event2size_map[eid])  # 事件所包含的实体数量 size
#         ]
#         event_features.append(feat)
#     x = torch.tensor(event_features, dtype=torch.float)

#     print(f"[EventGraph] 特征矩阵形状: {x.shape}, 边数: {edge_index.size(1)}")

#     return Data(
#         x=x,
#         edge_index=edge_index,
#         edge_attr=edge_attr,
#         event_edge_delta_t=event_edge_delta_t,   # ★ 新增
#         event_edge_delta_s=event_edge_delta_s    # ★ 新增
#     )



def generate_traj2traj_data(
        data,
        traj_stat,
        traj_column,
        threshold=0.02,
        filter_mode='min size',
        chunk_num=10,
        relation_type='intra'
):
    """
    生成超边到超边（轨迹到轨迹）的动态关系。

    :param data: 原始轨迹数据
    :param traj_stat: 超边（轨迹）统计信息
    :param traj_column: 轨迹列名
    :param threshold: 过滤噪声关系的阈值
    :param filter_mode: 过滤噪声关系的模式
    :param chunk_num: 快速过滤的块数量
    :param relation_type: 内部或外部，切换不同类型的超边到超边关系
    :return: 超边到超边元组数据(edge_index(coo), edge_type, edge_delta_t和edge_delta_s
    'ID', 'EventChain_id', 'Source_name_encoded', 'Target_name_encoded',
    'Source_Country_encoded', 
    'UTC_time', 'Location_encoded', 'Event_type', 
    'Intensity','latitude', 'longitude','check_ins_id','UTCTimeOffsetEpoch'
    """
    # 初始化原始度量值
    traj2traj_original_metric = None
    # 首先为轨迹->POI创建稀疏矩阵，然后生成用户间邻接表
    # 一个轨迹可能有多个相同的POI ID，我们先删除重复的
    traj_user_map = data[['Source_name_encoded', traj_column]].drop_duplicates().set_index(traj_column)
    traj_size_adjust = None
    if relation_type == 'inter':
        # 对于用户间关系，使用POI映射
        traj_poi_map = data[['PoiId', traj_column]].drop_duplicates()
        # 创建POI到轨迹的稀疏矩阵
        traj2node = coo_matrix((
            np.ones(traj_poi_map.shape[0]),
            (np.array(traj_poi_map['PoiId'], dtype=np.int64), np.array(traj_poi_map[traj_column], dtype=np.int64))
        )).tocsr()

        # 基于新的traj_poi_map调整轨迹ID大小
        traj_size_adjust = traj_poi_map.groupby(traj_column).apply(len).tolist()
    else:
        # 对于用户内关系，使用用户映射
        traj2node = coo_matrix((
            np.ones(traj_user_map.shape[0]),
            (np.array(traj_user_map['UserId'], dtype=np.int64), np.array(traj_user_map.index, dtype=np.int64))
        )).tocsr()

    # 计算节点到轨迹的转置矩阵
    node2traj = traj2node.T
    # 计算轨迹到轨迹的关系矩阵
    traj2traj = node2traj * traj2node
    traj2traj = traj2traj.tocoo()

    # 对于用户间类型，保存原始相似度度量
    if relation_type == 'inter':
        # 使用分块过滤
        row_filtered, col_filtered, data_filtered = filter_chunk(
            row=traj2traj.row,
            col=traj2traj.col,
            data=traj2traj.data,
            chunk_num=chunk_num,
            he_size=traj_size_adjust,
            threshold=0,
            filter_mode=filter_mode
        )
        # 创建原始度量矩阵
        traj2traj_original_metric = coo_matrix((data_filtered, (row_filtered, col_filtered)), shape=traj2traj.shape)

    # 过滤1：基于预定义条件过滤
    # 1. 不同的轨迹 2. 源结束时间 <= 目标开始时间
    mask_1 = traj2traj.row != traj2traj.col
    mask_2 = traj_stat.end_time[traj2traj.col].values <= traj_stat.start_time[traj2traj.row].values
    mask = mask_1 & mask_2
    if relation_type == 'inter':
        # 3. 不同用户
        mask_3 = traj_user_map['UserId'][traj2traj.row].values != traj_user_map['UserId'][traj2traj.col].values
        mask = mask & mask_3

    # 应用过滤掩码
    traj2traj.row = traj2traj.row[mask]
    traj2traj.col = traj2traj.col[mask]
    traj2traj.data = traj2traj.data[mask]

    if relation_type == 'inter':
        # 过滤2：基于预定义度量阈值过滤
        row_filtered, col_filtered, data_filtered = filter_chunk(
            row=traj2traj.row,
            col=traj2traj.col,
            data=traj2traj.data,
            chunk_num=chunk_num,
            he_size=traj_size_adjust,
            threshold=threshold,
            filter_mode=filter_mode
        )
        traj2traj.row = row_filtered
        traj2traj.col = col_filtered
        traj2traj.data = data_filtered
        # 用户间关系类型为1
        edge_type = np.ones_like(traj2traj.row)
    else:
        # 用户内关系类型为0
        edge_type = np.zeros_like(traj2traj.row)

    # 计算edge_delta_t和edge_delta_s
    # 时间差：目标轨迹平均时间 - 源轨迹平均时间
    edge_delta_t = traj_stat.mean_time[traj2traj.row].values - traj_stat.mean_time[traj2traj.col].values
    # 空间坐标：源轨迹平均位置和目标轨迹平均位置
    edge_delta_s = np.stack([
        traj_stat.mean_lon[traj2traj.row].values,
        traj_stat.mean_lat[traj2traj.row].values,
        traj_stat.mean_lon[traj2traj.col].values,
        traj_stat.mean_lat[traj2traj.col].values],
        axis=1
    )

    # 转换为张量并计算地理距离
    edge_delta_s = torch.tensor(edge_delta_s)
    edge_delta_s = haversine(edge_delta_s[:, 0], edge_delta_s[:, 1], edge_delta_s[:, 2], edge_delta_s[:, 3])

    # 记录生成的超边到超边关系数量
    logging.info(
        f"[Preprocess - Generate Hypergraph] Number of {relation_type}-user hyperedge2hyperedge(traj2traj) "
        f"relation has been generated: {traj2traj.row.shape[0]}, while threshold={threshold} and mode={filter_mode}."
    )

    return traj2traj, traj2traj_original_metric, edge_type, edge_delta_t, edge_delta_s.numpy()


def merge_traj2traj_data(traj_stat, intra_u_data, inter_u_data, checkin_offset):
    """
    合并用户内和用户间的超边到超边（轨迹到轨迹）动态关系。

    :param traj_stat: 超边（轨迹）统计信息
    :param intra_u_data: 同一用户内的超边到超边（轨迹到轨迹）关系，由元组组成：
        edge_index(coo), edge_attr(np.array), edge_type(np.array), edge_delta_t(np.array), edge_delta_s(np.array)
    :param inter_u_data: 不同用户间的超边到超边（轨迹到轨迹）关系，组成类似intra_u_data
    :param checkin_offset: 最大签到点索引加1
    :return: 轨迹到轨迹的PyG数据
    """
    # 提取轨迹特征
    traj_feature = traj_stat[['size', 'mean_lon', 'mean_lat', 'mean_time', 'start_time', 'end_time']].to_numpy()

    # 添加两个额外的特征列，确保轨迹特征与签到点特征具有相同的维度大小
    padding_feature = np.zeros([traj_feature.shape[0], 2])
    traj_feature = np.concatenate([traj_feature, padding_feature], axis=1)

    # 解包用户内和用户间数据
    intra_edge_index, _, intra_edge_type, intra_edge_delta_t, intra_edge_delta_s = intra_u_data
    inter_edge_index, traj2traj_orginal_metric, inter_edge_type, inter_edge_delta_t, inter_edge_delta_s = inter_u_data
    # 合并行和列索引
    row = np.concatenate([intra_edge_index.row, inter_edge_index.row])
    col = np.concatenate([intra_edge_index.col, inter_edge_index.col])

    # 用度量值替换数据
    metric_data = coo_matrix((np.ones(row.shape[0]), (row, col)), shape=traj2traj_orginal_metric.shape)
    epsilon = coo_matrix((np.zeros(row.shape[0]) + 1e-6, (row, col)), shape=traj2traj_orginal_metric.shape)
    metric_data = metric_data.multiply(traj2traj_orginal_metric)
    metric_data += epsilon

    # 创建稀疏张量
    adj_t = SparseTensor(
        row=torch.as_tensor(row, dtype=torch.long),
        col=torch.as_tensor(col, dtype=torch.long),
        value=torch.as_tensor(range(0, row.shape[0]), dtype=torch.long)
    )
    # 获取排列索引
    perm = adj_t.storage.value()

    # 创建特征张量
    x = torch.tensor(traj_feature)
    # 合并边类型
    edge_type = torch.tensor(np.concatenate([intra_edge_type, inter_edge_type]))[perm]
    # 合并时间差
    edge_delta_t = torch.tensor(np.concatenate([intra_edge_delta_t, inter_edge_delta_t]))[perm]
    # 合并空间距离
    edge_delta_s = torch.tensor(np.concatenate([intra_edge_delta_s, inter_edge_delta_s]))[perm]

    # 创建边索引（加上偏移量）
    edge_index = torch.stack([
        adj_t.storage.col() + checkin_offset,
        adj_t.storage.row() + checkin_offset
    ])

    # edge_attr: 源大小, 目标大小, jaccard相似度
    source_size = x[edge_index[0] - checkin_offset][:, 0] / x[:, 0].max()
    target_size = x[edge_index[1] - checkin_offset][:, 0] / x[:, 0].max()
    edge_attr = torch.stack([source_size, target_size, torch.tensor(metric_data.data)], dim=1)

    # 创建轨迹到轨迹的PyG数据对象
    traj2traj_pyg_data = Data(
        edge_index=edge_index,
        x=x,
        edge_attr=edge_attr,
        edge_type=edge_type,
        edge_delta_t=edge_delta_t,
        edge_delta_s=edge_delta_s
    )
    return traj2traj_pyg_data


def filter_chunk(row, col, data, he_size, chunk_num=10, threshold=0.02, filter_mode='min size'):
    """
    基于度量阈值过滤噪声超边到超边连接

    :param row: 行，超边到超边scipy.sparse coo矩阵
    :param col: 列，超边到超边scipy.sparse coo矩阵
    :param data: 数据，超边到超边scipy.sparse coo矩阵
    :param he_size: 超边大小列表（删除重复项）
    :param chunk_num: 块数量，防止内存溢出问题
    :param threshold: 度量阈值，只有当度量值大于阈值时才保留关系
    :param filter_mode: min_size - 与最小大小成比例, 'jaccard' - jaccard相似度
        min_size, 当E2E_{ij} \ge \theta\min(|\mathcal{E}_i|,|\mathcal{E}_j|)时保留E2E_{ij}
        jaccard, 当\frac{E2E_{ij}}{|\mathcal{E}_i|+|\mathcal{E}_j| - E2E_{ij}} \ge \theta时保留
    :return: 过滤后的行、列、数据
    """
    print(f"[FilterChunk] 开始分块过滤 | chunks={chunk_num} | threshold={threshold} | mode={filter_mode}")
    # 将数据分割为多个块以处理大数据
    chunk_bin = np.linspace(0, row.shape[0], chunk_num, dtype=np.int64)
    rows, cols, datas = [], [], []
    # 遍历每个块
    for i in tqdm(range(len(chunk_bin) - 1)):
        # 获取当前块的数据
        row_chunk = row[chunk_bin[i]:chunk_bin[i + 1]]
        col_chunk = col[chunk_bin[i]:chunk_bin[i + 1]]
        data_chunk = data[chunk_bin[i]:chunk_bin[i + 1]]
        # 获取源和目标的大小
        source_size = np.array(list(map(he_size.__getitem__, row_chunk.tolist())))
        target_size = np.array(list(map(he_size.__getitem__, col_chunk.tolist())))
        if filter_mode == 'min size':
            # 与最小大小成比例
            metric = data_chunk / np.minimum(source_size, target_size)
        else:
            # jaccard相似度
            metric = data_chunk / (source_size + target_size - data_chunk)
        # 应用阈值过滤
        filter_mask = metric >= threshold
        # 保存过滤后的数据
        rows.append(row_chunk[filter_mask])
        cols.append(col_chunk[filter_mask])
        datas.append(metric[filter_mask])

    # 连接所有块的结果
    out_rows, out_cols, out_datas = np.concatenate(rows), np.concatenate(cols), np.concatenate(datas)
    print(f"[FilterChunk] 过滤完成 | 保留边数={out_rows.shape[0]}")
    return out_rows, out_cols, out_datas


def generate_ci2traj_pyg_data(data, traj_stat, traj_column, checkin_offset, threshold=0.02, filter_mode='min size'):
    return 0
