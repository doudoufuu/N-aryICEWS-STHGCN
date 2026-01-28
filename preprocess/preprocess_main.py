
#1995-2022 icews_nary数据集使用的预处理文件
# 导入必要的库
import os  # 操作系统接口

import json
import math
import csv
import pickle  # 用于序列化和反序列化Python对象
import pandas as pd  # 数据处理库
import numpy as np  # 数值计算库
from collections import defaultdict
from datetime import datetime, time  # 日期时间处理
import os.path as osp  # 路径处理工具
from typing import Dict, Iterable, List, Optional, Set, Tuple
import random

from utils import Cfg, get_root_dir  # 导入配置类和根目录获取函数
from preprocess import (
    ignore_first,  # 忽略轨迹中的第一个签到点
    only_keep_last,  # 只保留轨迹中的最后一个签到点
    id_encode,  # ID编码函数
    remove_unseen_user_poi,  # 移除未见过的用户和POI
    FileReader,  # 文件读取器
    generate_hypergraph_from_file  # 从文件生成超图
)
import logging  # 日志记录
from sklearn.preprocessing import LabelEncoder

import time


def _to_list(value: Optional[Iterable], *, allow_str: bool = False) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str) and not allow_str:
        return [value]
    return [value]


CSV_EVENTS_KEEP_COLS = [
    'latitude', 'longitude', 'timezone', 'UTC_time', 'local_time', 'day_of_week',
    'norm_in_day_time', 'EventChain_id', 'norm_day_shift', 'norm_relative_time',
    'Intensity', 'ID', 'Event_type', 'EventText', 'Structure_Type',
    'Source_name_encoded', 'Target_name_encoded', 'Source_Country_encoded',
    'Target_Country_encoded', 'Location_encoded', 'UTCTime', 'UTCTimeOffset',
    'UTCTimeOffsetEpoch', 'SplitTag'
]

RAW_EVENTS_CSV_COLUMNS = [
    'latitude', 'longitude', 'timezone', 'UTC_time', 'local_time', 'day_of_week',
    'norm_in_day_time', 'EventChain_id', 'norm_day_shift', 'norm_relative_time',
    'Intensity', 'ID', 'Event_type', 'EventText', 'Structure_Type',
    'Source_name_encoded', 'Target_name_encoded', 'Source_Country_encoded',
    'Target_Country_encoded', 'Location_encoded', 'Source_Sectors_encoded',
    'Target_Sectors_encoded', 'Chain_order'
]

DEFAULT_COMBINED_EVENTS_JSONL = "/home/beihang/hsy/ICEWS/1995-2022_combined_events.jsonl"
DEFAULT_DEMO_COMBINED_EVENTS_JSONL = osp.join(
    osp.dirname(__file__),
    "1995-2022_combined_events_demo_3000.jsonl",
)


def _coerce_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_event_type(value):
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _parse_day_of_week(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str), "%Y-%m-%d").weekday()
    except Exception:
        try:
            return datetime.fromisoformat(str(date_str)).weekday()
        except Exception:
            return None


def _extract_lat_lon(record, fallback_location):
    lat = lon = None
    if isinstance(record, dict):
        lat = _coerce_float(record.get("Latitude"))
        lon = _coerce_float(record.get("Longitude"))
        if lat is None or lon is None:
            loc = record.get("Location")
            if isinstance(loc, dict):
                if lat is None:
                    lat = _coerce_float(loc.get("Latitude"))
                if lon is None:
                    lon = _coerce_float(loc.get("Longitude"))
    if (lat is None or lon is None) and isinstance(fallback_location, dict):
        if lat is None:
            lat = _coerce_float(fallback_location.get("Latitude"))
        if lon is None:
            lon = _coerce_float(fallback_location.get("Longitude"))
    return lat, lon


def _iter_combined_event_rows(event):
    structure_type = event.get("Structure_Type")
    event_type = _parse_event_type(event.get("EventType"))
    if event_type is None:
        event_type = -1
    event_text = event.get("EventText") or ""
    time_str = str(event.get("Time") or "")
    location_key = event.get("Location_Key") or ""
    location = event.get("Location") if isinstance(event.get("Location"), dict) else {}
    day_of_week = _parse_day_of_week(time_str)

    base_row = {
        "timezone": 0,
        "UTC_time": time_str,
        "local_time": time_str,
        "day_of_week": day_of_week if day_of_week is not None else "",
        "norm_in_day_time": 0.0,
        "norm_day_shift": 0.0,
        "norm_relative_time": 0.0,
        "Event_type": event_type,
        "EventText": event_text,
        "Structure_Type": structure_type,
        "Location_encoded": location_key,
        "EventChain_id": "",
        "Source_Sectors_encoded": "",
        "Target_Sectors_encoded": "",
        "Chain_order": "",
    }

    def _emit_row(record, source_name, target_name, source_country, target_country, intensity, event_id):
        lat, lon = _extract_lat_lon(record, location)
        row = base_row.copy()
        row.update({
            "latitude": lat if lat is not None else "",
            "longitude": lon if lon is not None else "",
            "Intensity": intensity,
            "ID": event_id,
            "Source_name_encoded": source_name,
            "Target_name_encoded": target_name,
            "Source_Country_encoded": source_country,
            "Target_Country_encoded": target_country,
        })
        return row

    if structure_type == "multi-to-multi":
        records = event.get("Records") or []
        for record in records:
            if not isinstance(record, dict):
                continue
            yield _emit_row(
                record,
                record.get("Source_Name"),
                record.get("Target_Name"),
                record.get("Source_Country"),
                record.get("Target_Country"),
                record.get("Intensity"),
                record.get("Event_ID"),
            )
    elif structure_type == "one-to-many":
        targets = event.get("Targets") or []
        source_name = event.get("Source")
        source_country = event.get("Source_Country")
        for record in targets:
            if not isinstance(record, dict):
                continue
            yield _emit_row(
                record,
                source_name,
                record.get("Target_Name"),
                source_country,
                record.get("Target_Country"),
                record.get("Intensity"),
                record.get("Event_ID"),
            )
    elif structure_type == "many-to-one":
        sources = event.get("Sources") or []
        target_name = event.get("Target_Name")
        target_country = event.get("Target_Country")
        for record in sources:
            if not isinstance(record, dict):
                continue
            yield _emit_row(
                record,
                record.get("Source"),
                target_name,
                record.get("Source_Country"),
                target_country,
                record.get("Intensity"),
                record.get("Event_ID"),
            )
    else:
        logging.warning("[CSV Bridge] Unknown Structure_Type: %s", structure_type)


def _generate_eventsid_chainid_csv(jsonl_path, csv_path):
    print(f"[CSV Bridge] Generating {csv_path} from {jsonl_path}")
    total_events = 0
    total_rows = 0
    bad_lines = 0
    with open(jsonl_path, "r", encoding="utf-8") as src, open(csv_path, "w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=RAW_EVENTS_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for line_no, line in enumerate(src, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                bad_lines += 1
                if bad_lines <= 5:
                    logging.warning("[CSV Bridge] JSON decode error at line %d: %s", line_no, exc)
                continue
            total_events += 1
            for row in _iter_combined_event_rows(event):
                writer.writerow(row)
                total_rows += 1
            if line_no % 10000 == 0:
                print(f"[CSV Bridge] Processed {line_no} events, rows={total_rows}")
    print(f"[CSV Bridge] Done. events={total_events}, rows={total_rows}, bad_lines={bad_lines}")


def _ensure_eventsid_chainid_csv(raw_path, cfg):
    csv_path = osp.join(raw_path, "eventsid_chainid.csv")
    if osp.exists(csv_path):
        return csv_path

    dataset_args = getattr(cfg, "dataset_args", None)
    jsonl_path = None
    if dataset_args is not None:
        jsonl_path = getattr(dataset_args, "combined_events_jsonl", None) or getattr(
            dataset_args, "combined_events_jsonl_path", None
        )
    if not jsonl_path:
        jsonl_path = DEFAULT_COMBINED_EVENTS_JSONL
    if not osp.exists(jsonl_path) and osp.exists(DEFAULT_DEMO_COMBINED_EVENTS_JSONL):
        jsonl_path = DEFAULT_DEMO_COMBINED_EVENTS_JSONL
    if not osp.exists(jsonl_path):
        raise FileNotFoundError(f"[CSV Bridge] JSONL not found: {jsonl_path}")

    os.makedirs(raw_path, exist_ok=True)
    _generate_eventsid_chainid_csv(jsonl_path, csv_path)
    return csv_path


def _restore_progress_df(preprocessed_path, csv_file, stage):
    temp_file = osp.join(preprocessed_path, "temp_df.pkl")
    if osp.exists(temp_file):
        print(f"[CSV Events] Restore df for {stage}: {temp_file}")
        return pd.read_pickle(temp_file)
    print(f"[CSV Events] temp_df.pkl missing for {stage}, re-reading CSV.")
    return pd.read_csv(csv_file)


def _ensure_raw_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        'latitude': '',
        'longitude': '',
        'timezone': 0,
        'UTC_time': '',
        'local_time': '',
        'day_of_week': '',
        'norm_in_day_time': 0.0,
        'norm_day_shift': 0.0,
        'norm_relative_time': 0.0,
        'Intensity': '',
        'ID': '',
        'Event_type': '',
        'EventText': '',
        'Structure_Type': '',
        'Source_name_encoded': '',
        'Target_name_encoded': '',
        'Source_Country_encoded': '',
        'Target_Country_encoded': '',
        'Location_encoded': '',
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def _collect_min_occurrence_indices(
    df: pd.DataFrame,
    columns: Iterable[str],
    min_occurrence: int,
) -> Tuple[Set[int], Dict[str, int]]:
    """
    Return row indices that must stay in the training split so every value
    in the provided columns appears at least ``min_occurrence`` times.
    """
    forced_indices: Set[int] = set()
    per_column_forced: Dict[str, int] = defaultdict(int)
    if min_occurrence <= 0:
        return forced_indices, per_column_forced

    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        return forced_indices, per_column_forced

    column_arrays = {col: df[col].to_numpy() for col in valid_columns}
    occurrence_counters: Dict[Tuple[str, object], int] = defaultdict(int)

    total_rows = len(df)
    for row_idx in range(total_rows):
        for col in valid_columns:
            val = column_arrays[col][row_idx]
            if pd.isna(val):
                continue
            key = (col, val)
            count = occurrence_counters[key]
            if count < min_occurrence:
                if row_idx not in forced_indices:
                    forced_indices.add(row_idx)
                per_column_forced[col] += 1
            occurrence_counters[key] = count + 1
    return forced_indices, per_column_forced


def _deduplicate_eval_by_event(
    df: pd.DataFrame,
    split_tag: str,
    event_field: str,
    chain_field: Optional[str] = None,
) -> Tuple[pd.DataFrame, int]:
    if event_field not in df.columns:
        return df, 0
    mask = df["SplitTag"] == split_tag
    subset = df.loc[mask]
    if subset.empty:
        return df, 0
    dedup_columns = [event_field]
    if chain_field and chain_field in df.columns:
        dedup_columns.append(chain_field)
    duplicated = subset.duplicated(dedup_columns, keep="first")
    drop_indices = subset[duplicated].index
    if not len(drop_indices):
        return df, 0
    df = df.drop(index=drop_indices)
    return df, len(drop_indices)


def _sort_events_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "UTCTimeOffsetEpoch" in df.columns:
        sort_series = pd.to_numeric(df["UTCTimeOffsetEpoch"], errors="coerce")
        df = df.assign(_sort_time=sort_series)
        df = df.sort_values(
            ["_sort_time", "UTCTimeOffset", "check_ins_id"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        df = df.drop(columns=["_sort_time"])
    elif "UTCTimeOffset" in df.columns:
        df = df.sort_values(["UTCTimeOffset", "check_ins_id"], ascending=[True, True]).reset_index(drop=True)
    else:
        df = df.sort_values("check_ins_id").reset_index(drop=True)
    return df


def _apply_chain_position_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    if "EventChain_id" not in df.columns or "UTCTimeOffset" not in df.columns:
        return df, 0, 0
    work_df = df.copy()
    work_df["EventChain_id"] = work_df["EventChain_id"].fillna("unknown_chain")
    work_df = work_df.sort_values(["EventChain_id", "UTCTimeOffset", "check_ins_id"]).reset_index(drop=True)
    work_df["chain_rank"] = work_df.groupby("EventChain_id").cumcount() + 1
    work_df["chain_size"] = work_df.groupby("EventChain_id")["chain_rank"].transform("max")

    ignored_first = int((work_df["chain_rank"] == 1).sum())
    work_df.loc[work_df["chain_rank"] == 1, "SplitTag"] = "ignore"

    val_test_mask = work_df["SplitTag"].isin(["validation", "test"])
    non_last_mask = work_df["chain_rank"] != work_df["chain_size"]
    ignored_non_tail = int((val_test_mask & non_last_mask).sum())
    work_df.loc[val_test_mask & non_last_mask, "SplitTag"] = "ignore"

    work_df = work_df[work_df["SplitTag"] != "ignore"].copy()
    work_df.drop(columns=["chain_rank", "chain_size"], inplace=True, errors="ignore")
    work_df["EventChain_id"] = work_df["EventChain_id"].replace("unknown_chain", np.nan)
    work_df = work_df.sort_values("UTCTimeOffset").reset_index(drop=True)
    return work_df, ignored_first, ignored_non_tail


def _sanitize_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure UTCTimeOffsetEpoch stays numeric and UTCTimeOffset remains datetime.
    """
    if "UTCTimeOffsetEpoch" in df.columns:
        epoch_series = pd.to_numeric(df["UTCTimeOffsetEpoch"], errors="coerce")
        null_count = int(epoch_series.isna().sum())
        if null_count:
            logging.warning("[Preprocess] UTCTimeOffsetEpoch had %d non-numeric entries; filling with 0.", null_count)
        df["UTCTimeOffsetEpoch"] = epoch_series.fillna(0).astype(np.int64)
    if "UTCTimeOffset" in df.columns:
        df["UTCTimeOffset"] = pd.to_datetime(df["UTCTimeOffset"], errors="coerce")
    return df


def _split_with_coverage(df: pd.DataFrame, cfg: Cfg) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = _sort_events_dataframe(df)
    dataset_args = getattr(cfg, "dataset_args", None)
    seed_val = int(getattr(cfg.run_args, "seed", 42) or 42)

    coverage_strategy = str(getattr(dataset_args, "coverage_strategy", "entity")).lower()
    if coverage_strategy not in {"entity", "event", "none"}:
        logging.warning("[Split] Unknown coverage_strategy=%s, fallback to 'entity'", coverage_strategy)
        coverage_strategy = "entity"

    entity_fields = _to_list(getattr(dataset_args, "entity_coverage_fields", ["Source_name_encoded"]))
    event_field = getattr(dataset_args, "coverage_event_field", "ID")
    min_entity_occ = int(getattr(dataset_args, "min_entity_train_occurrence", 1))
    min_event_occ = int(getattr(dataset_args, "min_event_train_occurrence", 1))
    dedup_eval = bool(getattr(dataset_args, "dedup_eval_by_event", True))
    dedup_chain_field = getattr(dataset_args, "dedup_chain_field", "EventChain_id")

    total_rows = len(df)
    if total_rows == 0:
        return df, {}

    if "EventChain_id" not in df.columns:
        df["EventChain_id"] = -1

    chain_times = (
        df.groupby("EventChain_id")["UTCTimeOffsetEpoch"]
        .max()
        .sort_values()
    )
    chain_order = chain_times.index.tolist()
    total_chains = len(chain_order)
    if total_chains == 0:
        df["SplitTag"] = "train"
        return df, {"total_rows": total_rows, "train_rows": total_rows, "validation_rows": 0, "test_rows": 0}

    def _compute_targets(n_chains: int) -> Tuple[int, int, int]:
        tgt_train = max(1, int(round(n_chains * 0.7)))
        tgt_val = max(1, int(round(n_chains * 0.2)))
        tgt_test = n_chains - tgt_train - tgt_val
        if tgt_test <= 0:
            tgt_test = 1
            if tgt_train > tgt_val:
                tgt_train -= 1
            else:
                tgt_val -= 1
        if tgt_val <= 0 and n_chains >= 2:
            tgt_val = 1
            tgt_train = max(1, n_chains - tgt_val - tgt_test)
        return tgt_train, tgt_val, max(1, n_chains - tgt_train - tgt_val)

    target_train, target_val, target_test = _compute_targets(total_chains)
    chain_splits: Dict[str, Set] = {
        "train": set(chain_order[:target_train]),
        "validation": set(chain_order[target_train:target_train + target_val]),
        "test": set(chain_order[target_train + target_val:]),
    }

    report: Dict[str, int] = {
        "total_rows": total_rows,
        "total_chains": total_chains,
        "target_train_chains": target_train,
        "target_val_chains": target_val,
        "target_test_chains": target_test,
        "coverage_strategy": coverage_strategy,
        "min_entity_train_occurrence": min_entity_occ,
        "min_event_train_occurrence": min_event_occ,
    }

    def _collect_entity_sets(chain_ids: Set) -> Dict[str, Set]:
        subset = df[df["EventChain_id"].isin(chain_ids)]
        entity_sets_local: Dict[str, Set] = {}
        for field in entity_fields:
            if field in subset.columns:
                entity_sets_local[field] = set(subset[field].dropna().tolist())
        return entity_sets_local

    forced_train_chains: Set = set()
    coverage_details: Dict[str, Dict[str, int]] = {}

    if coverage_strategy == "entity":
        entity_sets = _collect_entity_sets(chain_splits["train"])
        coverage_details["entity"] = {field: 0 for field in entity_fields}

        for split in ("validation", "test"):
            split_chains = chain_splits[split].copy()
            if not split_chains:
                continue
            subset = df[df["EventChain_id"].isin(split_chains)]
            violations: Set = set()
            for field in entity_fields:
                if field not in subset.columns:
                    continue
                allowed = entity_sets.get(field, set())
                invalid_chains = set(subset.loc[~subset[field].isin(allowed), "EventChain_id"])
                if invalid_chains:
                    coverage_details["entity"][field] = coverage_details["entity"].get(field, 0) + len(invalid_chains)
                violations |= invalid_chains
            if violations:
                chain_splits[split] -= violations
                chain_splits["train"].update(violations)
                forced_train_chains.update(violations)
                entity_sets = _collect_entity_sets(chain_splits["train"])
    elif coverage_strategy == "event":
        # 对事件覆盖的实现：确保验证/测试中的事件至少在训练链出现 min_event_occ 次
        per_chain = df.groupby("EventChain_id")
        forced_chains = set()
        for split in ("validation", "test"):
            split_chains = chain_splits[split].copy()
            if not split_chains:
                continue
            for chain_id in split_chains:
                chain_df = per_chain.get_group(chain_id)
                events = chain_df[event_field]
                occurrences = events.value_counts()
                if (occurrences < min_event_occ).any():
                    forced_chains.add(chain_id)
        if forced_chains:
            chain_splits["validation"] -= forced_chains
            chain_splits["test"] -= forced_chains
            chain_splits["train"].update(forced_chains)
            forced_train_chains.update(forced_chains)
            coverage_details["event"] = {event_field: len(forced_chains)}

    chain_position = {chain_id: idx for idx, chain_id in enumerate(chain_order)}

    def _rebalance_split(split: str, target: int) -> int:
        current = len(chain_splits[split])
        deficit = target - current
        if deficit <= 0:
            return 0
        transferable = sorted(
            [cid for cid in chain_splits["train"] if cid not in forced_train_chains],
            key=lambda cid: chain_position[cid],
            reverse=True,
        )
        moved = 0
        for cid in transferable:
            if moved >= deficit:
                break
            chain_splits["train"].remove(cid)
            chain_splits[split].add(cid)
            moved += 1
        return deficit - moved

    remaining_val = _rebalance_split("validation", target_val)
    remaining_test = _rebalance_split("test", target_test)
    if remaining_val > 0 or remaining_test > 0:
        logging.warning(
            "[Split] Unable to fully rebalance splits (val deficit=%d, test deficit=%d).",
            max(0, remaining_val),
            max(0, remaining_test),
        )

    report.update({
        "forced_train_chains": len(forced_train_chains),
        "final_train_chains": len(chain_splits["train"]),
        "final_val_chains": len(chain_splits["validation"]),
        "final_test_chains": len(chain_splits["test"]),
    })

    for key, value in coverage_details.items():
        for col, count in value.items():
            report_key = f"{key}_forced_{col}"
            report[report_key] = count

    chain_to_split = {}
    for split, chains in chain_splits.items():
        for cid in chains:
            chain_to_split[cid] = split

    df["SplitTag"] = df["EventChain_id"].map(chain_to_split).fillna("train")

    tail_time = df.groupby("EventChain_id")["UTCTimeOffsetEpoch"].transform("max")
    is_tail = df["UTCTimeOffsetEpoch"] == tail_time
    eval_mask = df["SplitTag"].isin(["validation", "test"])
    non_tail_eval = eval_mask & (~is_tail)
    ignored_eval_rows = int(non_tail_eval.sum())
    if ignored_eval_rows > 0:
        # 保留完整 sample，但将验证/测试链的非尾事件标记为 ignore，避免参与评测
        df.loc[non_tail_eval, "SplitTag"] = "ignore"
    report["ignored_eval_non_tail"] = ignored_eval_rows

    eval_mask = df["SplitTag"].isin(["validation", "test"])
    rng = random.Random(seed_val)
    duplicates: List[int] = []
    eval_subset = df.loc[eval_mask]
    for chain_id, grp in eval_subset.groupby("EventChain_id"):
        if grp.empty:
            continue
        max_time = grp["UTCTimeOffsetEpoch"].max()
        tail_rows = grp[grp["UTCTimeOffsetEpoch"] == max_time]
        if len(tail_rows) > 1:
            keep_idx = rng.choice(tail_rows.index.tolist())
            duplicates.extend(idx for idx in tail_rows.index if idx != keep_idx)
    if duplicates:
        df = df.drop(index=duplicates).reset_index(drop=True)

    return df, report
def create_event_chains(df: pd.DataFrame, *, chunk_size: int = 8) -> pd.Series:
    """
    按“同一 Source_name_encoded 的连续 chunk_size 个事件”切分事件链。
    时间窗口不再使用，直接按时间顺序将每个 source 的事件按 8 条一组分链。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if "Source_name_encoded" not in df.columns:
        raise ValueError("缺少 Source_name_encoded 列，无法构建事件链。")

    print(f"[Event Chains] 基于 Source_name，每 {chunk_size} 个事件切分事件链...")

    # 对时间列做健壮排序
    sort_time = pd.to_numeric(df.get("UTCTimeOffsetEpoch", pd.Series(index=df.index, dtype="float")), errors="coerce")
    df_sorted = df.assign(_sort_time=sort_time).sort_values("_sort_time").drop(columns=["_sort_time"])

    event_chains: Dict[int, str] = {}
    # 按 source 分组，再按时间排序后每 chunk_size 条切分
    for src_idx, (source, group) in enumerate(
        df_sorted.groupby("Source_name_encoded", dropna=False), start=1
    ):
        group = group.sort_values("UTCTimeOffsetEpoch")
        idx_list = list(group.index)
        if not idx_list:
            continue
        total = len(idx_list)
        for i in range(0, total, chunk_size):
            chunk = idx_list[i : i + chunk_size]
            chain_id = f"{source}_{i // chunk_size}"
            for idx in chunk:
                event_chains[idx] = chain_id

        if src_idx % 100 == 0:
            print(f"[Event Chains] 已处理 Source {src_idx}")

    result_series = pd.Series(event_chains, name="EventChain_id")
    result_series = result_series.reindex(df.index)
    result_series = result_series.fillna("unassigned")
    print(f"[Event Chains] 创建完毕，链数量: {result_series.nunique()} (含 unassigned)")
    return result_series


def filter_symmetric_events(events_df: pd.DataFrame, target_country: int) -> pd.DataFrame:
    """
        过滤对称事件，只保留以指定国家为源的事件

    Args:
        events_df: 事件数据框
        target_country: 目标国家编码

    Returns:
        过滤对称事件，只保留以指定国家为源的事件，并根据Event_type对称表进一步筛选。
        只有主体互换且Event_type为对称型、时间/地点/Intensity相同才算对称事件，否则只做去重。
    """
    print(f"[Symmetric Filter] 开始对称事件过滤，目标国家: {target_country}")
    print(f"[Symmetric Filter] 输入事件数量: {len(events_df)}")

    symmetric_type_map = {
        42: 43, 43: 42,
        73: 74, 74: 73,
        112: 113, 113: 112,
        31: 32, 32: 31,
        53: 52, 52: 53,
        70: 71, 71: 70,
        121: 121
    }

    def _ensure_dict(row):
        if isinstance(row, dict):
            return row
        return row.to_dict() if hasattr(row, 'to_dict') else dict(row)

    # 按事件特征分组，识别对称事件
    print("[Symmetric Filter] 创建事件特征键...")
    events_df = events_df.copy()
    events_df['event_key'] = events_df.apply(lambda row: create_event_key(row), axis=1)

    print("[Symmetric Filter] 按事件特征分组...")
    symmetric_groups = events_df.groupby('event_key')
    print(f"[Symmetric Filter] 分组数量: {len(symmetric_groups)}")

    filtered_events = []
    group_count = 0
    symmetric_count = 0

    for event_key, group in symmetric_groups:
        group_count += 1
        if group_count % 1000 == 0:
            print(f"[Symmetric Filter] 处理进度: {group_count}/{len(symmetric_groups)} 组")

        if len(group) == 1:
            event = _ensure_dict(group.iloc[0])
            if target_country is None or event.get('Source_Country_encoded') == target_country:
                filtered_events.append(event)
        else:
            used = set()
            group_list = list(group.to_dict('records'))
            for i, row_i in enumerate(group_list):
                if i in used:
                    continue
                found_symmetric = False
                for j, row_j in enumerate(group_list):
                    if i == j or j in used:
                        continue
                    if (row_i.get('Source_name_encoded') == row_j.get('Target_name_encoded') and
                        row_i.get('Target_name_encoded') == row_j.get('Source_name_encoded')):
                        et_i = row_i.get('Event_type')
                        et_j = row_j.get('Event_type')
                        if symmetric_type_map.get(et_i) == et_j:
                            loc_i = row_i.get('Location_encoded', 'unknown')
                            loc_j = row_j.get('Location_encoded', 'unknown')
                            if (row_i.get('UTC_time') == row_j.get('UTC_time') and
                                loc_i == loc_j and
                                row_i.get('Intensity') == row_j.get('Intensity')):
                                symmetric_count += 1
                                if target_country is None:
                                    keep = row_i if et_i <= et_j else row_j
                                    filtered_events.append(_ensure_dict(keep))
                                else:
                                    if row_i.get('Source_Country_encoded') == target_country and et_i <= et_j:
                                        filtered_events.append(_ensure_dict(row_i))
                                    elif row_j.get('Source_Country_encoded') == target_country and et_j < et_i:
                                        filtered_events.append(_ensure_dict(row_j))
                                    else:
                                        keep = row_i if et_i <= et_j else row_j
                                        filtered_events.append(_ensure_dict(keep))
                                used.add(i)
                                used.add(j)
                                found_symmetric = True
                                break
                if not found_symmetric and i not in used:
                    if target_country is None:
                        filtered_events.append(_ensure_dict(row_i))
                    elif row_i.get('Source_Country_encoded') == target_country:
                        filtered_events.append(_ensure_dict(row_i))
                    elif row_i.get('Target_Country_encoded') == target_country:
                        filtered_events.append(_ensure_dict(row_i))
                    else:
                        filtered_events.append(_ensure_dict(row_i))
                    used.add(i)

    if filtered_events:
        result_df = pd.DataFrame(filtered_events)
        print(f"[Symmetric Filter] 对称事件过滤完成")
        print(f"[Symmetric Filter] 发现对称事件对: {symmetric_count}")
        print(f"[Symmetric Filter] 原始事件: {len(events_df)} 个，过滤后: {len(result_df)} 个")
        print(f"[Symmetric Filter] 过滤率: {(len(events_df) - len(result_df)) / len(events_df) * 100:.2f}%")
        return result_df
    else:
        print(f"[Symmetric Filter] 过滤后无事件")
        return pd.DataFrame()


def create_event_key(row):
    """
    创建事件特征键，用于识别对称事件
    
    Args:
        row: 事件行数据
    
    Returns:
        事件特征键
    """
    # 获取事件特征
    source = row['Source_name_encoded']
    target = row['Target_name_encoded']
    location = row.get('Location_encoded', 'unknown')  # 如果不存在Location_encoded，使用unknown
    time = row['UTC_time']
    intensity = row['Intensity']
    
    # 创建对称事件的特征键（实体无序对）
    entity_pair = "_".join(sorted([str(source), str(target)]))
    
    # 组合特征：实体对_位置_时间_强度
    event_key = f"{entity_pair}_{location}_{time}_{intensity}"
    
    return event_key


def haversine(lat1, lon1, lat2, lon2):
    """
    计算两个地理坐标之间的距离（公里）
    
    Args:
        lat1, lon1: 第一个坐标的纬度和经度
        lat2, lon2: 第二个坐标的纬度和经度
    
    Returns:
        距离（公里）
    """
    from math import radians, cos, sin, asin, sqrt
    
    # 将经纬度转换为弧度
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # 计算差值
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine公式
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # 地球半径（公里）
    r = 6371
    
    return c * r

def time_series_split_with_id_check(df, id_columns, time_col='UTCTimeOffset', 
                                  test_size=0.1, val_size=0.1):
    """
    时间序列友好的分割方法，同时保证ID一致性
    
    参数:
        df: 原始DataFrame
        id_columns: 需要检查的ID列名列表
        time_col: 时间戳列名
        test_size: 测试集比例
        val_size: 验证集比例
    
    返回:
        添加了SplitTag列的DataFrame
    """
    # 1. 按时间排序
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # 2. 获取每个ID组合的最后出现时间
    id_last_occurrence = df.groupby(id_columns)[time_col].max().reset_index()
    
    # 3. 计算分割时间点（基于ID的最后出现时间）
    total_size = test_size + val_size
    split_time = id_last_occurrence[time_col].quantile(1 - total_size)
    val_time = id_last_occurrence[time_col].quantile(1 - test_size)
    
    # 4. 分配ID组合到不同数据集
    train_ids = id_last_occurrence[id_last_occurrence[time_col] <= split_time][id_columns]
    val_ids = id_last_occurrence[(id_last_occurrence[time_col] > split_time) & 
                               (id_last_occurrence[time_col] <= val_time)][id_columns]
    test_ids = id_last_occurrence[id_last_occurrence[time_col] > val_time][id_columns]
    
    # 5. 标记原始数据
    df['SplitTag'] = 'train'  # 默认全部训练集
    
    # 使用merge筛选验证集和测试集
    val_mask = df.merge(val_ids, on=id_columns, how='inner').index
    test_mask = df.merge(test_ids, on=id_columns, how='inner').index
    
    df.loc[val_mask, 'SplitTag'] = 'validation'
    df.loc[test_mask, 'SplitTag'] = 'test'
    
    # 6. 确保时间顺序严格性（验证集时间>=训练集最大时间）
    min_val_time = df[df['SplitTag']=='validation'][time_col].min()
    max_train_time = df[df['SplitTag']=='train'][time_col].max()
    if min_val_time < max_train_time:
        print(f"  警告: 调整时间边界以保持严格时间顺序 (原验证集最早时间: {min_val_time}, 训练集最晚时间: {max_train_time})")
        # 将时间早于训练集最大时间的验证样本划归训练集
        df.loc[(df['SplitTag']=='validation') & (df[time_col] <= max_train_time), 'SplitTag'] = 'train'
    
    return df

# 打印数据基本信息函数
def print_data_info(df, df_train, fields_to_encode):
    print("\n===== 数据基本信息 =====")
    # 1. 数据规模信息
    print(f"\n1. 数据规模:")
    print(f"总数据集形状: {df.shape} (行数: {len(df):,}, 列数: {len(df.columns)})")
    print(f"训练集形状: {df_train.shape} (行数: {len(df_train):,}, 列数: {len(df_train.columns)})")
    
    # 2. 内存使用情况
    print(f"\n2. 内存使用:")
    print(f"总数据集内存使用: {df.memory_usage(deep=True).sum()/1024/1024:.2f} MB")
    print(f"训练集内存使用: {df_train.memory_usage(deep=True).sum()/1024/1024:.2f} MB")
    
    # 3. 字段信息
    print("\n3. 待编码字段详细信息:")
    for field in fields_to_encode:
        if field in df.columns:
            print(f"\n字段: {field}")
            print(f"  类型: {df[field].dtype}")
            print(f"  训练集唯一值数量: {df_train[field].nunique():,}")
            print(f"  全量数据唯一值数量: {df[field].nunique():,}")
            print(f"  缺失值数量: {df[field].isna().sum():,}")
            
            # 对于对象类型，显示示例值
            if df[field].dtype == 'object':
                print(f"  示例值: {df[field].dropna().unique()[:5]}")
        else:
            print(f"\n字段: {field} (不存在于数据中)")
    
    # 4. 系统内存信息
    try:
        import psutil
        print("\n4. 系统内存信息:")
        mem = psutil.virtual_memory()
        print(f"  总内存: {mem.total/1024/1024:.2f} MB")
        print(f"  可用内存: {mem.available/1024/1024:.2f} MB")
        print(f"  使用比例: {mem.percent}%")
    except ImportError:
        print("\n4. 系统内存信息: (需要安装psutil库)")

def event_aware_time_series_split(df, id_columns, time_col='UTCTimeOffset',
                                chain_col='EventChain_id',
                                test_size=0.1, val_size=0.1):
    """
    最终修正版事件链感知分割函数
    """
    # 1. 检查必要列是否存在
    required_cols = [time_col, chain_col] + id_columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列: {missing_cols}")

    # 2. 按时间排序
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # 3. 获取每条事件链的元信息
    chain_meta = df.groupby(chain_col).agg({
        time_col: ['min', 'max'],
        **{col: 'first' for col in id_columns}
    })
    chain_meta.columns = ['_'.join(col).strip('_') for col in chain_meta.columns.values]
    chain_meta = chain_meta.reset_index()
    
    # 4. 按事件链的最后时间排序
    chain_meta = chain_meta.sort_values(f'{time_col}_max')
    
    # 5. 计算分割点
    total_size = test_size + val_size
    split_idx = int(len(chain_meta) * (1 - total_size))
    val_idx = int(len(chain_meta) * (1 - test_size))
    
    # 6. 标记事件链的分割标签 (确保列名正确)
    chain_meta = chain_meta.assign(SplitTag='test')
    chain_meta.loc[:split_idx, 'SplitTag'] = 'train'
    chain_meta.loc[split_idx:val_idx, 'SplitTag'] = 'validation'

    # 确保 chain_col 类型一致
    df[chain_col] = df[chain_col].astype(str)
    chain_meta[chain_col] = chain_meta[chain_col].astype(str)

    # 防止 NaN 导致匹配失败
    df[chain_col] = df[chain_col].fillna('NA')
    chain_meta[chain_col] = chain_meta[chain_col].fillna('NA')

    print("=== DEBUG MERGE ===")
    print("df columns:", df.columns.tolist())
    print("chain_meta columns:", chain_meta.columns.tolist())
    print("df chain_col sample:", df[chain_col].head().tolist())
    print("chain_meta chain_col sample:", chain_meta[chain_col].head().tolist())
    print("df chain_col dtype:", df[chain_col].dtype)
    print("chain_meta chain_col dtype:", chain_meta[chain_col].dtype)
    print("df chain_col unique count:", df[chain_col].nunique())
    print("chain_meta chain_col unique count:", chain_meta[chain_col].nunique())

    if 'SplitTag' in df.columns:
        df = df.drop(columns=['SplitTag'])

    # 7. 合并回原始数据 (确保使用正确的列名)
    result = df.merge(
        chain_meta[[chain_col, 'SplitTag']],
        on=chain_col,
        how='left'
    )
        
    if result['SplitTag'].isna().any():
        missing_count = result['SplitTag'].isna().sum()
        raise ValueError(f"{missing_count} 条记录未能匹配到 SplitTag，可能是 EventChain_id 在 chain_meta 中不存在")

    # 8. 验证合并结果
    if 'SplitTag' not in result.columns:
        raise ValueError("SplitTag列未正确合并，请检查merge操作")
    
    
    
    # 9. 验证事件链完整性
    try:
        chain_check = result.groupby(chain_col)['SplitTag'].nunique()
        if chain_check.max() > 1:
            bad_chains = chain_check[chain_check > 1].index.tolist()
            raise ValueError(f"存在被拆分的事件链！共{len(bad_chains)}条，示例: {bad_chains[:5]}")
    except KeyError as e:
        raise ValueError(f"验证失败: {str(e)}，可用列: {result.columns.tolist()}")
    
    return result

def preprocess_csv_events(data_path: bytes, preprocessed_path: bytes, cfg: Cfg) -> pd.DataFrame:
    if isinstance(data_path, (bytes, bytearray)):
        data_path = data_path.decode()
    if isinstance(preprocessed_path, (bytes, bytearray)):
        preprocessed_path = preprocessed_path.decode()

    os.makedirs(preprocessed_path, exist_ok=True)
    dataset_args = getattr(cfg, 'dataset_args', None)
    progress_file = osp.join(preprocessed_path, 'preprocess_progress.pkl')
    progress = {
        'raw_loaded': False,
        'time_converted': False,
        'check_ins_calculated': False,
        'filtered': False,
        'event_chains_created': False,
        'split_done': False,
        'encoded': False,
        'small_dataset_selected': False,
    }
    if osp.exists(progress_file):
        with open(progress_file, 'rb') as f:
            saved_progress = pickle.load(f)
            progress.update(saved_progress)
            print('[CSV Events] 加载已有进度:', progress)

    raw_path = osp.join(data_path, 'raw')
    csv_file = _ensure_eventsid_chainid_csv(raw_path, cfg)
    temp_path = osp.join(preprocessed_path, 'temp_df.pkl')

    # 1. 读取或恢复原始 CSV
    if not progress['raw_loaded']:
        print(f"[CSV Events] 读取CSV文件: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"[CSV Events] 成功读取数据，共 {len(df)} 行")
        df = _ensure_raw_event_columns(df)
        progress['raw_loaded'] = True
        with open(progress_file, 'wb') as f:
            pickle.dump(progress, f)
        df.to_pickle(temp_path)
    else:
        df = _restore_progress_df(preprocessed_path, csv_file, 'raw_loaded')
        df = _ensure_raw_event_columns(df)

    # 2. 时间转换
    if not progress['time_converted']:
        print('[CSV Events] 转换UTC时间...')
        df['UTCTime'] = pd.to_datetime(df['UTC_time'])
        df['UTCTimeOffset'] = df['UTCTime']
        df['UTCTimeOffsetEpoch'] = df['UTCTimeOffset'].astype(np.int64) // 10**9
        print(f"[CSV Events] 时间转换完成，时间范围: {df['UTCTimeOffsetEpoch'].min()} - {df['UTCTimeOffsetEpoch'].max()}")
        progress['time_converted'] = True
        with open(progress_file, 'wb') as f:
            pickle.dump(progress, f)
        df.to_pickle(temp_path)
    else:
        df = _restore_progress_df(preprocessed_path, csv_file, 'time_converted')

    # 先强制 Event_type 为 int，便于后续对称过滤
    df['Event_type'] = pd.to_numeric(df['Event_type'], errors='coerce').fillna(-1).astype(int)

    # 3. 对称事件过滤
    if not progress['filtered']:
        print('[CSV Events] 开始全局对称事件过滤...')
        df = filter_symmetric_events(df, target_country=None)
        print(f"[CSV Events] 对称事件过滤完成，剩余 {len(df)} 行数据")
        progress['filtered'] = True
        with open(progress_file, 'wb') as f:
            pickle.dump(progress, f)
        df.to_pickle(temp_path)
    else:
        df = _restore_progress_df(preprocessed_path, csv_file, 'filtered')

    # 4. 创建事件链
    if not progress['event_chains_created']:
        print('[CSV Events] 开始创建事件链...')
        df['EventChain_id'] = create_event_chains(df, chunk_size=8)
        print(f"[CSV Events] 事件链创建完成，共 {df['EventChain_id'].nunique()} 个事件链")
        progress['event_chains_created'] = True
        with open(progress_file, 'wb') as f:
            pickle.dump(progress, f)
        df.to_pickle(temp_path)
    else:
        df = _restore_progress_df(preprocessed_path, csv_file, 'event_chains_created')

    # 4.1 筛选小规模数据集（保持事件链完整）
    if not progress.get('small_dataset_selected'):
        # 可选截断：默认不截断，若在配置中设置 csv_events_row_cap / max_csv_events_rows 则启用
        row_cap = None
        if dataset_args is not None:
            row_cap = getattr(dataset_args, 'csv_events_row_cap', None)
            if row_cap is None:
                row_cap = getattr(dataset_args, 'max_csv_events_rows', None)

        if row_cap is None or row_cap <= 0:
            print('[CSV Events] 未设置行数截断，保留全量数据。')
        else:
            print(f'[CSV Events] 按配置截断到最多 {row_cap} 行（保持事件链完整）...')
            chain_sizes = (
                df.groupby('EventChain_id')
                  .size()
                  .reset_index(name='count')
                  .sort_values(by='EventChain_id')
            )
            selected_chains, total_rows = [], 0
            for chain_id, count in zip(chain_sizes['EventChain_id'], chain_sizes['count']):
                if total_rows + count > row_cap:
                    break
                selected_chains.append(chain_id)
                total_rows += count
            df = df[df['EventChain_id'].isin(selected_chains)].copy()
            print(f"[CSV Events] 截断完成: {len(df)} 行, {len(selected_chains)} 条事件链")

        progress['small_dataset_selected'] = True
        with open(progress_file, 'wb') as f:
            pickle.dump(progress, f)
        df.to_pickle(temp_path)
    else:
        df = _restore_progress_df(preprocessed_path, csv_file, 'small_dataset_selected')

    # 5. 计算 check_ins_id
    if not progress['check_ins_calculated']:
        print('[CSV Events] 计算check_ins_id...')
        df['check_ins_id'] = np.arange(len(df), dtype=np.int64)
        print(f"[CSV Events] check_ins_id计算完成，范围: {df['check_ins_id'].min()} - {df['check_ins_id'].max()}")
        progress['check_ins_calculated'] = True
        with open(progress_file, 'wb') as f:
            pickle.dump(progress, f)
        df.to_pickle(temp_path)
    else:
        df = _restore_progress_df(preprocessed_path, csv_file, 'check_ins_calculated')

    # 6. 划分 train/val/test
    if not progress['split_done']:
        print('[CSV Events] 开始基于覆盖约束的训练/验证/测试划分...')
        df, split_report = _split_with_coverage(df, cfg)
        dataset_args = getattr(cfg, 'dataset_args', None)
        dedup_eval = bool(getattr(dataset_args, 'dedup_eval_by_event', True))
        event_field = getattr(dataset_args, 'coverage_event_field', 'ID')
        dedup_chain_field = getattr(dataset_args, 'dedup_chain_field', 'EventChain_id')
        dedup_stats = {'validation': 0, 'test': 0}
        if dedup_eval:
            df, dropped_val = _deduplicate_eval_by_event(df, 'validation', event_field, dedup_chain_field)
            df, dropped_test = _deduplicate_eval_by_event(df, 'test', event_field, dedup_chain_field)
            dedup_stats['validation'] = dropped_val
            dedup_stats['test'] = dropped_test
            if dropped_val or dropped_test:
                print(f"[CSV Events] 去重事件ID：验证集移除 {dropped_val} 条，测试集移除 {dropped_test} 条")
        df = _sanitize_time_columns(df)

        def _split_stats(tag: str):
            part = df[df['SplitTag'] == tag]
            n_rows = int(len(part))
            n_chains = int(part['EventChain_id'].nunique()) if 'EventChain_id' in part.columns else 0
            t_min = str(part['UTCTime'].min()) if 'UTCTime' in part.columns and n_rows > 0 else 'NA'
            t_max = str(part['UTCTime'].max()) if 'UTCTime' in part.columns and n_rows > 0 else 'NA'
            return n_rows, n_chains, t_min, t_max

        tr_n, tr_c, tr_min, tr_max = _split_stats('train')
        va_n, va_c, va_min, va_max = _split_stats('validation')
        te_n, te_c, te_min, te_max = _split_stats('test')
        print(f"[CSV Events] 划分完成 - 训练集: {tr_n} 条, 链数: {tr_c}, 时间: {tr_min} ~ {tr_max}")
        print(f"[CSV Events] 划分完成 - 验证集: {va_n} 条, 链数: {va_c}, 时间: {va_min} ~ {va_max}")
        print(f"[CSV Events] 划分完成 - 测试集: {te_n} 条, 链数: {te_c}, 时间: {te_min} ~ {te_max}")

        if split_report is None:
            split_report = {}
        split_report.update({
            'dedup_validation': dedup_stats['validation'],
            'dedup_test': dedup_stats['test'],
        })
        report_path = osp.join(preprocessed_path, 'split_report.json')
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(split_report, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.warning('[Split] Failed to write split report: %s', exc)
        logging.info('[Split] coverage report: %s', split_report)

        progress['split_done'] = True
        with open(progress_file, 'wb') as f:
            pickle.dump(progress, f)
        df.to_pickle(temp_path)
    else:
        df = _restore_progress_df(preprocessed_path, csv_file, 'split_done')

    # 7. 字段编码
    if not progress['encoded']:
        print('[CSV Events] 开始字段编码...')
        fields_to_encode = [
            'ID',
            'EventChain_id',
            'Source_name_encoded', 'Target_name_encoded',
            'Source_Country_encoded', 'Target_Country_encoded',
            'Location_encoded'
        ]
        encoders = {}
        for i, field in enumerate(fields_to_encode):
            if field not in df.columns:
                print(f"[CSV Events] 跳过不存在的字段: {field}")
                continue
            print(f"\n[CSV Events] 正在处理字段 {i+1}/{len(fields_to_encode)}: {field}")
            start_time = time.time()
            if field == 'ID':
                le = LabelEncoder()
                all_ids = df[field].astype(str).unique()
                le.fit(all_ids)
                chunk_size = 5000
                for j in range(0, len(df), chunk_size):
                    chunk = df.iloc[j:j + chunk_size]
                    transformed = le.transform(chunk[field].astype(str))
                    df.loc[chunk.index, field] = transformed
                    if (j // chunk_size) % 20 == 0:
                        print(f"  已完成 {min(j + chunk_size, len(df))}/{len(df)} 行")
                padding = len(le.classes_)
                encoders[field] = (le, padding)
            elif field == 'EventChain_id':
                le = LabelEncoder()
                chain_values = df[field].fillna('unknown_chain').astype(str)
                df[field] = le.fit_transform(chain_values).astype(np.int64)
                padding = len(le.classes_)
                encoders[field] = (le, padding)
                print(f"  EventChain_id 已重新编号: 0~{padding-1}")
            else:
                le = LabelEncoder()
                train_vals = df[df['SplitTag'] == 'train'][field].astype(str)
                le.fit(train_vals)
                new_labels = np.setdiff1d(df[field].astype(str).unique(), le.classes_)
                if new_labels.size > 0:
                    le.classes_ = np.concatenate([le.classes_, new_labels])
                df[field] = le.transform(df[field].astype(str))
                padding = len(le.classes_)
                encoders[field] = (le, padding)
            print(f"  编码完成，唯一值数量: {padding:,} (耗时: {time.time() - start_time:.2f}s)")
        print('\n[CSV Events] 保存编码器...')
        with open(osp.join(preprocessed_path, 'label_encoding.pkl'), 'wb') as f:
            pickle.dump(encoders, f, protocol=pickle.HIGHEST_PROTOCOL)
        progress['encoded'] = True
        with open(progress_file, 'wb') as f:
            pickle.dump(progress, f)
        df.to_pickle(temp_path)
    else:
        df = _restore_progress_df(preprocessed_path, csv_file, 'encoded')

    missing = [c for c in CSV_EVENTS_KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV events missing columns after preprocess: {missing}")

    if osp.exists(temp_path):
        os.remove(temp_path)

    print(f"\n[CSV Events] 预处理完成，最终数据形状: {df.shape}")
    return df
# def preprocess_csv_events(path: bytes, preprocessed_path: bytes, cfg: Cfg) -> pd.DataFrame:
#     """
#     预处理CSV事件数据集，构建三层超图结构
    
#     第一层：实体节点图 - 以Source_name_encoded和Target_name_encoded为节点，事件为超边
#     第二层：事件节点图 - 以事件ID为节点，事件间关系为边
#     第三层：事件链节点图 - 以事件链为节点，事件链间关系为边
    
#     Args:
#         path: 原始数据路径
#         preprocessed_path: 预处理后数据保存路径
#         TIME_GAP_DAYS: 时间间隔天数
    
#     Returns:
#         预处理后的DataFrame
#     """
#     print("[CSV Events] 开始预处理CSV事件数据集...")
    
    # # 构建原始数据文件路径
    # raw_path = osp.join(path, 'raw')
    # csv_file = osp.join(raw_path, 'eventsid_chainid.csv')
    # print(f"[CSV Events] 读取CSV文件: {csv_file}")
    
    # # 读取CSV文件
    # print("[CSV Events] 正在读取CSV文件...")
    # df = pd.read_csv(csv_file)
    # print(f"[CSV Events] 成功读取数据，共 {len(df)} 行")
    # print(f"[CSV Events] 数据列: {list(df.columns)}")
    
    # # 数据清洗和转换
    # print("[CSV Events] 开始数据清洗和转换...")
    
    # # 将UTC时间字符串转换为datetime对象
    # print("[CSV Events] 转换UTC时间...")
    # df['UTCTime'] = pd.to_datetime(df['UTC_time'])
    # df['UTCTimeOffset'] = df['UTCTime']
    # df['UTCTimeOffsetEpoch'] = df['UTCTimeOffset'].astype(np.int64) // 10**9
    # print(f"[CSV Events] 时间转换完成，时间范围: {df['UTCTimeOffsetEpoch'].min()} - {df['UTCTimeOffsetEpoch'].max()}")
    
    # print("[CSV Events] 计算check_ins_id...")
    # df['check_ins_id'] = df['UTCTimeOffset'].rank(ascending=True, method='first') - 1
    # print(f"[CSV Events] check_ins_id计算完成，范围: {df['check_ins_id'].min()} - {df['check_ins_id'].max()}")
    
    # # 全局去除对称事件
    # print("[CSV Events] 开始全局对称事件过滤...")
    # df = filter_symmetric_events(df, target_country=None) # target_country=None for global filter
    # print(f"[CSV Events] 对称事件过滤完成，剩余 {len(df)} 行数据")
    
    # # 计算check_ins_id：Source_name_encoded和Target_name_encoded的最大值+1
    # print("[CSV Events] 重新计算check_ins_id...")
    # max_source = df['Source_name_encoded'].max()
    # max_target = df['Target_name_encoded'].max()
    # max_entity_id = max(max_source, max_target)
    # df['check_ins_id'] = max_entity_id + 1
    # print(f"[CSV Events] check_ins_id重新计算完成: {df['check_ins_id'].iloc[0]}")
    
    # # 3. 事件链ID生成
    # print("[CSV Events] 开始创建事件链...")
    # df['EventChain_id'] = create_event_chains(df)
    # print(f"[CSV Events] 事件链创建完成，共 {df['EventChain_id'].nunique()} 个事件链")

    # # 4. 分割训练/验证/测试集（SplitTag）
    # print("[CSV Events] 开始数据集分割...")
    # df['SplitTag'] = 'train'  # 默认全部为训练集
    # if len(df) > 10000:
    #     # 按时间排序后分割
    #     df_sorted = df.sort_values('UTCTimeOffset')
    #     split_idx = int(len(df_sorted) * 0.8)  # 80%用于训练
    #     val_split_idx = int(len(df_sorted) * 0.9)  # 10%用于验证
        
    #     # 分配数据集标签
    #     df_sorted.iloc[:split_idx, df_sorted.columns.get_loc('SplitTag')] = 'train'
    #     df_sorted.iloc[split_idx:val_split_idx, df_sorted.columns.get_loc('SplitTag')] = 'validation'
    #     df_sorted.iloc[val_split_idx:, df_sorted.columns.get_loc('SplitTag')] = 'test'
        

    #     df = df_sorted
    #     print(f"[CSV Events] 数据集分割完成 - 训练集: {len(df[df['SplitTag']=='train'])} 行, "
    #           f"验证集: {len(df[df['SplitTag']=='validation'])} 行, "
    #           f"测试集: {len(df[df['SplitTag']=='test'])} 行")
    # else:
    #     print(f"[CSV Events] 数据量较小({len(df)}行)，全部作为训练集")

    # # 5. 统一列名（如有必要，可选，当前不做重命名）

    # # # 6. ID编码（fit on train, transform all）
    # # print("[CSV Events] 开始字段编码...")
    # # fields_to_encode = [
    # #     'latitude', 'longitude', 'timezone', 'UTC_time', 'local_time', 'day_of_week',
    # #     'norm_in_day_time', 'EventChain_id', 'norm_day_shift', 'norm_relative_time',
    # #     'Intensity', 'ID', 'Event_type', 'EventText', 'Structure_Type',
    # #     'Source_name_encoded', 'Target_name_encoded', 'Source_Country_encoded',
    # #     'Target_Country_encoded', 'Location_encoded'
    # # ]
    
    # # df_train = df[df['SplitTag'] == 'train']
    # # encoders = {}
    # # for i, field in enumerate(fields_to_encode):
    # #     if field in df.columns:
    # #         print(f"[CSV Events] 编码字段 {i+1}/{len(fields_to_encode)}: {field}")
    # #         le, padding = id_encode(df_train, df, field)
    # #         encoders[field] = (le, padding)
    # #     else:
    # #         print(f"[CSV Events] 跳过不存在的字段: {field}")
    # # 6. ID编码（fit on train, transform all）

    
    # # 使用函数打印信息
    # #print_data_info(df, df_train, fields_to_encode)
    
    # # print("[CSV Events] 开始字段编码...")
    # # fields_to_encode = [
    # #     'ID', 'EventChain_id', 
    # #     'Source_name_encoded', 'Target_name_encoded', 
    # #     'Source_Country_encoded', 'Target_Country_encoded', 
    # #     'Location_encoded'
    # # ]

    # # df_train = df[df['SplitTag'] == 'train']
    # # print_data_info(df, df_train, fields_to_encode)
    # # encoders = {}
    # # for i, field in enumerate(fields_to_encode):
    # #     if field in df.columns:
    # #         print(f"[CSV Events] 编码字段 {i+1}/{len(fields_to_encode)}: {field}")
    # #         le, padding = id_encode(df_train, df, field)
    # #         encoders[field] = (le, padding)
    # #     else:
    # #         print(f"[CSV Events] 跳过不存在的字段: {field}")

    # # # 7. 保存编码器
    # # print("[CSV Events] 保存编码器...")
    # # with open(osp.join(preprocessed_path, 'label_encoding.pkl'), 'wb') as f:
    # #     pickle.dump(encoders, f)
    # # print(f"[CSV Events] 编码器已保存到: {osp.join(preprocessed_path, 'label_encoding.pkl')}")

    # # #8. 返回df（保留所有字段，兼容NYC结构）
    # # print(f"[CSV Events] 预处理完成，最终数据形状: {df.shape}")
    # # print(f"[CSV Events] 最终数据列: {list(df.columns)}")
    # # return df

    # print("[CSV Events] 开始字段编码...")
    # fields_to_encode = [
    #     'ID', 'EventChain_id', 
    #     'Source_name_encoded', 'Target_name_encoded', 
    #     'Source_Country_encoded', 'Target_Country_encoded', 
    #     'Location_encoded'
    # ]

    # df_train = df[df['SplitTag'] == 'train']

    # # 打印数据诊断信息（优化版）
    # def print_optimized_info(df, df_train, fields):
    #     print("\n===== 优化诊断信息 =====")
    #     # 关键字段信息
    #     print("\n关键字段统计:")
    #     for field in fields:
    #         if field in df.columns:
    #             uniq_train = df_train[field].nunique()
    #             uniq_all = df[field].nunique()
    #             dtype = df[field].dtype
    #             print(f"{field:>20}: {dtype} | 唯一值(训练/全量): {uniq_train:,}/{uniq_all:,} | 内存: {df[field].memory_usage(deep=True)/1024/1024:.1f}MB")
        
    #     # 系统资源
    #     try:
    #         import psutil
    #         mem = psutil.virtual_memory()
    #         print(f"\n系统内存: 可用{mem.available/1024/1024:.0f}MB/{mem.total/1024/1024:.0f}MB ({mem.percent}%)")
    #     except:
    #         print("\n(安装psutil可查看内存信息)")

    # print_optimized_info(df, df_train, fields_to_encode)

    # # 优化后的编码逻辑
    # encoders = {}
    # for i, field in enumerate(fields_to_encode):
    #     if field not in df.columns:
    #         print(f"[CSV Events] 跳过不存在的字段: {field}")
    #         continue
        
    #     print(f"\n[CSV Events] 正在处理字段 {i+1}/{len(fields_to_encode)}: {field}")
    #     start_time = time.time()
        
    #     # 优化策略选择
    #     if np.issubdtype(df[field].dtype, np.integer):
    #         # 已经是数值型的字段快速处理
    #         padding = df[field].max() + 1
    #         encoders[field] = (None, padding)
    #         print(f"  已为数值型字段设置padding: {padding} (耗时: {time.time()-start_time:.2f}s)")
    #     else:
    #         # 对象类型字段优化处理
    #         le = LabelEncoder()
            
    #         # 分批处理大数据（针对ID字段）
    #         if field == 'ID' and len(df) > 1e6:
    #             print("  检测到大尺寸ID字段，启用分批编码...")
    #             chunk_size = 500000
    #             le.fit(df_train[field].astype(str))
                
    #             for j in range(0, len(df), chunk_size):
    #                 df.loc[j:j+chunk_size-1, field] = le.transform(
    #                     df.iloc[j:j+chunk_size][field].astype(str))
    #         else:
    #             # 常规处理
    #             le.fit(df_train[field].astype(str))
    #             df[field] = le.transform(df[field].astype(str))
            
    #         padding = len(le.classes_)
    #         encoders[field] = (le, padding)
    #         print(f"  编码完成，唯一值数量: {padding:,} (耗时: {time.time()-start_time:.2f}s)")

    # # 7. 保存编码器（添加压缩优化）
    # print("\n[CSV Events] 保存编码器...")
    # save_start = time.time()
    # with open(osp.join(preprocessed_path, 'label_encoding.pkl'), 'wb') as f:
    #     # 使用最高压缩率
    #     pickle.dump(encoders, f, protocol=pickle.HIGHEST_PROTOCOL)
    # print(f"  保存完成，耗时: {time.time()-save_start:.2f}s")
    # print(f"  文件大小: {os.path.getsize(osp.join(preprocessed_path, 'label_encoding.pkl'))/1024/1024:.2f}MB")
    # print(f"  保存路径: {osp.join(preprocessed_path, 'label_encoding.pkl')}")

    # # 8. 返回结果（添加内存清理）
    # print(f"\n[CSV Events] 预处理完成，最终数据形状: {df.shape}")
    # print(f"内存使用情况:")
    # print(f"- 对象类型列: {len(df.select_dtypes(include='object').columns)}")
    # print(f"- 数值类型列: {len(df.select_dtypes(include=np.number).columns)}")

    # # 清理临时内存
    # del df_train
    # gc.collect()

    # return df


def preprocess_nyc(path: bytes, preprocessed_path: bytes, TIME_GAP_DAYS: int) -> pd.DataFrame:
    """
    预处理NYC数据集
    
    Args:
        path: 原始数据路径
        preprocessed_path: 预处理后数据保存路径
        TIME_GAP_DAYS: 时间间隔天数
    
    Returns:
        预处理后的DataFrame
    """
    # 构建原始数据文件路径
    raw_path = osp.join(path, 'raw')
    gap_str = str(int(TIME_GAP_DAYS))  # 强制转为整数字符串

    # 读取训练、验证、测试数据
    df_train = pd.read_csv(osp.join(raw_path, f'NYC_train_{gap_str}.csv'))
    df_val = pd.read_csv(osp.join(raw_path, f'NYC_val_{gap_str}.csv'))
    df_test = pd.read_csv(osp.join(raw_path, f'NYC_test_{gap_str}.csv'))
    
    # 添加数据集分割标签
    df_train['SplitTag'] = 'train'
    df_val['SplitTag'] = 'validation'
    df_test['SplitTag'] = 'test'
    
    # 合并所有数据
    df = pd.concat([df_train, df_val, df_test])
    
    # 需要删除的字段列表
    columns_to_drop = [
        'Intensity',
        'Event_ID',
        'POI_catid_encoded',
        'POI_catid_code_encoded',
        'POI_catname_encoded',
        'split'
    ]

    # 删除字段（若存在）
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    df_train = df_train.drop(columns=[col for col in columns_to_drop if col in df.columns])
    df_val = df_val.drop(columns=[col for col in columns_to_drop if col in df.columns])
    df_test = df_test.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # 重命名列
    df.columns = [
        'UserId', 'PoiId', 'PoiCategoryId', 'PoiCategoryCode', 'PoiCategoryName', 'Latitude', 'Longitude',
        'TimezoneOffset', 'UTCTime', 'UTCTimeOffset', 'UTCTimeOffsetWeekday', 'UTCTimeOffsetNormInDayTime',
        'pseudo_session_trajectory_id', 'UTCTimeOffsetNormDayShift', 'UTCTimeOffsetNormRelativeTime', 'SplitTag'
    ]

    # 数据转换
    df['trajectory_id'] = df['pseudo_session_trajectory_id']  # 轨迹ID
    # 将时间字符串转换为datetime对象
    df['UTCTimeOffset'] = df['UTCTimeOffset'].apply(lambda x: datetime.strptime(x[:19], "%Y-%m-%d %H:%M:%S"))
    # 转换为Unix时间戳
    df['UTCTimeOffsetEpoch'] = df['UTCTimeOffset'].apply(lambda x: x.strftime('%s'))
    # 提取星期几
    df['UTCTimeOffsetWeekday'] = df['UTCTimeOffset'].apply(lambda x: x.weekday())
    # 提取小时
    df['UTCTimeOffsetHour'] = df['UTCTimeOffset'].apply(lambda x: x.hour)
    # 提取日期
    df['UTCTimeOffsetDay'] = df['UTCTimeOffset'].apply(lambda x: x.strftime('%Y-%m-%d'))
    # 用户内时间排名
    df['UserRank'] = df.groupby('UserId')['UTCTimeOffset'].rank(method='first')
    # 按用户ID和时间排序
    df = df.sort_values(by=['UserId', 'UTCTimeOffset'], ascending=True)

    # ID编码
    # 创建签到点ID
    df['check_ins_id'] = df['UTCTimeOffset'].rank(ascending=True, method='first') - 1
    # 轨迹ID编码
    traj_id_le, padding_traj_id = id_encode(df, df, 'pseudo_session_trajectory_id')

    # 使用训练集进行编码器拟合
    df_train = df[df['SplitTag'] == 'train']
    # POI ID编码
    poi_id_le, padding_poi_id = id_encode(df_train, df, 'PoiId')
    # POI类别编码
    poi_category_le, padding_poi_category = id_encode(df_train, df, 'PoiCategoryId')
    # 用户ID编码
    user_id_le, padding_user_id = id_encode(df_train, df, 'UserId')
    # 小时编码
    hour_id_le, padding_hour_id = id_encode(df_train, df, 'UTCTimeOffsetHour')
    # 星期几编码
    weekday_id_le, padding_weekday_id = id_encode(df_train, df, 'UTCTimeOffsetWeekday')

    # 保存编码映射逻辑
    with open(osp.join(preprocessed_path, 'label_encoding.pkl'), 'wb') as f:
        pickle.dump([
            poi_id_le, poi_category_le, user_id_le, hour_id_le, weekday_id_le,
            padding_poi_id, padding_poi_category, padding_user_id, padding_hour_id, padding_weekday_id
        ], f)

    # 忽略第一个签到点并只保留最后一个
    df = ignore_first(df)
    df = only_keep_last(df)
    return df


def preprocess_tky_ca(cfg: Cfg, path: bytes) -> pd.DataFrame:
    """
    预处理TKY和CA数据集
    
    Args:
        cfg: 配置对象
        path: 数据路径
    
    Returns:
        预处理后的DataFrame
    """
    # 根据数据集名称选择原始文件
    if cfg.dataset_args.dataset_name == 'tky':
        raw_file = 'dataset_TSMC2014_TKY.txt'
    else:
        raw_file = 'dataset_gowalla_ca_ne.csv'

    # 设置文件读取器根路径
    FileReader.root_path = path
    # 读取数据集
    data = FileReader.read_dataset(raw_file, cfg.dataset_args.dataset_name)
    # 过滤低频POI和用户
    data = FileReader.do_filter(data, cfg.dataset_args.min_poi_freq, cfg.dataset_args.min_user_freq)
    # 分割训练测试集
    data = FileReader.split_train_test(data)

    # 对于CA数据集，由于过滤后仍有大量低频POI和用户，进行二次过滤
    if cfg.dataset_args.dataset_name == 'ca':
        data = FileReader.do_filter(data, cfg.dataset_args.min_poi_freq, cfg.dataset_args.min_user_freq)
        data = FileReader.split_train_test(data)

    # 生成ID编码
    data = FileReader.generate_id(
        data,
        cfg.dataset_args.session_time_interval,
        cfg.dataset_args.do_label_encode,
        cfg.dataset_args.only_last_metric
    )
    return data


def preprocess(cfg: Cfg):
    """
    主预处理函数
    
    Args:
        cfg: 配置对象
    """
    print(f"[Preprocess] 开始预处理，数据集: {cfg.dataset_args.dataset_name}")
    
    # 获取根目录
    root_path = get_root_dir()
    dataset_name = cfg.dataset_args.dataset_name
    # 构建数据路径
    data_path = osp.join(root_path, 'data', dataset_name)
    # 使用参数构造预处理数据路径
    preprocessed_path = osp.join(data_path, f'preprocessed_6')
    
    print(f"[Preprocess] 数据路径: {data_path}")
    print(f"[Preprocess] 预处理路径: {preprocessed_path}")

    
    
    # 定义输出文件路径
    sample_file = osp.join(preprocessed_path, 'sample.csv')
    train_file = osp.join(preprocessed_path, 'train_sample.csv')
    validate_file = osp.join(preprocessed_path, 'validate_sample.csv')
    test_file = osp.join(preprocessed_path, 'test_sample.csv')

    # 创建预处理目录（如果不存在）
    if not osp.exists(preprocessed_path):
        print(f"[Preprocess] 创建预处理目录: {preprocessed_path}")
        os.makedirs(preprocessed_path)
    else:
        print(f"[Preprocess] 预处理目录已存在: {preprocessed_path}")

    # 步骤1：预处理原始文件并创建样本文件
    # 包括：1. 数据转换; 2. ID编码; 3. 训练/验证/测试分割; 4. 移除未见过的用户或POI
    if not osp.exists(sample_file):
        print(f"[Preprocess] 样本文件不存在，开始预处理: {sample_file}")
        
        if 'nyc' == dataset_name:
            print("[Preprocess] 处理NYC数据集...")
            # NYC数据集的保留列
            keep_cols = [
                'check_ins_id', 'UTCTimeOffset', 'UTCTimeOffsetEpoch', 'pseudo_session_trajectory_id',
                'query_pseudo_session_trajectory_id', 'UserId', 'Latitude', 'Longitude', 'PoiId', 'PoiCategoryId',
                'PoiCategoryName', 'last_checkin_epoch_time', 'trajectory_id'
            ]
            preprocessed_data = preprocess_nyc(data_path, preprocessed_path, cfg.TIME_GAP_DAYS)
            # 移除未见过的用户和POI
            print("[Preprocess] 移除未见过的用户和POI...")
            preprocessed_result = remove_unseen_user_poi(preprocessed_data)
        elif 'tky' == dataset_name or 'ca' == dataset_name:
            print(f"[Preprocess] 处理{dataset_name.upper()}数据集...")
            # TKY/CA数据集的保留列
            keep_cols = [
                'check_ins_id', 'UTCTimeOffset', 'UTCTimeOffsetEpoch', 'pseudo_session_trajectory_id',
                'query_pseudo_session_trajectory_id', 'UserId', 'Latitude', 'Longitude', 'PoiId', 'PoiCategoryId',
                'PoiCategoryName', 'last_checkin_epoch_time'
            ]
            preprocessed_data = preprocess_tky_ca(cfg, data_path)
            # 移除未见过的用户和POI
            print("[Preprocess] 移除未见过的用户和POI...")
            preprocessed_result = remove_unseen_user_poi(preprocessed_data)
        elif 'csv_events' == dataset_name:
            print("[Preprocess] 处理CSV事件数据集...")
            # CSV事件数据集的保留列
            keep_cols = [
                'latitude', 'longitude', 'timezone', 'UTC_time', 'local_time', 'day_of_week',
                'norm_in_day_time', 'EventChain_id', 'norm_day_shift', 'norm_relative_time',
                'Intensity', 'ID', 'Event_type', 'EventText', 'Structure_Type',
                'Source_name_encoded', 'Target_name_encoded', 'Source_Country_encoded',
                'Target_Country_encoded', 'Location_encoded', 'UTCTime', 'UTCTimeOffset', 'UTCTimeOffsetEpoch',
                'SplitTag'
            ]
            preprocessed_data = preprocess_csv_events(data_path, preprocessed_path, cfg)
            # CSV事件数据不需要remove_unseen_user_poi处理
            preprocessed_result = {
                'sample': preprocessed_data,
                'train_sample': preprocessed_data[preprocessed_data['SplitTag'] == 'train'],
                'validate_sample': preprocessed_data[preprocessed_data['SplitTag'] == 'validation'],
                'test_sample': preprocessed_data[preprocessed_data['SplitTag'] == 'test']
            }
        else:
            raise ValueError(f'Wrong dataset name: {dataset_name} ')
        
        # 保存处理后的数据
        print("[Preprocess] 保存处理后的数据...")
        preprocessed_result['sample'].to_csv(sample_file, index=False)
        print(f"[Preprocess] 样本文件已保存: {sample_file}")
        
        preprocessed_result['train_sample'][keep_cols].to_csv(train_file, index=False)
        print(f"[Preprocess] 训练文件已保存: {train_file}")
        
        preprocessed_result['validate_sample'][keep_cols].to_csv(validate_file, index=False)
        print(f"[Preprocess] 验证文件已保存: {validate_file}")
        
        preprocessed_result['test_sample'][keep_cols].to_csv(test_file, index=False)
        print(f"[Preprocess] 测试文件已保存: {test_file}")
    else:
        print(f"[Preprocess] 样本文件已存在，跳过预处理: {sample_file}")

    # 步骤2：生成超图相关数据
    if dataset_name == 'csv_events':
        # CSV事件数据集生成三层超图，分别保存三层图
        print("[Preprocess] 检查CSV事件数据集的三层超图文件...")
        entity_graph_file = osp.join(preprocessed_path, 'entity_graph.pt')
        event_graph_file = osp.join(preprocessed_path, 'event_graph.pt')
        chain_graph_file = osp.join(preprocessed_path, 'chain_graph.pt')
        
        if not (osp.exists(entity_graph_file) and osp.exists(event_graph_file) and osp.exists(chain_graph_file)):
            print("[Preprocess] 三层超图文件不完整，开始生成...")
            generate_hypergraph_from_file(sample_file, preprocessed_path, cfg.dataset_args)
            print("[Preprocess] 三层超图生成完成")
        else:
            print("[Preprocess] 三层超图文件已存在，跳过生成")
    else:
        # 其他数据集生成原有的超图
        ci2traj_file = osp.join(preprocessed_path, 'ci2traj_pyg_data.pt')
        if not osp.exists(ci2traj_file):
            print(f"[Preprocess] 超图文件不存在，开始生成: {ci2traj_file}")
            generate_hypergraph_from_file(sample_file, preprocessed_path, cfg.dataset_args)
            print("[Preprocess] 超图生成完成")
        else:
            print(f"[Preprocess] 超图文件已存在，跳过生成: {ci2traj_file}")

    print("[Preprocess] 预处理完成！")
    logging.info('[Preprocess] Done preprocessing.')
