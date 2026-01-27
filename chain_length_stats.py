import argparse
import os.path as osp

import pandas as pd


def chain_length_stats(df: pd.DataFrame) -> pd.Series:
    """Return length (size) of each EventChain_id."""
    if "EventChain_id" not in df.columns:
        raise ValueError("Missing EventChain_id column.")
    return df.groupby("EventChain_id").size()


def summarize(lengths: pd.Series) -> str:
    if lengths.empty:
        return "no chains"
    desc = lengths.describe()
    head_freq = lengths.value_counts().sort_index().head(10)
    return (
        f"#chains={len(lengths)}, "
        f"min={desc['min']:.0f}, "
        f"mean={desc['mean']:.2f}, "
        f"median={desc['50%']:.0f}, "
        f"max={desc['max']:.0f}\n"
        f"length freq (first 10 lengths):\n{head_freq}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chain length distribution per split.")
    parser.add_argument(
        "--base-dir",
        default="/home/beihang/hsy/Spatio-Temporal-Hypergraph-Model/data/csv_events/preprocessed_6",
        help="Directory containing sample.csv, train_sample.csv, validate_sample.csv, test_sample.csv",
    )
    parser.add_argument(
        "--length",
        type=int,
        action="append",
        dest="query_lengths",
        help="Specific chain length(s) to query count for (can be passed multiple times).",
    )
    args = parser.parse_args()
    base = args.base_dir

    # Use full sample.csv to get true chain lengths (val/test CSVs only keep tails)
    sample_path = osp.join(base, "sample.csv")
    try:
        df_all = pd.read_csv(sample_path)
    except Exception as exc:
        print(f"[sample] failed to read {sample_path}: {exc}")
        return
    lengths_all = chain_length_stats(df_all)
    freq_all = lengths_all.value_counts().sort_index()

    print(f"\n[sample] rows={len(df_all)}")
    print(summarize(lengths_all))
    print("length -> count (sorted by length):")
    print(freq_all.to_string())

    files = {
        "train": "train_sample.csv",
        "validation": "validate_sample.csv",
        "test": "test_sample.csv",
    }

    for split, fname in files.items():
        path = osp.join(base, fname)
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[{split}] failed to read {path}: {exc}")
            continue
        if "EventChain_id" not in df.columns:
            print(f"[{split}] missing EventChain_id column, skip.")
            continue
        chain_ids = pd.unique(df["EventChain_id"])
        lengths = lengths_all[lengths_all.index.isin(chain_ids)]
        freq = lengths.value_counts().sort_index()

        print(f"\n[{split}] rows={len(df)}")
        print(summarize(lengths))

        # Length distribution table
        print("length -> count (sorted by length):")
        print(freq.to_string())

        # Specific length queries
        if args.query_lengths:
            for q_len in args.query_lengths:
                cnt = int(freq.get(q_len, 0))
                print(f"length={q_len} count={cnt}")


if __name__ == "__main__":
    main()
