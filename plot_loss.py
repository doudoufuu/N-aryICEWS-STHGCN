import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # adjust if you place the script elsewhere
LOG_DIR = ROOT / "Spatio-Temporal-Hypergraph-Model" / "log" / "20251027_224105" / "csv_events"

TRAIN_CSV = LOG_DIR / "loss_train_step.csv"
VAL_CSV = LOG_DIR / "loss_valid.csv"
OUTPUT_FIG = LOG_DIR / "loss_curve.png"

def plot_loss_curves(train_csv: Path, val_csv: Path, output_path: Path) -> None:
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_x = train_df.columns[0] if train_df.columns[0] != "loss" else "step"
    val_x = val_df.columns[0] if val_df.columns[0] != "loss" else "epoch"

    plt.figure(figsize=(8, 5))
    plt.plot(train_df[train_x], train_df["loss"], label="Train Loss", linewidth=2)
    plt.plot(val_df[val_x], val_df["loss"], label="Validation Loss", linewidth=2)

    plt.title("Train vs Validation Loss")
    plt.xlabel(train_x.capitalize())
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_loss_curves(TRAIN_CSV, VAL_CSV, OUTPUT_FIG)
    print(f"Saved loss plot to: {OUTPUT_FIG}")
