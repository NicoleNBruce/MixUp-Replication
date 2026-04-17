import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser("Plot Corruption Results")
    parser.add_argument("--save-dir", type=str, default="./outputs", help="Directory containing run outputs")
    return parser.parse_args()

def main():
    args = parse_args()
    base_dir = Path(args.save_dir)
    
    # Expected run names based on the fast strategy
    history_paths = {
        "ERM (20% Corrupt)": base_dir / "corrupted_20" / "history.json",
        "MixUp (20% Corrupt)": base_dir / "corrupted_20_mixup" / "history.json",
        "ERM (50% Corrupt)": base_dir / "corrupted_50" / "history.json",
        "MixUp (50% Corrupt)": base_dir / "corrupted_50_mixup" / "history.json",
        "ERM (80% Corrupt)": base_dir / "corrupted_80" / "history.json",
        "MixUp (80% Corrupt)": base_dir / "corrupted_80_mixup" / "history.json"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    corruption_levels = ["20", "50", "80"]
    
    for i, level in enumerate(corruption_levels):
        erm_path = base_dir / f"corrupted_{level}" / "history.json"
        mixup_path = base_dir / f"corrupted_{level}_mixup" / "history.json"
        
        if erm_path.exists():
            with erm_path.open("r") as f:
                h = json.load(f)
                axes[i].plot(h["epoch"], h["test_acc"], label="ERM Baseline", color='blue')
        else:
            print(f"Missing {erm_path}")
        
        if mixup_path.exists():
            with mixup_path.open("r") as f:
                h = json.load(f)
                axes[i].plot(h["epoch"], h["test_acc"], label="MixUp (alpha=8.0)", color='orange')
        else:
            print(f"Missing {mixup_path}")
                
        axes[i].set_title(f"{level}% Label Corruption")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Test Accuracy (%)")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    fig.tight_layout()
    save_path = base_dir / "corruption_results.png"
    fig.savefig(save_path, dpi=200)
    print(f"Saved corruption plots to {save_path}")

if __name__ == "__main__":
    main()
