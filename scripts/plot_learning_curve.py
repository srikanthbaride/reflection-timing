import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/plot_learning_curve.py <metrics.csv> <out.png>")
        sys.exit(1)
    csv_path = sys.argv[1]
    out_path = sys.argv[2]

    df = pd.read_csv(csv_path)
    # pick episodic rows with 'return'
    if "return" in df.columns:
        df2 = df.dropna(subset=["return"])
    else:
        df2 = df

    plt.figure()
    plt.plot(df2["episode"], df2["return"], label="Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
