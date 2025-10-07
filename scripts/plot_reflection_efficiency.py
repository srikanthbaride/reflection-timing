import sys, os, pandas as pd, matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/plot_reflection_efficiency.py <metrics.csv> <out.png>")
        sys.exit(1)
    csv_path, out_path = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df = pd.read_csv(csv_path)

    # Aggregate by mode
    agg = df.groupby("mode").agg(
        success_rate=("success", "mean"),
        mean_steps=("steps", "mean"),
        mean_reflections=("reflections", "mean"),
        mean_return=("return", "mean")
    ).reset_index()
    # efficiency: success per reflection (avoid div-by-zero)
    agg["efficiency"] = agg["success_rate"] / (agg["mean_reflections"].replace(0, 1))

    # Plot 1: success vs reflections (scatter)
    plt.figure()
    for _, row in agg.iterrows():
        plt.scatter(row["mean_reflections"], row["success_rate"], label=row["mode"])
    plt.xlabel("Mean Reflections per Episode")
    plt.ylabel("Success Rate")
    plt.title("Success vs Reflection Budget")
    plt.legend()
    plt.savefig(out_path.replace(".png", "_success.png"), bbox_inches="tight", dpi=150)

    # Plot 2: efficiency bar
    plt.figure()
    plt.bar(agg["mode"], agg["efficiency"])
    plt.xlabel("Mode")
    plt.ylabel("Success per Reflection")
    plt.title("Reflection Efficiency")
    plt.savefig(out_path.replace(".png", "_efficiency.png"), bbox_inches="tight", dpi=150)

    # Plot 3: steps
    plt.figure()
    plt.bar(agg["mode"], agg["mean_steps"])
    plt.xlabel("Mode")
    plt.ylabel("Mean Steps")
    plt.title("Steps to Finish (lower is better)")
    plt.savefig(out_path.replace(".png", "_steps.png"), bbox_inches="tight", dpi=150)

    print("Saved plots:",
          out_path.replace(".png", "_success.png"),
          out_path.replace(".png", "_efficiency.png"),
          out_path.replace(".png", "_steps.png"))

if __name__ == "__main__":
    main()
