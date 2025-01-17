import json
from pathlib import Path

import pandas as pd
import seaborn as sns


def load_results_json(results_dir: Path) -> list:
    data = []
    for seed_dir in results_dir.glob("seed_*"):
        seed = int(seed_dir.name.split("_")[-1])

        with open(seed_dir / "results.json") as f:
            results = json.load(f)
            for model, metrics in results.items():
                data.append(
                    {
                        "Model": model.replace("_", " "),
                        "Error": metrics["prediction_error"] * 100,
                        "Seed": seed,
                    }
                )
    return data


def plot_comparison_results(
    results_dir: Path,
    save_dir: Path,
):
    """Plot comparison results of different models.

    Args:
        results_dir: Directory containing seed_* subdirectories
        save_dir: Directory to save the plot
    """
    # Load results
    data = load_results_json(results_dir)
    df = pd.DataFrame(data)

    # Sort models by error
    model_order = df.groupby("Model")["Error"].mean().sort_values().index.tolist()
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)

    # Create figure
    sns.set_theme(style="ticks", font_scale=2.2)
    g = sns.catplot(
        data=df,
        y="Model",
        x="Error",
        color="#0984E3",  # blue
        kind="bar",
        height=6,
        aspect=1.2,
        orient="h",
        errorbar="se",  # Show standard error
        capsize=0.25,  # Add caps to error bars
    )

    g.ax.set_xlabel("Test prediction error (%)")
    g.ax.set_xlim(0, 100)
    g.ax.set_ylabel("")

    # Add minor ticks
    g.ax.set_xticks([10, 20, 30, 40, 60, 70, 80, 90], minor=True)

    # Save the plot
    save_dir.mkdir(parents=True, exist_ok=True)
    g.savefig(save_dir / "model_comparison.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent / "aloi/output/model_comparison"
    plot_comparison_results(
        results_dir=base_path,
        save_dir=base_path,
    )
