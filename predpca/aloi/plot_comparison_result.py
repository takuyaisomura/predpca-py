import json
from pathlib import Path

import pandas as pd
import seaborn as sns


def plot_comparison_results(
    results_path: Path,
    save_dir: Path,
):
    """Plot comparison results of different models.

    Args:
        results_path: Path to results.json
        save_dir: Directory to save the plot
    """
    # Load results
    with open(results_path) as f:
        results = json.load(f)

    # Create DataFrame
    data = []
    for model in results.keys():
        data.append(
            {
                "Model": model,
                "Error": results[model]["prediction_error"] * 100,
            }
        )
    df = pd.DataFrame(data)

    # Sort models by error
    model_order = df.groupby("Model")["Error"].mean().sort_values().index.tolist()
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)

    # Create figure
    sns.set_theme(style="whitegrid", font_scale=2.5)
    g = sns.catplot(
        data=df,
        y="Model",
        x="Error",
        color="#0984E3",  # blue
        kind="bar",
        height=6,
        aspect=1.2,
        orient="h",
    )

    g.ax.set_xlabel("Prediction error (%)")
    g.ax.set_xlim(0, 100)
    g.ax.set_ylabel("")

    # Save the plot
    save_dir.mkdir(parents=True, exist_ok=True)
    g.savefig(save_dir / "model_comparison.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent / "aloi/output/model_comparison"
    plot_comparison_results(
        results_path=base_path / "results.json",
        save_dir=base_path,
    )
