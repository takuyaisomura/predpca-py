import json
from pathlib import Path

import pandas as pd
import seaborn as sns


def plot_comparison_results(
    results_path1: Path,
    results_path2: Path,
    save_dir: Path,
):
    """Plot comparison results of two sequence types.

    Args:
        results_path1: Path to results.json for sequence type 1
        results_path2: Path to results.json for sequence type 2
        save_dir: Directory to save the plot
    """
    # Load results
    with open(results_path1) as f:
        results1 = json.load(f)
    with open(results_path2) as f:
        results2 = json.load(f)

    # Create DataFrame
    data = []
    for model in results1.keys():
        data.append(
            {
                "Model": model,
                "Error": results1[model]["categorization_error"] * 100,
                "Sequence": "Ascending",
            }
        )
        data.append(
            {
                "Model": model,
                "Error": results2[model]["categorization_error"] * 100,
                "Sequence": "Fibonacci",
            }
        )
    df = pd.DataFrame(data)

    # Sort models by average error
    model_order = df.groupby("Model")["Error"].mean().sort_values().index.tolist()
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)

    # Create figure
    sns.set_theme(style="whitegrid", font_scale=2)
    g = sns.catplot(
        data=df,
        y="Model",
        x="Error",
        hue="Sequence",
        palette=["#0984E3", "#D63031"],  # blue and red
        kind="bar",
        height=8,
        aspect=1.5,
        orient="h",
    )

    g.ax.set_xlabel("Categorization error (%)")
    g.ax.set_xlim(0, 100)
    g.ax.set_ylabel("")

    # Save the plot
    save_dir.mkdir(parents=True, exist_ok=True)
    g.savefig(save_dir / "model_comparison.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent / "mnist/output/model_comparison"
    plot_comparison_results(
        results_path1=base_path / "sequence_type_1/results.json",
        results_path2=base_path / "sequence_type_2/results.json",
        save_dir=base_path,
    )
