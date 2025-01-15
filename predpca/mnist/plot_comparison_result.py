import json
from pathlib import Path

import pandas as pd
import seaborn as sns


def load_results_json(results_dir: Path, sequence_type: str) -> list:
    data = []
    for seed_dir in results_dir.glob("seed_*"):
        seed = int(seed_dir.name.split("_")[-1])

        with open(seed_dir / "results.json") as f:
            results = json.load(f)
            for model, metrics in results.items():
                data.append(
                    {
                        "Model": model,
                        "Error": metrics["categorization_error"] * 100,
                        "Sequence": sequence_type,
                        "Seed": seed,
                    }
                )
    return data


def create_comparison_dataframe(data1: list, data2: list) -> pd.DataFrame:
    df = pd.DataFrame(data1 + data2)

    # Sort models by average error
    model_order = df.groupby("Model")["Error"].mean().sort_values().index.tolist()
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)

    return df


def plot_comparison_results(
    results_dir1: Path,
    results_dir2: Path,
    save_dir: Path,
):
    """Plot comparison results of two sequence types with error bars.

    Args:
        results_dir1: Directory containing results for sequence type 1 (multiple seeds)
        results_dir2: Directory containing results for sequence type 2 (multiple seeds)
        save_dir: Directory to save the plot
    """
    data1 = load_results_json(results_dir1, "Ascending")
    data2 = load_results_json(results_dir2, "Fibonacci")

    df = create_comparison_dataframe(data1, data2)

    sns.set_theme(style="ticks", font_scale=2.2)
    g = sns.catplot(
        data=df,
        y="Model",
        x="Error",
        hue="Sequence",
        palette=["#0984E3", "#D63031"],  # blue and red
        kind="bar",
        height=6,
        aspect=1.2,
        orient="h",
        errorbar="se",  # Show standard error
        capsize=0.25,  # Add caps to error bars
    )

    g.ax.set_xlabel("Categorization error (%)")
    g.ax.set_xlim(0, 100)
    g.ax.set_ylabel("")

    # Add minor ticks
    g.ax.set_xticks([10, 20, 30, 40, 60, 70, 80, 90], minor=True)

    # Save the plot
    save_dir.mkdir(parents=True, exist_ok=True)
    g.savefig(save_dir / "model_comparison.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent / "mnist/output/model_comparison"
    plot_comparison_results(
        results_dir1=base_path / "sequence_type_1",
        results_dir2=base_path / "sequence_type_2",
        save_dir=base_path,
    )
