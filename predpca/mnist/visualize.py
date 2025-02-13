import matplotlib.pyplot as plt
import numpy as np

color_map = {
    0: [1.0, 0.0, 0.0],
    1: [1.0, 0.5, 0.0],
    2: [1.0, 1.0, 0.0],
    3: [0.0, 1.0, 0.0],
    4: [0.0, 1.0, 0.5],
    5: [0.0, 1.0, 1.0],
    6: [0.0, 0.5, 1.0],
    7: [0.0, 0.0, 1.0],
    8: [0.5, 0.0, 1.0],
    9: [1.0, 0.0, 1.0],
}


def visualize_encodings(
    ui_test,  # (Nu, T_test)
    label_test,  # (1, T_test)
    ts,  # time steps to plot
    figsize=(10, 15),
):
    T_test = ui_test.shape[1]
    colors = np.zeros((T_test, 3))

    for label, color in color_map.items():
        colors[label_test[0] == label] = color

    # compare continuous latent variables (0 vs 1, 2 vs 3, ...)
    fig, axs = plt.subplots(3, 2, figsize=figsize)
    for i, ax in enumerate(axs.flatten()):
        if i == axs.size - 1:
            # Create a legend for all digits in the last subplot
            legend_elements = [plt.scatter([], [], c=[color_map[digit]], label=str(digit)) for digit in range(10)]
            ax.legend(handles=legend_elements, loc="center", ncol=2, title="Digit Colors")
            ax.axis("off")
            break
        # Use consecutive pairs since the input is already sorted
        idx1, idx2 = i * 2, i * 2 + 1
        ax.scatter(ui_test[idx1, ts], ui_test[idx2, ts], c=colors[ts], s=10)
        ax.axis("equal")
        ax.axis("square")
        ax.set_title(f"Latent {idx1} vs {idx1 + 1}")

    plt.tight_layout()

    return fig


def digit_image(
    input: np.ndarray,  # (784, T)
) -> np.ndarray:
    T = input.shape[1]
    img = np.zeros((28, 28 * T), dtype=np.uint8)

    for t in range(T):
        img[:, 28 * t : 28 * (t + 1)] = input[:, t].reshape((28, 28), order="F").T

    img = ((1 - img.clip(0, 1)) * 255).astype(np.uint8)
    img = img.repeat(5, axis=0).repeat(5, axis=1)  # (28 * 5, 28 * T * 5)
    dstimg = np.stack([img, img, img], axis=-1)  # (28 * 5, 28 * T * 5, 3)

    return dstimg
