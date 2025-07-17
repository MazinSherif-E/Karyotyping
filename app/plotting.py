import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_instance_segmentation(res, class_names):
    img_orig = res[0].orig_img
    img_annotated = res[0].plot(
        font_size=9,
        line_width=2,
        probs=False,
        conf=False,
        txt_color=(0, 0, 0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    axes[0].imshow(img_annotated)
    axes[0].axis('off')
    axes[0].set_title("Predictions")

    axes[1].imshow(img_orig)
    axes[1].axis('off')
    axes[1].set_title("Original Image")

    fig.tight_layout()
    return fig


def save_matplotlib_fig(fig, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
