import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

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
    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving figure to: {save_path}")
        fig.savefig(str(save_path), dpi=300, bbox_inches='tight', pad_inches=0.1)
        logger.info("Figure saved successfully")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error saving figure: {str(e)}")
        plt.close(fig)
        raise
