import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Use absolute path for chromosome pool
PROJECT_ROOT = Path(__file__).parent.parent
CHROMS_POOL_DIR = PROJECT_ROOT / "chroms_pool"

logger.info(f"Chromosome pool directory: {CHROMS_POOL_DIR}")

def draw_karyogram(res, classes):
    logger.info("Starting karyogram creation")
    kayrogram = np.ones((300, 2000, 3), dtype=np.uint8) * 255
    counts = {i: 0 for i in classes}
    space_in = 10
    space_out_x = 50
    classes_inds = res[0].boxes.cls.cpu().numpy()
    classes_inds = [int(i) for i in classes_inds]
    classes_inds = sorted(classes_inds)
    startx = 0
    starty = 150
    max_x = 0

    pair_info = {}

    logger.info(f"Processing {len(classes_inds)} chromosomes")

    for cls in classes_inds:
        cls_ind = cls
        if cls_ind == 12 and starty == 150:
            starty = 260
            startx=0
        counts[classes[cls_ind]] += 1

        if classes[cls_ind] != 'y':
            chrom_path = CHROMS_POOL_DIR / f'{classes[cls_ind]}.{counts[classes[cls_ind]] % 2}.png'
        else:
            chrom_path = CHROMS_POOL_DIR / f'{classes[cls_ind]}.png'
        
        logger.debug(f"Loading chromosome image: {chrom_path}")
        
        if not chrom_path.exists():
            logger.error(f"Chromosome image not found: {chrom_path}")
            continue
            
        chrom = cv2.imread(str(chrom_path))
        
        if chrom is None:
            logger.error(f"Failed to load chromosome image: {chrom_path}")
            continue

        ys, xs = np.where(chrom[:, :, 0] < 230)
        min_y, max_y, min_x, max_x_crop = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
        chrom = chrom[min_y:max_y, min_x:max_x_crop]

        if counts[classes[cls_ind]] == 1:
            startx += space_out_x
            if startx >= 2000:
                startx = space_out_x
                starty = 250
            pair_info[classes[cls_ind]] = {
                "x_start": startx,
                "y_start": starty,
                "width_total": 0
            }
        else:
            startx += space_in

        h, w = chrom.shape[:2]
        kayrogram[starty - h:starty, startx:startx + w, :] = chrom

        pair_info[classes[cls_ind]]["width_total"] += w + (space_in if counts[classes[cls_ind]] == 2 else 0)

        if counts[classes[cls_ind]] == 2:
            label = classes[cls_ind]
            pair_x = pair_info[label]["x_start"]
            pair_w = pair_info[label]["width_total"]
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = pair_x + (pair_w - text_size[0]) // 2
            text_y = pair_info[label]['y_start'] + 30
            cv2.putText(
                kayrogram, label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
            )

            line_y = starty + 5
            cv2.line(
                kayrogram,
                (pair_x, line_y),
                (pair_x + pair_w, line_y),
                (0, 0, 0), 2
            )

        startx += w

        if startx + w > max_x:
            max_x = startx + w + space_out_x

    for label, info in pair_info.items():
        if counts[label] == 1:
            pair_x = info["x_start"]
            pair_w = info["width_total"]
            pair_y = info["y_start"]

            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = pair_x + (pair_w - text_size[0]) // 2
            text_y = pair_y + 30

            cv2.putText(
                kayrogram, label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
            )

            line_y = pair_y + 5
            cv2.line(
                kayrogram,
                (pair_x, line_y),
                (pair_x + pair_w, line_y),
                (0, 0, 0), 2
            )

    kayrogram = kayrogram[:, :max_x]
    fig, ax = plt.subplots(1, 1)
    ax.imshow(kayrogram)
    ax.axis('off')
    ax.set_title("Karyogram")
    fig.tight_layout()
    
    logger.info("Karyogram creation completed")
    return fig
