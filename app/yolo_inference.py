import torch
from ultralytics import YOLO

# Load model once globally
model_path = r'../models/best.pt'
model = YOLO(model_path)

NUM_CLASSES = 24
CLASS_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
               '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
               '21', '22', 'x', 'y']


def apply_nms(res, iou_thresh=0.7, conf_thresh=0.2):
    masks = res[0].masks
    boxes = res[0].boxes
    to_remove = []

    for i in range(len(masks)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(masks)):
            if j in to_remove:
                continue
            intersection = torch.logical_and(masks[i].data, masks[j].data)
            union = torch.logical_or(masks[i].data, masks[j].data)
            iou = torch.sum(intersection) / torch.sum(union) if torch.sum(union) > 0 else 0.0
            if iou > iou_thresh:
                if boxes[i].conf >= boxes[j].conf:
                    to_remove.append(j)
                else:
                    to_remove.append(i)



    to_keep = [i for i in range(len(masks)) if i not in to_remove]
    masks = masks[to_keep]
    boxes = boxes[to_keep]

    res[0].masks = masks
    res[0].boxes = boxes
    return res


def detect_chromosomes(image_path):
    res = model.predict(image_path, conf=0.2, multi_scale=True, verbose=False)
    res = apply_nms(res)
    return res
