from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import uuid
import cv2
import numpy as np
import os
from yolo_inference import detect_chromosomes, CLASS_NAMES
from plotting import plot_instance_segmentation, save_matplotlib_fig
from create_karyogram import  draw_karyogram

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = FastAPI()

STATIC_DIR = "static/karyograms"
os.makedirs(STATIC_DIR, exist_ok=True)

@app.post("/karyogram/")
async def generate_karyogram(image_file: UploadFile = File(...)):
    # Save uploaded image temporarily
    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image_file.file, buffer)

    # Run detection
    res = detect_chromosomes(temp_path)

    karyo_filename = f"karyogram_{uuid.uuid4().hex}.png"
    karyo_path = os.path.join(STATIC_DIR, karyo_filename)
    kayrogram=draw_karyogram(res, CLASS_NAMES)
    
    save_matplotlib_fig(kayrogram,karyo_path)

    # Remove temp image
    os.remove(temp_path)

    return FileResponse(
        path=karyo_path,
        media_type="image/png",
        filename=karyo_filename
    )
