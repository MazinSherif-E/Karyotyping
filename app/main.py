from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import uuid
import cv2
import numpy as np
import os
import logging
from pathlib import Path
from yolo_inference import detect_chromosomes, CLASS_NAMES
from plotting import plot_instance_segmentation, save_matplotlib_fig
from create_karyogram import draw_karyogram

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = FastAPI()

# Use absolute paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
STATIC_DIR = PROJECT_ROOT / "static" / "karyograms"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Static directory: {STATIC_DIR}")

@app.post("/karyogram/")
async def generate_karyogram(image_file: UploadFile = File(...)):
    logger.info(f"Received file: {image_file.filename}")
    
    try:
        # Save uploaded image temporarily
        temp_path = PROJECT_ROOT / f"temp_{uuid.uuid4().hex}.jpg"
        logger.info(f"Saving temp file to: {temp_path}")
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
        
        logger.info("Temp file saved successfully")
        
        # Run detection
        logger.info("Starting chromosome detection...")
        res = detect_chromosomes(str(temp_path))
        logger.info(f"Detection completed. Found {len(res[0].boxes)} chromosomes")

        # Generate karyogram
        karyo_filename = f"karyogram_{uuid.uuid4().hex}.png"
        karyo_path = STATIC_DIR / karyo_filename
        logger.info(f"Creating karyogram at: {karyo_path}")
        
        karyogram = draw_karyogram(res, CLASS_NAMES)
        logger.info("Karyogram created successfully")
        
        save_matplotlib_fig(karyogram, str(karyo_path))
        logger.info("Karyogram saved successfully")

        # Remove temp image
        if temp_path.exists():
            temp_path.unlink()
            logger.info("Temp file cleaned up")

        return FileResponse(
            path=str(karyo_path),
            media_type="image/png",
            filename=karyo_filename
        )
        
    except Exception as e:
        logger.error(f"Error processing karyogram: {str(e)}")
        # Clean up temp file if it exists
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing karyogram: {str(e)}")

@app.get("/")
def read_root():
    logger.info("Health check endpoint called")
    return {"status": "ok"}

@app.get("/health")
def health_check():
    """Extended health check with system info"""
    try:
        logger.info("Extended health check called")
        
        # Check if model file exists
        model_path = PROJECT_ROOT / "models" / "best.pt"
        model_exists = model_path.exists()
        
        # Check if chroms_pool exists
        chroms_pool_path = PROJECT_ROOT / "chroms_pool"
        chroms_pool_exists = chroms_pool_path.exists()
        
        # Count chromosome images
        chrom_count = 0
        if chroms_pool_exists:
            chrom_count = len(list(chroms_pool_path.glob("*.png")))
        
        return {
            "status": "ok",
            "project_root": str(PROJECT_ROOT),
            "model_exists": model_exists,
            "model_path": str(model_path),
            "chroms_pool_exists": chroms_pool_exists,
            "chromosome_images_count": chrom_count,
            "static_dir": str(STATIC_DIR)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "message": str(e)}