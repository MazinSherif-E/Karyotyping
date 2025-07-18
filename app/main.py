from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import uuid
import cv2
import numpy as np
import os
import logging
import gc
import traceback
from pathlib import Path
from yolo_inference import detect_chromosomes, CLASS_NAMES
from plotting import plot_instance_segmentation, save_matplotlib_fig
from create_karyogram import draw_karyogram

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    temp_path = None
    try:
        logger.info(f"🔥 STARTING: Received file: {image_file.filename}")
        
        # Check file size (limit to 10MB)
        file_size = 0
        content = await image_file.read()
        file_size = len(content)
        logger.info(f"📦 File size: {file_size} bytes")
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=413, detail="File too large")
        
        # Reset file pointer
        await image_file.seek(0)
        
        # Save uploaded image temporarily
        temp_path = PROJECT_ROOT / f"temp_{uuid.uuid4().hex}.jpg"
        logger.info(f"💾 Saving temp file to: {temp_path}")
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
        
        logger.info(f"✅ Temp file saved successfully, size: {temp_path.stat().st_size} bytes")
        
        # Verify file exists and is readable
        if not temp_path.exists():
            raise Exception("Temp file was not created")
            
        # Test if image can be read
        test_img = cv2.imread(str(temp_path))
        if test_img is None:
            raise Exception("Cannot read uploaded image - may be corrupted")
        logger.info(f"✅ Image verified: {test_img.shape}")
        del test_img  # Free memory
        gc.collect()
        
        # Run detection with error handling
        logger.info("🔍 CRITICAL: Starting chromosome detection...")
        try:
            res = detect_chromosomes(str(temp_path))
            logger.info(f"🧬 CRITICAL: Detection completed. Found {len(res[0].boxes)} chromosomes")
        except Exception as e:
            logger.error(f"❌ YOLO inference failed: {str(e)}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise Exception(f"YOLO inference failed: {str(e)}")

        # Force garbage collection after YOLO
        gc.collect()

        # Generate karyogram with error handling
        logger.info("🎨 CRITICAL: Starting karyogram creation...")
        try:
            karyo_filename = f"karyogram_{uuid.uuid4().hex}.png"
            karyo_path = STATIC_DIR / karyo_filename
            logger.info(f"📍 Creating karyogram at: {karyo_path}")
            
            karyogram = draw_karyogram(res, CLASS_NAMES)
            logger.info("✅ CRITICAL: Karyogram created successfully")
            
            save_matplotlib_fig(karyogram, str(karyo_path))
            logger.info("💾 CRITICAL: Karyogram saved successfully")
            
            # Verify the saved file
            if not karyo_path.exists():
                raise Exception("Karyogram file was not created")
            
            file_size = karyo_path.stat().st_size
            if file_size == 0:
                raise Exception("Karyogram file is empty")
                
            logger.info(f"✅ CRITICAL: Karyogram verified, size: {file_size} bytes")
            
        except Exception as e:
            logger.error(f"❌ Karyogram creation failed: {str(e)}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise Exception(f"Karyogram creation failed: {str(e)}")

        # Clean up temp image
        if temp_path and temp_path.exists():
            temp_path.unlink()
            logger.info("🧹 Temp file cleaned up")

        # Force final garbage collection
        gc.collect()
        
        logger.info(f"🎉 SUCCESS: Karyogram generation completed: {karyo_filename}")
        return FileResponse(
            path=str(karyo_path),
            media_type="image/png",
            filename=karyo_filename
        )
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {str(e)}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        
        # Clean up temp file if it exists
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
                logger.info("🧹 Cleaned up temp file after error")
            except:
                logger.error("Failed to clean up temp file")
        
        # Force garbage collection
        gc.collect()
        
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
        
        # Check available memory (if psutil available)
        memory_info = "Not available"
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_info = f"Available: {memory.available // (1024*1024)}MB, Total: {memory.total // (1024*1024)}MB"
        except ImportError:
            pass
        
        return {
            "status": "ok",
            "project_root": str(PROJECT_ROOT),
            "model_exists": model_exists,
            "model_path": str(model_path),
            "chroms_pool_exists": chroms_pool_exists,
            "chromosome_images_count": chrom_count,
            "static_dir": str(STATIC_DIR),
            "memory_info": memory_info
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/test")
def test_components():
    """Test individual components"""
    try:
        results = {}
        
        # Test 1: Model loading
        try:
            from .yolo_inference import model
            results["model_loaded"] = True
            results["model_type"] = str(type(model))
        except Exception as e:
            results["model_loaded"] = False
            results["model_error"] = str(e)
        
        # Test 2: Chromosome pool
        chroms_pool_path = PROJECT_ROOT / "chroms_pool"
        if chroms_pool_path.exists():
            png_files = list(chroms_pool_path.glob("*.png"))
            results["chroms_pool_count"] = len(png_files)
            
            # Test loading a sample chromosome
            if png_files:
                try:
                    sample_chrom = cv2.imread(str(png_files[0]))
                    results["sample_chrom_loaded"] = sample_chrom is not None
                    if sample_chrom is not None:
                        results["sample_chrom_shape"] = sample_chrom.shape
                except Exception as e:
                    results["sample_chrom_error"] = str(e)
        
        return {"status": "ok", "test_results": results}
        
    except Exception as e:
        logger.error(f"Component test failed: {str(e)}")
        return {"status": "error", "message": str(e)}