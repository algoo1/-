#!/usr/bin/env python3
"""
SigLIP-2 Server Implementation
خادم لتشغيل نموذج SigLIP-2 مع FastAPI
"""

import os
import io
import base64
import logging
from typing import List, Optional, Dict, Any, Union
import asyncio
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoModel, AutoProcessor, pipeline
from transformers.image_utils import load_image
import numpy as np

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تكوين النموذج
MODEL_NAME = "google/siglip2-so400m-patch14-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# متغيرات عامة للنموذج
model = None
processor = None
pipeline_classifier = None

class ImageClassificationRequest(BaseModel):
    """طلب تصنيف الصور"""
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    candidate_labels: List[str] = Field(..., description="قائمة التصنيفات المرشحة")
    threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="حد الثقة الأدنى")

class ImageEmbeddingRequest(BaseModel):
    """طلب استخراج المميزات من الصورة"""
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    normalize: bool = Field(default=True, description="تطبيع المتجهات")

class TextEmbeddingRequest(BaseModel):
    """طلب استخراج المميزات من النص"""
    texts: List[str] = Field(..., description="قائمة النصوص")
    normalize: bool = Field(default=True, description="تطبيع المتجهات")

class SimilarityRequest(BaseModel):
    """طلب حساب التشابه بين الصورة والنص"""
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    texts: List[str] = Field(..., description="قائمة النصوص للمقارنة")

class ModelResponse(BaseModel):
    """استجابة عامة من النموذج"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None

async def load_model():
    """تحميل النموذج والمعالج"""
    global model, processor, pipeline_classifier
    
    try:
        logger.info(f"تحميل النموذج: {MODEL_NAME}")
        
        # تحميل النموذج الأساسي والمعالج
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            device_map="auto" if DEVICE == "cuda" else None,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            trust_remote_code=True
        ).eval()
        
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        
        # تحميل pipeline للتصنيف
        pipeline_classifier = pipeline(
            model=MODEL_NAME,
            task="zero-shot-image-classification",
            device=0 if DEVICE == "cuda" else -1
        )
        
        if DEVICE == "cuda":
            model = model.to(DEVICE)
        
        logger.info(f"تم تحميل النموذج بنجاح على الجهاز: {DEVICE}")
        
    except Exception as e:
        logger.error(f"خطأ في تحميل النموذج: {str(e)}")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    """إدارة دورة حياة التطبيق"""
    # تحميل النموذج عند البدء
    await load_model()
    yield
    # تنظيف الذاكرة عند الإغلاق
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# إنشاء تطبيق FastAPI
app = FastAPI(
    title="SigLIP-2 Vision-Language Server",
    description="خادم لتشغيل نموذج SigLIP-2 للرؤية واللغة",
    version="1.0.0",
    lifespan=lifespan
)

# إضافة CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_image_from_input(image_url: Optional[str] = None, 
                         image_base64: Optional[str] = None, 
                         uploaded_file: Optional[UploadFile] = None) -> Image.Image:
    """تحميل الصورة من مصادر مختلفة"""
    try:
        if uploaded_file:
            # تحميل من ملف مرفوع
            if uploaded_file.size > MAX_IMAGE_SIZE:
                raise HTTPException(status_code=413, detail="حجم الصورة كبير جداً")
            
            image_data = uploaded_file.file.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
        elif image_base64:
            # فك تشفير base64
            try:
                image_data = base64.b64decode(image_base64.split(',')[-1])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            except Exception:
                raise HTTPException(status_code=400, detail="تنسيق base64 غير صحيح")
                
        elif image_url:
            # تحميل من URL
            image = load_image(image_url)
            
        else:
            raise HTTPException(status_code=400, detail="يجب توفير صورة (URL، base64، أو ملف)")
        
        return image
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ في تحميل الصورة: {str(e)}")
        raise HTTPException(status_code=400, detail=f"فشل في تحميل الصورة: {str(e)}")

@app.get("/")
async def root():
    """الصفحة الرئيسية"""
    return {
        "message": "SigLIP-2 Vision-Language Server",
        "model": MODEL_NAME,
        "device": DEVICE,
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """فحص صحة الخادم"""
    try:
        # اختبار بسيط للنموذج
        is_healthy = model is not None and processor is not None
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "model_loaded": model is not None,
            "processor_loaded": processor is not None,
            "device": DEVICE,
            "gpu_available": torch.cuda.is_available()
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/classify", response_model=ModelResponse)
async def classify_image(request: ImageClassificationRequest):
    """تصنيف الصورة باستخدام zero-shot classification"""
    import time
    start_time = time.time()
    
    try:
        # تحميل الصورة
        image = load_image_from_input(request.image_url, request.image_base64)
        
        # تشغيل التصنيف
        results = pipeline_classifier(image, request.candidate_labels)
        
        # تطبيق عتبة الثقة
        filtered_results = [
            result for result in results 
            if result['score'] >= request.threshold
        ]
        
        processing_time = time.time() - start_time
        
        return ModelResponse(
            success=True,
            message="تم التصنيف بنجاح",
            data={
                "classifications": filtered_results,
                "total_candidates": len(request.candidate_labels),
                "filtered_results": len(filtered_results)
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"خطأ في التصنيف: {str(e)}")
        return ModelResponse(
            success=False,
            message=f"فشل في التصنيف: {str(e)}"
        )

@app.post("/classify/upload")
async def classify_uploaded_image(
    file: UploadFile = File(...),
    candidate_labels: str = Form(...),
    threshold: float = Form(default=0.1)
):
    """تصنيف صورة مرفوعة"""
    try:
        # تحويل النصوص إلى قائمة
        labels_list = [label.strip() for label in candidate_labels.split(',')]
        
        # تحميل الصورة
        image = load_image_from_input(uploaded_file=file)
        
        # تشغيل التصنيف
        results = pipeline_classifier(image, labels_list)
        
        # تطبيق عتبة الثقة
        filtered_results = [
            result for result in results 
            if result['score'] >= threshold
        ]
        
        return JSONResponse({
            "success": True,
            "message": "تم التصنيف بنجاح",
            "data": {
                "classifications": filtered_results,
                "filename": file.filename
            }
        })
        
    except Exception as e:
        logger.error(f"خطأ في تصنيف الملف المرفوع: {str(e)}")
        return JSONResponse({
            "success": False,
            "message": f"فشل في التصنيف: {str(e)}"
        })

@app.post("/embeddings/image", response_model=ModelResponse)
async def get_image_embeddings(request: ImageEmbeddingRequest):
    """استخراج المميزات من الصورة"""
    import time
    start_time = time.time()
    
    try:
        # تحميل الصورة
        image = load_image_from_input(request.image_url, request.image_base64)
        
        # معالجة الصورة
        inputs = processor(images=[image], return_tensors="pt").to(model.device)
        
        # استخراج المميزات
        with torch.no_grad():
            image_embeddings = model.get_image_features(**inputs)
            
            if request.normalize:
                image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
        
        # تحويل إلى قائمة
        embeddings_list = image_embeddings.cpu().numpy().tolist()
        processing_time = time.time() - start_time
        
        return ModelResponse(
            success=True,
            message="تم استخراج المميزات بنجاح",
            data={
                "embeddings": embeddings_list,
                "shape": list(image_embeddings.shape),
                "normalized": request.normalize
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"خطأ في استخراج مميزات الصورة: {str(e)}")
        return ModelResponse(
            success=False,
            message=f"فشل في استخراج المميزات: {str(e)}"
        )

@app.post("/embeddings/text", response_model=ModelResponse)
async def get_text_embeddings(request: TextEmbeddingRequest):
    """استخراج المميزات من النص"""
    import time
    start_time = time.time()
    
    try:
        # معالجة النصوص
        inputs = processor(text=request.texts, return_tensors="pt", padding=True).to(model.device)
        
        # استخراج المميزات
        with torch.no_grad():
            text_embeddings = model.get_text_features(**inputs)
            
            if request.normalize:
                text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)
        
        # تحويل إلى قائمة
        embeddings_list = text_embeddings.cpu().numpy().tolist()
        processing_time = time.time() - start_time
        
        return ModelResponse(
            success=True,
            message="تم استخراج المميزات بنجاح",
            data={
                "embeddings": embeddings_list,
                "shape": list(text_embeddings.shape),
                "texts": request.texts,
                "normalized": request.normalize
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"خطأ في استخراج مميزات النص: {str(e)}")
        return ModelResponse(
            success=False,
            message=f"فشل في استخراج المميزات: {str(e)}"
        )

@app.post("/similarity", response_model=ModelResponse)
async def calculate_similarity(request: SimilarityRequest):
    """حساب التشابه بين الصورة والنصوص"""
    import time
    start_time = time.time()
    
    try:
        # تحميل الصورة
        image = load_image_from_input(request.image_url, request.image_base64)
        
        # معالجة الصورة والنصوص
        inputs = processor(
            images=[image], 
            text=request.texts, 
            return_tensors="pt", 
            padding=True
        ).to(model.device)
        
        # حساب التشابه
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.softmax(logits_per_image, dim=-1)
        
        # تحضير النتائج
        similarities = []
        for i, text in enumerate(request.texts):
            similarities.append({
                "text": text,
                "similarity_score": float(probs[0][i].cpu()),
                "logit": float(logits_per_image[0][i].cpu())
            })
        
        # ترتيب حسب النتيجة
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        processing_time = time.time() - start_time
        
        return ModelResponse(
            success=True,
            message="تم حساب التشابه بنجاح",
            data={
                "similarities": similarities,
                "best_match": similarities[0] if similarities else None,
                "total_comparisons": len(request.texts)
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"خطأ في حساب التشابه: {str(e)}")
        return ModelResponse(
            success=False,
            message=f"فشل في حساب التشابه: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info():
    """معلومات عن النموذج"""
    return {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "model_loaded": model is not None,
        "processor_loaded": processor is not None,
        "pipeline_loaded": pipeline_classifier is not None,
        "capabilities": [
            "zero-shot image classification",
            "image embeddings extraction",
            "text embeddings extraction", 
            "image-text similarity calculation"
        ],
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "webp"],
        "max_image_size_mb": MAX_IMAGE_SIZE // (1024 * 1024)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SigLIP-2 Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "main:app" if args.reload else app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )
