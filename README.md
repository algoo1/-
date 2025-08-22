# SigLIP-2 Vision-Language Server

خادم FastAPI لتشغيل نموذج Google SigLIP-2 للرؤية واللغة مع دعم العربية.

## المميزات

- 🔍 **Zero-shot Image Classification**: تصنيف الصور بدون تدريب مسبق
- 🎯 **Image Embeddings**: استخراج المميزات من الصور
- 📝 **Text Embeddings**: استخراج المميزات من النصوص
- 🔗 **Similarity Calculation**: حساب التشابه بين الصور والنصوص
- 🚀 **Fast API**: واجهة برمجية سريعة وموثقة
- 🐳 **Docker Support**: دعم كامل لـ Docker
- ⚡ **GPU Acceleration**: تسريع تلقائي باستخدام GPU
- 🌍 **Multi-input Support**: دعم URLs، base64، ورفع الملفات

## التثبيت السريع

### 1. استنساخ المشروع
```bash
git clone https://github.com/YOUR_USERNAME/siglip2-server.git
cd siglip2-server
```

### 2. تثبيت المتطلبات
```bash
pip install -r requirements.txt
```

### 3. تشغيل الخادم
```bash
python main.py
```

## التشغيل على RunPod

### الطريقة الأولى: استخدام الكود مباشرة
```bash
# في RunPod Terminal
git clone https://github.com/YOUR_USERNAME/siglip2-server.git
cd siglip2-server
pip install -r requirements.txt
python main.py --host 0.0.0.0 --port 8000
```

### الطريقة الثانية: استخدام Docker
```bash
# في RunPod Terminal
git clone https://github.com/YOUR_USERNAME/siglip2-server.git
cd siglip2-server
docker build -t siglip2-server .
docker run -p 8000:8000 --gpus all siglip2-server
```

### الطريقة الثالثة: استخدام سكريبت التثبيت
```bash
# في RunPod Terminal
git clone https://github.com/YOUR_USERNAME/siglip2-server.git
cd siglip2-server
chmod +x scripts/install.sh
./scripts/install.sh
chmod +x scripts/run_server.sh
./scripts/run_server.sh
```

## استخدام API

### 1. تصنيف صورة
```bash
curl -X POST "http://YOUR_RUNPOD_URL:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "candidate_labels": ["قطة", "كلب", "سيارة"],
    "threshold": 0.1
  }'
```

### 2. رفع صورة للتصنيف
```bash
curl -X POST "http://YOUR_RUNPOD_URL:8000/classify/upload" \
  -F "file=@image.jpg" \
  -F "candidate_labels=قطة,كلب,سيارة" \
  -F "threshold=0.1"
```

### 3. حساب التشابه
```bash
curl -X POST "http://YOUR_RUNPOD_URL:8000/similarity" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "texts": ["قطة جميلة", "كلب لطيف", "سيارة سريعة"]
  }'
```

## Python Client

```python
from examples.client import SigLIPClient

client = SigLIPClient("http://YOUR_RUNPOD_URL:8000")

# تصنيف صورة
result = client.classify_image_url(
    "https://example.com/cat.jpg",
    ["قطة", "كلب", "سيارة"]
)
print(result)
```

## متطلبات النظام

- **RAM**: 8GB+ (16GB مفضل)
- **GPU**: NVIDIA GPU مع 8GB+ VRAM (اختياري)
- **Storage**: 10GB+ للنموذج والتبعيات
- **Python**: 3.8+

## الوثائق

- [API Documentation](docs/API_DOCS.md)
- [Examples](examples/)
- [Scripts](scripts/)

## المساهمة

مرحب بالمساهمات! يرجى فتح issue أو pull request.

## الرخصة

MIT License

## الدعم

إذا واجهت مشاكل، يرجى فتح issue على GitHub.

---

## RunPod Specific Notes

### تعيين المنافذ
- تأكد من تعريض البورت 8000 في إعدادات RunPod
- استخدم Public IP للوصول الخارجي

### متغيرات البيئة المفيدة
```bash
export CUDA_VISIBLE_DEVICES=0  # لاستخدام GPU محدد
export TRANSFORMERS_CACHE=/workspace/cache  # لحفظ النماذج
```

### نصائح الأداء
- استخدم SSD storage للأداء الأفضل
- احفظ النماذج المحملة في `/workspace` لتجنب إعادة التحميل
