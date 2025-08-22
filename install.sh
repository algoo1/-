#!/bin/bash

# SigLIP-2 Server Installation Script for RunPod
# سكريبت تثبيت خادم SigLIP-2 على RunPod

set -e

echo "🚀 بدء تثبيت SigLIP-2 Server..."

# التحقق من Python
echo "📋 التحقق من Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python غير مثبت"
    exit 1
fi

python_version=$(python3 --version | cut -d' ' -f2)
echo "✅ Python version: $python_version"

# التحقق من pip
echo "📋 التحقق من pip..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip غير مثبت"
    exit 1
fi

# تحديث pip
echo "🔄 تحديث pip..."
pip3 install --upgrade pip

# التحقق من CUDA
echo "📋 التحقق من CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU متاح:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    CUDA_AVAILABLE=true
else
    echo "⚠️ NVIDIA GPU غير متاح - سيتم استخدام CPU"
    CUDA_AVAILABLE=false
fi

# تثبيت PyTorch
echo "🔄 تثبيت PyTorch..."
if [ "$CUDA_AVAILABLE" = true ]; then
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo "✅ تم تثبيت PyTorch مع دعم CUDA"
else
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo "✅ تم تثبيت PyTorch للـ CPU"
fi

# تثبيت المتطلبات
echo "🔄 تثبيت متطلبات المشروع..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    echo "✅ تم تثبيت جميع المتطلبات"
else
    echo "❌ ملف requirements.txt غير موجود"
    exit 1
fi

# تحقق من تثبيت transformers
echo "📋 التحقق من مكتبة transformers..."
python3 -c "import transformers; print('✅ transformers version:', transformers.__version__)"

# تحقق من تثبيت fastapi
echo "📋 التحقق من مكتبة FastAPI..."
python3 -c "import fastapi; print('✅ FastAPI version:', fastapi.__version__)"

# إنشاء مجلد cache للنماذج
echo "📁 إنشاء مجلد cache..."
mkdir -p /workspace/model_cache
export TRANSFORMERS_CACHE=/workspace/model_cache
echo "export TRANSFORMERS_CACHE=/workspace/model_cache" >> ~/.bashrc

# تحميل النموذج مسبقاً (اختياري)
echo "🤖 هل تريد تحميل النموذج مسبقاً؟ (y/n)"
read -r preload_model

if [ "$preload_model" = "y" ] || [ "$preload_model" = "Y" ]; then
    echo "⬇️ تحميل نموذج SigLIP-2..."
    python3 -c "
from transformers import AutoModel, AutoProcessor
import torch

print('تحميل النموذج...')
model_name = 'google/siglip2-so400m-patch14-384'
try:
    model = AutoModel.from_pretrained(model_name, cache_dir='/workspace/model_cache')
    processor = AutoProcessor.from_pretrained(model_name, cache_dir='/workspace/model_cache')
    print('✅ تم تحميل النموذج بنجاح')
except Exception as e:
    print(f'❌ خطأ في تحميل النموذج: {e}')
"
fi

# إنشاء سكريبت بدء سريع
echo "📝 إنشاء سكريپت البدء السريع..."
cat > quick_start.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    print("🚀 SigLIP-2 Server Quick Start")
    print("=" * 40)
    
    # تحقق من الملفات المطلوبة
    if not os.path.exists('main.py'):
        print("❌ main.py غير موجود")
        return
    
    # تعيين متغيرات البيئة
    os.environ['TRANSFORMERS_CACHE'] = '/workspace/model_cache'
    
    print("🔄 بدء تشغيل الخادم...")
    print("📡 الخادم متاح على: http://0.0.0.0:8000")
    print("📚 الوثائق متاحة على: http://0.0.0.0:8000/docs")
    print("💡 اضغط Ctrl+C لإيقاف الخادم")
    print("-" * 40)
    
    # تشغيل الخادم
    try:
        subprocess.run([
            sys.executable, 'main.py',
            '--host', '0.0.0.0',
            '--port', '8000'
        ])
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف الخادم")

if __name__ == "__main__":
    main()
EOF

chmod +x quick_start.py

# اختبار سريع
echo "🧪 إجراء اختبار سريع..."
python3 -c "
try:
    import torch
    import transformers
    import fastapi
    import uvicorn
    from PIL import Image
    print('✅ جميع المكتبات تعمل بشكل صحيح')
    
    if torch.cuda.is_available():
        print(f'✅ CUDA متاح - الجهاز: {torch.cuda.get_device_name(0)}')
        print(f'✅ ذاكرة GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        print('ℹ️ CUDA غير متاح - سيتم استخدام CPU')
        
except ImportError as e:
    print(f'❌ خطأ في استيراد المكتبات: {e}')
    exit(1)
"

echo ""
echo "🎉 تم التثبيت بنجاح!"
echo "=" * 50
echo "للبدء السريع، استخدم:"
echo "  python3 quick_start.py"
echo ""
echo "أو تشغيل الخادم مباشرة:"
echo "  python3 main.py"
echo ""
echo "أو استخدام سكريپت التشغيل:"
echo "  ./scripts/run_server.sh"
echo "=" * 50
