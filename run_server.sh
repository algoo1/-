#!/bin/bash

# SigLIP-2 Server Runner Script for RunPod
# سكريپت تشغيل خادم SigLIP-2 على RunPod

set -e

# ألوان للإخراج
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# الإعدادات الافتراضية
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-1}
RELOAD=${RELOAD:-false}

echo -e "${BLUE}🚀 SigLIP-2 Server Runner${NC}"
echo "=================================="

# التحقق من وجود الملف الرئيسي
if [ ! -f "main.py" ]; then
    echo -e "${RED}❌ main.py غير موجود في المجلد الحالي${NC}"
    exit 1
fi

# تعيين متغيرات البيئة
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"/workspace/model_cache"}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

echo -e "${YELLOW}📋 إعدادات التشغيل:${NC}"
echo "   Host: $HOST"
echo "   Port: $PORT"  
echo "   Workers: $WORKERS"
echo "   Reload: $RELOAD"
echo "   Model Cache: $TRANSFORMERS_CACHE"
echo "   CUDA Device: $CUDA_VISIBLE_DEVICES"

# التحقق من المتطلبات
echo -e "\n${YELLOW}🔍 التحقق من المتطلبات...${NC}"

# التحقق من Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 غير مثبت${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python3 متاح${NC}"

# التحقق من المكتبات المطلوبة
python3 -c "
import sys
required_modules = ['fastapi', 'uvicorn', 'torch', 'transformers', 'PIL']
missing_modules = []

for module in required_modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError:
        missing_modules.append(module)
        print(f'❌ {module} مفقود')

if missing_modules:
    print(f'❌ المكتبات المفقودة: {missing_modules}')
    print('يرجى تشغيل: pip install -r requirements.txt')
    sys.exit(1)
"

# التحقق من GPU
echo -e "\n${YELLOW}🎮 فحص GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU متاح:${NC}"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | while read line; do
        echo "   GPU: $line"
    done
else
    echo -e "${YELLOW}⚠️ NVIDIA GPU غير متاح - سيتم استخدام CPU${NC}"
fi

# إنشاء مجلد cache إذا لم يكن موجوداً
mkdir -p "$TRANSFORMERS_CACHE"

# فحص المساحة المتاحة
available_space=$(df -h . | awk 'NR==2 {print $4}')
echo -e "${BLUE}💾 المساحة المتاحة: $available_space${NC}"

# إنشاء ملف log
LOG_FILE="/workspace/siglip2_server.log"
touch "$LOG_FILE"

echo -e "\n${YELLOW}📝 الـ logs متاحة في: $LOG_FILE${NC}"

# دالة لتنظيف العمليات عند الخروج
cleanup() {
    echo -e "\n${YELLOW}🧹 تنظيف العمليات...${NC}"
    pkill -f "python3 main.py" || true
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | xargs -r kill || true
    fi
    echo -e "${GREEN}✅ تم التنظيف${NC}"
}

trap cleanup EXIT

# بدء الخادم
echo -e "\n${GREEN}🚀 بدء تشغيل SigLIP-2 Server...${NC}"
echo -e "${BLUE}📡 الخادم متاح على: http://$HOST:$PORT${NC}"
echo -e "${BLUE}📚 الوثائق متاحة على: http://$HOST:$PORT/docs${NC}"
echo -e "${BLUE}❤️ فحص الصحة: http://$HOST:$PORT/health${NC}"
echo -e "${YELLOW}💡 اضغط Ctrl+C لإيقاف الخادم${NC}"
echo "=================================="

# اختيار طريقة التشغيل
if [ "$RELOAD" = "true" ]; then
    # تشغيل مع إعادة التحميل (للتطوير)
    python3 main.py --host "$HOST" --port "$PORT" --reload 2>&1 | tee "$LOG_FILE"
else
    # تشغيل عادي
    if [ "$WORKERS" -gt 1 ]; then
        # تشغيل متعدد العمليات
        uvicorn main:app --host "$HOST" --port "$PORT" --workers "$WORKERS" 2>&1 | tee "$LOG_FILE"
    else
        # تشغيل عملية واحدة
        python3 main.py --host "$HOST" --port "$PORT" 2>&1 | tee "$LOG_FILE"
    fi
fi
