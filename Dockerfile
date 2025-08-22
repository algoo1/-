# Dockerfile for SigLIP-2 Server
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# تعيين متغيرات البيئة
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf-8

# تثبيت النظام
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# تحديث pip
RUN pip3 install --upgrade pip

# إنشاء مجلد العمل
WORKDIR /app

# نسخ متطلبات المشروع
COPY requirements.txt .

# تثبيت المكتبات Python
RUN pip3 install -r requirements.txt

# تثبيت PyTorch مع دعم CUDA
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# نسخ ملفات المشروع
COPY . .

# تعريض البورت
EXPOSE 8000

# الأمر الافتراضي
CMD ["python3", "main.py", "--host", "0.0.0.0", "--port", "8000"]
