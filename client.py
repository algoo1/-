#!/usr/bin/env python3
"""
SigLIP-2 Python Client
عميل Python للتفاعل مع خادم SigLIP-2
"""

import requests
import base64
import json
import time
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SigLIPClient:
    """عميل Python للتفاعل مع خادم SigLIP-2"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        تهيئة العميل
        
        Args:
            base_url: رابط الخادم الأساسي
            timeout: مهلة الطلب بالثواني
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """إرسال طلب HTTP مع معالجة الأخطاء"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method, 
                url, 
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"خطأ في الطلب: {e}")
            return {
                "success": False,
                "message": f"خطأ في الاتصال: {str(e)}",
                "data": None
            }
    
    def health_check(self) -> Dict[str, Any]:
        """فحص صحة الخادم"""
        return self._make_request("GET", "/health")
    
    def get_server_info(self) -> Dict[str, Any]:
        """الحصول على معلومات الخادم"""
        return self._make_request("GET", "/")
    
    def get_model_info(self) -> Dict[str, Any]:
        """الحصول على معلومات النموذج"""
        return self._make_request("GET", "/model/info")
    
    def image_to_base64(self, image_path: Union[str, Path]) -> str:
        """تحويل صورة إلى base64"""
        try:
            with open(image_path, "rb") as img_file:
