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
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{img_base64}"
        except Exception as e:
            logger.error(f"خطأ في تحويل الصورة إلى base64: {e}")
            raise
    
    def classify_image_url(self, image_url: str, candidate_labels: List[str], 
                          threshold: float = 0.1) -> Dict[str, Any]:
        """تصنيف صورة من URL"""
        data = {
            "image_url": image_url,
            "candidate_labels": candidate_labels,
            "threshold": threshold
        }
        return self._make_request("POST", "/classify", json=data)
    
    def classify_image_base64(self, image_base64: str, candidate_labels: List[str], 
                             threshold: float = 0.1) -> Dict[str, Any]:
        """تصنيف صورة من base64"""
        data = {
            "image_base64": image_base64,
            "candidate_labels": candidate_labels,
            "threshold": threshold
        }
        return self._make_request("POST", "/classify", json=data)
    
    def classify_image_file(self, image_path: Union[str, Path], 
                           candidate_labels: List[str], threshold: float = 0.1) -> Dict[str, Any]:
        """تصنيف صورة من ملف محلي"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
                data = {
                    'candidate_labels': ','.join(candidate_labels),
                    'threshold': threshold
                }
                return self._make_request("POST", "/classify/upload", files=files, data=data)
        except Exception as e:
            logger.error(f"خطأ في رفع الملف: {e}")
            return {
                "success": False,
                "message": f"خطأ في رفع الملف: {str(e)}",
                "data": None
            }
    
    def get_image_embeddings_url(self, image_url: str, normalize: bool = True) -> Dict[str, Any]:
        """استخراج مميزات صورة من URL"""
        data = {
            "image_url": image_url,
            "normalize": normalize
        }
        return self._make_request("POST", "/embeddings/image", json=data)
    
    def get_image_embeddings_base64(self, image_base64: str, normalize: bool = True) -> Dict[str, Any]:
        """استخراج مميزات صورة من base64"""
        data = {
            "image_base64": image_base64,
            "normalize": normalize
        }
        return self._make_request("POST", "/embeddings/image", json=data)
    
    def get_image_embeddings_file(self, image_path: Union[str, Path], 
                                 normalize: bool = True) -> Dict[str, Any]:
        """استخراج مميزات صورة من ملف محلي"""
        image_base64 = self.image_to_base64(image_path)
        return self.get_image_embeddings_base64(image_base64, normalize)
    
    def get_text_embeddings(self, texts: List[str], normalize: bool = True) -> Dict[str, Any]:
        """استخراج مميزات النصوص"""
        data = {
            "texts": texts,
            "normalize": normalize
        }
        return self._make_request("POST", "/embeddings/text", json=data)
    
    def calculate_similarity_url(self, image_url: str, texts: List[str]) -> Dict[str, Any]:
        """حساب التشابه بين صورة (URL) ونصوص"""
        data = {
            "image_url": image_url,
            "texts": texts
        }
        return self._make_request("POST", "/similarity", json=data)
    
    def calculate_similarity_base64(self, image_base64: str, texts: List[str]) -> Dict[str, Any]:
        """حساب التشابه بين صورة (base64) ونصوص"""
        data = {
            "image_base64": image_base64,
            "texts": texts
        }
        return self._make_request("POST", "/similarity", json=data)
    
    def calculate_similarity_file(self, image_path: Union[str, Path], 
                                 texts: List[str]) -> Dict[str, Any]:
        """حساب التشابه بين صورة (ملف محلي) ونصوص"""
        image_base64 = self.image_to_base64(image_path)
        return self.calculate_similarity_base64(image_base64, texts)
    
    def batch_classify(self, images: List[Dict[str, Any]], 
                      candidate_labels: List[str], threshold: float = 0.1) -> List[Dict[str, Any]]:
        """تصنيف متعدد للصور"""
        results = []
        
        for i, image_data in enumerate(images):
            logger.info(f"معالجة الصورة {i+1}/{len(images)}")
            
            if 'url' in image_data:
                result = self.classify_image_url(image_data['url'], candidate_labels, threshold)
            elif 'path' in image_data:
                result = self.classify_image_file(image_data['path'], candidate_labels, threshold)
            elif 'base64' in image_data:
                result = self.classify_image_base64(image_data['base64'], candidate_labels, threshold)
            else:
                result = {
                    "success": False,
                    "message": "نوع الصورة غير مدعوم",
                    "data": None
                }
            
            result['image_index'] = i
            if 'id' in image_data:
                result['image_id'] = image_data['id']
                
            results.append(result)
            
        return results
    
    def benchmark_server(self, test_image_url: str = None, 
                        test_texts: List[str] = None, iterations: int = 5) -> Dict[str, Any]:
        """اختبار أداء الخادم"""
        if test_image_url is None:
            test_image_url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba"
        
        if test_texts is None:
            test_texts = ["a cat", "a dog", "a car", "a tree"]
        
        results = {
            "health_check": [],
            "classification": [],
            "similarity": [],
            "image_embeddings": [],
            "text_embeddings": []
        }
        
        logger.info(f"بدء اختبار الأداء مع {iterations} تكرار")
        
        for i in range(iterations):
            logger.info(f"التكرار {i+1}/{iterations}")
            
            # فحص الصحة
            start = time.time()
            health = self.health_check()
            results["health_check"].append({
                "success": health.get("success", True),
                "time": time.time() - start
            })
            
            # التصنيف
            start = time.time()
            classification = self.classify_image_url(test_image_url, test_texts)
            results["classification"].append({
                "success": classification.get("success", False),
                "time": time.time() - start
            })
            
            # التشابه
            start = time.time()
            similarity = self.calculate_similarity_url(test_image_url, test_texts)
            results["similarity"].append({
                "success": similarity.get("success", False),
                "time": time.time() - start
            })
            
            # مميزات الصورة
            start = time.time()
            img_emb = self.get_image_embeddings_url(test_image_url)
            results["image_embeddings"].append({
                "success": img_emb.get("success", False),
                "time": time.time() - start
            })
            
            # مميزات النص
            start = time.time()
            text_emb = self.get_text_embeddings(test_texts)
            results["text_embeddings"].append({
                "success": text_emb.get("success", False),
                "time": time.time() - start
            })
        
        # حساب الإحصائيات
        stats = {}
        for test_type, test_results in results.items():
            times = [r["time"] for r in test_results if r["success"]]
            success_rate = sum(1 for r in test_results if r["success"]) / len(test_results)
            
            if times:
                stats[test_type] = {
                    "success_rate": success_rate,
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_requests": len(test_results)
                }
            else:
                stats[test_type] = {
                    "success_rate": 0,
                    "avg_time": 0,
                    "min_time": 0,
                    "max_time": 0,
                    "total_requests": len(test_results)
                }
        
        return {
            "summary": stats,
            "detailed_results": results,
            "test_config": {
                "iterations": iterations,
                "test_image_url": test_image_url,
                "test_texts": test_texts
            }
        }

def main():
    """مثال على الاستخدام"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SigLIP-2 Client")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--image", help="Image path or URL")
    parser.add_argument("--labels", nargs="+", help="Classification labels")
    parser.add_argument("--texts", nargs="+", help="Texts for similarity")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    
    args = parser.parse_args()
    
    # إنشاء العميل
    client = SigLIPClient(args.server)
    
    # فحص الصحة
    print("🔍 فحص صحة الخادم...")
    health = client.health_check()
    print(f"حالة الخادم: {'صحي' if health.get('status') == 'healthy' else 'غير صحي'}")
    
    if args.benchmark:
        print("\n📊 تشغيل اختبار الأداء...")
        benchmark = client.benchmark_server()
        
        print("\n📈 نتائج اختبار الأداء:")
        for test_type, stats in benchmark["summary"].items():
            print(f"  {test_type}:")
            print(f"    معدل النجاح: {stats['success_rate']:.2%}")
            print(f"    متوسط الوقت: {stats['avg_time']:.3f}s")
            print(f"    أسرع وقت: {stats['min_time']:.3f}s")
            print(f"    أبطأ وقت: {stats['max_time']:.3f}s")
    
    if args.image and args.labels:
        print(f"\n🎯 تصنيف الصورة: {args.image}")
        
        if args.image.startswith(('http://', 'https://')):
            result = client.classify_image_url(args.image, args.labels)
        else:
            result = client.classify_image_file(args.image, args.labels)
        
        if result.get("success"):
            print("نتائج التصنيف:")
            for classification in result["data"]["classifications"]:
                print(f"  {classification['label']}: {classification['score']:.3f}")
        else:
            print(f"خطأ: {result.get('message')}")
    
    if args.image and args.texts:
        print(f"\n🔗 حساب التشابه للصورة: {args.image}")
        
        if args.image.startswith(('http://', 'https://')):
            result = client.calculate_similarity_url(args.image, args.texts)
        else:
            result = client.calculate_similarity_file(args.image, args.texts)
        
        if result.get("success"):
            print("نتائج التشابه:")
            for similarity in result["data"]["similarities"]:
                print(f"  '{similarity['text']}': {similarity['similarity_score']:.3f}")
        else:
            print(f"خطأ: {result.get('message')}")

if __name__ == "__main__":
    main()
