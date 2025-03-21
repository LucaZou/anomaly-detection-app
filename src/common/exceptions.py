# exceptions.py
class AnomalyDetectionError(Exception):
    """异常检测基础异常类"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class ModelLoadError(AnomalyDetectionError):
    """模型加载异常"""
    pass

class DetectionError(AnomalyDetectionError):
    """图像检测异常"""
    pass