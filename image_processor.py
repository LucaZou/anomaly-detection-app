import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import glob
from tqdm import tqdm
import logging
from PyQt5.QtCore import pyqtSignal, QObject, QThread

logger = logging.getLogger(__name__)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(329), # 保持长宽比缩放到 329
    transforms.CenterCrop((288, 288)), # 中心裁剪到 288x288
    transforms.ToTensor(), # 转为 Tensor张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 标准化
])

# 定义批量检测的工作线程
class BatchDetectWorker(QThread):
    progress_updated = pyqtSignal(int)  # 进度信号
    log_message = pyqtSignal(str)      # 日志信号
    batch_finished = pyqtSignal(list)  # 完成信号

    def __init__(self, processor, input_dir):
        super().__init__()
        self.processor = processor
        self.input_dir = input_dir

    def run(self):
        # 在线程中执行批量检测
        image_paths = glob.glob(os.path.join(self.input_dir, "*.jpg")) + glob.glob(os.path.join(self.input_dir, "*.png"))
        if not image_paths:
            self.log_message.emit(f"在 {self.input_dir} 中未找到任何图片")
            return

        output_paths = []
        detection_infos = []  # 新增：存储每张图片的检测信息
        for i, input_image_path in enumerate(tqdm(image_paths, desc="批量检测图片")):
            output_path, info = self.processor.detect_single_image(input_image_path)  # 修改：接收返回的提示信息
            # output_path = self.processor.detect_single_image(input_image_path)
            if output_path:
                output_paths.append(output_path)
                detection_infos.append(info)  # 新增：存储检测信息
            self.progress_updated.emit(int((i + 1) / len(image_paths) * 100))

        self.log_message.emit(f"批量检测结果已保存到 {self.processor.output_base_dir}")
        # self.batch_finished.emit(output_paths)
        self.batch_finished.emit([output_paths, detection_infos])  # 修改：传递路径和信息列表

class ImageProcessor(QObject):
    # 图像处理器类，用于处理图像检测任务
    progress_updated = pyqtSignal(int) # 进度更新信号
    log_message = pyqtSignal(str) # 日志消息信号
    batch_finished = pyqtSignal(list) # 批量处理完成信号

    def __init__(self, device, models=None):
        # 初始化处理器
        super().__init__()
        self.device = device # 设备（CPU/GPU）
        self.model_cache = models if models else {}  # 缓存预加载的模型
        self.current_model_name = None # 当前模型名称
        self.model_path = None # 当前模型路径
        self.output_base_dir = "./output" # 输出目录
        self.batch_worker = None # 存储工作线程对象

    def update_output_dir(self):
        # 根据模型路径更新输出目录
        if self.model_path:
            model_dir = os.path.basename(os.path.dirname(self.model_path)) # 模型目录名
            self.output_base_dir = os.path.join("./output", model_dir)
            os.makedirs(self.output_base_dir, exist_ok=True) # 创建输出目录

    def set_model(self, model_name, model_path=None):
        # 设置当前模型，支持预加载和按需加载
        if model_name in self.model_cache: # 模型已预加载
            self.model = self.model_cache[model_name] # 直接从缓存中获取
            self.model_path = model_path or list(self.model_cache.keys())[list(self.model_cache.values()).index(self.model)]
        elif model_path: # 模型未预加载但提供了路径
            from model_loader import load_model
            self.model = load_model(model_path, self.device)
            self.model_cache[model_name] = self.model
            self.model_path = model_path
        else: # 模型未找到且无路径提供
            self.log_message.emit(f"模型 {model_name} 未找到且无路径提供！")
            return
        self.current_model_name = model_name
        self.update_output_dir()

    def detect_single_image(self, input_image_path):
        # 检测单张图片
        if not hasattr(self, 'model') or self.model is None:
            self.log_message.emit("请先选择模型！")
            return None
        try:
            image = Image.open(input_image_path).convert("RGB") # 读取图片并转为 RGB 模式
            image_tensor = transform(image).unsqueeze(0).to(self.device) # 图像预处理并移动到设备

            with torch.no_grad():
                scores, masks, _ = self.model.predict(image_tensor) # 模型推理
                # 日志输出scores
                # self.log_message.emit(f"检测结果: {scores[0].item()}") # 输出第一个元素的值
                anomaly_map = masks[0] # 获取异常图

            # scores 是单值列表，直接取第一个元素
            score = float(scores[0]) if not torch.is_tensor(scores[0]) else scores[0].item()

            # 处理 scores 列表的情况(处理多种类型的模型)
            # if isinstance(scores, list):  # 检查 scores 是否为列表
            #     if len(scores) == 1:  # 如果列表只有一项，取第一个值
            #         score = float(scores[0]) if not torch.is_tensor(scores[0]) else scores[0].item()
            #     else:  # 如果列表有多项，取最大值（假设最大值代表最可能的异常）
            #         score = max([float(s) if not torch.is_tensor(s) else s.item() for s in scores])
            # elif torch.is_tensor(scores):  # 如果是张量，取标量值
            #     score = scores.item()
            # else:  # 其他情况，直接尝试转换为浮点数
            #     score = float(scores)
            
            # 生成检测信息
            threshold = 1.3  # 可调整的阈值
            detection_info = f"异常得分: {score:.2f} - "
            if score > threshold:
                detection_info += "检测到异常"
            else:
                detection_info += "图像正常"
            
            # 归一化异常图
            anomaly_map = anomaly_map.squeeze()
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

            # 生成热力图
            original_image = np.array(image.resize((288, 288)))
            heatmap = plt.cm.jet(anomaly_map)[:, :, :3] # 使用 jet 颜色映射
            heatmap = (original_image * 0.5 + heatmap * 255 * 0.5).astype(np.uint8) # 叠加到原图上
            combined_image = np.hstack((original_image, heatmap)) # 水平拼接

            # 保存结果图像
            input_filename = os.path.splitext(os.path.basename(input_image_path))[0]
            output_path = os.path.join(self.output_base_dir, f"detection_{input_filename}.png")
            plt.imsave(output_path, combined_image)
            self.log_message.emit(f"检测结果已保存到 {output_path}")
            return output_path, detection_info  # 修改：返回输出路径和提示信息
        except Exception as e:
            error_msg = f"检测单张图片时发生错误: {str(e)}"
            self.log_message.emit(error_msg)
            return None, error_msg
            

    def detect_batch_images(self, input_dir):
        # 批量检测图片
        if not hasattr(self, 'model') or self.model is None:
            self.log_message.emit("请先选择模型！")
            return None
        # 创建并启动工作线程
        self.batch_worker = BatchDetectWorker(self, input_dir)
        self.batch_worker.progress_updated.connect(self.progress_updated.emit)
        self.batch_worker.log_message.connect(self.log_message.emit)
        self.batch_worker.batch_finished.connect(self.batch_finished.emit)
        self.batch_worker.start()
        return None  # 返回 None，因为结果通过信号异步传递
        # try:
        #     # 获取目录下所有图片路径
        #     image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
        #     if not image_paths:
        #         self.log_message.emit(f"在 {input_dir} 中未找到任何图片")
        #         return None
            
        #     output_paths = []
        #     for i, input_image_path in enumerate(tqdm(image_paths, desc="批量检测图片")):
        #         output_path = self.detect_single_image(input_image_path)
        #         if output_path:
        #             output_paths.append(output_path)
        #         self.progress_updated.emit(int((i + 1) / len(image_paths) * 100))

        #     self.log_message.emit(f"批量检测结果已保存到 {self.output_base_dir}")
        #     self.batch_finished.emit(output_paths)
        #     return self.output_base_dir
        # except Exception as e:
        #     self.log_message.emit(f"批量检测过程中发生错误: {str(e)}")
        #     raise