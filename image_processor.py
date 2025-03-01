import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import glob
from tqdm import tqdm
import logging
from PyQt5.QtCore import pyqtSignal, QObject

logger = logging.getLogger(__name__)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(329),
    transforms.CenterCrop((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageProcessor(QObject):
    progress_updated = pyqtSignal(int)  # 批量处理进度信号
    log_message = pyqtSignal(str)      # 日志信号
    batch_finished = pyqtSignal(list)  # 批量处理完成，返回结果路径列表

    def __init__(self, device):
        super().__init__()
        self.model = None
        self.device = device
        self.model_path = None
        self.output_base_dir = "./output"

    def update_output_dir(self):
        if self.model_path:
            model_dir = os.path.basename(os.path.dirname(self.model_path))
            self.output_base_dir = os.path.join("./output", model_dir)
            os.makedirs(self.output_base_dir, exist_ok=True)

    def set_model(self, model, model_path):
        self.model = model
        self.model_path = model_path
        self.update_output_dir()

    def detect_single_image(self, input_image_path):
        if not self.model:
            self.log_message.emit("请先选择模型！")
            return None
        try:
            image = Image.open(input_image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                scores, masks, _ = self.model.predict(image_tensor)
                anomaly_map = masks[0]

            anomaly_map = anomaly_map.squeeze()
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

            original_image = np.array(image.resize((288, 288)))
            heatmap = plt.cm.jet(anomaly_map)[:, :, :3]
            heatmap = (original_image * 0.5 + heatmap * 255 * 0.5).astype(np.uint8)
            combined_image = np.hstack((original_image, heatmap))

            input_filename = os.path.splitext(os.path.basename(input_image_path))[0]
            output_path = os.path.join(self.output_base_dir, f"detection_{input_filename}.png")
            plt.imsave(output_path, combined_image)
            self.log_message.emit(f"检测结果已保存到 {output_path}")
            return output_path
        except Exception as e:
            self.log_message.emit(f"检测单张图片时发生错误: {str(e)}")
            raise

    def detect_batch_images(self, input_dir):
        if not self.model:
            self.log_message.emit("请先选择模型！")
            return None
        try:
            image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
            if not image_paths:
                self.log_message.emit(f"在 {input_dir} 中未找到任何图片")
                return None

            output_paths = []
            for i, input_image_path in enumerate(tqdm(image_paths, desc="批量检测图片")):
                output_path = self.detect_single_image(input_image_path)
                if output_path:
                    output_paths.append(output_path)
                self.progress_updated.emit(int((i + 1) / len(image_paths) * 100))

            self.log_message.emit(f"批量检测结果已保存到 {self.output_base_dir}")
            self.batch_finished.emit(output_paths)
            return self.output_base_dir
        except Exception as e:
            self.log_message.emit(f"批量检测过程中发生错误: {str(e)}")
            raise