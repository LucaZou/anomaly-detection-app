import os
import sys
import torch
import logging
from PyQt5.QtWidgets import QApplication
from gui import MainWindow
from model_loader import load_model
from image_processor import ImageProcessor

# 配置日志
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "detection_log.txt"), mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def main():
    # 初始化设备和模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "models/mvtec_metal_nut/ckpt.pth"
    model = load_model(model_path, device)
    
    # 初始化图像处理器
    processor = ImageProcessor(model, device)

    # 启动 GUI
    app = QApplication(sys.argv)
    window = MainWindow(processor)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()