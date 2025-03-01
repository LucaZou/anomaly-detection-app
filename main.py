import os
import sys
import torch
import logging
from PyQt5.QtWidgets import QApplication
from gui import MainWindow
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = ImageProcessor(device)  # 不加载默认模型

    app = QApplication(sys.argv)
    window = MainWindow(processor)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()