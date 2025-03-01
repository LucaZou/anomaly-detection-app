import os
import sys
import json
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
logger = logging.getLogger(__name__)

def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def preload_models(device, config):
    models = {}
    for name, path in config["models"].items():
        try:
            models[name] = load_model(path, device)
            logger.info(f"预加载模型: {name} ({path})")
        except Exception as e:
            logger.error(f"预加载模型 {name} 失败: {str(e)}")
    return models

def main():
    config = load_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config["load_mode"] == "preload":
        model_cache = preload_models(device, config)
        processor = ImageProcessor(device, model_cache)
    else:  # "ondemand"
        processor = ImageProcessor(device)

    app = QApplication(sys.argv)
    window = MainWindow(processor, config)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()