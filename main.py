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
os.makedirs(log_dir, exist_ok=True) # 创建日志目录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "detection_log.txt"), mode="a", encoding="utf-8"),
        logging.StreamHandler() # 控制台输出
    ]
)
logger = logging.getLogger(__name__) # 获取日志记录器

def load_config():
    # 加载配置文件
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def preload_models(device, config):
    # 预加载模型
    models = {}
    for name, path in config["models"].items():
        try:
            models[name] = load_model(path, device)
            logger.info(f"预加载模型: {name} ({path})")
        except Exception as e:
            logger.error(f"预加载模型 {name} 失败: {str(e)}")
    return models

def main():
    # 加载配置文件
    config = load_config()
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 根据加载模式初始化处理器
    if config["load_mode"] == "preload":
        model_cache = preload_models(device, config) # 预加载模型
        processor = ImageProcessor(device, model_cache) # 初始化处理器并传入模型缓存
    else:  # "ondemand"
        processor = ImageProcessor(device) # 不预加载模型，需要时再加载

    # 启动GUI
    app = QApplication(sys.argv)
    window = MainWindow(processor, config)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()