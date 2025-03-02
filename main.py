import os
import sys
import json
import torch
import logging
from PyQt5.QtWidgets import QApplication
from gui import MainWindow
from model_loader import load_model
from image_processor import ImageProcessor
import yaml
from progress_dialog import ProgressDialog, ProgressWorker  # 新增：导入进度对话框

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
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)  # 使用 yaml.safe_load 解析 YAML 文件
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        raise

# def preload_models(device, config):
#     # 预加载模型
#     models = {}
#     for name, path in config["models"].items():
#         try:
#             models[name] = load_model(path, device)
#             logger.info(f"预加载模型: {name} ({path})")
#         except Exception as e:
#             logger.error(f"预加载模型 {name} 失败: {str(e)}")
#     return models

def preload_models(device, config, progress_dialog=None):
    """预加载所有模型，支持进度更新"""
    models = {}
    model_configs = config["models"]
    total_tasks = len(model_configs)

    for i, (name, path) in enumerate(model_configs.items()):
        try:
            if progress_dialog:
                progress_dialog.set_description(f"Loading model: {name} ({i+1}/{total_tasks})")
            models[name] = load_model(path, device)
            logger.info(f"预加载模型: {name} ({path})")
            if progress_dialog:
                progress_dialog.update_progress(1)  # 每次加载完成更新进度
        except Exception as e:
            logger.error(f"预加载模型 {name} 失败: {str(e)}")
            # 如果有进度对话框，继续加载其他模型；否则抛出异常
            if not progress_dialog:
                raise
    return models

def main():
    # 加载配置文件
    config = load_config()
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 启动GUI
    app = QApplication(sys.argv)

    # 根据加载模式初始化处理器
    if config["load_mode"] == "preload":
        # 新增：在预加载模式下显示进度对话框
        progress_dialog = ProgressDialog(
            total_tasks=len(config["models"]),
            description="Preloading models..."
        )
        progress_dialog.show()
        model_cache = preload_models(device, config,progress_dialog) # 预加载模型
        processor = ImageProcessor(device, model_cache) # 初始化处理器并传入模型缓存
        progress_dialog.accept()  # 加载完成后关闭对话框
    else:  # "ondemand"
        processor = ImageProcessor(device) # 不预加载模型，需要时再加载

    
    window = MainWindow(processor, config)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()