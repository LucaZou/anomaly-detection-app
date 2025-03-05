import os
import sys
import torch
import logging
import logging.handlers
from PyQt5.QtWidgets import QApplication
from gui import MainWindow
from model_loader import load_model
from image_processor import ImageProcessor
import yaml
from progress_dialog import ProgressDialog
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

def setup_logging():
    # 主日志器
    logger = logging.getLogger('AnomalyDetection')
    logger.setLevel(logging.DEBUG)  # 修改：降低日志级别为DEBUG，捕获更多信息

    # 清空已有处理器，避免重复添加
    if logger.handlers:
        logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # 修改：确保控制台输出所有日志

    # 文件处理器（按大小轮转）
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "detection_log.txt"),
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=10,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)  # 修改：确保文件记录所有日志

    # 日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 为子模块设置日志器并绑定处理器
    for module in ['ModelLoader', 'ImageProcessor', 'GUI']:
        sub_logger = logging.getLogger(module)
        sub_logger.setLevel(logging.DEBUG)
        if not sub_logger.handlers:  # 避免重复添加处理器
            sub_logger.addHandler(console_handler)
            sub_logger.addHandler(file_handler)

    return logger

logger = setup_logging()

def load_config():
    logger.debug("开始加载配置文件 config.yaml")  # 新增：调试日志
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("配置文件加载成功")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}", exc_info=True)
        raise

def preload_models(device, config, progress_dialog=None):
    logger.debug("开始预加载模型")  # 新增：调试日志
    models = {}
    model_configs = config["models"]
    total_tasks = len(model_configs)
    model_logger = logging.getLogger('ModelLoader')

    def load_single_model(name, path):
        try:
            model = load_model(path, device)
            model_logger.info(f"预加载模型: {name} ({path})")
            return name, model
        except Exception as e:
            model_logger.error(f"预加载模型 {name} 失败: {str(e)}", exc_info=True)
            return name, e

    with ThreadPoolExecutor(max_workers=min(4, total_tasks)) as executor:
        future_to_model = {executor.submit(load_single_model, name, path): name 
                          for name, path in model_configs.items()}
        completed_tasks = 0
        for future in as_completed(future_to_model):
            name = future_to_model[future]
            try:
                result_name, result = future.result()
                if not isinstance(result, Exception):
                    models[result_name] = result
                completed_tasks += 1
                if progress_dialog:
                    progress_dialog.set_description(f"Loading model: {name} ({completed_tasks}/{total_tasks})")
                    progress_dialog.update_progress(1)
            except Exception as e:
                model_logger.error(f"加载模型 {name} 时线程异常: {str(e)}", exc_info=True)
    logger.info(f"模型预加载完成，共加载 {len(models)} 个模型")
    return models

def main():
    logger.debug("程序启动")  # 新增：调试日志
    config = load_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    app = QApplication(sys.argv)
    logger.debug("QApplication 初始化完成")  # 新增：调试日志

    if config["load_mode"] == "preload":
        logger.debug("进入预加载模式")  # 新增：调试日志
        progress_dialog = ProgressDialog(
            total_tasks=len(config["models"]),
            description="Preloading models..."
        )
        progress_dialog.show()
        model_cache = preload_models(device, config, progress_dialog)
        processor = ImageProcessor(device, model_cache, config)
        progress_dialog.accept()
    else:
        logger.debug("进入按需加载模式")  # 新增：调试日志
        processor = ImageProcessor(device, config)

    logger.debug("开始初始化 MainWindow")  # 新增：调试日志
    window = MainWindow(processor, config)
    logger.debug("MainWindow 初始化完成")  # 新增：调试日志
    window.show()
    logger.info("窗口显示")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()