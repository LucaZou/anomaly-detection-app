import os
import sys
import torch
import logging
import logging.handlers
from typing import Dict, Optional, Any  # 新增：类型提示支持
from PyQt5.QtWidgets import QApplication
from gui import MainWindow
from model_loader import load_model
from image_processor import ImageProcessor
import yaml
from progress_dialog import ProgressDialog
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
log_dir: str = "./logs"  # 日志存储目录
os.makedirs(log_dir, exist_ok=True)

def setup_logging() -> logging.Logger:
    """
    配置全局日志系统,包括控制台和文件输出。
    
    Returns:
        logging.Logger: 配置好的主日志器对象
    """
    # 主日志器
    logger: logging.Logger = logging.getLogger('AnomalyDetection')
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG以捕获所有信息

    # 清空已有处理器,避免重复添加
    if logger.handlers:
        logger.handlers.clear()

    # 控制台处理器
    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 文件处理器(按大小轮转)
    file_handler: logging.handlers.RotatingFileHandler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "detection_log.txt"),
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=10,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)

    # 日志格式
    formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 为子模块设置日志器并绑定处理器
    for module in ['ModelLoader', 'ImageProcessor', 'GUI']:
        sub_logger: logging.Logger = logging.getLogger(module)
        sub_logger.setLevel(logging.DEBUG)
        if not sub_logger.handlers:
            sub_logger.addHandler(console_handler)
            sub_logger.addHandler(file_handler)

    return logger

logger: logging.Logger = setup_logging()

def load_config() -> Dict[str, Any]:
    """
    加载配置文件 config.yaml。

    Returns:
        Dict[str, Any]: 解析后的配置字典

    Raises:
        Exception: 如果文件读取或解析失败
    """
    logger.debug("开始加载配置文件 config.yaml")
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config: Dict[str, Any] = yaml.safe_load(f)
        logger.info("配置文件加载成功")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}", exc_info=True)
        raise

def preload_models(device: torch.device, config: Dict[str, Any], progress_dialog: Optional[ProgressDialog] = None) -> Dict[str, Any]:
    """
    预加载所有模型并缓存。

    Args:
        device (torch.device): 运行设备(CPU或GPU)
        config (Dict[str, Any]): 包含模型配置的字典
        progress_dialog (Optional[ProgressDialog]): 进度对话框,用于显示加载进度

    Returns:
        Dict[str, Any]: 模型名称到模型对象的映射
    """
    logger.debug("开始预加载模型")
    models: Dict[str, Any] = {}
    model_configs: Dict[str, str] = config["models"]
    total_tasks: int = len(model_configs)
    model_logger: logging.Logger = logging.getLogger('ModelLoader')

    def load_single_model(name: str, path: str) -> tuple[str, Any]:
        """
        加载单个模型。

        Args:
            name (str): 模型名称
            path (str): 模型文件路径

        Returns:
            tuple[str, Any]: (模型名称, 模型对象或异常)
        """
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
        completed_tasks: int = 0
        for future in as_completed(future_to_model):
            name: str = future_to_model[future]
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
    logger.info(f"模型预加载完成,共加载 {len(models)} 个模型")
    return models

def main() -> None:
    """
    程序主入口,初始化设备、GUI并启动应用。

    Returns:
        None
    """
    logger.debug("程序启动")
    config: Dict[str, Any] = load_config()
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    app: QApplication = QApplication(sys.argv)
    logger.debug("QApplication 初始化完成")

    if config["load_mode"] == "preload":
        logger.debug("进入预加载模式")
        progress_dialog: ProgressDialog = ProgressDialog(
            total_tasks=len(config["models"]),
            description="Preloading models..."
        )
        progress_dialog.show()
        model_cache: Dict[str, Any] = preload_models(device, config, progress_dialog)
        processor: ImageProcessor = ImageProcessor(device, model_cache, config)
        progress_dialog.accept()
    else:
        logger.debug("进入按需加载模式")
        processor: ImageProcessor = ImageProcessor(device, config)

    logger.debug("开始初始化 MainWindow")
    window: MainWindow = MainWindow(processor, config)
    logger.debug("MainWindow 初始化完成")
    window.show()
    logger.info("窗口显示")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()