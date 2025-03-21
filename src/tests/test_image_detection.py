import argparse
import os
import time
import torch
import logging
import logging.handlers
from typing import Dict, Optional, Any
import yaml
from src.image_processing.image_processor import ImageProcessor
from src.model_loading.model_loader import load_model
import glob

# 配置日志
log_dir: str = "../logs"
os.makedirs(log_dir, exist_ok=True)

def setup_logging() -> logging.Logger:
    """
    配置日志系统，输出到控制台和文件。

    Returns:
        logging.Logger: 配置好的日志器
    """
    logger: logging.Logger = logging.getLogger('BatchTest')
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler: logging.handlers.RotatingFileHandler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "batch_test_log.txt"),
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=10,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)

    formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger: logging.Logger = setup_logging()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件。

    Args:
        config_path (str): 配置文件路径

    Returns:
        Dict[str, Any]: 配置字典

    Raises:
        FileNotFoundError: 如果配置文件不存在
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    logger.info(f"配置文件 {config_path} 加载成功")
    return config

def test_batch_detection(input_dir: str, model_path: str, config_path: str, model_name: str = "SimpleNet") -> None:
    """
    测试图像批量检测性能。

    Args:
        input_dir (str): 输入图像目录
        model_path (str): 模型文件路径
        config_path (str): 配置文件路径
        model_name (str): 模型名称，默认为 "SimpleNet"
    """
    # 初始化设备
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载配置
    config: Dict[str, Any] = load_config(config_path)

    # 初始化图像处理器
    logger.info("初始化 ImageProcessor")
    processor: ImageProcessor = ImageProcessor(device, config=config)

    # 加载模型
    logger.info(f"加载模型: {model_name} 从 {model_path}")
    processor.set_model(model_name, model_path)

    # 检查输入目录
    if not os.path.isdir(input_dir):
        logger.error(f"输入目录 {input_dir} 不存在或不是目录")
        raise ValueError(f"输入目录 {input_dir} 无效")
    image_count: int = len(glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png")))
    if image_count == 0:
        logger.error(f"输入目录 {input_dir} 中未找到任何图像")
        raise ValueError(f"输入目录 {input_dir} 中无图像")
    logger.info(f"输入目录 {input_dir} 包含 {image_count} 张图像")

    # 开始性能测试
    logger.info("开始批量检测性能测试")
    start_time: float = time.time()

    # 记录初始显存
    if torch.cuda.is_available():
        initial_memory: float = torch.cuda.memory_allocated(device) / 1024**3  # GiB
        logger.info(f"初始显存使用: {initial_memory:.2f} GiB")

    # 执行批量检测
    processor.detect_batch_images(input_dir, config.get("threshold", 1.20))

    # 等待批量检测完成（由于是异步线程，需手动等待）
    while processor.batch_worker and processor.batch_worker.isRunning():
        time.sleep(0.1)

    # 计算耗时
    end_time: float = time.time()
    total_time: float = end_time - start_time
    avg_time_per_image: float = total_time / image_count if image_count > 0 else 0

    # 记录结束显存
    if torch.cuda.is_available():
        final_memory: float = torch.cuda.memory_allocated(device) / 1024**3  # GiB
        memory_used: float = final_memory - initial_memory
        logger.info(f"结束显存使用: {final_memory:.2f} GiB, 检测过程中显存增加: {memory_used:.2f} GiB")

    # 输出性能结果
    logger.info(f"批量检测总耗时: {total_time:.2f} 秒")
    logger.info(f"每张图像平均耗时: {avg_time_per_image:.4f} 秒")
    logger.info(f"性能测试完成，结果保存在 {processor.output_base_dir}")

def main() -> None:
    """主函数，解析命令行参数并运行测试"""
    parser = argparse.ArgumentParser(description="测试图像批量检测性能")
    # parser.add_argument("--input-dir", type=str, required=True, help="输入图像目录")
    # parser.add_argument("--model-path", type=str, required=True, help="模型文件路径")
    parser.add_argument("--config-path", type=str, default="config.yaml", help="配置文件路径，默认为 config.yaml")
    parser.add_argument("--model-name", type=str, default="SimpleNet", help="模型名称，默认为 SimpleNet")
    args = parser.parse_args()
    args.input_dir = "data/mvtec_anomaly_detection/metal_nut/test/bent"
    args.model_path = "../models/mvtec_metal_nut/ckpt.pth"

    try:
        test_batch_detection(args.input_dir, args.model_path, args.config_path, args.model_name)
    except Exception as e:
        logger.error(f"性能测试失败: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()