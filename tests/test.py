# encoding:utf-8
import argparse
import os
import sys
import torch
import yaml
import logging
from src.image_processing.image_processor import ImageProcessor
from src.common.exceptions import DetectionError

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('../logs/test_batch_detection.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    """加载配置文件并返回配置字典"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"配置文件 {config_path} 未找到，使用默认配置")
        return {}
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}", exc_info=True)
        return {}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批量图像异常检测脚本")
    parser.add_argument('--input-dir', type=str, required=True,
                        help="输入图像目录路径")
    parser.add_argument('--model-path', type=str, required=True,
                        help="模型权重文件路径")
    parser.add_argument('--threshold', type=float, default=1.20,
                        help="异常检测阈值（默认: 1.20）")
    parser.add_argument('--device', type=str, default=None,
                        help="运行设备（例如: cuda:0 或 cpu），默认自动选择")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help="配置文件路径（默认: config.yaml）")
    return parser.parse_args()

def main():
    """主函数，执行批量检测"""
    # 解析参数和配置
    args = parse_args()
    config = load_config(args.config)

    # 确定设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 初始化图像处理器
    processor = ImageProcessor(device=device, config=config)

    # 设置模型
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    try:
        processor.set_model(model_name, args.model_path)
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}", exc_info=True)
        sys.exit(1)

    # 检查输入目录
    if not os.path.isdir(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        sys.exit(1)

    # 启动批量检测
    logger.info(f"开始批量检测，输入目录: {args.input_dir}, 阈值: {args.threshold}")
    try:
        processor.detect_batch_images(args.input_dir, args.threshold)

        # 等待批量检测完成（由于是异步线程，需手动模拟等待或连接信号）
        if processor.batch_worker:
            processor.batch_worker.wait()  # 等待线程完成
            logger.info("批量检测完成")
    except DetectionError as e:
        logger.error(f"批量检测失败: {str(e)}", exc_info=True)
        sys.exit(1)

    # 输出结果路径
    output_dir = processor.output_base_dir
    logger.info(f"检测结果已保存至: {output_dir}")
    report_dir = os.path.join(output_dir, "reports")
    if os.path.exists(report_dir):
        logger.info(f"检测报告已保存至: {report_dir}")

if __name__ == "__main__":
    main()