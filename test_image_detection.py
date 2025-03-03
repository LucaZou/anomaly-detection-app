import os
import sys
import time
import argparse
import logging
import torch
import psutil
import pandas as pd
from PIL import Image
from image_processor import ImageProcessor
from model_loader import load_model
import yaml
from tqdm import tqdm

# 配置日志
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "test_detection_log.txt"), mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """加载配置文件"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        sys.exit(1)

def get_memory_usage():
    """获取当前进程的内存使用量（单位：MB）"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # 转换为MB

def test_single_image(processor, image_path, threshold):
    """测试单张图像检测性能"""
    start_time = time.time()
    start_memory = get_memory_usage()

    output_path, detection_info = processor.detect_single_image(image_path, threshold)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    processing_time = end_time - start_time
    memory_used = end_memory - start_memory
    
    logger.info(f"单张图像检测: {image_path}")
    logger.info(f"检测信息: {detection_info}")
    logger.info(f"处理时间: {processing_time:.3f}秒")
    logger.info(f"内存变化: {memory_used:.2f}MB")
    
    return {
        "image_path": image_path,
        "detection_info": detection_info,
        "processing_time": processing_time,
        "memory_used": memory_used
    }

def test_batch_images(processor, input_dir, threshold):
    """测试批量图像检测性能"""
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                   if f.endswith((".jpg", ".png"))]
    if not image_paths:
        logger.error(f"在 {input_dir} 中未找到任何图片")
        return []

    results = []
    total_start_time = time.time()
    start_memory = get_memory_usage()
    
    # 模拟GUI中的批量检测逻辑，手动调用BatchDetectWorker的run方法
    from image_processor import BatchDetectWorker
    worker = BatchDetectWorker(processor, input_dir, threshold)
    worker.run()  # 直接运行以获取结果
    
    # 从处理器中获取结果（假设batch_finished信号已触发）
    output_paths = processor.batch_worker.result_paths if hasattr(worker, 'result_paths') else []
    detection_infos = processor.batch_worker.detection_infos if hasattr(worker, 'detection_infos') else []
    
    total_end_time = time.time()
    end_memory = get_memory_usage()
    
    total_time = total_end_time - total_start_time
    total_memory = end_memory - start_memory
    
    for i, (img_path, output, info) in enumerate(zip(image_paths, output_paths, detection_infos)):
        results.append({
            "image_path": img_path,
            "output_path": output,
            "detection_info": info,
            "processing_time": total_time / len(image_paths),  # 平均每张时间
            "memory_used": total_memory / len(image_paths)     # 平均每张内存
        })
    
    logger.info(f"批量检测完成: {len(image_paths)} 张图像")
    logger.info(f"总处理时间: {total_time:.3f}秒")
    logger.info(f"平均处理时间: {total_time / len(image_paths):.3f}秒/张")
    logger.info(f"总内存变化: {total_memory:.2f}MB")
    
    return results

def save_results(results, output_csv="test_results.csv"):
    """将测试结果保存到CSV文件"""
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info(f"测试结果已保存到 {output_csv}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Test image detection performance without GUI")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    # parser.add_argument("--model", required=True, help="Model name from config")
    # parser.add_argument("--input", required=True, help="Path to single image or directory")
    # parser.add_argument("--threshold", type=float, default=1.2, help="Anomaly detection threshold")
    args = parser.parse_args()

    args.model = 'Metal Nut'
    args.input = 'data/mvtec_anomaly_detection/metal_nut/test/bent'
    args.threshold = 1.2 

    # 加载配置
    config = load_config(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 初始化图像处理器
    processor = ImageProcessor(device)
    model_path = config["models"].get(args.model)
    if not model_path:
        logger.error(f"模型 {args.model} 未在配置文件中找到")
        sys.exit(1)
    
    logger.info(f"加载模型: {args.model} ({model_path})")
    processor.set_model(args.model, model_path)

    # 测试逻辑
    results = []
    if os.path.isfile(args.input):
        # 单张图像测试
        result = test_single_image(processor, args.input, args.threshold)
        results.append(result)
    elif os.path.isdir(args.input):
        # 批量图像测试
        results = test_batch_images(processor, args.input, args.threshold)
    else:
        logger.error(f"输入路径无效: {args.input}")
        sys.exit(1)

    # 保存结果
    save_results(results)

if __name__ == "__main__":
    main()