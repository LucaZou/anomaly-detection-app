import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import glob
from tqdm import tqdm
import logging
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import tempfile
import time
from typing import List, Optional, Dict, Any, Tuple  # 新增：类型提示支持
from exceptions import DetectionError
from performance_monitor import PerformanceMonitor

# 配置模块日志器
logger: logging.Logger = logging.getLogger('ImageProcessor')

# 图像预处理变换
transform: transforms.Compose = transforms.Compose([
    transforms.Resize(329),  # 调整图像大小到329x329
    transforms.CenterCrop((288, 288)),  # 中心裁剪到288x288
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])

class ImagePreloader:
    """图像预加载器,支持磁盘缓存或内存模式以优化批量检测性能"""

    def __init__(self, max_memory_mb: int = 2048, use_disk_cache: bool = True, device: Optional[torch.device] = None):
        """
        初始化图像预加载器。

        Args:
            max_memory_mb (int): 最大内存使用量（MB）,默认为2048
            use_disk_cache (bool): 是否使用磁盘缓存,默认为True
            device (Optional[torch.device]): 运行设备,默认为None
        """
        self.max_memory_mb: int = max_memory_mb  # 内存限制
        self.use_disk_cache: bool = use_disk_cache  # 缓存模式标志
        self.device: Optional[torch.device] = device  # 设备引用
        self.perf_monitor: PerformanceMonitor = PerformanceMonitor(device)  # 性能监控实例
        if self.use_disk_cache:
            self.temp_dir: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()  # 磁盘缓存临时目录
            logger.info(f"使用磁盘缓存,临时目录: {self.temp_dir.name}")
        else:
            self.memory_chunks: List[Dict[str, Any]] = []  # 内存缓存分片列表
            logger.info("使用内存缓存")

    def _load_image(self, path: str) -> Tuple[Optional[torch.Tensor], str]:
        """
        加载并预处理单张图像。

        Args:
            path (str): 图像文件路径

        Returns:
            Tuple[Optional[torch.Tensor], str]: (图像张量或None, 文件路径)
        """
        try:
            img: Image.Image = Image.open(path).convert("RGB")  # 打开并转换为RGB格式
            tensor: torch.Tensor = transform(img)  # 应用预处理变换
            return tensor, path
        except Exception as e:
            logger.error(f"预加载图片失败 {path}: {str(e)}", exc_info=True)
            return None, path

    def preload(self, image_paths: List[str], chunk_size: int = 100) -> List[Any]:
        """
        预加载图像并分片存储。

        Args:
            image_paths (List[str]): 图像路径列表
            chunk_size (int): 每个分片的最大图像数量,默认为100

        Returns:
            List[Any]: 分片引用列表（磁盘模式为文件路径,内存模式为索引）
        """
        max_workers: int = min(multiprocessing.cpu_count(), 8)  # 最大线程数
        logger.info(f"开始预加载 {len(image_paths)} 张图片,使用 {max_workers} 个线程,"
                    f"缓存模式: {'磁盘' if self.use_disk_cache else '内存'}")
        self.perf_monitor.start_timer()
        preloaded_chunks: List[Any] = []
        total_images: int = len(image_paths)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_index: int = 0  # 内存模式下的分片索引
            for start_idx in range(0, total_images, chunk_size):
                end_idx: int = min(start_idx + chunk_size, total_images)
                chunk_paths: List[str] = image_paths[start_idx:end_idx]
                logger.debug(f"预加载分片 {start_idx}-{end_idx},包含 {len(chunk_paths)} 张图片")
                self.perf_monitor.start_wait_timer()
                future_to_path = {executor.submit(self._load_image, path): path for path in chunk_paths}
                chunk_tensors: List[torch.Tensor] = []
                chunk_paths_valid: List[str] = []

                for future in as_completed(future_to_path):
                    tensor, path = future.result()
                    if tensor is not None:
                        chunk_tensors.append(tensor)
                        chunk_paths_valid.append(path)

                self.perf_monitor.log_wait_time(f"Preload Chunk {start_idx}-{end_idx} Thread Wait")
                if chunk_tensors:
                    if self.use_disk_cache:
                        self.perf_monitor.start_timer()
                        chunk_array: np.ndarray = torch.stack(chunk_tensors).cpu().numpy()
                        chunk_file: str = os.path.join(self.temp_dir.name, f"chunk_{start_idx}.npz")
                        np.savez_compressed(chunk_file, images=chunk_array, paths=chunk_paths_valid)
                        file_size: float = os.path.getsize(chunk_file) / 1024**2  # MB
                        self.perf_monitor.log_io_stats(f"Save Chunk {chunk_file}", file_size)
                        preloaded_chunks.append(chunk_file)
                        logger.info(f"保存磁盘分片 {chunk_file},包含 {len(chunk_tensors)} 张图片,"
                                    f"大小: {file_size:.2f} MB")
                    else:
                        self.memory_chunks.append({
                            'images': torch.stack(chunk_tensors),
                            'paths': chunk_paths_valid
                        })
                        preloaded_chunks.append(chunk_index)
                        logger.info(f"内存存储分片 {chunk_index},包含 {len(chunk_tensors)} 张图片")
                        chunk_index += 1

        self.perf_monitor.log_time("Image Preloading Total")
        return preloaded_chunks

    def cleanup(self) -> None:
        """清理缓存资源（磁盘文件或内存分片）"""
        if self.use_disk_cache:
            for attempt in range(3):
                try:
                    self.temp_dir.cleanup()
                    logger.info("成功清理磁盘缓存临时目录")
                    break
                except PermissionError as e:
                    logger.warning(f"清理磁盘缓存失败 (尝试 {attempt+1}/3): {str(e)}")
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"清理磁盘缓存时发生未知错误: {str(e)}", exc_info=True)
                    break
            else:
                logger.error("多次尝试后仍无法清理磁盘缓存,可能需要手动删除")
        else:
            self.memory_chunks.clear()
            logger.info("成功清理内存缓存")

class BatchDetectWorker(QThread):
    """批量检测工作线程,异步处理图像检测任务"""
    progress_updated = pyqtSignal(int)  # 进度更新信号
    log_message = pyqtSignal(str)  # 日志消息信号
    batch_finished = pyqtSignal(list)  # 批量检测完成信号

    def __init__(self, processor: 'ImageProcessor', input_dir: str, threshold: float):
        """
        初始化批量检测工作线程。

        Args:
            processor (ImageProcessor): 图像处理器实例
            input_dir (str): 输入图像目录
            threshold (float): 异常检测阈值
        """
        super().__init__()
        self.processor: 'ImageProcessor' = processor
        self.input_dir: str = input_dir
        self.threshold: float = threshold
        self.device: torch.device = processor.device
        self.config: Dict[str, Any] = processor.config
        self.batch_size: int = self._estimate_batch_size()
        self.preloader: ImagePreloader = ImagePreloader(use_disk_cache=self.config.get('use_disk_cache', True), device=self.device)
        self.perf_monitor: PerformanceMonitor = PerformanceMonitor(self.device)

    def _estimate_batch_size(self) -> int:
        """
        根据可用显存估计批量大小。

        Returns:
            int: 计算出的批量大小
        """
        total_memory: float = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GiB
        reserved_memory: float = torch.cuda.memory_reserved(self.device) / 1024**3  # GiB
        available_memory: float = total_memory - reserved_memory
        image_memory: float = 0.236  # 单张图像显存占用（GiB）
        safe_memory: float = available_memory - 0.5  # 保留安全余量
        max_batch_size: int = self.config.get('max_batch_size', 32)
        batch_size: int = max(1, min(max_batch_size, int(safe_memory / image_memory)))
        logger.info(f"动态调整 batch_size 为 {batch_size},根据可用内存 {available_memory:.2f} GiB")
        return batch_size

    def run(self) -> None:
        """执行批量检测任务"""
        image_paths: List[str] = glob.glob(os.path.join(self.input_dir, "*.jpg")) + glob.glob(os.path.join(self.input_dir, "*.png"))
        if not image_paths:
            self.log_message.emit(f"ERROR:在 {self.input_dir} 中未找到任何图片")
            logger.error(f"批量检测输入目录 {self.input_dir} 为空")
            return

        self.perf_monitor.log_memory("Batch Detection Start")
        self.perf_monitor.start_timer()

        chunk_size: int = self.config.get('preload_chunk_size', 100)
        preloaded_chunks: List[Any] = self.preloader.preload(image_paths, chunk_size)
        self.perf_monitor.log_time("Image Preloading")

        output_paths: List[str] = []
        detection_infos: List[str] = []
        total_images: int = sum(len(np.load(chunk_file, allow_pickle=True)['paths']) if self.preloader.use_disk_cache 
                               else len(self.preloader.memory_chunks[idx]['paths']) 
                               for idx, chunk_file in enumerate(preloaded_chunks))

        with ThreadPoolExecutor(max_workers=12) as executor:
            processed_images: int = 0
            for chunk_idx, chunk_ref in enumerate(tqdm(preloaded_chunks, desc="处理图像分片")):
                if self.preloader.use_disk_cache:
                    self.perf_monitor.start_timer()
                    with np.load(chunk_ref, allow_pickle=True) as chunk_data:
                        chunk_tensors: torch.Tensor = torch.from_numpy(chunk_data['images']).to(self.device)
                        chunk_paths: List[str] = chunk_data['paths']
                    file_size: float = os.path.getsize(chunk_ref) / 1024**2  # MB
                    self.perf_monitor.log_io_stats(f"Load Chunk {chunk_ref}", file_size)
                else:
                    if chunk_ref >= len(self.preloader.memory_chunks):
                        self.log_message.emit(f"ERROR: 内存分片索引 {chunk_ref} 超出范围,最大索引为 {len(self.preloader.memory_chunks)-1}")
                        logger.error(f"内存分片索引错误: {chunk_ref}")
                        break
                    chunk_data: Dict[str, Any] = self.preloader.memory_chunks[chunk_ref]
                    chunk_tensors: torch.Tensor = chunk_data['images'].to(self.device)
                    chunk_paths: List[str] = chunk_data['paths']

                for start_idx in range(0, len(chunk_tensors), self.batch_size):
                    end_idx: int = min(start_idx + self.batch_size, len(chunk_tensors))
                    batch_tensors: torch.Tensor = chunk_tensors[start_idx:end_idx]
                    batch_paths: List[str] = chunk_paths[start_idx:end_idx]

                    try:
                        self.perf_monitor.start_timer()
                        with torch.no_grad():
                            scores_list, masks_list, _ = self.processor.model.predict(batch_tensors)
                        self.perf_monitor.log_time(f"Batch Inference (Chunk {chunk_idx}, Batch {start_idx//self.batch_size})")

                        futures: List[Any] = []
                        self.perf_monitor.start_wait_timer()
                        for i, (scores, mask, image_path) in enumerate(zip(scores_list, masks_list, batch_paths)):
                            score: float = float(scores[0]) if isinstance(scores, list) else scores.item()
                            detection_info: str = f"异常得分: {score:.2f} - {'检测到异常' if score > self.threshold else '图像正常'}"
                            image: Image.Image = Image.open(image_path).convert("RGB")
                            futures.append(executor.submit(self.processor._generate_heatmap, image, mask, image_path))
                            detection_infos.append(detection_info)

                        for i, future in enumerate(as_completed(futures)):
                            output_path: Optional[str] = future.result()
                            if output_path:
                                output_paths.append(output_path)
                            processed_images += 1
                            progress: int = int((processed_images / total_images) * 100)
                            self.progress_updated.emit(progress)

                        self.perf_monitor.log_wait_time(f"Heatmap Generation Wait (Chunk {chunk_idx}, Batch {start_idx//self.batch_size})")
                        torch.cuda.empty_cache()
                    except Exception as e:
                        error_msg: str = f"ERROR: 批量检测错误: {str(e)}"
                        self.log_message.emit(error_msg)
                        logger.error(f"批量检测异常: {str(e)},输入路径: {self.input_dir}", exc_info=True)
                        break

                if self.preloader.use_disk_cache:
                    try:
                        os.remove(chunk_ref)
                        logger.info(f"成功删除分片文件 {chunk_ref}")
                    except PermissionError as e:
                        logger.warning(f"无法删除分片文件 {chunk_ref}: {str(e)}")

        self.perf_monitor.log_memory("Batch Detection End")
        self.log_message.emit(f"批量检测结果已保存到 {self.processor.output_base_dir}")
        self.batch_finished.emit([output_paths, detection_infos])
        self.preloader.cleanup()

class ImageProcessor(QObject):
    """图像处理器,负责模型管理和异常检测逻辑"""
    progress_updated = pyqtSignal(int)  # 进度更新信号
    log_message = pyqtSignal(str)  # 日志消息信号
    batch_finished = pyqtSignal(list)  # 批量检测完成信号

    def __init__(self, device: torch.device, models: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None):
        """
        初始化图像处理器。

        Args:
            device (torch.device): 运行设备（CPU或GPU）
            models (Optional[Dict[str, Any]]): 预加载的模型缓存,默认为None
            config (Optional[Dict[str, Any]]): 配置字典,默认为None
        """
        super().__init__()
        self.device: torch.device = device
        self.model_cache: Dict[str, Any] = models if models else {}
        self.current_model_name: Optional[str] = None
        self.model_path: Optional[str] = None
        self.output_base_dir: str = "./output"  # 输出结果基础目录
        self.batch_worker: Optional[BatchDetectWorker] = None
        self.config: Dict[str, Any] = config or {}
        self.perf_monitor: PerformanceMonitor = PerformanceMonitor(device)
        logger.debug("ImageProcessor 初始化完成")

    def update_output_dir(self) -> None:
        """根据模型路径更新输出目录"""
        if self.model_path:
            model_dir: str = os.path.basename(os.path.dirname(self.model_path))
            self.output_base_dir = os.path.join("./output", model_dir)
            os.makedirs(self.output_base_dir, exist_ok=True)
            logger.info(f"更新输出目录: {self.output_base_dir}")

    def set_model(self, model_name: str, model_path: Optional[str] = None) -> None:
        """
        设置当前使用的模型。

        Args:
            model_name (str): 模型名称
            model_path (Optional[str]): 模型文件路径,默认为None
        """
        from model_loader import load_model  # 延迟导入避免循环依赖
        if model_name in self.model_cache:
            self.model = self.model_cache[model_name]
            self.model_path = model_path or list(self.model_cache.keys())[list(self.model_cache.values()).index(self.model)]
        elif model_path:
            self.perf_monitor.log_memory("Before Model Load")
            self.perf_monitor.start_timer()
            self.model = load_model(model_path, self.device)
            self.perf_monitor.log_time(f"Load Model {model_name}")
            self.perf_monitor.log_memory("After Model Load")
            self.model_cache[model_name] = self.model
            self.model_path = model_path
        else:
            self.log_message.emit(f"ERROR:模型 {model_name} 未找到且无路径提供！")
            logger.error(f"模型 {model_name} 未找到且无路径提供")
            return
        self.current_model_name = model_name
        self.update_output_dir()
        logger.info(f"当前模型设置为: {model_name} ({self.model_path})")

    def _generate_heatmap(self, image: Image.Image, anomaly_map: np.ndarray, input_image_path: str) -> Optional[str]:
        """
        生成异常检测热图并保存。

        Args:
            image (Image.Image): 输入图像
            anomaly_map (np.ndarray): 异常得分图
            input_image_path (str): 输入图像路径

        Returns:
            Optional[str]: 输出图像路径,或None如果失败
        """
        try:
            anomaly_map = anomaly_map.squeeze()
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)  # 归一化
            original_image: np.ndarray = np.array(image.resize((288, 288)))
            heatmap: np.ndarray = plt.cm.jet(anomaly_map)[:, :, :3]  # 转换为热图颜色
            heatmap = (original_image * 0.5 + heatmap * 255 * 0.5).astype(np.uint8)  # 叠加原图
            combined_image: np.ndarray = np.hstack((original_image, heatmap))
            input_filename: str = os.path.splitext(os.path.basename(input_image_path))[0]
            output_path: str = os.path.join(self.output_base_dir, f"detection_{input_filename}.png")
            plt.imsave(output_path, combined_image)
            return output_path
        except Exception as e:
            error_msg: str = f"ERROR: 生成热力图失败: {str(e)}"
            self.log_message.emit(error_msg)
            logger.error(f"生成热力图失败: {str(e)},输入路径: {input_image_path}", exc_info=True)
            return None

    def detect_single_image(self, input_image_path: str, threshold: float) -> Tuple[Optional[str], str]:
        """
        检测单张图像的异常。

        Args:
            input_image_path (str): 输入图像路径
            threshold (float): 异常检测阈值

        Returns:
            Tuple[Optional[str], str]: (输出图像路径, 检测信息)
        
        Raises:
            DetectionError: 如果检测过程中发生异常
        """
        if not hasattr(self, 'model') or self.model is None:
            self.log_message.emit("ERROR:请先选择模型！")
            logger.error("未选择模型,无法进行检测")
            return None, "ERROR:请先选择模型！"
        try:
            self.perf_monitor.log_memory("Before Single Detection")
            self.perf_monitor.start_timer()

            image: Image.Image = Image.open(input_image_path).convert("RGB")
            image_tensor: torch.Tensor = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                scores, masks, _ = self.model.predict(image_tensor)
                anomaly_map: np.ndarray = masks[0]
            score: float = float(scores[0]) if not torch.is_tensor(scores[0]) else scores[0].item()
            detection_info: str = f"异常得分: {score:.2f} - {'检测到异常' if score > threshold else '图像正常'}"
            output_path: Optional[str] = self._generate_heatmap(image, anomaly_map, input_image_path)

            self.perf_monitor.log_time("Single Image Detection")
            self.perf_monitor.log_memory("After Single Detection")

            if output_path:
                self.log_message.emit(f"检测结果已保存到 {output_path}")
                logger.info(f"单张图像检测完成: {input_image_path}, 得分: {score:.2f}")
            return output_path, detection_info
        except Exception as e:
            error_msg: str = f"ERROR: 检测单张图片时发生错误: {str(e)}"
            self.log_message.emit(error_msg)
            logger.error(f"单张图像检测异常: {str(e)},输入路径: {input_image_path}", exc_info=True)
            raise DetectionError(f"单张图像检测失败: {str(e)}")

    def detect_batch_images(self, input_dir: str, threshold: float) -> None:
        """
        启动批量图像检测任务。

        Args:
            input_dir (str): 输入图像目录
            threshold (float): 异常检测阈值
        """
        if not hasattr(self, 'model') or self.model is None:
            self.log_message.emit("ERROR:请先选择模型！")
            logger.error("未选择模型,无法进行批量检测")
            return
        self.batch_worker = BatchDetectWorker(self, input_dir, threshold)
        self.batch_worker.progress_updated.connect(self.progress_updated.emit)
        self.batch_worker.log_message.connect(self.log_message.emit)
        self.batch_worker.batch_finished.connect(self.batch_finished.emit)
        self.batch_worker.start()
        logger.info(f"启动批量检测任务: {input_dir}")