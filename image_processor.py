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
from exceptions import DetectionError
from performance_monitor import PerformanceMonitor

logger = logging.getLogger('ImageProcessor')

transform = transforms.Compose([
    transforms.Resize(329),
    transforms.CenterCrop((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImagePreloader:
    def __init__(self, max_memory_mb=2048, use_disk_cache=True, device=None):
        self.max_memory_mb = max_memory_mb
        self.use_disk_cache = use_disk_cache
        self.device = device
        self.perf_monitor = PerformanceMonitor(device)  # 新增：性能监控
        if self.use_disk_cache:
            self.temp_dir = tempfile.TemporaryDirectory()
            logger.info(f"使用磁盘缓存，临时目录: {self.temp_dir.name}")
        else:
            self.memory_chunks = []
            logger.info("使用内存缓存")

    def _load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            tensor = transform(img)
            return tensor, path
        except Exception as e:
            logger.error(f"预加载图片失败 {path}: {str(e)}", exc_info=True)
            return None, path

    def preload(self, image_paths, chunk_size=100):
        max_workers = min(multiprocessing.cpu_count(), 8)
        logger.info(f"开始预加载 {len(image_paths)} 张图片，使用 {max_workers} 个线程，"
                    f"缓存模式: {'磁盘' if self.use_disk_cache else '内存'}")
        self.perf_monitor.start_timer()
        preloaded_chunks = []
        total_images = len(image_paths)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_index = 0
            for start_idx in range(0, total_images, chunk_size):
                end_idx = min(start_idx + chunk_size, total_images)
                chunk_paths = image_paths[start_idx:end_idx]
                logger.debug(f"预加载分片 {start_idx}-{end_idx}，包含 {len(chunk_paths)} 张图片")
                self.perf_monitor.start_wait_timer()  # 新增：记录线程池等待时间
                future_to_path = {executor.submit(self._load_image, path): path for path in chunk_paths}
                chunk_tensors = []
                chunk_paths_valid = []

                for future in as_completed(future_to_path):
                    tensor, path = future.result()
                    if tensor is not None:
                        chunk_tensors.append(tensor)
                        chunk_paths_valid.append(path)

                wait_time = self.perf_monitor.log_wait_time(f"Preload Chunk {start_idx}-{end_idx} Thread Wait")
                if chunk_tensors:
                    if self.use_disk_cache:
                        self.perf_monitor.start_timer()
                        chunk_array = torch.stack(chunk_tensors).cpu().numpy()
                        chunk_file = os.path.join(self.temp_dir.name, f"chunk_{start_idx}.npz")
                        np.savez_compressed(chunk_file, images=chunk_array, paths=chunk_paths_valid)
                        file_size = os.path.getsize(chunk_file) / 1024**2  # MB
                        self.perf_monitor.log_io_stats(f"Save Chunk {chunk_file}", file_size)
                        preloaded_chunks.append(chunk_file)
                        logger.info(f"保存磁盘分片 {chunk_file}，包含 {len(chunk_tensors)} 张图片，"
                                    f"大小: {file_size:.2f} MB")
                    else:
                        self.memory_chunks.append({
                            'images': torch.stack(chunk_tensors),
                            'paths': chunk_paths_valid
                        })
                        preloaded_chunks.append(chunk_index)
                        logger.info(f"内存存储分片 {chunk_index}，包含 {len(chunk_tensors)} 张图片")
                        chunk_index += 1

        self.perf_monitor.log_time("Image Preloading Total")
        return preloaded_chunks

    def cleanup(self):
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
                logger.error("多次尝试后仍无法清理磁盘缓存，可能需要手动删除")
        else:
            self.memory_chunks.clear()
            logger.info("成功清理内存缓存")

class BatchDetectWorker(QThread):
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str)
    batch_finished = pyqtSignal(list)

    def __init__(self, processor, input_dir, threshold):
        super().__init__()
        self.processor = processor
        self.input_dir = input_dir
        self.threshold = threshold
        self.device = processor.device
        self.config = processor.config
        self.batch_size = self._estimate_batch_size()
        self.preloader = ImagePreloader(use_disk_cache=self.config.get('use_disk_cache', True), device=self.device)
        self.perf_monitor = PerformanceMonitor(self.device)

    def _estimate_batch_size(self):
        total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        reserved_memory = torch.cuda.memory_reserved(self.device) / 1024**3
        available_memory = total_memory - reserved_memory
        image_memory = 0.236
        safe_memory = available_memory - 0.5
        max_batch_size = self.config.get('max_batch_size', 32)
        batch_size = max(1, min(max_batch_size, int(safe_memory / image_memory)))
        logger.info(f"动态调整 batch_size 为 {batch_size}，根据可用内存 {available_memory:.2f} GiB")
        return batch_size

    def run(self):
        image_paths = glob.glob(os.path.join(self.input_dir, "*.jpg")) + glob.glob(os.path.join(self.input_dir, "*.png"))
        if not image_paths:
            self.log_message.emit(f"ERROR:在 {self.input_dir} 中未找到任何图片")
            logger.error(f"批量检测输入目录 {self.input_dir} 为空")
            return

        self.perf_monitor.log_memory("Batch Detection Start")
        self.perf_monitor.start_timer()

        chunk_size = self.config.get('preload_chunk_size', 100)
        preloaded_chunks = self.preloader.preload(image_paths, chunk_size)
        self.perf_monitor.log_time("Image Preloading")

        output_paths = []
        detection_infos = []
        total_images = sum(len(np.load(chunk_file, allow_pickle=True)['paths']) if self.preloader.use_disk_cache 
                           else len(self.preloader.memory_chunks[idx]['paths']) 
                           for idx, chunk_file in enumerate(preloaded_chunks))

        with ThreadPoolExecutor(max_workers=12) as executor:
            processed_images = 0
            for chunk_idx, chunk_ref in enumerate(tqdm(preloaded_chunks, desc="处理图像分片")):
                if self.preloader.use_disk_cache:
                    self.perf_monitor.start_timer()
                    with np.load(chunk_ref, allow_pickle=True) as chunk_data:
                        chunk_tensors = torch.from_numpy(chunk_data['images']).to(self.device)
                        chunk_paths = chunk_data['paths']
                    file_size = os.path.getsize(chunk_ref) / 1024**2  # MB
                    self.perf_monitor.log_io_stats(f"Load Chunk {chunk_ref}", file_size)
                else:
                    if chunk_ref >= len(self.preloader.memory_chunks):
                        self.log_message.emit(f"ERROR: 内存分片索引 {chunk_ref} 超出范围，最大索引为 {len(self.preloader.memory_chunks)-1}")
                        logger.error(f"内存分片索引错误: {chunk_ref}")
                        break
                    chunk_data = self.preloader.memory_chunks[chunk_ref]
                    chunk_tensors = chunk_data['images'].to(self.device)
                    chunk_paths = chunk_data['paths']

                for start_idx in range(0, len(chunk_tensors), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, len(chunk_tensors))
                    batch_tensors = chunk_tensors[start_idx:end_idx]
                    batch_paths = chunk_paths[start_idx:end_idx]

                    try:
                        self.perf_monitor.start_timer()
                        with torch.no_grad():
                            scores_list, masks_list, _ = self.processor.model.predict(batch_tensors)
                        self.perf_monitor.log_time(f"Batch Inference (Chunk {chunk_idx}, Batch {start_idx//self.batch_size})")

                        futures = []
                        self.perf_monitor.start_wait_timer()  # 新增：记录热图生成等待时间
                        for i, (scores, mask, image_path) in enumerate(zip(scores_list, masks_list, batch_paths)):
                            score = float(scores[0]) if isinstance(scores, list) else scores.item()
                            detection_info = f"异常得分: {score:.2f} - {'检测到异常' if score > self.threshold else '图像正常'}"
                            image = Image.open(image_path).convert("RGB")
                            futures.append(executor.submit(self.processor._generate_heatmap, image, mask, image_path))
                            detection_infos.append(detection_info)

                        for i, future in enumerate(as_completed(futures)):
                            output_path = future.result()
                            if output_path:
                                output_paths.append(output_path)
                            processed_images += 1
                            progress = int((processed_images / total_images) * 100)
                            self.progress_updated.emit(progress)

                        wait_time = self.perf_monitor.log_wait_time(f"Heatmap Generation Wait (Chunk {chunk_idx}, Batch {start_idx//self.batch_size})")
                        torch.cuda.empty_cache()
                    except Exception as e:
                        error_msg = f"ERROR: 批量检测错误: {str(e)}"
                        self.log_message.emit(error_msg)
                        logger.error(f"批量检测异常: {str(e)}，输入路径: {self.input_dir}", exc_info=True)
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
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str)
    batch_finished = pyqtSignal(list)

    def __init__(self, device, models=None, config=None):
        super().__init__()
        self.device = device
        self.model_cache = models if models else {}
        self.current_model_name = None
        self.model_path = None
        self.output_base_dir = "./output"
        self.batch_worker = None
        self.config = config or {}
        self.perf_monitor = PerformanceMonitor(device)

    def update_output_dir(self):
        if self.model_path:
            model_dir = os.path.basename(os.path.dirname(self.model_path))
            self.output_base_dir = os.path.join("./output", model_dir)
            os.makedirs(self.output_base_dir, exist_ok=True)
            logger.info(f"更新输出目录: {self.output_base_dir}")

    def set_model(self, model_name, model_path=None):
        from model_loader import load_model
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
        return None

    def _generate_heatmap(self, image, anomaly_map, input_image_path):
        try:
            anomaly_map = anomaly_map.squeeze()
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            original_image = np.array(image.resize((288, 288)))
            heatmap = plt.cm.jet(anomaly_map)[:, :, :3]
            heatmap = (original_image * 0.5 + heatmap * 255 * 0.5).astype(np.uint8)
            combined_image = np.hstack((original_image, heatmap))
            input_filename = os.path.splitext(os.path.basename(input_image_path))[0]
            output_path = os.path.join(self.output_base_dir, f"detection_{input_filename}.png")
            plt.imsave(output_path, combined_image)
            return output_path
        except Exception as e:
            error_msg = f"ERROR: 生成热力图失败: {str(e)}"
            self.log_message.emit(error_msg)
            logger.error(f"生成热力图失败: {str(e)}，输入路径: {input_image_path}", exc_info=True)
            return None

    def detect_single_image(self, input_image_path, threshold):
        if not hasattr(self, 'model') or self.model is None:
            self.log_message.emit("ERROR:请先选择模型！")
            logger.error("未选择模型，无法进行检测")
            return None, "ERROR:请先选择模型！"
        try:
            self.perf_monitor.log_memory("Before Single Detection")
            self.perf_monitor.start_timer()

            image = Image.open(input_image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                scores, masks, _ = self.model.predict(image_tensor)
                anomaly_map = masks[0]
            score = float(scores[0]) if not torch.is_tensor(scores[0]) else scores[0].item()
            detection_info = f"异常得分: {score:.2f} - {'检测到异常' if score > threshold else '图像正常'}"
            output_path = self._generate_heatmap(image, anomaly_map, input_image_path)

            self.perf_monitor.log_time("Single Image Detection")
            self.perf_monitor.log_memory("After Single Detection")

            if output_path:
                self.log_message.emit(f"检测结果已保存到 {output_path}")
                logger.info(f"单张图像检测完成: {input_image_path}, 得分: {score:.2f}")
            return output_path, detection_info
        except Exception as e:
            error_msg = f"ERROR: 检测单张图片时发生错误: {str(e)}"
            self.log_message.emit(error_msg)
            logger.error(f"单张图像检测异常: {str(e)}，输入路径: {input_image_path}", exc_info=True)
            raise DetectionError(f"单张图像检测失败: {str(e)}")

    def detect_batch_images(self, input_dir, threshold):
        if not hasattr(self, 'model') or self.model is None:
            self.log_message.emit("ERROR:请先选择模型！")
            logger.error("未选择模型，无法进行批量检测")
            return None
        self.batch_worker = BatchDetectWorker(self, input_dir, threshold)
        self.batch_worker.progress_updated.connect(self.progress_updated.emit)
        self.batch_worker.log_message.connect(self.log_message.emit)
        self.batch_worker.batch_finished.connect(self.batch_finished.emit)
        self.batch_worker.start()
        logger.info(f"启动批量检测任务: {input_dir}")
        return None