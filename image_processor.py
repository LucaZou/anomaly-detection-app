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

logger = logging.getLogger(__name__)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(329),
    transforms.CenterCrop((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImagePreloader:
    """独立的图像预加载器，支持磁盘缓存或内存模式"""
    def __init__(self, max_memory_mb=2048, use_disk_cache=True):
        self.max_memory_mb = max_memory_mb
        self.use_disk_cache = use_disk_cache
        if self.use_disk_cache:
            self.temp_dir = tempfile.TemporaryDirectory()
        else:
            self.memory_chunks = []

    def _load_image(self, path):
        """加载单张图片并返回 tensor 和路径"""
        try:
            img = Image.open(path).convert("RGB")
            tensor = transform(img)
            return tensor, path
        except Exception as e:
            logger.error(f"预加载图片失败 {path}: {str(e)}")
            return None, path

    def preload(self, image_paths, chunk_size=100):
        """预加载图片，支持磁盘缓存或内存模式"""
        max_workers = min(multiprocessing.cpu_count(), 8)
        logger.info(f"使用 {max_workers} 个线程进行预加载，模式: {'磁盘缓存' if self.use_disk_cache else '内存'}")
        preloaded_chunks = []
        total_images = len(image_paths)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_index = 0  # 修改1：跟踪内存分片的索引
            for start_idx in range(0, total_images, chunk_size):
                end_idx = min(start_idx + chunk_size, total_images)
                chunk_paths = image_paths[start_idx:end_idx]
                future_to_path = {executor.submit(self._load_image, path): path for path in chunk_paths}
                chunk_tensors = []
                chunk_paths_valid = []

                for future in as_completed(future_to_path):
                    tensor, path = future.result()
                    if tensor is not None:
                        chunk_tensors.append(tensor)
                        chunk_paths_valid.append(path)

                if chunk_tensors:
                    if self.use_disk_cache:
                        chunk_array = torch.stack(chunk_tensors).cpu().numpy()
                        chunk_file = os.path.join(self.temp_dir.name, f"chunk_{start_idx}.npz")
                        np.savez_compressed(chunk_file, images=chunk_array, paths=chunk_paths_valid)
                        preloaded_chunks.append(chunk_file)
                        logger.info(f"保存分片 {chunk_file}，包含 {len(chunk_tensors)} 张图片")
                    else:
                        self.memory_chunks.append({
                            'images': torch.stack(chunk_tensors),
                            'paths': chunk_paths_valid
                        })
                        preloaded_chunks.append(chunk_index)  # 修改1：使用连续索引
                        logger.info(f"内存存储分片 {chunk_index}，包含 {len(chunk_tensors)} 张图片")
                        chunk_index += 1  # 修改1：递增索引

        return preloaded_chunks

    def cleanup(self):
        """清理临时文件或内存数据"""
        if self.use_disk_cache:
            for attempt in range(3):
                try:
                    self.temp_dir.cleanup()
                    logger.info("成功清理临时目录")
                    break
                except PermissionError as e:
                    logger.warning(f"清理临时目录失败 (尝试 {attempt+1}/3): {str(e)}")
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"清理临时目录时发生未知错误: {str(e)}")
                    break
            else:
                logger.error("多次尝试后仍无法清理临时目录，可能需要手动删除")
        else:
            self.memory_chunks.clear()
            logger.info("成功清理内存分片")

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
        self.preloader = ImagePreloader(use_disk_cache=self.config.get('use_disk_cache', True))

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
            return

        chunk_size = self.config.get('preload_chunk_size', 100)
        preloaded_chunks = self.preloader.preload(image_paths, chunk_size)
        output_paths = []
        detection_infos = []
        total_images = sum(len(np.load(chunk_file, allow_pickle=True)['paths']) if self.preloader.use_disk_cache 
                           else len(self.preloader.memory_chunks[idx]['paths']) 
                           for idx, chunk_file in enumerate(preloaded_chunks))

        with ThreadPoolExecutor(max_workers=12) as executor:
            processed_images = 0
            for chunk_idx, chunk_ref in enumerate(tqdm(preloaded_chunks, desc="处理图像分片")):
                if self.preloader.use_disk_cache:
                    with np.load(chunk_ref, allow_pickle=True) as chunk_data:
                        chunk_tensors = torch.from_numpy(chunk_data['images']).to(self.device)
                        chunk_paths = chunk_data['paths']
                else:
                    if chunk_ref >= len(self.preloader.memory_chunks):  # 修改2：添加索引检查
                        self.log_message.emit(f"ERROR: 内存分片索引 {chunk_ref} 超出范围，最大索引为 {len(self.preloader.memory_chunks)-1}")
                        break
                    chunk_data = self.preloader.memory_chunks[chunk_ref]
                    chunk_tensors = chunk_data['images'].to(self.device)
                    chunk_paths = chunk_data['paths']

                for start_idx in range(0, len(chunk_tensors), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, len(chunk_tensors))
                    batch_tensors = chunk_tensors[start_idx:end_idx]
                    batch_paths = chunk_paths[start_idx:end_idx]

                    try:
                        with torch.no_grad():
                            scores_list, masks_list, _ = self.processor.model.predict(batch_tensors)

                        futures = []
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

                        torch.cuda.empty_cache()
                    except Exception as e:
                        error_msg = f"ERROR: 批量检测错误: {str(e)}"
                        self.log_message.emit(error_msg)
                        break

                if self.preloader.use_disk_cache:
                    try:
                        os.remove(chunk_ref)
                        logger.info(f"成功删除分片文件 {chunk_ref}")
                    except PermissionError as e:
                        logger.warning(f"无法删除分片文件 {chunk_ref}: {str(e)}，将在最后清理")

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

    def update_output_dir(self):
        if self.model_path:
            model_dir = os.path.basename(os.path.dirname(self.model_path))
            self.output_base_dir = os.path.join("./output", model_dir)
            os.makedirs(self.output_base_dir, exist_ok=True)

    def set_model(self, model_name, model_path=None):
        if model_name in self.model_cache:
            self.model = self.model_cache[model_name]
            self.model_path = model_path or list(self.model_cache.keys())[list(self.model_cache.values()).index(self.model)]
        elif model_path:
            from model_loader import load_model
            self.model = load_model(model_path, self.device)
            self.model_cache[model_name] = self.model
            self.model_path = model_path
        else:
            self.log_message.emit(f"ERROR:模型 {model_name} 未找到且无路径提供！")
            return
        self.current_model_name = model_name
        self.update_output_dir()
        return None

    def _generate_heatmap(self, image, anomaly_map, input_image_path):
        """生成热图"""
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
            return None

    def detect_single_image(self, input_image_path, threshold):
        if not hasattr(self, 'model') or self.model is None:
            self.log_message.emit("ERROR:请先选择模型！")
            return None, "ERROR:请先选择模型！"
        try:
            image = Image.open(input_image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                scores, masks, _ = self.model.predict(image_tensor)
                anomaly_map = masks[0]
            score = float(scores[0]) if not torch.is_tensor(scores[0]) else scores[0].item()
            detection_info = f"异常得分: {score:.2f} - {'检测到异常' if score > threshold else '图像正常'}"
            output_path = self._generate_heatmap(image, anomaly_map, input_image_path)
            if output_path:
                self.log_message.emit(f"检测结果已保存到 {output_path}")
            return output_path, detection_info
        except Exception as e:
            error_msg = f"ERROR: 检测单张图片时发生错误: {str(e)}"
            self.log_message.emit(error_msg)
            return None, error_msg

    def detect_batch_images(self, input_dir, threshold):
        if not hasattr(self, 'model') or self.model is None:
            self.log_message.emit("ERROR:请先选择模型！")
            return None
        self.batch_worker = BatchDetectWorker(self, input_dir, threshold)
        self.batch_worker.progress_updated.connect(self.progress_updated.emit)
        self.batch_worker.log_message.connect(self.log_message.emit)
        self.batch_worker.batch_finished.connect(self.batch_finished.emit)
        self.batch_worker.start()
        return None