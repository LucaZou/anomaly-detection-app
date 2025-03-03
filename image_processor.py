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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(329), # 保持长宽比缩放到 329
    transforms.CenterCrop((288, 288)), # 中心裁剪到 288x288
    transforms.ToTensor(), # 转为 Tensor张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 标准化
])

# 定义批量检测的工作线程
class BatchDetectWorker(QThread):
    progress_updated = pyqtSignal(int)  # 进度信号
    log_message = pyqtSignal(str)      # 日志信号
    batch_finished = pyqtSignal(list)  # 完成信号


    def __init__(self, processor, input_dir, threshold):  # 新增：接收阈值
        super().__init__()
        self.processor = processor
        self.input_dir = input_dir
        self.threshold = threshold  # 接收阈值
        self.device = processor.device
        # self.batch_size = 8  # 设置分批大小
        self.batch_size = self._estimate_batch_size() # 动态调整 batch_size

    def _estimate_batch_size(self):
        """动态估计适合的 batch_size,根据 GPU 可用内存"""
        total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GiB
        reserved_memory = torch.cuda.memory_reserved(self.device) / 1024**3  # GiB
        available_memory = total_memory - reserved_memory  # 可用内存

        # 假设每张图片（3通道，288x288）占用约 3 * 288 * 288 * 4 / 1024 / 1024 ≈ 0.236 MiB
        image_memory = 0.236  # 每张图片的内存占用（GiB，粗略估算）
        # 预留一定内存以避免碎片，假设预留 0.5 GiB
        safe_memory = available_memory - 0.5
        max_batch_size = int(safe_memory / image_memory)

        # 限制 batch_size 在 1 到 32 之间
        batch_size = max(1, min(32, max_batch_size))
        logger.info(f"动态调整 batch_size 为 {batch_size}，根据可用内存 {available_memory:.2f} GiB")
        return batch_size
    
    def _preload_images(self, image_paths): # 多线程预加载图片，减少 I/O 等待
        def load_image(path):
            try:
                img = Image.open(path).convert("RGB")
                return transform(img), path
            except Exception as e:
                logger.error(f"预加载图片失败 {path}: {str(e)}")
                return None, path

        self.preloaded_images = []
        self.preloaded_paths = []
        with ThreadPoolExecutor(max_workers=8) as executor:  # 固定线程数，优化 I/O
            future_to_path = {executor.submit(load_image, path): path for path in image_paths}
            for future in as_completed(future_to_path):
                img, path = future.result()
                if img is not None:
                    self.preloaded_images.append(img)
                    self.preloaded_paths.append(path)
        logger.info(f"成功预加载 {len(self.preloaded_images)} 张图片")
    
    # def _preload_images(self, image_paths):
    #     """多线程预加载图片，减少 I/O 等待"""
    #     def load_image(path):
    #         try:
    #             img = Image.open(path).convert("RGB")
    #             return transform(img), path
    #         except Exception as e:
    #             logger.error(f"预加载图片失败 {path}: {str(e)}")
    #             return None, path

    #     self.preloaded_images = []
    #     self.preloaded_paths = []
    #     threads = []
    #     for path in image_paths:
    #         thread = threading.Thread(target=lambda p, i=self.preloaded_images, ps=self.preloaded_paths: 
    #                                  [i.append(t[0]), ps.append(p)] if (t := load_image(p))[0] is not None else None, 
    #                                  args=(path,))
    #         threads.append(thread)
    #         thread.start()

    #     for thread in threads:
    #         thread.join()

    #     # 过滤失败的图片
    #     valid_indices = [i for i, img in enumerate(self.preloaded_images) if img is not None]
    #     self.preloaded_images = [self.preloaded_images[i] for i in valid_indices]
    #     self.preloaded_paths = [self.preloaded_paths[i] for i in valid_indices]
    #     logger.info(f"成功预加载 {len(self.preloaded_images)} 张图片")

    # def run(self): # 旧版本(单张逐一检测)
    #     # 在线程中执行批量检测
    #     image_paths = glob.glob(os.path.join(self.input_dir, "*.jpg")) + glob.glob(os.path.join(self.input_dir, "*.png"))
    #     if not image_paths:
    #         self.log_message.emit(f"在 {self.input_dir} 中未找到任何图片")
    #         return

    #     output_paths = []
    #     detection_infos = []  # 新增：存储每张图片的检测信息
    #     for i, input_image_path in enumerate(tqdm(image_paths, desc="批量检测图片")):
    #         # output_path, info = self.processor.detect_single_image(input_image_path)  # 接收返回的提示信息
    #         output_path, info = self.processor.detect_single_image(input_image_path, self.threshold)  # 修改：传递阈值
    #         if output_path:
    #             output_paths.append(output_path)
    #             detection_infos.append(info)  # 新增：存储检测信息
    #         self.progress_updated.emit(int((i + 1) / len(image_paths) * 100))

    #     self.log_message.emit(f"批量检测结果已保存到 {self.processor.output_base_dir}")
    #     # self.batch_finished.emit(output_paths)
    #     self.batch_finished.emit([output_paths, detection_infos])  # 修改：传递路径和信息列表

    def run(self):
        # 在线程中执行批量检测
        image_paths = glob.glob(os.path.join(self.input_dir, "*.jpg")) + glob.glob(os.path.join(self.input_dir, "*.png"))
        if not image_paths:
            self.log_message.emit(f"ERROR:在 {self.input_dir} 中未找到任何图片")
            return
        # 预加载图片
        self._preload_images(image_paths)
        output_paths = []
        detection_infos = []
        total_images = len(self.preloaded_images)

        with ThreadPoolExecutor(max_workers=12) as executor:  # 新增：并行处理热力图生成
            for start_idx in tqdm(range(0, total_images, self.batch_size), desc="批量检测图片"):
                end_idx = min(start_idx + self.batch_size, total_images)
                batch_tensors = torch.stack(self.preloaded_images[start_idx:end_idx]).to(self.device)
                try:
                    with torch.no_grad():
                        scores_list, masks_list, _ = self.processor.model.predict(batch_tensors)
                    
                    futures = []
                    for i, (scores, mask) in enumerate(zip(scores_list, masks_list)):
                        image_path = self.preloaded_paths[start_idx + i]
                        score = float(scores[0]) if isinstance(scores, list) else scores.item()
                        detection_info = f"异常得分: {score:.2f} - {'检测到异常' if score > self.threshold else '图像正常'}"
                        image = Image.open(image_path).convert("RGB")
                        futures.append(executor.submit(self.processor._generate_heatmap, image, mask, image_path))
                        detection_infos.append(detection_info)
                    
                    # 收集并行结果
                    for i, future in enumerate(as_completed(futures)):
                        output_path = future.result()
                        if output_path:
                            output_paths.append(output_path)
                        progress = int((start_idx + i + 1) / total_images * 100)
                        self.progress_updated.emit(progress)
                    
                    torch.cuda.empty_cache()
                except Exception as e:
                    error_msg = f"ERROR: 批量检测错误: {str(e)}"
                    self.log_message.emit(error_msg)
                    break

        # 分批处理预加载的图片
        # for start_idx in tqdm(range(0, total_images, self.batch_size), desc="批量检测图片"):
        #     end_idx = min(start_idx + self.batch_size, total_images)
        #     batch_tensors = torch.stack(self.preloaded_images[start_idx:end_idx]).to(self.device)

        #     try:
        #         # 批量推理
        #         with torch.no_grad():
        #             scores_list, masks_list, _ = self.processor.model.predict(batch_tensors)
        #         for i, (scores, mask) in enumerate(zip(scores_list, masks_list)): # 逐一后处理每张图片
        #             image_path = self.preloaded_paths[start_idx + i]
        #             # 提取单个图片的得分（假设 scores 是列表或张量中的单个值）
        #             score = float(scores[0]) if isinstance(scores, list) and len(scores) > 0 else scores.item() if torch.is_tensor(scores) else float(scores)
        #             detection_info = f"异常得分: {score:.2f} - {'检测到异常' if score > self.threshold else '图像正常'}" # 生成检测信息
        #             # 从预加载的图片生成热力图和输出
        #             image = Image.open(image_path).convert("RGB")  # 重新加载原始图片以保持一致性
        #             output_path = self.processor._generate_heatmap(image, mask, image_path)  # 修改：调用公共方法
        #             output_paths.append(output_path)
        #             detection_infos.append(detection_info)
        #             progress = int((start_idx + i + 1) / total_images * 100)
        #             self.progress_updated.emit(progress)
        #         # 清理 GPU 缓存
        #         torch.cuda.empty_cache()

        #     except RuntimeError as e:
        #         if "out of memory" in str(e):
        #             self.batch_size = max(1, self.batch_size // 2)  # 动态减小 batch_size
        #             self.log_message.emit(f"GPU 内存不足，调整 batch_size 为 {self.batch_size}")
        #             continue
        #         else:
        #             self.log_message.emit(f"批量检测错误: {str(e)}")
        #             self.log_message.emit(f"分批检测过程中发生错误（批次 {start_idx}-{end_idx}): {str(e)}")
        #             raise

        self.log_message.emit(f"批量检测结果已保存到 {self.processor.output_base_dir}")
        self.batch_finished.emit([output_paths, detection_infos])


class ImageProcessor(QObject):
    # 图像处理器类，用于处理图像检测任务
    progress_updated = pyqtSignal(int) # 进度更新信号
    log_message = pyqtSignal(str) # 日志消息信号
    batch_finished = pyqtSignal(list) # 批量处理完成信号

    def __init__(self, device, models=None):
        # 初始化处理器
        super().__init__()
        self.device = device # 设备（CPU/GPU）
        self.model_cache = models if models else {}  # 缓存预加载的模型
        self.current_model_name = None # 当前模型名称
        self.model_path = None # 当前模型路径
        self.output_base_dir = "./output" # 输出目录
        self.batch_worker = None # 存储工作线程对象

    def update_output_dir(self):
        # 根据模型路径更新输出目录
        if self.model_path:
            model_dir = os.path.basename(os.path.dirname(self.model_path)) # 模型目录名
            self.output_base_dir = os.path.join("./output", model_dir)
            os.makedirs(self.output_base_dir, exist_ok=True) # 创建输出目录

    def set_model(self, model_name, model_path=None):
        # 设置当前模型，支持预加载和按需加载
        if model_name in self.model_cache: # 模型已预加载
            self.model = self.model_cache[model_name] # 直接从缓存中获取
            self.model_path = model_path or list(self.model_cache.keys())[list(self.model_cache.values()).index(self.model)]
        elif model_path: # 模型未预加载但提供了路径
            from model_loader import load_model
            self.model = load_model(model_path, self.device)
            self.model_cache[model_name] = self.model
            self.model_path = model_path
        else: # 模型未找到且无路径提供
            self.log_message.emit(f"ERROR:模型 {model_name} 未找到且无路径提供！")
            return
        self.current_model_name = model_name
        self.update_output_dir()
        return None
    
    def _generate_heatmap(self, image, anomaly_map, input_image_path):
        """生成并保存热力图（新增公共方法）"""
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
            error_msg = f"ERROR: 生成热力图失败: {str(e)}"  # 修改：添加 ERROR 前缀
            self.log_message.emit(error_msg)
            return None

    def detect_single_image(self, input_image_path, threshold):
        # 检测单张图片
        if not hasattr(self, 'model') or self.model is None:
            self.log_message.emit("ERROR:请先选择模型！")
            return None, "ERROR:请先选择模型！"  # 修改：返回错误信息
        try:
            image = Image.open(input_image_path).convert("RGB") # 读取图片并转为 RGB 模式
            image_tensor = transform(image).unsqueeze(0).to(self.device) # 图像预处理并移动到设备
            with torch.no_grad():
                scores, masks, _ = self.model.predict(image_tensor) # 模型推理
                anomaly_map = masks[0] # 获取异常图
            score = float(scores[0]) if not torch.is_tensor(scores[0]) else scores[0].item() # scores 是单值列表，直接取第一个元素
            detection_info = f"异常得分: {score:.2f} - {'检测到异常' if score > threshold else '图像正常'}" # 生成检测信息
            output_path = self._generate_heatmap(image, anomaly_map, input_image_path)  # 修改：调用公共方法
            if output_path:
                self.log_message.emit(f"检测结果已保存到 {output_path}")
            return output_path, detection_info
        except Exception as e:
            error_msg = f"ERROR: 检测单张图片时发生错误: {str(e)}"  # 修改：添加 ERROR 前缀
            self.log_message.emit(error_msg)
            return None, error_msg
            

    def detect_batch_images(self, input_dir, threshold):  # 新增：接收阈值
        # 批量检测图片
        if not hasattr(self, 'model') or self.model is None:
            self.log_message.emit("ERROR:请先选择模型！")
            return None
        # 创建并启动工作线程
        self.batch_worker = BatchDetectWorker(self, input_dir, threshold)  # 修改：传递阈值
        # self.batch_worker = BatchDetectWorker(self, input_dir)
        self.batch_worker.progress_updated.connect(self.progress_updated.emit)
        self.batch_worker.log_message.connect(self.log_message.emit)
        self.batch_worker.batch_finished.connect(self.batch_finished.emit)
        self.batch_worker.start()
        return None  # 返回 None，因为结果通过信号异步传递