# performance_monitor.py
import torch
import time
import logging
import psutil
import os
from threading import active_count

class PerformanceMonitor:
    def __init__(self, device, log_file="./logs/performance_log.txt"):
        self.device = device
        self.logger = logging.getLogger('PerformanceMonitor')
        self.logger.setLevel(logging.INFO)
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=10,
            encoding="utf-8"
        )
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.start_time = None
        self.wait_start_time = None

    def log_memory(self, stage="Unknown"):
        """记录当前内存和显存使用情况"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            reserved_memory = torch.cuda.memory_reserved(self.device) / 1024**3
            allocated_memory = torch.cuda.memory_allocated(self.device) / 1024**3
            self.logger.info(f"{stage} - GPU Memory: Total={total_memory:.2f} GiB, "
                            f"Reserved={reserved_memory:.2f} GiB, "
                            f"Allocated={allocated_memory:.2f} GiB")
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss / 1024**3  # 转换为GiB
        self.logger.info(f"{stage} - CPU Memory: {cpu_memory:.2f} GiB")

    def start_timer(self):
        """开始计时"""
        self.start_time = time.time()

    def log_time(self, task_name):
        """记录任务耗时"""
        if self.start_time is None:
            self.logger.warning(f"计时未开始，无法记录 {task_name} 的耗时")
            return
        elapsed_time = time.time() - self.start_time
        self.logger.info(f"{task_name} 耗时: {elapsed_time:.2f} 秒")
        self.start_time = None

    def start_wait_timer(self):
        """开始等待计时"""
        self.wait_start_time = time.time()

    def log_wait_time(self, wait_name):
        """记录等待时间"""
        if self.wait_start_time is None:
            self.logger.warning(f"等待计时未开始，无法记录 {wait_name} 的等待时间")
            return
        wait_time = time.time() - self.wait_start_time
        self.logger.info(f"{wait_name} 等待时间: {wait_time:.2f} 秒")
        self.wait_start_time = None
        return wait_time

    def log_thread_usage(self, stage="Unknown"):
        """记录当前线程利用率"""
        active_threads = active_count()
        cpu_cores = psutil.cpu_count()
        self.logger.info(f"{stage} - Active Threads: {active_threads}, CPU Cores: {cpu_cores}")

    def log_io_stats(self, stage="Unknown", file_size=None):
        """记录IO性能(仅在有文件操作时调用）"""
        if file_size:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            if elapsed_time > 0:
                throughput = file_size / elapsed_time / 1024**2  # MB/s
                self.logger.info(f"{stage} - IO Throughput: {throughput:.2f} MB/s")

    def wrap_function(self, func, stage_name):
        """装饰器，用于监控函数性能"""
        def wrapper(*args, **kwargs):
            self.log_memory(f"Before {stage_name}")
            self.log_thread_usage(f"Before {stage_name}")
            self.start_timer()
            result = func(*args, **kwargs)
            self.log_time(stage_name)
            self.log_memory(f"After {stage_name}")
            self.log_thread_usage(f"After {stage_name}")
            return result
        return wrapper