from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QFormLayout, QDialog,
    QDoubleSpinBox, QTextEdit, QProgressBar, QFileDialog, QAction, QToolBar, QMenu, QPushButton,
    QListWidget, QListWidgetItem, QSizePolicy, QStatusBar, QScrollBar, QSlider)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QSize
import os
import yaml
from typing import List, Optional, Dict, Any  # 新增：类型提示支持
from progress_dialog import ProgressDialog, ProgressWorker
import logging
from image_processor import ImageProcessor
import torch

logger: logging.Logger = logging.getLogger('GUI')

class SettingsDialog(QDialog):
    def __init__(self, parent: Optional[QMainWindow] = None):
        """
        设置对话框,用于未来扩展设置选项。

        Args:
            parent (Optional[QMainWindow]): 父窗口,默认为None
        """
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.init_ui()

    def init_ui(self) -> None:
        """初始化设置对话框的UI"""
        layout: QFormLayout = QFormLayout()
        placeholder_label: QLabel = QLabel("Settings will be added here in the future.")
        layout.addWidget(placeholder_label)
        save_button: QPushButton = QPushButton("Close")
        save_button.clicked.connect(self.accept)
        layout.addWidget(save_button)
        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self, processor: ImageProcessor, config: Dict[str, Any]):
        """
        主窗口,负责GUI交互和异常检测功能。

        Args:
            processor (ImageProcessor): 图像处理器实例
            config (Dict[str, Any]): 配置字典
        """
        super().__init__()
        self.processor: ImageProcessor = processor
        self.config: Dict[str, Any] = config
        self.result_paths: List[str] = []  # 检测结果图像路径列表
        self.current_index: int = 0  # 当前显示的图像索引
        self.current_model_name: str = "未选择模型"  # 当前选中的模型名称
        self.detection_infos: List[str] = []  # 检测信息列表
        self.threshold: float = config.get("threshold", 1.20)  # 异常检测阈值
        self.performance_info: str = ""  # 性能信息（如耗时、等待时间）
        self.setAcceptDrops(True)  # 启用拖放功能
        self.init_ui()
        self.connect_signals()
        logger.debug("MainWindow 初始化完成")

    def init_ui(self) -> None:
        """初始化主窗口的UI布局"""
        self.setWindowTitle("Anomaly Detection App")
        self.setGeometry(100, 100, 800, 600)

        toolbar: QToolBar = QToolBar("Tools")
        self.addToolBar(toolbar)

        # 模型选择菜单
        self.model_menu: QMenu = QMenu("Select Model", self)
        model_action: QAction = QAction("Select Model", self)
        model_action.setMenu(self.model_menu)
        self.models: Dict[str, str] = self.config["models"]
        for i, model_name in enumerate(self.models.keys()):
            action: QAction = self.model_menu.addAction(model_name, lambda name=model_name: self.select_model(name))
            action.setShortcut(f"Ctrl+{i+1}")
        toolbar.addAction(model_action)

        # 检测菜单
        detect_menu: QMenu = QMenu("Detect", self)
        detect_action: QAction = QAction("Detect", self)
        detect_action.setMenu(detect_menu)
        detect_menu.addAction("Single Image", self.detect_single)
        detect_menu.addAction("Batch Images", self.detect_batch)
        toolbar.addAction(detect_action)

        # 选项菜单
        options_menu: QMenu = QMenu("Options", self)
        options_action: QAction = QAction("Options", self)
        options_action.setMenu(options_menu)
        options_menu.addAction("Settings", self.open_settings)
        toolbar.addAction(options_action)

        main_widget: QWidget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout: QHBoxLayout = QHBoxLayout(main_widget)

        # 左侧布局：阈值调整和检测信息
        left_widget: QWidget = QWidget()
        left_layout: QVBoxLayout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignTop)

        threshold_label: QLabel = QLabel("Anomaly Threshold:")
        self.threshold_slider: QSlider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 200)
        self.threshold_slider.setSingleStep(5)
        self.threshold_slider.setValue(int(self.threshold * 100))
        self.threshold_slider.valueChanged.connect(self.update_threshold_directly)
        threshold_value_label: QLabel = QLabel(f"{self.threshold:.2f}")
        self.threshold_slider.valueChanged.connect(lambda v: threshold_value_label.setText(f"{v/100:.2f}"))
        left_layout.addWidget(threshold_label)
        left_layout.addWidget(self.threshold_slider)
        left_layout.addWidget(threshold_value_label)

        self.detection_info_label: QLabel = QLabel("检测信息: 未检测")
        left_layout.addWidget(self.detection_info_label)
        left_layout.addStretch()

        # 右侧布局：图像显示和日志
        right_widget: QWidget = QWidget()
        right_layout: QVBoxLayout = QVBoxLayout(right_widget)

        self.status_bar: QStatusBar = QStatusBar()
        self.update_status_bar()
        right_layout.addWidget(self.status_bar)

        self.image_label: QLabel = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setAcceptDrops(True)
        right_layout.addWidget(self.image_label)

        self.thumbnail_list: QListWidget = QListWidget()
        self.thumbnail_list.setFlow(QListWidget.LeftToRight)
        self.thumbnail_list.setIconSize(QSize(80, 80))
        self.thumbnail_list.setFixedHeight(100)
        self.thumbnail_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thumbnail_list.itemClicked.connect(self.select_thumbnail)
        self.thumbnail_list.setToolTip("Click to view image")
        right_layout.addWidget(self.thumbnail_list)

        self.log_container: QWidget = QWidget()
        log_layout: QVBoxLayout = QVBoxLayout(self.log_container)
        self.log_toggle_button: QPushButton = QPushButton("Show Log")
        self.log_toggle_button.clicked.connect(self.toggle_log)
        log_layout.addWidget(self.log_toggle_button)
        self.log_text: QTextEdit = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setVisible(False)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(self.log_container)

        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 3)

        self.update_thumbnails()

    def connect_signals(self) -> None:
        """连接处理器信号到GUI槽函数"""
        self.processor.log_message.connect(self.update_log)
        self.processor.progress_updated.connect(self.update_progress)
        self.processor.batch_finished.connect(self.show_batch_results)

    def dragEnterEvent(self, event: Any) -> None:
        """处理拖入事件"""
        mime_data = event.mimeData()
        if mime_data.hasUrls():
            logger.info("检测到拖放事件: 有 URL 数据")
            event.acceptProposedAction()
        else:
            logger.debug("检测到拖放事件: 无 URL 数据")
            event.ignore()

    def dropEvent(self, event: Any) -> None:
        """处理放下事件"""
        mime_data = event.mimeData()
        if not mime_data.hasUrls():
            logger.debug("丢弃事件: 无 URL 数据")
            return
        urls = mime_data.urls()
        if not urls:
            logger.debug("丢弃事件: URL 列表为空")
            return
        path: str = urls[0].toLocalFile()
        logger.info(f"丢弃事件: 处理路径 {path}")
        if not path:
            logger.debug("丢弃事件: 路径无效")
            return
        if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg')):
            self.detect_single_drop(path)
        elif os.path.isdir(path):
            self.detect_batch_drop(path)
        else:
            logger.warning(f"不支持的文件或路径: {path}")
        event.acceptProposedAction()

    def detect_single_drop(self, file_path: str) -> None:
        """处理单张图像拖放检测"""
        output_path, detection_info = self.processor.detect_single_image(file_path, self.threshold)
        if output_path:
            self.show_result(output_path)
            self.result_paths = [output_path]
            self.detection_infos = [detection_info]
            self.current_index = 0
            self.update_status_bar()
            self.detection_info_label.setText(f"检测信息: {detection_info}")
            self.thumbnail_list.setVisible(True)
            self.update_thumbnails()

    def detect_batch_drop(self, folder_path: str) -> None:
        """处理批量图像拖放检测"""
        self.processor.detect_batch_images(folder_path, self.threshold)

    def select_model(self, model_name: str) -> None:
        """选择并加载指定模型"""
        model_path: Optional[str] = self.models.get(model_name)
        if not model_path:
            logger.error(f"模型 {model_name} 未找到！")
            QMessageBox.critical(self, "错误", f"模型 {model_name} 未找到！")
            return

        if model_name in self.processor.model_cache:
            self.processor.set_model(model_name, model_path)
            self.current_model_name = model_name
            self.update_status_bar()
            logger.info(f"模型已切换为: {model_name} (已缓存)")
            return

        progress_dialog: ProgressDialog = ProgressDialog(
            total_tasks=1,
            description=f"Loading model: {model_name}",
            parent=self
        )
        worker: ProgressWorker = ProgressWorker(
            self.processor.set_model,
            model_name,
            model_path
        )

        def on_finished(result: Any) -> None:
            progress_dialog.update_progress(1)
            if isinstance(result, Exception):
                error_msg: str = f"加载模型失败: {str(result)}"
                logger.error(error_msg, exc_info=True)
                QMessageBox.critical(self, "错误", error_msg)
            else:
                self.current_model_name = model_name
                self.update_status_bar()
                logger.info(f"模型已切换为: {model_name} ({model_path})")
            progress_dialog.accept()

        worker.finished.connect(on_finished)
        worker.start()
        progress_dialog.exec_()

    def detect_single(self) -> None:
        """通过文件对话框选择并检测单张图像"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg)")
        if file_path:
            output_path, detection_info = self.processor.detect_single_image(file_path, self.threshold)
            if output_path:
                self.show_result(output_path)
                self.result_paths = [output_path]
                self.detection_infos = [detection_info]
                self.current_index = 0
                self.update_status_bar()
                self.detection_info_label.setText(f"检测信息: {detection_info}")
                self.thumbnail_list.setVisible(True)
                self.update_thumbnails()

    def detect_batch(self) -> None:
        """通过文件夹对话框选择并检测批量图像"""
        folder_path: str = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.processor.detect_batch_images(folder_path, self.threshold)

    def show_result(self, output_path: str) -> None:
        """显示检测结果图像"""
        pixmap: QPixmap = QPixmap(output_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def show_batch_results(self, results: List[Any]) -> None:
        """显示批量检测结果"""
        output_paths, detection_infos = results
        self.result_paths = output_paths
        self.detection_infos = detection_infos
        self.current_index = 0
        if self.result_paths:
            self.show_result(self.result_paths[self.current_index])
            self.update_status_bar()
            self.detection_info_label.setText(f"检测信息: {self.detection_infos[self.current_index]}")
            self.thumbnail_list.setVisible(True)
            self.update_thumbnails()

    def select_thumbnail(self, item: QListWidgetItem) -> None:
        """选择缩略图切换显示图像"""
        self.current_index = item.data(Qt.UserRole)
        self.show_result(self.result_paths[self.current_index])
        self.update_status_bar()
        self.detection_info_label.setText(f"检测信息: {self.detection_infos[self.current_index]}")

    def update_status_bar(self) -> None:
        """更新状态栏信息"""
        image_name: str = os.path.basename(self.result_paths[self.current_index]) if self.result_paths else "未加载"
        if torch.cuda.is_available():
            allocated_memory: float = torch.cuda.memory_allocated(self.processor.device) / 1024**3
            status_text: str = f"Model: {self.current_model_name} | Image: {image_name} | GPU Memory: {allocated_memory:.2f} GiB"
            if self.performance_info:
                status_text += f" | {self.performance_info}"
        else:
            status_text: str = f"Model: {self.current_model_name} | Image: {image_name} | CPU Mode"
        self.status_bar.showMessage(status_text)

    def update_log(self, message: str) -> None:
        """更新日志显示"""
        self.log_text.append(message)
        if "耗时" in message or "等待时间" in message:
            self.performance_info = message.split(" - ")[-1]
            self.update_status_bar()
        if message.startswith("ERROR:"):
            error_msg: str = message[len("ERROR:"):].strip()
            QMessageBox.critical(self, "错误", error_msg)
            logger.error(f"GUI反馈错误: {error_msg}")

    def update_progress(self, value: int) -> None:
        """更新进度条"""
        if not hasattr(self, 'progress_dialog') or not self.progress_dialog.isVisible():
            self.progress_dialog: ProgressDialog = ProgressDialog(1, "Processing...", self)
            self.progress_dialog.show()
        self.progress_dialog.update_progress(value / 100)
        if value >= 100:
            self.progress_dialog.accept()

    def open_settings(self) -> None:
        """打开设置对话框"""
        dialog: SettingsDialog = SettingsDialog(self)
        dialog.exec_()
        logger.info("打开设置对话框")

    def update_threshold(self, new_threshold: float) -> None:
        """更新异常检测阈值并保存到配置"""
        self.threshold = new_threshold
        self.config["threshold"] = self.threshold
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True)
        if self.result_paths and self.detection_infos:
            for i in range(len(self.detection_infos)):
                score: float = float(self.detection_infos[i].split("异常得分: ")[1].split(" - ")[0])
                self.detection_infos[i] = f"异常得分: {score:.2f} - {'检测到异常' if score > self.threshold else '图像正常'}"
            self.detection_info_label.setText(f"检测信息: {self.detection_infos[self.current_index]}")
            self.update_thumbnails()
        logger.info(f"阈值已更新为: {self.threshold}")

    def update_threshold_directly(self, value: int) -> None:
        """直接从滑块更新阈值"""
        self.update_threshold(value / 100.0)

    def update_thumbnails(self) -> None:
        """更新缩略图列表"""
        self.thumbnail_list.clear()
        if not self.result_paths:
            return
        for i, path in enumerate(self.result_paths):
            pixmap: QPixmap = QPixmap(path).scaled(80, 80, Qt.KeepAspectRatio)
            item: QListWidgetItem = QListWidgetItem(QIcon(pixmap), "")
            item.setData(Qt.UserRole, i)
            item.setToolTip(f"{os.path.basename(path)}\n{self.detection_infos[i]}")
            self.thumbnail_list.addItem(item)
        if self.current_index < self.thumbnail_list.count():
            self.thumbnail_list.setCurrentRow(self.current_index)

    def toggle_log(self) -> None:
        """切换日志显示状态"""
        is_visible: bool = self.log_text.isVisible()
        self.log_text.setVisible(not is_visible)
        self.log_toggle_button.setText("Hide Log" if not is_visible else "Show Log")