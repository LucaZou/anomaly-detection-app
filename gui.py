from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QFormLayout, QDialog,
    QDoubleSpinBox, QTextEdit, QProgressBar, QFileDialog, QAction, QToolBar, QMenu, QPushButton,
    QListWidget, QListWidgetItem, QSizePolicy, QStatusBar, QScrollBar, QSlider)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QSize
import os
import yaml
from progress_dialog import ProgressDialog, ProgressWorker

# 定义设置窗口类（保持不变）
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()
        # 暂时留空，供后续扩展
        placeholder_label = QLabel("Settings will be added here in the future.")
        layout.addWidget(placeholder_label)
        save_button = QPushButton("Close")
        save_button.clicked.connect(self.accept)
        layout.addWidget(save_button)
        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self, processor, config):
        super().__init__()
        self.processor = processor
        self.config = config
        self.result_paths = []
        self.current_index = 0
        self.current_model_name = "未选择模型"
        self.detection_infos = []
        self.threshold = config.get("threshold", 1.20)
        self.setAcceptDrops(True)  # 修改1：在窗口级别启用拖放
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        self.setWindowTitle("Anomaly Detection App")
        self.setGeometry(100, 100, 800, 600)  # 调整窗口大小以适应新布局

        # 修改1：简化工具栏
        toolbar = QToolBar("Tools")
        self.addToolBar(toolbar)

        # 模型选择菜单
        self.model_menu = QMenu("Select Model", self)
        model_action = QAction("Select Model", self)
        model_action.setMenu(self.model_menu)
        self.models = self.config["models"]
        for i, model_name in enumerate(self.models.keys()):
            action = self.model_menu.addAction(model_name, lambda name=model_name: self.select_model(name))
            action.setShortcut(f"Ctrl+{i+1}")  # 修改5：添加快捷键
        toolbar.addAction(model_action)

        # 检测菜单（合并单张和批量检测）
        detect_menu = QMenu("Detect", self)
        detect_action = QAction("Detect", self)
        detect_action.setMenu(detect_menu)
        detect_menu.addAction("Single Image", self.detect_single)
        detect_menu.addAction("Batch Images", self.detect_batch)
        toolbar.addAction(detect_action)

        # 选项菜单（包含设置）
        options_menu = QMenu("Options", self)
        options_action = QAction("Options", self)
        options_action.setMenu(options_menu)
        options_menu.addAction("Settings", self.open_settings)
        toolbar.addAction(options_action)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)  # 修改2：改为水平布局，分为左右两部分

        # 左边：配置参数和检测信息
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignTop)

        # 配置参数（阈值）
        threshold_label = QLabel("Anomaly Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 200)  # 0.0 - 2.0，放大100倍以支持整数滑动
        self.threshold_slider.setSingleStep(5)  # 步长 0.05
        self.threshold_slider.setValue(int(self.threshold * 100))  # 默认值
        self.threshold_slider.valueChanged.connect(self.update_threshold_directly)
        threshold_value_label = QLabel(f"{self.threshold:.2f}")  # 显示当前值
        self.threshold_slider.valueChanged.connect(lambda v: threshold_value_label.setText(f"{v/100:.2f}"))
        left_layout.addWidget(threshold_label)
        left_layout.addWidget(self.threshold_slider)
        left_layout.addWidget(threshold_value_label)

        # 检测信息（占位，未来可扩展）
        self.detection_info_label = QLabel("检测信息: 未检测")
        left_layout.addWidget(self.detection_info_label)
        left_layout.addStretch()  # 添加伸缩项，使内容靠上对齐

        # 右边：图像显示和缩略图
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 修改2：状态栏替代独立标签
        self.status_bar = QStatusBar()
        self.update_status_bar()
        right_layout.addWidget(self.status_bar)

        # 图像显示区（支持拖放）
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setAcceptDrops(True)  # 修改5：支持拖放
        right_layout.addWidget(self.image_label)

        # 修改3：改进缩略图导航（移除切换按钮）
        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setFlow(QListWidget.LeftToRight)
        self.thumbnail_list.setIconSize(QSize(80, 80))
        self.thumbnail_list.setFixedHeight(100)  # 固定高度
        self.thumbnail_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thumbnail_list.itemClicked.connect(self.select_thumbnail)
        self.thumbnail_list.setToolTip("Click to view image")  # 添加提示
        right_layout.addWidget(self.thumbnail_list)

        # 将左右布局添加到主布局
        main_layout.addWidget(left_widget, 1)  # 左边占1份宽度
        main_layout.addWidget(right_widget, 3)  # 右边占3份宽度

        # 修改4：日志区改为可折叠
        self.log_container = QWidget()
        log_layout = QVBoxLayout(self.log_container)
        self.log_toggle_button = QPushButton("Show Log")
        self.log_toggle_button.clicked.connect(self.toggle_log)
        log_layout.addWidget(self.log_toggle_button)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setVisible(False)  # 默认隐藏
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(self.log_container)

        # 修改4：进度条移至浮动窗口，此处无需添加

        self.update_thumbnails()

    def connect_signals(self):
        self.processor.log_message.connect(self.update_log)
        self.processor.progress_updated.connect(self.update_progress)
        self.processor.batch_finished.connect(self.show_batch_results)

    def dragEnterEvent(self, event):
        # 修改2：优化拖放事件接受逻辑
        mime_data = event.mimeData()
        if mime_data.hasUrls():
            self.log_text.append("检测到拖放事件: 有 URL 数据")
            event.acceptProposedAction()
        else:
            self.log_text.append("检测到拖放事件: 无 URL 数据")
            event.ignore()

    def dropEvent(self, event):
        # 修改2：优化拖放处理并添加调试日志
        mime_data = event.mimeData()
        if not mime_data.hasUrls():
            self.log_text.append("丢弃事件: 无 URL 数据")
            return
        urls = mime_data.urls()
        if not urls:
            self.log_text.append("丢弃事件: URL 列表为空")
            return
        path = urls[0].toLocalFile()
        self.log_text.append(f"丢弃事件: 处理路径 {path}")
        if not path:
            self.log_text.append("丢弃事件: 路径无效")
            return
        if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg')):
            self.log_text.append(f"触发单张检测: {path}")
            self.detect_single_drop(path)
        elif os.path.isdir(path):
            self.log_text.append(f"触发批量检测: {path}")
            self.detect_batch_drop(path)
        else:
            self.log_text.append(f"不支持的文件或路径: {path}")
        event.acceptProposedAction()  # 明确接受事件

    def detect_single_drop(self, file_path):
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

    def detect_batch_drop(self, folder_path):
        self.processor.detect_batch_images(folder_path, self.threshold)

    def select_model(self, model_name):
        """选择模型并显示加载进度"""
        model_path = self.models.get(model_name)
        if not model_path:
            self.log_text.append(f"模型 {model_name} 未找到！")
            return

        # 如果模型已加载，直接切换
        if model_name in self.processor.model_cache:
            self.processor.set_model(model_name, model_path)
            self.current_model_name = model_name
            self.update_status_bar()  # 修改：使用状态栏更新模型名
            self.log_text.append(f"模型已切换为: {model_name} (已缓存)")
            return

        # 新增：显示进度对话框并加载模型
        progress_dialog = ProgressDialog(
            total_tasks=1,  # 单个模型加载
            description=f"Loading model: {model_name}",
            parent=self
        )
        worker = ProgressWorker(
            self.processor.set_model,
            model_name,
            model_path
        )

        def on_finished(result):
            progress_dialog.update_progress(1)  # 加载完成，进度满
            if isinstance(result, Exception):
                error_msg = f"加载模型失败: {str(result)}"
                self.log_text.append(error_msg)
                QMessageBox.critical(self, "错误", error_msg)
            else:
                self.current_model_name = model_name
                self.update_status_bar()  # 修改：使用状态栏更新模型名
                self.log_text.append(f"模型已切换为: {model_name} ({model_path})")
            progress_dialog.accept()

        worker.finished.connect(on_finished)
        worker.start()
        progress_dialog.exec_()  # 模态显示，直到加载完成

    def detect_single(self):
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

    def detect_batch(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.processor.detect_batch_images(folder_path, self.threshold)

    def show_result(self, output_path):
        pixmap = QPixmap(output_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def show_batch_results(self, results):
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

    def select_thumbnail(self, item):
        self.current_index = item.data(Qt.UserRole)
        self.show_result(self.result_paths[self.current_index])
        self.update_status_bar()
        self.detection_info_label.setText(f"检测信息: {self.detection_infos[self.current_index]}")

    def update_status_bar(self):
        # 修改2：更新状态栏
        image_name = os.path.basename(self.result_paths[self.current_index]) if self.result_paths else "未加载"
        status_text = f"Model: {self.current_model_name} | Image: {image_name}"
        self.status_bar.showMessage(status_text)

    def update_log(self, message):
        self.log_text.append(message)
        if message.startswith("ERROR:"):
            error_msg = message[len("ERROR:"):].strip()
            QMessageBox.critical(self, "错误", error_msg)

    def update_progress(self, value):
        # 修改4：进度条改为浮动窗口
        if not hasattr(self, 'progress_dialog') or not self.progress_dialog.isVisible():
            self.progress_dialog = ProgressDialog(1, "Processing...", self)
            self.progress_dialog.show()
        self.progress_dialog.update_progress(value / 100)
        if value >= 100:
            self.progress_dialog.accept()

    def open_settings(self):
        dialog = SettingsDialog(self.threshold, self)
        dialog.exec_()

    def update_threshold(self, new_threshold):
        self.threshold = new_threshold
        self.config["threshold"] = self.threshold
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True)
        if self.result_paths and self.detection_infos:
            for i in range(len(self.detection_infos)):
                score = float(self.detection_infos[i].split("异常得分: ")[1].split(" - ")[0])
                self.detection_infos[i] = f"异常得分: {score:.2f} - {'检测到异常' if score > self.threshold else '图像正常'}"
            self.detection_info_label.setText(f"检测信息: {self.detection_infos[self.current_index]}")
            self.update_thumbnails()
        self.log_text.append(f"阈值已更新为: {self.threshold}")

    def update_threshold_directly(self, value):
        # 修改2：处理滑块值的转换（整数转为浮点数）
        self.update_threshold(value / 100.0)

    def update_thumbnails(self):
        self.thumbnail_list.clear()
        if not self.result_paths:
            return
        for i, path in enumerate(self.result_paths):
            pixmap = QPixmap(path).scaled(80, 80, Qt.KeepAspectRatio)
            item = QListWidgetItem(QIcon(pixmap), "")
            item.setData(Qt.UserRole, i)
            item.setToolTip(f"{os.path.basename(path)}\n{self.detection_infos[i]}")  # 修改3：添加悬浮提示
            self.thumbnail_list.addItem(item)
        if self.current_index < self.thumbnail_list.count():
            self.thumbnail_list.setCurrentRow(self.current_index)

    def toggle_log(self):
        # 修改4：切换日志区显示
        is_visible = self.log_text.isVisible()
        self.log_text.setVisible(not is_visible)
        self.log_toggle_button.setText("Hide Log" if not is_visible else "Show Log")