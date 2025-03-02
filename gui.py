from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QFormLayout, QDialog,
    QDoubleSpinBox,QTextEdit, QProgressBar, QFileDialog, QAction, QToolBar, QMenu, QPushButton,
    QListWidget, QListWidgetItem, QSizePolicy, QSpacerItem, QScrollArea)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QSize
import os

# 定义设置窗口类
class SettingsDialog(QDialog):
    threshold_changed = pyqtSignal(float)  # 信号：阈值更改时通知主窗口

    def __init__(self, current_threshold, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.init_ui(current_threshold)

    def init_ui(self, current_threshold):
        layout = QFormLayout()

        # 动态阈值配置
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.0, 2.0)  # 阈值范围 0.0 - 1.0
        self.threshold_spinbox.setSingleStep(0.05)  # 步长 0.01
        self.threshold_spinbox.setValue(current_threshold)  # 默认值
        layout.addRow("Anomaly Threshold:", self.threshold_spinbox)

        # 保存按钮
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)

        self.setLayout(layout)

    def save_settings(self):
        threshold = self.threshold_spinbox.value()
        self.threshold_changed.emit(threshold)  # 发射信号通知阈值更新
        self.accept()  # 关闭对话框

class MainWindow(QMainWindow):
    # 主窗口类，定义 GUI 界面和交互逻辑
    def __init__(self, processor, config):
        super().__init__()
        self.processor = processor # 图像处理器
        self.config = config # 配置信息
        self.result_paths = [] # 批量检测结果路径
        self.current_index = 0 # 当前显示的结果索引
        self.current_model_name = "未选择模型" # 当前模型名称
        self.detection_infos = []  # 存储检测信息
        self.threshold = 1.2  # 新增：默认阈值，初始为 0.5
        # 缩略图分页相关属性
        # self.thumbnails_per_page = 5  # 每页显示 5 个缩略图
        # self.current_page = 0  # 当前页码
        self.init_ui() # 初始化界面
        self.connect_signals() # 连接信号和槽

    def init_ui(self):
        # 初始化界面
        self.setWindowTitle("Anomaly Detection App")
        self.setGeometry(100, 100, 600, 700) # 设置窗口大小

        # 工具栏
        toolbar = QToolBar("Tools")
        self.addToolBar(toolbar)
        
        # 模型选择菜单
        self.model_menu = QMenu("Select Model", self)
        model_action = QAction("Select Model", self)
        model_action.setMenu(self.model_menu)
        self.models = self.config["models"]
        for model_name in self.models.keys():
            self.model_menu.addAction(model_name, lambda name=model_name: self.select_model(name))
        toolbar.addAction(model_action)

        # 检测按钮
        self.single_action = QAction("Detect Single Image", self)
        self.batch_action = QAction("Detect Batch Images", self)
        toolbar.addAction(self.single_action)
        toolbar.addAction(self.batch_action)

        # 设置菜单
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.open_settings)
        toolbar.addAction(settings_action)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 状态信息
        self.model_label = QLabel(f"当前模型: {self.current_model_name}")
        self.image_label_info = QLabel("当前图片: 未加载")
        self.detection_info_label = QLabel("检测信息: 未检测")
        layout.addWidget(self.model_label)
        layout.addWidget(self.image_label_info)
        layout.addWidget(self.detection_info_label)

        # 图像显示区
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(576, 288) # 固定显示区域大小
        layout.addWidget(self.image_label)

        # 切换按钮
        button_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.prev_button.clicked.connect(self.prev_image)
        self.next_button.clicked.connect(self.next_image)
        self.prev_button.setEnabled(False) # 默认不可用
        self.next_button.setEnabled(False)
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        layout.addLayout(button_layout)

        # 修改：使用 QScrollArea 包裹缩略图列表，确保可见
        self.thumbnail_scroll = QScrollArea()
        thumbnail_widget = QWidget()
        thumbnail_layout = QVBoxLayout(thumbnail_widget)

        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setFlow(QListWidget.LeftToRight)
        self.thumbnail_list.setIconSize(QSize(100, 100))
        self.thumbnail_list.setMinimumHeight(120)  # 最小高度
        self.thumbnail_list.setMaximumWidth(600)  # 最大宽度，避免过长
        self.thumbnail_list.itemClicked.connect(self.select_thumbnail)
        thumbnail_layout.addWidget(self.thumbnail_list)

        thumbnail_widget.setLayout(thumbnail_layout)
        self.thumbnail_scroll.setWidget(thumbnail_widget)
        self.thumbnail_scroll.setWidgetResizable(True)  # 允许滚动
        self.thumbnail_scroll.setMinimumHeight(150)  # 确保可见
        self.thumbnail_scroll.setVisible(False)  # 初始隐藏
        layout.addWidget(self.thumbnail_scroll)

        # 修改：添加间隔，确保缩略图区域不被压缩
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False) # 默认隐藏
        layout.addWidget(self.progress_bar)

        # 日志区
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True) # 只读
        layout.addWidget(self.log_text)

        # 初始化时更新缩略图,初始化时不显示缩略图
        self.update_thumbnails()

    def connect_signals(self):
        # 连接信号和槽
        self.single_action.triggered.connect(self.detect_single)
        self.batch_action.triggered.connect(self.detect_batch)
        self.processor.log_message.connect(self.update_log)
        self.processor.progress_updated.connect(self.update_progress)
        self.processor.batch_finished.connect(self.show_batch_results)

    def select_model(self, model_name):
        # 选择模型
        model_path = self.models.get(model_name)
        if model_path:
            try:
                self.processor.set_model(model_name, model_path) # 设置当前模型
                self.current_model_name = model_name
                self.model_label.setText(f"当前模型: {self.current_model_name}")
                self.log_text.append(f"模型已切换为: {model_name} ({model_path})")
            except Exception as e:
                error_msg = f"加载模型失败: {str(e)}"
                self.log_text.append(error_msg)
                # 新增：显示错误提示框
                QMessageBox.critical(self, "错误", error_msg)

    def detect_single(self):
        # 检测单张图片
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg)")
        if file_path:
            output_path, detection_info = self.processor.detect_single_image(file_path, self.threshold) # 接收检测信息，传递阈值
            if output_path:
                self.show_result(output_path)
                self.result_paths = [output_path]
                self.detection_infos = [detection_info]  # 存储检测信息
                self.current_index = 0
                self.update_buttons(single_mode=False)
                self.image_label_info.setText(f"当前图片: {os.path.basename(output_path)}")
                self.detection_info_label.setText(f"检测信息: {detection_info}")  # 新增：更新检测信息
                # 修改：单张检测时隐藏缩略图
                self.thumbnail_scroll.setVisible(False)
                self.update_thumbnails()  # 更新缩略图

    def detect_batch(self):
        # 批量检测文件夹中的图片
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.progress_bar.setVisible(True)
            self.processor.detect_batch_images(folder_path, self.threshold)  # 传递阈值

    def show_result(self, output_path):
        # 显示检测结果
        pixmap = QPixmap(output_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def show_batch_results(self, results):
        # 接收路径和检测信息
        output_paths, detection_infos = results
        self.result_paths = output_paths
        self.detection_infos = detection_infos
        self.current_index = 0
        if self.result_paths:
            self.show_result(self.result_paths[self.current_index])
            self.update_buttons(single_mode=False)
            self.image_label_info.setText(f"当前图片: {os.path.basename(self.result_paths[self.current_index])}")
            self.detection_info_label.setText(f"检测信息: {self.detection_infos[self.current_index]}")  # 新增：显示检测信息
            # 修改：批量检测时显示缩略图
            self.thumbnail_scroll.setVisible(True)
            self.update_thumbnails()  # 更新缩略图
        self.progress_bar.setVisible(False)

    def prev_image(self):
        # 显示上一张图片
        if self.current_index > 0:
            self.current_index -= 1
            self.show_result(self.result_paths[self.current_index])
            self.image_label_info.setText(f"当前图片: {os.path.basename(self.result_paths[self.current_index])}")
            # 修改：同步更新检测信息
            self.detection_info_label.setText(f"检测信息: {self.detection_infos[self.current_index]}")
            self.update_buttons(single_mode=False)
            self.sync_thumbnail_selection()  # 同步缩略图选中状态

    def next_image(self):
        # 显示下一张图片
        if self.current_index < len(self.result_paths) - 1:
            self.current_index += 1
            self.show_result(self.result_paths[self.current_index])
            self.image_label_info.setText(f"当前图片: {os.path.basename(self.result_paths[self.current_index])}")
            # 修改：同步更新检测信息
            self.detection_info_label.setText(f"检测信息: {self.detection_infos[self.current_index]}")
            self.update_buttons(single_mode=False)
            self.sync_thumbnail_selection()  # 同步缩略图选中状态

    def update_buttons(self, single_mode=False):
        # 更新切换按钮状态
        if single_mode:
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(self.current_index < len(self.result_paths) - 1)
        else:
            self.prev_button.setEnabled(self.current_index > 0)
            self.next_button.setEnabled(self.current_index < len(self.result_paths) - 1)

    def update_log(self, message):
        # 更新日志信息
        self.log_text.append(message)

    def update_progress(self, value):
        # 更新进度条
        self.progress_bar.setValue(value)

    # 新增：打开设置窗口
    def open_settings(self):
        dialog = SettingsDialog(self.threshold, self)
        dialog.threshold_changed.connect(self.update_threshold)  # 连接信号
        dialog.exec_()

    # 新增：更新阈值并刷新检测信息
    def update_threshold(self, new_threshold):
        self.threshold = new_threshold
        self.log_text.append(f"阈值已更新为: {self.threshold}")
        # 如果已有检测结果，重新计算检测信息
        if self.result_paths and self.detection_infos:
            for i in range(len(self.detection_infos)):
                score = float(self.detection_infos[i].split("异常得分: ")[1].split(" - ")[0])
                self.detection_infos[i] = f"异常得分: {score:.2f} - {'检测到异常' if score > self.threshold else '图像正常'}"
            self.detection_info_label.setText(f"检测信息: {self.detection_infos[self.current_index]}")
            self.update_thumbnails()  # 更新缩略图显示

    # 新增：更新缩略图列表
    def update_thumbnails(self):
        self.thumbnail_list.clear()
        if not self.result_paths:
            return

        self.log_text.append(f"更新缩略图: 显示所有 {len(self.result_paths)} 张图片")
        for i in range(len(self.result_paths)):
            try:
                pixmap = QPixmap(self.result_paths[i]).scaled(100, 100, Qt.KeepAspectRatio)
                item = QListWidgetItem(QIcon(pixmap),"")
                item.setData(Qt.UserRole, i)
                self.thumbnail_list.addItem(item)
            except Exception as e:
                self.log_text.append(f"加载缩略图失败 {self.result_paths[i]}: {str(e)}")
        
        # 选中当前图片
        self.sync_thumbnail_selection()

    # 同步缩略图选中状态
    def sync_thumbnail_selection(self):
        for i in range(self.thumbnail_list.count()):
            item = self.thumbnail_list.item(i)
            idx = item.data(Qt.UserRole)
            if idx == self.current_index:
                self.thumbnail_list.setCurrentItem(item)
                break

    # 选择缩略图跳转
    def select_thumbnail(self, item):
        self.current_index = item.data(Qt.UserRole)
        self.show_result(self.result_paths[self.current_index])
        self.image_label_info.setText(f"当前图片: {os.path.basename(self.result_paths[self.current_index])}")
        self.detection_info_label.setText(f"检测信息: {self.detection_infos[self.current_index]}")
        self.update_buttons(single_mode=False)