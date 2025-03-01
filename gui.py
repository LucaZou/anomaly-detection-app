from PyQt5.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, 
                             QTextEdit, QProgressBar, QFileDialog, QAction, QToolBar, QMenu, QPushButton)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os

class MainWindow(QMainWindow):
    # 主窗口类，定义 GUI 界面和交互逻辑
    def __init__(self, processor, config):
        super().__init__()
        self.processor = processor # 图像处理器
        self.config = config # 配置信息
        self.result_paths = [] # 批量检测结果路径
        self.current_index = 0 # 当前显示的结果索引
        self.current_model_name = "未选择模型" # 当前模型名称
        self.init_ui() # 初始化界面
        self.connect_signals() # 连接信号和槽

    def init_ui(self):
        # 初始化界面
        self.setWindowTitle("Anomaly Detection App")
        self.setGeometry(100, 100, 800, 600)

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

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 状态信息
        self.model_label = QLabel(f"当前模型: {self.current_model_name}")
        self.image_label_info = QLabel("当前图片: 未加载")
        layout.addWidget(self.model_label)
        layout.addWidget(self.image_label_info)

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

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False) # 默认隐藏
        layout.addWidget(self.progress_bar)

        # 日志区
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True) # 只读
        layout.addWidget(self.log_text)

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
                self.log_text.append(f"加载模型失败: {str(e)}")

    def detect_single(self):
        # 检测单张图片
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg)")
        if file_path:
            output_path = self.processor.detect_single_image(file_path)
            if output_path:
                self.show_result(output_path)
                self.result_paths = [output_path]
                self.current_index = 0
                self.update_buttons(single_mode=False)
                self.image_label_info.setText(f"当前图片: {os.path.basename(output_path)}")

    def detect_batch(self):
        # 批量检测文件夹中的图片
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.progress_bar.setVisible(True)
            self.processor.detect_batch_images(folder_path)
            self.progress_bar.setVisible(False)

    def show_result(self, output_path):
        # 显示检测结果
        pixmap = QPixmap(output_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def show_batch_results(self, output_paths):
        # 显示批量检测结果
        self.result_paths = output_paths
        self.current_index = 0
        if self.result_paths:
            self.show_result(self.result_paths[self.current_index])
            self.update_buttons(single_mode=False)
            self.image_label_info.setText(f"当前图片: {os.path.basename(self.result_paths[self.current_index])}")

    def prev_image(self):
        # 显示上一张图片
        if self.current_index > 0:
            self.current_index -= 1
            self.show_result(self.result_paths[self.current_index])
            self.image_label_info.setText(f"当前图片: {os.path.basename(self.result_paths[self.current_index])}")
            self.update_buttons(single_mode=False)

    def next_image(self):
        # 显示下一张图片
        if self.current_index < len(self.result_paths) - 1:
            self.current_index += 1
            self.show_result(self.result_paths[self.current_index])
            self.image_label_info.setText(f"当前图片: {os.path.basename(self.result_paths[self.current_index])}")
            self.update_buttons(single_mode=False)

    def update_buttons(self, single_mode=False):
        # 更新切换按钮状态
        if single_mode:
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
        else:
            self.prev_button.setEnabled(self.current_index > 0)
            self.next_button.setEnabled(self.current_index < len(self.result_paths) - 1)

    def update_log(self, message):
        # 更新日志信息
        self.log_text.append(message)

    def update_progress(self, value):
        # 更新进度条
        self.progress_bar.setValue(value)