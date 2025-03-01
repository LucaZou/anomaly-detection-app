from PyQt5.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, 
                             QTextEdit, QProgressBar, QFileDialog, QAction, QToolBar)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        self.setWindowTitle("Anomaly Detection App")
        self.setGeometry(100, 100, 800, 600)

        # 工具栏
        toolbar = QToolBar("Tools")
        self.addToolBar(toolbar)
        self.single_action = QAction("Detect Single Image", self)
        self.batch_action = QAction("Detect Batch Images", self)
        toolbar.addAction(self.single_action)
        toolbar.addAction(self.batch_action)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 图像显示区
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(576, 288)  # 适应并排显示的combined_image
        layout.addWidget(self.image_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 日志区
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

    def connect_signals(self):
        self.single_action.triggered.connect(self.detect_single)
        self.batch_action.triggered.connect(self.detect_batch)
        self.processor.log_message.connect(self.update_log)
        self.processor.progress_updated.connect(self.update_progress)

    def detect_single(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg)")
        if file_path:
            output_path = self.processor.detect_single_image(file_path)
            if output_path:
                pixmap = QPixmap(output_path)
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def detect_batch(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.progress_bar.setVisible(True)
            self.processor.detect_batch_images(folder_path)
            self.progress_bar.setVisible(False)

    def update_log(self, message):
        self.log_text.append(message)

    def update_progress(self, value):
        self.progress_bar.setValue(value)