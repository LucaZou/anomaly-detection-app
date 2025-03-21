from PyQt5.QtWidgets import QDialog, QVBoxLayout, QProgressBar, QLabel, QPushButton, QApplication
from PyQt5.QtCore import Qt, pyqtSignal, QThread

class ProgressWorker(QThread):
    """模型加载的工作线程"""
    progress_updated = pyqtSignal(int)  # 进度更新信号
    finished = pyqtSignal(object)      # 完成信号，传递加载结果

    def __init__(self, load_func, *args, **kwargs):
        super().__init__()
        self.load_func = load_func  # 加载函数
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.load_func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(e)

class ProgressDialog(QDialog):
    """通用的进度对话框，支持单个或多个任务"""
    def __init__(self, total_tasks, description="Loading...", parent=None):
        super().__init__(parent)
        self.total_tasks = total_tasks  # 总任务数
        self.current_progress = 0       # 当前进度
        self.setWindowTitle("Loading Progress")
        self.setModal(True)             # 模态对话框，阻塞父窗口
        self.setFixedSize(300, 150)
        self.init_ui(description)

    def init_ui(self, description):
        layout = QVBoxLayout()

        # 描述标签
        self.label = QLabel(description, self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # 进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 取消按钮（可选）
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

    def update_progress(self, increment):
        """更新进度，每次任务完成时调用"""
        self.current_progress += increment
        progress = int((self.current_progress / self.total_tasks) * 100)
        self.progress_bar.setValue(progress)
        QApplication.processEvents()  # 强制刷新 UI

    def set_description(self, text):
        """动态更新描述文本"""
        self.label.setText(text)
        QApplication.processEvents()  # 强制刷新 UI