# Anomaly Detection App 文档

## 项目概述

**Anomaly Detection App** 是一个基于 Python 和 PyQt5 的桌面应用程序，利用深度学习模型（如 Wide ResNet50 和 SimpleNet）进行图像异常检测。它支持单张和批量图片处理，提供直观的用户界面，用户可以选择模型、调整异常阈值、查看检测结果、分析报告并管理检测历史。应用通过多线程、GPU 加速和动态内存管理优化了性能，适用于工业检测、质量控制或其他需要图像异常分析的场景。

### 示例图

- **主界面**  

![image.png](https://img-1313049298.cos.ap-shanghai.myqcloud.com/note-img/202503212158873.png)


- **单张检测**  

![image.png](https://img-1313049298.cos.ap-shanghai.myqcloud.com/note-img/202503212158874.png)


- **批量检测与报告**  

![image.png](https://img-1313049298.cos.ap-shanghai.myqcloud.com/note-img/202503212158875.png)

![image.png](https://img-1313049298.cos.ap-shanghai.myqcloud.com/note-img/202503212158876.png)


  *注：新版本增加了“Report”选项卡，显示统计和图表。*

- **历史记录查看**  

![image.png](https://img-1313049298.cos.ap-shanghai.myqcloud.com/note-img/202503212158877.png)


### 主要功能
- **模型选择**：通过“Select Model”菜单选择预定义模型（如“Metal Nut”或“Screw”），支持快捷键（如 Ctrl+1）。
- **单张检测**：检测单张图片，显示结果和异常信息，支持拖放操作。
- **批量检测**：处理文件夹中的多张图片，生成检测结果、缩略图和分析报告，支持拖放文件夹。
- **结果导航**：通过缩略图点击切换批量检测结果，鼠标悬停显示详细信息。
- **报告生成**：批量检测后生成统计报告（异常数量、得分分布等）和图表（直方图、箱线图），支持导出为 CSV 或 PDF。
- **历史记录**：保存每次检测的报告，支持查看和重新加载历史结果。
- **动态阈值**：通过左侧滑块实时调整异常检测阈值（默认范围 0.0-2.0，默认值 1.2）。
- **性能优化**：支持动态批次大小、GPU 加速、多线程预加载和磁盘缓存，减少内存占用和处理时间。
- **日志记录**：可折叠日志区显示操作和错误信息，保存至文件。
- **拖放支持**：支持拖放图片或文件夹触发检测。

### 技术栈
- **编程语言**：Python 3.8+
- **GUI 框架**：PyQt5
- **深度学习**：PyTorch, torchvision
- **图像处理**：Pillow, NumPy, Matplotlib
- **数据分析**：pandas, NumPy
- **报告生成**：reportlab（PDF 导出）
- **其他**：tqdm（进度条）、logging（日志）、ruamel.yaml（配置文件）、concurrent.futures（多线程）

---

## 安装与运行

### 环境要求
- **操作系统**：Windows、macOS 或 Linux（跨平台支持）
- **Python**：3.8 或更高版本
- **硬件**：支持 CUDA 的 GPU（可选，推荐至少 6 GiB 显存以处理批量检测）

### 依赖安装
安装所需库：
```bash
pip install PyQt5 torch torchvision pillow numpy matplotlib tqdm ruamel.yaml pandas reportlab
```
- **torch**：若需 GPU 支持，请根据硬件安装对应 CUDA 版本（参考 [PyTorch 官网](https://pytorch.org/)）。
- **simplenet**：从 [GitHub](https://github.com/DonaldRR/SimpleNet) 克隆并置于项目目录，或根据需求修改 `simplenet.py`。

### 项目文件
下载项目代码并确保以下结构完整：
```
anomaly_detection_app/
├── README.md
├── .gitignore
├── config.yaml
├── requirements.txt
├── main.py
├── data/
│   └── (数据文件)
├── logs/
│   └── (日志文件)
├── output/
│   └── (输出文件)
├── models/
│   └── (训练好的模型文件)
├── src/
│   ├── __init__.py
│   ├── backbones/
│   │   ├── __init__.py
│   │   ├── resnet.py
│   │   ├── simplenet.py
│   │   └── backones.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── exceptions.py
│   │   ├── utils.py
│   │   └── common.py
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── gui.py
│   │   └── progress_dialog.py
│   ├── image_processing/
│   │   ├── __init__.py
│   │   └── image_processor.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── model_loading/
│   │   ├── __init__.py
│   │   └── model_loader.py
│   ├── performance_monitoring/
│   │   ├── __init__.py
│   │   └── performance_monitor.py
│   ├── report_generation/
│   │   ├── __init__.py
│   │   └── report_generator.py
│   └── tests/
│       ├── __init__.py
│       └── test_image_detection.py
```

### 配置
编辑 `config.yaml`：
```yaml
# Anomaly Detection App 的配置文件
load_mode: preload  # 可选：preload（预加载）或 ondemand（按需加载）
threshold: 1.2      # 默认阈值，范围 0.0-2.0
models:
  Metal Nut: models/mvtec_metal_nut/ckpt.pth  # 金属螺母检测模型
  Screw: models/mvtec_screw/ckpt.pth          # 螺丝检测模型
preload:
  max_preload_threads: 8    # 预加载最大线程数
  preload_chunk_size: 100   # 预加载分片大小
  use_disk_cache: true      # 是否使用磁盘缓存
  max_memory_mb: 2048       # 最大内存使用量（MB）
batch:
  max_batch_threads: 12     # 批量检测最大线程数
  max_batch_size: 32        # 最大批次大小
```
- **load_mode**：`preload` 在启动时加载所有模型，`ondemand` 在选择时加载。
- **threshold**：初始异常阈值，可通过界面调整。
- **models**：模型名称和路径，路径需指向有效的 `.pth` 文件。
- **preload/batch**：性能优化参数，可根据硬件调整。

### 运行
1. 确保模型文件路径正确。
2. 进入项目目录：
   ```bash
   cd anomaly_detection_app
   ```
3. 执行：
   ```bash
   python main.py
   ```

---

## 代码结构与模块说明

### 文件结构
- **`main.py`**：
  - 程序入口，配置日志，加载 `config.yaml`，初始化设备和处理器，启动 GUI。
- **`gui.py`**：
  - 定义图形界面，包括工具栏、选项卡（图像视图和报告）、缩略图、阈值滑块、历史记录对话框和日志区。
  - 处理用户交互（模型选择、检测、阈值调整、报告导出、历史查看）。
- **`image_processor.py`**：
  - 图像处理核心，支持单张和批量检测，管理模型缓存和性能优化。
- **`model_loader.py`**：
  - 模型加载逻辑，定义 `load_model` 函数。
- **`simplenet.py`**：
  - SimpleNet 模型实现，用于异常检测和热图生成。
- **`progress_dialog.py`**：
  - 进度对话框，用于模型加载和批量检测进度显示。
- **`report_generator.py`**：
  - 报告生成模块，统计检测结果，生成图表并支持导出。
- **`performance_monitor.py`**：
  - 性能监控模块，记录内存、线程和 I/O 使用情况。
- **`exceptions.py`**：
  - 自定义异常类，用于错误处理。
- **`config.yaml`**：
  - 配置文件，存储模型列表、加载模式、阈值和性能参数。

### 核心类与功能
1. **`MainWindow` (gui.py)**：
   - **属性**：
     - `processor`：图像处理器实例。
     - `result_paths`：检测结果路径列表。
     - `detection_infos`：检测信息列表。
     - `current_index`：当前显示图片索引。
     - `current_report`：当前报告数据。
     - `threshold`：异常阈值。
   - **方法**：
     - `select_model`：选择并加载模型。
     - `detect_single` / `detect_batch`：触发单张或批量检测。
     - `detect_single_drop` / `detect_batch_drop`：处理拖放检测。
     - `update_report`：更新报告选项卡和图像视图。
     - `export_report`：导出报告为 CSV 或 PDF。
     - `open_history`：打开历史记录对话框。
   - **界面元素**：
     - 工具栏（“Select Model”、“Detect”、“Options”）。
     - 选项卡（“Image View”和“Report”）。
     - 状态栏、缩略图、阈值滑块、日志区。

2. **`ImageProcessor` (image_processor.py)**：
   - **属性**：
     - `model_cache`：已加载模型缓存。
     - `current_model_name`：当前模型名称。
     - `report_generator`：报告生成器实例。
   - **方法**：
     - `set_model`：设置当前模型。
     - `detect_single_image`：单张检测，返回路径和信息。
     - `detect_batch_images`：批量检测，生成报告。
   - **信号**：
     - `report_generated`：批量检测完成后发出报告数据。
   - **优化**：
     - 动态批次大小（`BatchDetectWorker`）。
     - 多线程预加载（`ImagePreloader`）。

3. **`ReportGenerator` (report_generator.py)**：
   - **属性**：
     - `output_dir`：报告保存目录。
     - `history_file`：历史记录文件路径。
   - **方法**：
     - `generate_report`：生成统计信息和图表。
     - `save_history`：保存检测报告到历史记录。
     - `export_to_csv` / `export_to_pdf`：导出报告。
   - **功能**：
     - 统计异常数量、得分分布和异常程度。
     - 生成直方图和箱线图。

4. **`HistoryDialog` (gui.py)**：
   - **功能**：显示历史记录列表，支持双击或“OK”加载报告。
   - **方法**：
     - `load_history`：加载历史记录。
     - `load_selected_report`：加载选中的报告。

---

## 使用说明

### 界面布局
- **工具栏**：
  - “Select Model”：选择模型，支持快捷键。
  - “Detect”：下拉菜单，包含“Single Image”和“Batch Images”。
  - “Options”：包含“Settings”和“View History”。
- **状态栏**：显示“Model: [模型名] | Image: [图片名] | GPU Memory: [显存使用]”。
- **左侧区域**：
  - “Anomaly Threshold”：滑块调节阈值，实时更新。
  - “检测信息”：显示异常得分和判断。
- **右侧区域**：
  - **选项卡**：
    - “Image View”：显示检测结果（原图+热图）和缩略图。
    - “Report”：显示统计信息、图表和导出按钮。
  - **缩略图列表**：水平滚动，点击切换，悬停显示信息。
  - **日志区**：可折叠，默认隐藏。
- **浮动进度条**：检测或加载时显示进度。

### 操作流程
1. **启动程序**：
   - `preload` 模式：加载所有模型，显示进度。
   - `ondemand` 模式：显示“未选择模型”。
2. **选择模型**：
   - 点击“Select Model”或按快捷键。
   - 状态栏更新模型名。
3. **单张检测**：
   - 点击“Detect -> Single Image”或拖放图片。
   - 显示结果和信息，缩略图显示单张。
4. **批量检测**：
   - 点击“Detect -> Batch Images”或拖放文件夹。
   - 显示第一张结果，缩略图列出所有结果，报告选项卡显示统计和图表。
5. **查看报告**：
   - 批量检测后自动切换到“Report”选项卡。
   - 点击“Export Report”选择保存路径和格式（CSV/PDF）。
6. **查看历史**：
   - 点击“Options -> View History”。
   - 选择记录，双击或点击“OK”加载。
7. **调整阈值**：
   - 滑动阈值滑块，实时更新检测信息和 `config.yaml`。
8. **查看日志**：
   - 点击“Show Log”展开日志区。

### 输出结果
- **检测结果**：`./output/[模型目录名]/detection_[输入文件名].png`
- **报告文件**：
  - 默认保存至 `./output/reports/`，可通过导出选择其他路径。
  - CSV：包含图像路径、得分和状态。
  - PDF：包含统计信息和图表。
- **历史记录**：`./output/reports/detection_history.json`

---

## 维护与扩展

### 常见问题排查
1. **导出报告失败（KeyError: 'image_paths'）**：
   - 检查历史记录文件是否完整（需包含 `image_paths` 和 `scores`）。
   - 重新运行批量检测生成新记录。
2. **历史记录加载无效**：
   - 确保 `detection_history.json` 未损坏。
   - 检查日志确认加载错误。
3. **内存不足**：
   - 调整 `config.yaml` 中的 `max_batch_size` 或 `max_memory_mb`。
   - 设置 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`。
4. **拖放无效**：
   - 检查日志，确保拖放事件被识别。
   - 以管理员权限运行程序（Windows）。

### 优化建议
1. **性能提升**：
   - 使用异步图表生成（`ProgressWorker`）。
   - 优化 `SimpleNet.predict` 支持更大批量。
2. **界面增强**：
   - 添加图表交互（放大、悬停提示，使用 PyQtGraph）。
   - 支持历史记录筛选（按模型或日期）。
3. **功能扩展**：
   - 添加异常原因分析（需模型支持）。
   - 支持清理旧历史记录（按数量或时间）。
4. **模型支持**：
   - 抽象模型接口，支持其他架构（如 YOLO）。

### 代码维护
- **模块化**：GUI、处理、报告和模型逻辑分离。
- **配置文件**：支持动态调整参数（如异常程度阈值）。
- **日志**：详细记录操作和错误。
- **测试**：建议使用 `pytest` 测试检测、报告和历史功能。

---

## 版本信息
- **当前版本**：v1.8
- **更新日期**：2025年3月21日
- **作者**：LucaZou
- **更新记录**：
  - v1.0：基本检测功能。
  - v1.1：多线程支持，界面优化。
  - v1.2：动态阈值、缩略图导航。
  - v1.4：YAML 配置、GPU 加速、内存优化。
  - v1.5：拖放支持、滑块阈值调整。
  - v1.7：报告模块、历史记录管理、导出功能。
  - v1.8：重构项目文件结构