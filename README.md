# Anomaly Detection App 文档

## 项目概述

**Anomaly Detection App** 是一个基于 Python 和 PyQt5 的桌面应用程序，旨在利用深度学习模型（如 Wide ResNet50 和 SimpleNet）进行图像异常检测。它支持单张和批量图片处理，提供简洁高效的用户界面，用户可以选择模型、调整异常阈值、查看检测结果，并通过缩略图导航批量结果。应用优化了性能，支持动态内存管理和 GPU 加速，适用于工业检测或其他需要图像异常分析的场景。

### 示例图

- **主界面**  

![image.png](https://img-1313049298.cos.ap-shanghai.myqcloud.com/note-img/202503041541440.png)


- **单张检测**  

![image.png](https://img-1313049298.cos.ap-shanghai.myqcloud.com/note-img/202503041542105.png)

  
- **批量检测**  

![image.png](https://img-1313049298.cos.ap-shanghai.myqcloud.com/note-img/202503041544119.png)


### 主要功能
- **模型选择**：通过“Select Model”下拉菜单选择预定义模型（如“Metal Nut”或“Screw”），支持快捷键（如 Ctrl+1）。
- **单张检测**：检测单张图片，显示结果和异常信息，支持拖放操作。
- **批量检测**：处理文件夹中的多张图片，显示结果并提供缩略图导航，支持拖放文件夹。
- **结果导航**：通过缩略图点击切换批量检测结果（无独立切换按钮）。
- **状态显示**：状态栏显示当前模型和图片名称，左侧显示异常得分及判断。
- **动态阈值**：通过左侧滑块实时调整异常检测阈值（范围 0.0-2.0，默认 1.2）。
- **性能优化**：支持动态批次大小、GPU 加速和多线程预加载，减少内存占用和处理时间。
- **日志记录**：可折叠日志区显示操作和错误信息，保存至文件。
- **拖放支持**：支持将图片或文件夹拖入窗口触发检测。

### 技术栈
- **编程语言**：Python 3.8+
- **GUI 框架**：PyQt5
- **深度学习**：PyTorch, torchvision
- **图像处理**：Pillow, NumPy, Matplotlib
- **其他**：tqdm（进度条）、logging（日志）、pyyaml（配置文件）、concurrent.futures（多线程）

---

## 安装与运行

### 环境要求
- **操作系统**：Windows、macOS 或 Linux（跨平台支持）
- **Python**：3.8 或更高版本
- **硬件**：支持 CUDA 的 GPU（可选，推荐至少 4 GiB 显存以提升性能）

### 依赖安装
安装所需库：
```bash
pip install PyQt5 torch torchvision pillow numpy matplotlib tqdm pyyaml
```
- **torch**：若需 GPU 支持，请根据硬件安装对应 CUDA 版本（参考 [PyTorch 官网](https://pytorch.org/)）。
- **simplenet**：如需使用 `simplenet.py`，从 [GitHub](https://github.com/DonaldRR/SimpleNet) 克隆并安装。

### 项目文件
下载项目代码并确保以下结构完整：
```
anomaly_detection_app/
├── main.py
├── gui.py
├── image_processor.py
├── model_loader.py
├── simplenet.py
├── progress_dialog.py
├── config.yaml
├── logs/              # 日志保存目录（自动创建）
└── output/            # 检测结果保存目录（自动创建）
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
```
- **load_mode**：`preload` 在启动时加载所有模型，`ondemand` 在选择时加载。
- **threshold**：初始异常阈值，可通过界面滑块调整。
- **models**：模型名称和路径，路径需指向有效的 `.pth` 文件。

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
  - 程序入口，加载配置，初始化设备和处理器，启动 GUI。
- **`gui.py`**：
  - 定义图形界面，包括工具栏、状态栏、图像显示区、缩略图、阈值滑块和日志区。
  - 处理用户交互（模型选择、检测、阈值调整、拖放）。
- **`image_processor.py`**：
  - 图像处理逻辑，支持单张和批量检测，优化 GPU 内存和 I/O。
- **`model_loader.py`**：
  - 模型加载逻辑，定义 `load_model` 函数。
- **`simplenet.py`**：
  - 实现 SimpleNet 模型，用于异常检测和热图生成。
- **`progress_dialog.py`**：
  - 定义进度对话框，用于模型加载和检测进度显示。
- **`config.yaml`**：
  - 配置文件，存储模型列表、加载模式和阈值。

### 核心类与功能
1. **`MainWindow` (gui.py)**：
   - **属性**：
     - `processor`：图像处理器实例。
     - `result_paths`：检测结果路径列表。
     - `detection_infos`：检测信息列表。
     - `current_index`：当前显示图片索引。
     - `threshold`：异常阈值（通过滑块调整）。
   - **方法**：
     - `select_model`：选择并加载模型。
     - `detect_single` / `detect_batch`：触发单张或批量检测。
     - `detect_single_drop` / `detect_batch_drop`：处理拖放检测。
     - `update_threshold`：更新阈值并刷新结果。
     - `update_status_bar`：更新状态栏信息。
   - **界面元素**：
     - 工具栏（“Select Model”、“Detect”、“Options”菜单）。
     - 状态栏（模型名、图片名）。
     - 左侧（阈值滑块、检测信息）。
     - 右侧（图像显示区、缩略图、日志区）。

2. **`ImageProcessor` (image_processor.py)**：
   - **属性**：
     - `model_cache`：已加载模型缓存。
     - `current_model_name`：当前模型名称。
     - `output_base_dir`：结果输出目录。
   - **方法**：
     - `set_model`：设置当前模型。
     - `detect_single_image`：单张检测，返回路径和信息。
     - `detect_batch_images`：批量检测，使用线程处理。
   - **优化**：
     - 动态 `batch_size` 根据 GPU 内存。
     - 多线程预加载图片。
     - GPU 内存清理（`torch.cuda.empty_cache()`）。

3. **`BatchDetectWorker` (image_processor.py)**：
   - **功能**：异步批量检测线程。
   - **优化**：
     - 动态调整批次大小。
     - 并行生成热图。

4. **`SimpleNet` (simplenet.py)**：
   - **功能**：异常检测模型，基于 Wide ResNet50。
   - **方法**：
     - `load`：加载模型权重。
     - `predict`：预测异常得分和热图。

---

## 使用说明

### 界面布局
- **工具栏**：
  - “Select Model”：选择模型，支持快捷键（Ctrl+1 等）。
  - “Detect”：下拉菜单，包含“Single Image”和“Batch Images”。
  - “Options”：包含“Settings”（当前为空壳）。
- **状态栏**：显示“Model: [模型名] | Image: [图片名]”。
- **左侧区域**：
  - “Anomaly Threshold”：滑块调节阈值（0.0-2.0），显示当前值。
  - “检测信息”：显示异常得分和判断（如“异常得分: 1.50 - 检测到异常”）。
- **右侧区域**：
  - **图像显示区**：显示检测结果（原图+热图），支持拖放。
  - **缩略图列表**：固定高度，水平滚动，点击切换图片，悬浮显示信息。
  - **日志区**：可折叠，默认隐藏，显示操作和错误日志。
- **浮动进度条**：检测或加载时显示进度，完成后自动关闭。

### 操作流程
1. **启动程序**：
   - `preload` 模式：启动时加载所有模型，日志记录加载状态。
   - `ondemand` 模式：显示“未选择模型”，需手动选择。
2. **选择模型**：
   - 点击“Select Model”或按快捷键选择模型。
   - 状态栏更新模型名。
3. **单张检测**：
   - 点击“Detect -> Single Image”选择文件，或拖放图片到窗口。
   - 显示结果和检测信息，缩略图显示单张结果。
4. **批量检测**：
   - 点击“Detect -> Batch Images”选择文件夹，或拖放文件夹。
   - 显示第一张结果，缩略图列出所有结果，点击切换。
5. **调整阈值**：
   - 滑动左侧阈值滑块，实时更新检测信息和 `config.yaml`。
6. **查看日志**：
   - 点击“Show Log”展开日志区，查看操作详情。
   - 日志保存至 `logs/detection_log.txt`。

### 输出结果
- **保存路径**：`./output/[模型目录名]/detection_[输入文件名].png`
- **格式**：原图和异常热图并排显示。
- **检测信息**：格式如“异常得分: [得分] - [判断]”。

---

## 维护与扩展

### 常见问题排查
1. **拖放无效**：
   - 检查日志是否记录拖放事件（“检测到拖放事件”）。
   - 确保操作系统支持拖放（Windows 需管理员权限可能影响）。
   - 更新 PyQt5 至最新版本（`pip install --upgrade PyQt5`）。
2. **模型加载失败**：
   - 确认 `config.yaml` 中路径有效且 `.pth` 文件存在。
   - 检查 GPU 兼容性（`torch.cuda.is_available()`）。
3. **内存不足**：
   - 减小 `image_processor.py` 中 `max_batch_size`（默认 32）。
   - 设置环境变量 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`。
4. **检测结果异常**：
   - 验证模型输出格式，确保 `scores` 和 `masks` 正确解析。
   - 调整阈值范围（当前 0.0-2.0）适配模型得分。

### 优化建议
1. **性能提升**：
   - 使用异步 I/O（如 `asyncio`）替换线程预加载。
   - 优化 `SimpleNet.predict` 支持更大批量输入。
2. **界面增强**：
   - 添加缩略图右键菜单（导出、删除）。
   - 支持深色模式（通过 `QPalette`）。
3. **功能扩展**：
   - 添加结果导出功能（CSV/JSON），集成到“Options”菜单。
   - 支持检测历史记录和撤销操作。
4. **模型支持**：
   - 抽象模型加载接口，支持多种模型类型（不仅仅是 SimpleNet）。

### 代码维护
- **模块化**：GUI、处理和模型逻辑分离，易于独立修改。
- **配置文件**：`config.yaml` 支持扩展（如新增模型、参数）。
- **日志**：通过 `logging` 记录详细操作，便于调试。
- **测试**：建议使用 `pytest` 添加单元测试，覆盖检测和内存管理。

---

## 版本信息
- **当前版本**：v1.5
- **更新日期**：2025年3月3日
- **作者**：LucaZou（初始开发），xAI 协助优化
- **更新记录**：
  - v1.0：基本检测功能。
  - v1.1：多线程支持，界面优化。
  - v1.2：动态阈值、缩略图导航。
  - v1.4：YAML 配置、GPU 加速、内存优化。
  - v1.5：拖放支持、滑块阈值调整、简洁界面。
