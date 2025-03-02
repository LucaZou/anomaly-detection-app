# Anomaly Detection App 文档

## 项目概述

**Anomaly Detection App** 是一个基于 Python 和 PyQt5 的桌面应用程序，旨在通过深度学习模型（如 Wide ResNet50 和 SimpleNet）进行图像异常检测。它支持单张图片和批量图片处理，提供直观的用户界面，允许用户选择模型、查看检测结果、切换批量检测结果，并显示详细的检测信息。应用优化了性能，支持动态内存管理和 GPU 加速，适用于工业检测场景。

### 示例图

- **主界面**  
  ![主界面](https://img-1313049298.cos.ap-shanghai.myqcloud.com/note-img/202503021629304.png)  
- **单张检测**  
  ![单张检测](https://img-1313049298.cos.ap-shanghai.myqcloud.com/note-img/202503021629305.png)  
- **批量检测**  
  ![批量检测](https://img-1313049298.cos.ap-shanghai.myqcloud.com/note-img/202503021629306.png)

### 主要功能
- **模型选择**：通过下拉菜单选择预定义的异常检测模型（如“Metal Nut”或“Screw”）。
- **单张检测**：检测单张图片，显示结果和异常得分（无缩略图显示）。
- **批量检测**：处理文件夹中的多张图片，显示结果并提供滚动缩略图预览。
- **结果切换**：支持“Previous”和“Next”按钮切换批量检测结果。
- **状态显示**：显示当前模型、图片名称和检测信息（异常得分及判断）。
- **动态阈值**：通过设置菜单调整异常检测阈值（默认 0.5，可调整范围 0.0-1.0）。
- **性能优化**：支持动态批次大小和 GPU 加速，减少内存占用和等待时间。
- **日志记录**：操作和错误信息记录到界面及文件。

### 技术栈
- **编程语言**：Python 3.8+
- **GUI 框架**：PyQt5
- **深度学习**：PyTorch, torchvision
- **图像处理**：Pillow, NumPy, Matplotlib
- **其他**：tqdm（进度条）、logging（日志）、psutil（内存管理）、pyyaml（配置文件）

---

## 安装与运行

### 环境要求
- **操作系统**：Windows、macOS 或 Linux（跨平台支持）
- **Python**：3.8 或更高版本
- **硬件**：支持 CUDA 的 GPU（可选，提升性能，建议至少 4 GiB 显存）

### 依赖安装
安装所需库：
```bash
pip install PyQt5 torch torchvision pillow numpy matplotlib tqdm psutil pyyaml
```
- **torch**：若需 GPU 支持，请根据硬件安装对应版本（参考 [PyTorch 官网](https://pytorch.org/)）。
- **simplenet**：如果模型依赖自定义模块 `simplenet`，需从 [GitHub](https://github.com/DonaldRR/SimpleNet) 克隆并安装。

### 项目文件
下载项目代码并确保以下结构完整：
```
anomaly_detection_app/
├── main.py
├── gui.py
├── image_processor.py
├── model_loader.py
├── simplenet.py
├── common.py
├── config.yaml
├── logs/              # 日志保存目录（自动创建）
└── output/            # 检测结果保存目录（自动创建）
```

### 配置
编辑 `config.yaml`（替代原 `config.json`）：
```yaml
# Anomaly Detection App 的配置文件
# 定义模型加载模式，可选值：preload（预加载所有模型）或 ondemand（按需加载）
load_mode: preload

# 模型配置，键为模型名称，值为模型文件的路径
models:
  # 金属螺母检测模型，路径指向模型权重文件
  Metal Nut: models/mvtec_metal_nut/ckpt.pth
  # 螺丝检测模型，路径指向模型权重文件
  Screw: models/mvtec_screw/ckpt.pth
```
- **load_mode**：选择预加载或按需加载模型。
- **models**：模型名称和路径键值对，路径需指向实际 `.pth` 文件。

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
  - 程序入口，加载 `config.yaml`，初始化设备和处理器，启动 GUI。
- **`gui.py`**：
  - 定义图形界面，包括工具栏、图像显示区、切换按钮、状态标签、检测信息和缩略图区域。
  - 处理用户交互（如模型选择、检测、设置）。
- **`image_processor.py`**：
  - 图像处理逻辑，包括单张和批量检测，支持动态批次大小和 GPU 加速。
- **`model_loader.py`**：
  - 模型加载逻辑，定义 `load_model` 函数。
- **`simplenet.py`**：
  - 实现 SimpleNet 模型，用于图像异常检测和定位。
- **`common.py`**：
  - 提供辅助类（如 `PatchMaker`、`RescaleSegmentor`）和功能。
- **`config.yaml`**：
  - YAML 配置文件，存储模型列表和加载模式。

### 核心类与功能
1. **`ImageProcessor` (image_processor.py)**：
   - **属性**：
     - `model_cache`：存储已加载模型。
     - `current_model_name`：当前模型名称。
     - `output_base_dir`：输出目录。
     - `batch_worker`：批量检测线程对象。
   - **方法**：
     - `set_model`：设置当前模型，支持缓存或按需加载。
     - `detect_single_image`：检测单张图片，返回结果路径和检测信息。
     - `detect_batch_images`：异步批量检测，支持动态批次大小和 GPU 加速。
   - **优化**：
     - 动态调整 `batch_size` 根据 GPU 可用内存。
     - 多线程预加载图片，减少 I/O 等待。
     - 使用 `torch.cuda.empty_cache()` 管理 GPU 内存。

2. **`MainWindow` (gui.py)**：
   - **属性**：
     - `result_paths`：存储检测结果路径。
     - `detection_infos`：存储检测信息。
     - `current_index`：当前显示图片的索引。
     - `threshold`：动态异常检测阈值（默认 0.5）。
   - **方法**：
     - `select_model`：处理模型选择。
     - `detect_single` / `detect_batch`：触发检测。
     - `prev_image` / `next_image`：切换图片并更新检测信息。
     - `open_settings`：打开设置窗口，调整阈值。
   - **界面元素**：
     - 工具栏（模型选择、检测、设置）。
     - 状态标签（模型名、图片名、检测信息）。
     - 图像显示区、切换按钮、滚动缩略图、进度条、日志区。

3. **`SimpleNet` (simplenet.py)**：
   - 实现图像异常检测和定位，基于 Wide ResNet50 骨干网络和 SimpleNet 架构。
   - **方法**：
     - `load`：加载模型和配置。
     - `predict`：单张或批量预测异常得分和热图。
     - `fit`：训练判别器，优化异常检测性能。
   - **优化**：
     - 支持批量输入（需确保 `predict` 适配批量张量）。

---

## 使用说明

### 界面布局
- **工具栏**：
  - “Select Model”：下拉菜单选择模型。
  - “Detect Single Image”：单张检测（无缩略图）。
  - “Detect Batch Images”：批量检测（显示滚动缩略图）。
  - “Settings”：打开设置窗口，调整阈值。
- **状态区**：
  - “当前模型”：显示选择的模型名称。
  - “当前图片”：显示当前图片文件名。
  - “检测信息”：显示异常得分和判断（如“异常得分: 0.75 - 检测到异常”）。
- **图像显示区**：显示检测结果（原图+热图）。
- **切换按钮**：
  - “Previous”：切换到上一张图片（批量检测时可用）。
  - “Next”：切换到下一张图片（批量检测时可用）。
- **缩略图区域**（仅批量检测显示）：
  - 滚动显示所有批量检测结果的缩略图（仅图标，无文件名），点击跳转到对应图片。
- **进度条**：批量检测时显示进度。
- **日志区**：显示操作日志和错误信息。

### 操作流程
1. **启动程序**：
   - 预加载模式：启动时加载所有模型，日志记录加载状态。
   - 按需加载模式：显示“未选择模型”，需手动选择。
2. **选择模型**：
   - 点击“Select Model”，选择模型（如“Metal Nut”）。
   - 状态栏更新为“当前模型: [模型名]”。
3. **单张检测**：
   - 点击“Detect Single Image”，选择图片。
   - 显示结果和检测信息，缩略图区域隐藏，切换按钮禁用。
4. **批量检测**：
   - 点击“Detect Batch Images”，选择文件夹。
   - 处理完成后显示第一张结果及检测信息，缩略图区域显示所有结果，切换按钮启用。
   - 使用滚动条浏览缩略图，点击跳转到对应图片。
5. **调整阈值**：
   - 点击“Settings”，输入新阈值（范围 0.0-1.0），保存后更新检测信息。
6. **查看日志**：
   - 日志区实时显示操作信息，保存至 `logs/detection_log.txt`。

### 输出结果
- **保存路径**：`./output/[模型目录名]/detection_[输入文件名].png`
- **格式**：并排显示原图和异常热图。
- **检测信息**：包含异常得分和判断（例如“异常得分: 0.75 - 检测到异常”）。

---

## 维护与扩展

### 常见问题排查
1. **模型加载失败**：
   - 检查 `config.yaml` 中的路径是否正确。
   - 确保 `.pth` 文件存在且未损坏。
2. **内存不足**：
   - 调整 `image_processor.py` 中的 `batch_size`（默认 8），减少到 4 或 2。
   - 设置 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` 优化内存碎片。
3. **界面卡顿**：
   - 确保 GPU 驱动和 PyTorch 版本匹配，检查 `nvidia-smi` 监控 GPU 利用率。
4. **检测信息错误**：
   - 若异常得分显示异常，确认模型输出 `scores` 的格式并调整阈值。
5. **依赖缺失**：
   - 确保所有库已安装，尤其是 `simplenet` 和 `psutil`。

### 优化建议
1. **动态内存管理**：
   - 进一步优化 `_estimate_batch_size`，实时监控 GPU 可用内存，动态调整 `batch_size`。
2. **异步 I/O**：
   - 使用 `asyncio` 或 `concurrent.futures` 替换 `threading`，提升图片预加载效率。
3. **模型优化**：
   - 调整 `simplenet.SimpleNet.predict` 支持更高效的批量处理，减少内存峰值。
4. **结果预览增强**：
   - 添加缩略图工具提示，显示检测信息。
5. **多语言支持**：
   - 使用 `QTranslator` 支持界面多语言切换。

### 代码维护
- **模块化**：逻辑已拆分为独立模块（GUI、处理、加载、模型），便于单独修改。
- **配置文件**：`config.yaml` 支持复杂结构，新增模型只需更新配置文件。
- **日志**：通过 `logging` 模块记录详细运行信息，便于调试。
- **性能测试**：使用 `unittest` 或 `pytest` 编写单元测试，覆盖批量检测和内存管理。

---

## 版本信息
- **当前版本**：v1.4
- **更新日期**：2025年3月2日
- **作者**：LucaZou
- **更新记录**：
  - v1.0：初始版本，支持基本检测功能。
  - v1.1：添加多线程支持，优化界面响应。
  - v1.2：新增检测信息显示、动态阈值、缩略图预览。
  - v1.4：支持 `.yaml` 配置文件、动态批次大小、GPU 加速和内存优化。

如需进一步支持或功能扩展，请联系开发团队或参考代码注释。

---

### 文档说明
- **准确性**：文档反映了项目的最新状态，包括所有优化（如动态批次、预加载、GPU 加速）。
- **完整性**：涵盖了安装、运行、使用和维护的所有必要信息。
- **可扩展性**：提供了优化建议，便于未来开发。

