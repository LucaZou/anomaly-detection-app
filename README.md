- # Anomaly Detection App 文档

  ## 项目概述
  **Anomaly Detection App** 是一个基于 Python 和 PyQt5 的桌面应用程序，用于图像异常检测。它利用深度学习模型（如 Wide ResNet50）对输入图像进行异常检测，支持单张图片和批量图片处理。应用程序提供直观的图形界面，允许用户选择模型、查看检测结果、切换批量处理结果，并记录日志。

  ### 示例图

  ![image.png](https://img-1313049298.cos.ap-shanghai.myqcloud.com/note-img/202503011359862.png)

  - 单个检测
  ![image.png](https://img-1313049298.cos.ap-shanghai.myqcloud.com/note-img/202503011400905.png)

  - 批量检测
  ![image.png](https://img-1313049298.cos.ap-shanghai.myqcloud.com/note-img/202503011400586.png)


  ### 主要功能
  - **模型选择**: 通过下拉菜单选择预定义的异常检测模型。
  - **单张检测**: 检测单张图片并显示结果。
  - **批量检测**: 处理文件夹中的多张图片，逐张显示结果。
  - **结果切换**: 批量检测后支持通过“Next”按钮切换图片。
  - **状态显示**: 显示当前模型和图片名称。
  - **日志记录**: 操作和错误信息记录到界面及文件。
  - **灵活配置**: 通过 JSON 配置文件管理模型列表和加载模式。

  ### 技术栈
  - **编程语言**: Python 3.8+
  - **GUI 框架**: PyQt5
  - **深度学习**: PyTorch, torchvision
  - **图像处理**: Pillow, NumPy, Matplotlib
  - **其他**: tqdm（进度条）, logging（日志）

  ---

  ## 安装与运行

  ### 环境要求
  - **操作系统**: Windows, macOS, 或 Linux（跨平台支持）
  - **Python**: 3.8 或更高版本
  - **硬件**: 支持 CUDA 的 GPU（可选，提升性能）

  ### 依赖安装
  安装所需库：
  ```bash
  pip install PyQt5 torch torchvision pillow numpy matplotlib tqdm
  ```
  - **torch**: 若需 GPU 支持，请根据硬件安装对应版本（参考 [PyTorch 官网](https://pytorch.org/)）。
  - **simplenet**: 如果你的模型依赖自定义模块 `simplenet`，需自行提供或安装。

  ### 项目文件
  下载项目代码并确保以下结构完整：
  ```
  anomaly_detection_app/
  ├── main.py
  ├── gui.py
  ├── image_processor.py
  ├── model_loader.py
  ├── config.json
  ├── logs/              # 日志保存目录（自动创建）
  └── output/           # 检测结果保存目录（自动创建）
  ```

  ### 配置
  编辑 `config.json`：
  ```json
  {
    "load_mode": "preload",
    "models": {
      "Metal Nut": "models/mvtec_metal_nut/ckpt.pth",
      "Screw": "D:\\Projects_D\\Graduation-Project\\preject\\anomaly_detection_application\\models\\mvtec_screw\\ckpt.pth"
    }
  }
  ```
  - **load_mode**:  
    - `"preload"`: 启动时加载所有模型，切换无需等待。
    - `"ondemand"`: 启动后手动选择加载，加快启动速度。
  - **models**: 模型名称和路径键值对，路径需指向实际 `.pth` 文件。

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
  - **`main.py`**:  
    - 程序入口，加载配置，初始化设备和处理器，启动 GUI。
  - **`gui.py`**:  
    - 定义图形界面，包括工具栏、图像显示区、切换按钮、状态标签和日志区。
    - 处理用户交互（如模型选择、检测操作）。
  - **`image_processor.py`**:  
    - 图像处理逻辑，包括单张和批量检测，管理模型缓存和输出路径。
  - **`model_loader.py`**:  
    - 模型加载逻辑，定义 `load_model` 函数。
  - **`config.json`**:  
    - 配置文件，存储模型列表和加载模式。

  ### 核心类与功能
  1. **`ImageProcessor` (image_processor.py)**:
     - **属性**:  
       - `model_cache`: 存储已加载模型。
       - `current_model_name`: 当前模型名称。
       - `output_base_dir`: 根据模型路径动态生成的输出目录。
     - **方法**:  
       - `set_model`: 设置当前模型，支持缓存或按需加载。
       - `detect_single_image`: 处理单张图片。
       - `detect_batch_images`: 处理批量图片，返回结果路径列表。
     - **信号**:  
       - `progress_updated`: 更新批量处理进度。
       - `log_message`: 发送日志消息。
       - `batch_finished`: 批量处理完成，返回结果路径。

  2. **`MainWindow` (gui.py)**:
     - **属性**:  
       - `result_paths`: 存储检测结果路径。
       - `current_index`: 当前显示图片的索引。
       - `current_model_name`: 当前模型名称。
     - **方法**:  
       - `select_model`: 处理模型选择。
       - `detect_single` / `detect_batch`: 触发检测。
       - `prev_image` / `next_image`: 切换图片（Previous 禁用）。
       - `update_buttons`: 动态调整按钮状态。
     - **界面元素**:  
       - 工具栏（模型选择、检测操作）。
       - 状态标签（模型名、图片名）。
       - 图像显示区、切换按钮、进度条、日志区。

  3. **`load_model` (model_loader.py)**:
     - 加载 Wide ResNet50 模型及其权重，支持自定义参数。

  ---

  ## 使用说明

  ### 界面布局
  - **工具栏**:  
    - “Select Model”: 下拉菜单选择模型。
    - “Detect Single Image”: 单张检测。
    - “Detect Batch Images”: 批量检测。
  - **状态区**:  
    - “当前模型”: 显示选择的模型名称。
    - “当前图片”: 显示当前图片文件名。
  - **图像显示区**: 显示检测结果（原图+热图）。
  - **切换按钮**:  
    - “Previous”: 始终禁用。
    - “Next”: 批量检测时启用，切换下一张。
  - **进度条**: 批量检测时显示进度。
  - **日志区**: 显示操作日志和错误信息。

  ### 操作流程
  1. **启动程序**:  
     - 预加载模式：启动时加载所有模型，日志记录加载状态。
     - 按需加载模式：启动后显示“未选择模型”，需手动选择。
  2. **选择模型**:  
     - 点击“Select Model”，从下拉菜单选择模型（如“Metal Nut”）。
     - 状态栏更新为“当前模型: [模型名]”。
  3. **单张检测**:  
     - 点击“Detect Single Image”，选择图片文件。
     - 显示结果，切换按钮禁用，状态栏显示文件名。
  4. **批量检测**:  
     - 点击“Detect Batch Images”，选择文件夹。
     - 处理完成后显示第一张结果，进度条消失，“Next”按钮启用。
     - 点击“Next”切换下一张，状态栏更新文件名。
  5. **查看日志**:  
     - 日志区实时显示操作信息，保存至 `logs/detection_log.txt`。

  ### 输出结果
  - **保存路径**: `./output/[模型目录名]/detection_[输入文件名].png`。
  - **格式**: 并排显示原图和异常热图。

  ---

  ## 维护与扩展

  ### 常见问题排查
  1. **模型加载失败**:  
     - 检查 `config.json` 中的路径是否正确。
     - 确保 `.pth` 文件存在且未损坏。
  2. **界面卡顿**:  
     - 批量检测可能阻塞 GUI，考虑添加多线程支持。
  3. **依赖缺失**:  
     - 确认所有库已安装，尤其是 `simplenet`（若为自定义模块）。

  ### 扩展建议
  1. **多线程支持**:  
     - 使用 `QThread` 将检测任务移至后台，避免界面卡顿。
  2. **动态配置**:  
     - 添加界面修改 `config.json` 的功能，支持保存。
  3. **结果预览**:  
     - 添加缩略图列表，点击跳转到对应图片。
  4. **参数调整**:  
     - 提供界面调整图像预处理参数（如 `transform`）。
  5. **进度增强**:  
     - 预加载模型时显示进度条。

  ### 代码维护
  - **模块化**: 逻辑已拆分为独立模块（GUI、处理、加载），便于单独修改。
  - **配置文件**: 新增模型只需更新 `config.json`，无需改代码。
  - **日志**: 通过 `logging` 模块记录详细运行信息，便于调试。

  ---

  ## 版本信息
  - **当前版本**: v1.1
  - **更新日期**: 2025年2月28日
  - **作者**: LucaZou

  如需进一步支持或功能扩展，请联系开发团队或参考代码注释。