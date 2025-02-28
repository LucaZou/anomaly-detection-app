# 使用示例
## 单张检测
```powershell
detect_test1.py --input "data/mvtec_anomaly_detection/metal_nut/test/bent/001.png"

```

- 输出：
```text
2023-10-10 12:00:00,123 - INFO - 成功加载模型: models/mvtec_metal_nut/ckpt.pth
2023-10-10 12:00:00,456 - INFO - 检测结果已保存到 ./output/detection_001.png
```
## 批量检测
```powershell
& d:/Projects_D/Graduation-Project/preject/SimpleNet/.venv/Scripts/python.exe d:/Projects_D/Graduation-Project/preject/anomaly_detection_analysis/detect_test1.py --input "data/mvtec_anomaly_detection/metal_nut/test/bent" --batch
```
- 输出：
```text
2023-10-10 12:00:00,123 - INFO - 成功加载模型: models/mvtec_metal_nut/ckpt.pth
批量检测图片: 100%|██████████| 5/5 [00:02<00:00,  2.50image/s]
2023-10-10 12:00:02,789 - INFO - 检测结果已保存到 ./output/metal_nut
```
## 检测结果
### 日志文件
- 检查 ./logs/detection_log.txt，内容与控制台输出一致。