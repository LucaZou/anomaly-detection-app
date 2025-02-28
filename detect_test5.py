import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import simplenet
import argparse
import glob
import logging
from tqdm import tqdm

# 配置日志
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "detection_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file,mode="a",encoding="utf-8"),  # 保存到文件,追加（a）模式
        logging.StreamHandler()         # 输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# 配置参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前设备: {device}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
model_path = "models/mvtec_metal_nut/ckpt.pth"  # 替换为实际模型路径
output_base_dir = "./output"
imagesize = (288, 288)

# 加载模型
def load_model(model_path):
    try:
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        backbone.name = "wideresnet50"
        model = simplenet.SimpleNet(device)
        model.load(
            backbone=backbone,
            layers_to_extract_from=["layer2", "layer3"],
            device=device,
            input_shape=(3, 288, 288),
            pretrain_embed_dimension=1536,
            target_embed_dimension=1536,
            patchsize=3,
            embedding_size=256,
            meta_epochs=40,
            gan_epochs=4,
            noise_std=0.015,
            dsc_hidden=1024,
            dsc_layers=2,
            dsc_margin=0.5,
            pre_proj=1,
        )
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = {}
        for key in checkpoint:
            if key == "discriminator":
                state_dict.update({f"discriminator.{k}": v for k, v in checkpoint[key].items()})
            elif key == "pre_projection":
                state_dict.update({f"pre_projection.{k}": v for k, v in checkpoint[key].items()})
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        logger.info(f"成功加载模型: {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"模型文件未找到: {model_path}")
        raise
    except Exception as e:
        logger.error(f"加载模型时发生错误: {str(e)}")
        raise

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(329),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 检测单张图片
def detect_single_image(model, input_image_path, output_dir):
    try:
        image = Image.open(input_image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            scores, masks, _ = model.predict(image_tensor)
            anomaly_map = masks[0]

        # 生成热图
        anomaly_map = anomaly_map.squeeze()
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

        # 准备原图和热图
        original_image = np.array(image.resize(imagesize))
        heatmap = plt.cm.jet(anomaly_map)[:, :, :3]
        heatmap = (original_image * 0.5 + heatmap * 255 * 0.5).astype(np.uint8)

        # 并排放置
        combined_image = np.hstack((original_image, heatmap))

        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        input_filename = os.path.splitext(os.path.basename(input_image_path))[0]
        output_path = os.path.join(output_dir, f"detection_{input_filename}.png")
        plt.imsave(output_path, combined_image)
        logger.info(f"检测结果已保存到 {output_path}")
    except FileNotFoundError:
        logger.error(f"图片文件未找到: {input_image_path}")
    except Exception as e:
        logger.error(f"检测单张图片时发生错误: {input_image_path}, 错误信息: {str(e)}")

# 检测批量图片
def detect_batch_images(model, input_dir, output_base_dir):
    try:
        # 支持的图片格式
        image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
        if not image_paths:
            logger.warning(f"在 {input_dir} 中未找到任何图片")
            return

        # 确定输出目录
        parts = input_dir.replace('\\', '/').split('/')
        if len(parts) >= 4:  # 假设路径包含类别信息
            category = parts[-4]  # 例如 'metal_nut'
            output_dir = os.path.join(output_base_dir, category)
        else:
            output_dir = output_base_dir

        os.makedirs(output_dir, exist_ok=True)

        # 使用进度条遍历图片
        for input_image_path in tqdm(image_paths, desc="批量检测图片", unit="image"):
            try:
                image = Image.open(input_image_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)

                # 推理
                with torch.no_grad():
                    scores, masks, _ = model.predict(image_tensor)
                    anomaly_map = masks[0]

                # 生成热图
                anomaly_map = anomaly_map.squeeze()
                anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

                # 准备原图和热图
                original_image = np.array(image.resize(imagesize))
                heatmap = plt.cm.jet(anomaly_map)[:, :, :3]
                heatmap = (original_image * 0.5 + heatmap * 255 * 0.5).astype(np.uint8)

                # 并排放置
                combined_image = np.hstack((original_image, heatmap))

                # 保存结果
                input_filename = os.path.splitext(os.path.basename(input_image_path))[0]
                output_path = os.path.join(output_dir, f"detection_{input_filename}.png")
                plt.imsave(output_path, combined_image)

            except FileNotFoundError:
                logger.error(f"图片文件未找到: {input_image_path}")
            except Exception as e:
                logger.error(f"检测图片时发生错误: {input_image_path}, 错误信息: {str(e)}")

        # 批量检测完成后输出一次日志
        logger.info(f"检测结果已保存到 {output_dir}")
    except Exception as e:
        logger.error(f"批量检测过程中发生错误: {str(e)}")

# 主函数
def main():
    parser = argparse.ArgumentParser(description="异常检测脚本：支持单张或批量图片检测")
    parser.add_argument("--input", "-i", required=True, help="单张图片路径或批量图片目录")
    parser.add_argument("--batch", "-b", action="store_true", help="启用批量检测模式")
    parser.add_argument("--model", "-m", default=model_path, help="模型权重路径")

    args = parser.parse_args()

    try:
        # 加载模型
        model = load_model(args.model)

        if args.batch:
            # 批量检测
            detect_batch_images(model, args.input, output_base_dir)
        else:
            # 单张检测
            detect_single_image(model, args.input, output_base_dir)
    except Exception as e:
        logger.error(f"主程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()