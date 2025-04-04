import torch
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from src.backbones import simplenet
import logging

logger = logging.getLogger(__name__)

def load_model(model_path, device):
    # 加载模型
    try:
        # 初始化 WideResNet50-2 骨干网络
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        backbone.name = "wideresnet50"
        # 初始化 SimpleNet 检测器
        model = simplenet.SimpleNet(device)
        model.load(
            backbone=backbone,
            layers_to_extract_from=["layer2", "layer3"], # 从 layer2 和 layer3 提取特征
            device=device,
            input_shape=(3, 288, 288), # 输入图像尺寸
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
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = {}
        for key in checkpoint:
            if key == "discriminator":
                state_dict.update({f"discriminator.{k}": v for k, v in checkpoint[key].items()})
            elif key == "pre_projection":
                state_dict.update({f"pre_projection.{k}": v for k, v in checkpoint[key].items()})
        model.load_state_dict(state_dict, strict=False)
        model.eval() # 设置为评估模式
        logger.info(f"成功加载模型: {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"模型文件未找到: {model_path}")
        raise
    except Exception as e:
        logger.error(f"加载模型时发生错误: {str(e)}")
        raise