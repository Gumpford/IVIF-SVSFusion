import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from encoder0716 import DENet

def check_image_size(x, down_factor=8):
    """检查并调整图像尺寸以适应模型要求"""
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def restore_image_size(x, original_h, original_w):
    """恢复图像到原始尺寸"""
    return x[..., :original_h, :original_w]

class DENetFeatureExtractor:
    """
    DENet特征提取器
    用于从预训练的DENet模型中提取四个关键特征：f_d_vi, f_d_vis1, f_d_vis2, f_d_vis3
    """
    
    def __init__(self, weights_path='./detail_enhance/DeEn_model_epoch_1000.pth', base_dim=16):
        """
        初始化DENet特征提取器
        
        参数:
        weights_path (str): 预训练DENet模型的路径
        base_dim (int): DENet的基础维度，默认为16
        """
        self.base_dim = base_dim
        self.weights_path = weights_path
        
        # 直接加载预训练的DENet模型
        self.model = torch.load(self.weights_path, map_location='cpu')
        self.model.eval()
    
    def extract_features(self, x):
        # 获取输入数据的设备信息并将模型转移到相同设备
        _, _, original_h, original_w = x.size()
        device = x.device
        x = check_image_size(x)
        self.model = self.model.to(device)
        
        # 提取特征
        with torch.no_grad():
            f_d_vi = self.model(x)
        
        # 恢复特征图到原始尺寸
        f_d_vi = restore_image_size(f_d_vi, original_h, original_w)

        return f_d_vi

# 使用示例
if __name__ == '__main__':
    # 初始化特征提取器
    weights_path = 'd:/WVPFusion/detail_enhance/DeEn_model_epoch_1000.pth'  # 预训练模型路径
    extractor = DENetFeatureExtractor(weights_path, base_dim=16)
    
    # 创建测试输入
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 从文件加载图像
    from PIL import Image
    import numpy as np
    
    # 指定图像路径 - 可以替换为您的图像路径
    image_path = "detail_enhance/input/38.png"  # 使用示例图像
    
    try:
        # 加载图像并转换为灰度
        img = Image.open(image_path).convert('L')
        # 调整图像大小为256x256
        img = img.resize((256, 256))
        # 转换为numpy数组并归一化到[0,1]
        img_np = np.array(img).astype(np.float32) / 255.0
        # 转换为PyTorch张量并添加批次和通道维度
        image = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)  # 批次大小为1，单通道，256x256分辨率
        print(f"成功加载图像: {image_path}")
    except Exception as e:
        print(f"加载图像失败: {e}，使用随机生成的图像代替")
        image = torch.randn(1, 1, 256, 256).to(device)  # 批次大小为1，单通道，256x256分辨率
    
    print(f"使用设备: {device}")
    print(f"输入图像形状: {image.shape}")
    
    try:
        # 提取特征
        features = extractor.extract_features(image)
        
        # 打印特征形状
        print(f'f_d_vi shape: {features.shape}')
        
    except Exception as e:
        print(f"特征提取过程中出现错误: {str(e)}")