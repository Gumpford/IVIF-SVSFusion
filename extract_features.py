import torch
import torch.nn as nn
import os
import numpy as np
from detail_enhance.encoder0716 import DENet

def extract_denet_features(input_image, model_path, output_dir=None, device=None):
    """
    从预训练的DENet模型中提取特征
    
    参数:
    input_image (torch.Tensor): 输入图像张量，形状为 [B, 1, H, W]
    model_path (str): 预训练DENet模型的路径
    output_dir (str, optional): 保存特征的目录，如果为None则不保存
    device (torch.device, optional): 运行设备，如果为None则自动选择
    
    返回:
    tuple: (f_d_vi, f_d_vis1, f_d_vis2, f_d_vis3) 四个特征图
    """
    # 设置设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = DENet(base_dim=16)
    
    # 加载预训练权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        # 如果模型保存为state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        # 如果直接保存了整个模型的state_dict
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        print(f"成功加载模型: {model_path}")
    else:
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    # 将模型移至指定设备并设为评估模式
    model = model.to(device)
    model.eval()
    
    # 确保输入图像在正确的设备上
    input_image = input_image.to(device)
    
    # 提取特征
    with torch.no_grad():
        f_d_vi, f_d_vis1, f_d_vis2, f_d_vis3 = model(input_image)
    
    # 如果指定了输出目录，保存特征
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        # 保存为numpy数组
        np.save(os.path.join(output_dir, 'f_d_vi.npy'), f_d_vi.cpu().numpy())
        np.save(os.path.join(output_dir, 'f_d_vis1.npy'), f_d_vis1.cpu().numpy())
        np.save(os.path.join(output_dir, 'f_d_vis2.npy'), f_d_vis2.cpu().numpy())
        np.save(os.path.join(output_dir, 'f_d_vis3.npy'), f_d_vis3.cpu().numpy())
        print(f"特征已保存到: {output_dir}")
    
    return f_d_vi, f_d_vis1, f_d_vis2, f_d_vis3


def load_denet_features_as_prompt(features_dir):
    """
    从保存的特征文件中加载DENet特征作为提示
    
    参数:
    features_dir (str): 保存特征的目录
    
    返回:
    tuple: (f_d_vi, f_d_vis1, f_d_vis2, f_d_vis3) 四个特征图
    """
    f_d_vi = torch.from_numpy(np.load(os.path.join(features_dir, 'f_d_vi.npy')))
    f_d_vis1 = torch.from_numpy(np.load(os.path.join(features_dir, 'f_d_vis1.npy')))
    f_d_vis2 = torch.from_numpy(np.load(os.path.join(features_dir, 'f_d_vis2.npy')))
    f_d_vis3 = torch.from_numpy(np.load(os.path.join(features_dir, 'f_d_vis3.npy')))
    
    return f_d_vi, f_d_vis1, f_d_vis2, f_d_vis3


# 示例用法
if __name__ == "__main__":
    # 示例参数
    model_path = "path/to/denet_model.pth"  # 替换为实际的模型路径
    output_dir = "features_output"  # 特征输出目录
    
    # 创建示例输入
    input_image = torch.randn(1, 1, 256, 256)  # 批次大小为1，单通道，256x256图像
    
    # 提取特征
    f_d_vi, f_d_vis1, f_d_vis2, f_d_vis3 = extract_denet_features(
        input_image, model_path, output_dir)
    
    # 打印特征形状
    print(f"f_d_vi shape: {f_d_vi.shape}")
    print(f"f_d_vis1 shape: {f_d_vis1.shape}")
    print(f"f_d_vis2 shape: {f_d_vis2.shape}")
    print(f"f_d_vis3 shape: {f_d_vis3.shape}")
    
    # 示例：如何将提取的特征用作其他网络的提示
    print("\n演示如何将提取的特征用作其他网络的提示:")
    print("1. 直接使用提取的特征:")
    print("   other_network(input_tensor, f_d_vi, f_d_vis1, f_d_vis2, f_d_vis3)")
    
    print("\n2. 从保存的文件加载特征:")
    print("   features = load_denet_features_as_prompt('features_output')")
    print("   other_network(input_tensor, *features)")