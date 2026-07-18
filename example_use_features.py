import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from extract_features import extract_denet_features, load_denet_features_as_prompt

# 示例网络，接受DENet特征作为提示
class ExampleNetwork(nn.Module):
    def __init__(self, in_channels=1):
        super(ExampleNetwork, self).__init__()
        
        # 输入图像的处理
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 融合DENet的f_d_vi特征 (64通道)
        self.fusion_f_d_vi = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x, f_d_vi, f_d_vis1=None, f_d_vis2=None, f_d_vis3=None):
        # 处理输入图像
        x_feat = self.input_conv(x)
        
        # 融合DENet的f_d_vi特征
        combined = torch.cat([x_feat, f_d_vi], dim=1)
        fused = self.fusion_f_d_vi(combined)
        
        # 生成输出
        output = self.output_conv(fused)
        
        return output


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建示例输入图像
    input_image = torch.randn(1, 1, 256, 256)
    
    # 1. 从预训练DENet模型中提取特征
    denet_model_path = "path/to/denet_model.pth"  # 替换为实际的模型路径
    features_dir = "features_output"
    
    try:
        # 尝试提取特征
        print("\n方法1: 直接从预训练DENet模型提取特征")
        f_d_vi, f_d_vis1, f_d_vis2, f_d_vis3 = extract_denet_features(
            input_image, denet_model_path, features_dir, device)
        
        # 创建示例网络并使用提取的特征
        example_net = ExampleNetwork(in_channels=1).to(device)
        input_image = input_image.to(device)
        
        # 使用提取的特征作为提示
        output = example_net(input_image, f_d_vi, f_d_vis1, f_d_vis2, f_d_vis3)
        print(f"输出形状: {output.shape}")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("将尝试方法2...")
    
    # 2. 从保存的文件加载特征
    try:
        print("\n方法2: 从保存的文件加载特征")
        # 检查特征文件是否存在
        if os.path.exists(features_dir) and all(os.path.exists(os.path.join(features_dir, f)) 
                                              for f in ['f_d_vi.npy', 'f_d_vis1.npy', 'f_d_vis2.npy', 'f_d_vis3.npy']):
            # 加载特征
            features = load_denet_features_as_prompt(features_dir)
            
            # 创建示例网络
            example_net = ExampleNetwork(in_channels=1).to(device)
            input_image = input_image.to(device)
            features = [f.to(device) for f in features]
            
            # 使用加载的特征作为提示
            output = example_net(input_image, *features)
            print(f"输出形状: {output.shape}")
        else:
            print(f"特征文件不存在于 {features_dir}")
            print("请先运行方法1提取特征，或确保特征文件路径正确")
    
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()