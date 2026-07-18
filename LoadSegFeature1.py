import os

from Seg_build_BiSeNet import BiSeNet
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def selectHalfFeatureRand(input_tensor, N):
    # 获取输入张量的通道数
    num_channels = input_tensor.size(1)
    # 计算要选择的特征图数量
    selected_channels = num_channels // N
    # 使用torch.randperm生成一个随机排列索引
    random_indices = torch.randperm(num_channels)[:selected_channels]

    # 从输入张量中选择指定数量的特征图
    selected_features = input_tensor[:, random_indices, :, :]
    return selected_features


def selectNFeatureMapByImportant(data, N):
    # 计算通道的重要程度（这里采用通道的平均值作为重要程度）
    channel_mean = torch.mean(data, dim=(2, 3))  # 形状为(2, 1024)

    # 找到最重要的N个通道的索引
    top_channels = torch.topk(channel_mean, k=N, dim=1)[1]  # 形状为(2, N)

    # 生成向量形式的索引
    indices = torch.arange(data.shape[1], device=top_channels.device).unsqueeze(
        0)  # 形状为(1, 1024)，确保与top_channels张量在相同的设备上
    # 使用索引向量提取最重要的48个通道并保持原始形状
    output = torch.index_select(data, dim=1, index=indices.squeeze(0)[top_channels.flatten()]).view(
        top_channels.shape[0], -1, data.shape[2], data.shape[3])
    output = output.to('cpu')
    output = torch.index_select(output, dim=1, index=torch.arange(N))
    return output


def changeIrTo3Chanel(image_ir):
    rgb_tensor = image_ir.repeat(1, 3, 1, 1)
    return rgb_tensor


class LoadSegFeature:
    def __init__(self, image_ir):
        # 构造函数
        self.image_ir = changeIrTo3Chanel(image_ir)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        irModel = BiSeNet(12, "resnet18")

        # 加载预训练权重
        checkpoint_path = './pretrained_models/best_dice_loss_miou_0.655.pth'
        irModel.load_state_dict(torch.load(checkpoint_path))

        # 设置为评估模式
        irModel.eval()
        for param in irModel.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            self.irSegMode = torch.nn.DataParallel(irModel).cuda()
            self.irSegMode(self.image_ir)

    def getIrFeature3(self):
        if isinstance(self.irSegMode, torch.nn.DataParallel):
            irFeature3 = self.irSegMode.module.thirdFeature  # (Tensor (2,256,60,80)
        return irFeature3[0][0]



