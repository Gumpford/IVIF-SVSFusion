from __future__ import print_function
import os
import math
import numpy as np
import torch
from PIL import Image
import time
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--lambda2', type=float, default=0.2, help='weight on L1 term in objective')
opt = parser.parse_args()
from tqdm import tqdm

start = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_if_need_padding(h, w):
    """检查是否需要padding"""
    return h % 64 != 0 or w % 64 != 0

def pad_to_multiple_of_64(img_tensor):
    """将图像padding到64的倍数"""
    _, _, h, w = img_tensor.size()

    # 如果已经是64的倍数就直接返回
    if not check_if_need_padding(h, w):
        return img_tensor, None

    pad_h = (math.ceil(h / 64) * 64) - h
    pad_w = (math.ceil(w / 64) * 64) - w
    pad_h_top = pad_h // 2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    padding = (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)
    padded_img = torch.nn.functional.pad(img_tensor, padding, mode='reflect')
    return padded_img, padding

def unpad_image(padded_img_tensor, padding):
    """移除padding还原原始尺寸"""
    # 如果padding为None说明没有进行padding
    if padding is None:
        return padded_img_tensor

    pad_w_left, pad_w_right, pad_h_top, pad_h_bottom = padding
    _, _, h, w = padded_img_tensor.size()
    unpadded_img = padded_img_tensor[:, :,
                   pad_h_top:h - pad_h_bottom,
                   pad_w_left:w - pad_w_right]
    return unpadded_img

# 三个数据集的路径和图像对数量
dataset_paths = {

    'TNO': {
        'IR': 'D:/WVPFusion/DataSet/TestData/TNO/IR/',
        'VIS': 'D:/WVPFusion/DataSet/TestData/TNO/VIS/',
        'num_pairs': 40
    },
    'M3FD': {
        'IR': 'D:/WVPFusion/DataSet/TestData/M3FD/IR/',
        'VIS': 'D:/WVPFusion/DataSet/TestData/M3FD/VIS/',
        'num_pairs': 150
    },
    'MSRS': {
        'IR': 'D:/WVPFusion/DataSet/TestData/MSRS/IR/',
        'VIS': 'D:/WVPFusion/DataSet/TestData/MSRS/VIS/',
        'num_pairs': 361
    }
    # 'FMB': {
    #     'IR': 'D:/WVPFusion/DataSet/TestData/FMB/IR/',
    #     'VIS': 'D:/WVPFusion/DataSet/TestData/FMB/VIS/',
    #     'num_pairs': 280
    # }
}

# 创建目录
for dataset, paths in dataset_paths.items():
    if not os.path.exists(f"./dilated/fuse_results/{str(opt.lambda2)}/{dataset}"):
        os.makedirs(f"./dilated/fuse_results//{str(opt.lambda2)}/{dataset}")
        print(f"Created directory: {dataset}")

for dataset, paths in dataset_paths.items():
    num_pairs = paths['num_pairs']
    for i in tqdm(range(num_pairs)):
        with torch.no_grad():

            fuse_model_path = r'./fuse_parameter/{}/best_model_qabf.pth'.format(str(opt.lambda2))
            ae_model_path = r'./fuse_parameter/{}/best_encoder_qabf.pth'.format(str(opt.lambda2))
            ill_model_path = './enhancement/RDMFuse-main/Model/model.pth'

            net = torch.load(ae_model_path)
            f_net = torch.load(fuse_model_path)
            ill_en_net = torch.load(ill_model_path)

            net = net.to(device)
            f_net = f_net.to(device)
            ill_en_net = ill_en_net.to(device)

            net.eval()
            f_net.eval()

            imgA1_path = paths['IR'] + f"{i+1}.png"
            imgA2_path = paths['VIS'] + f"{i+1}.png"

            imgA1 = Image.open(imgA1_path)
            imgA2 = Image.open(imgA2_path)

            imgA1 = imgA1.convert('L')
            imgA2 = imgA2.convert('L')

            imgA1 = np.asarray(imgA1)
            imgA2 = np.asarray(imgA2)

            imgA1 = np.atleast_3d(imgA1).transpose(2, 0, 1).astype(float)
            imgA2 = np.atleast_3d(imgA2).transpose(2, 0, 1).astype(float)

            C, Row, Col = imgA1.shape

            imgA1 = imgA1 / float(255)
            imgA2 = imgA2 / float(255)

            imgA1 = torch.from_numpy(imgA1).float()
            imgA2 = torch.from_numpy(imgA2).float()

            imgA1 = imgA1.view(1, 1, Row, Col)
            imgA2 = imgA2.view(1, 1, Row, Col)

            f_net = f_net.to(device)
            net = net.to(device)
            imgA1, imgA2 = imgA1.to(device), imgA2.to(device)

            # 检查是否需要padding
            if check_if_need_padding(Row, Col):
                padded_imgA1, padding1 = pad_to_multiple_of_64(imgA1)
                padded_imgA2, padding2 = pad_to_multiple_of_64(imgA2)
                print(f"Image {i + 1} needs padding to size {padded_imgA1.size()}")
            else:
                padded_imgA1, padding1 = imgA1, None
                padded_imgA2, padding2 = imgA2, None
                print(f"Image {i + 1} already multiple of 32: {imgA1.size()}")

            tpA04, tpB04, fA04, fA14, fA24, fB04, fB14, fB24, f_P_ir_16, f_P_ir_32, f_P_ir_64, f_P_vis_16, f_P_vis_32, f_P_vis_64 = net(padded_imgA1, padded_imgA2)
            output = f_net(fA04, fA14, fA24, fB04, fB14, fB24, f_P_ir_16, f_P_ir_32, f_P_ir_64, f_P_vis_16, f_P_vis_32, f_P_vis_64, padded_imgA1, padded_imgA2)
            R, ill, IR, Fout, ill_en = ill_en_net(padded_imgA2, padded_imgA1)

            output = output * ill_en
            output = unpad_image(output, padding1)

            out1 = output[0].cpu()
            out_img = out1.data[0]
            out_img = out_img.squeeze()

            out_img_fuse = out_img.numpy()

            plt.imsave(f"./dilated/fuse_results/{str(opt.lambda2)}/{dataset}/{i+1}.png", out_img_fuse, cmap='gray')
            print(f"{dataset}: mask {i+1} has been saved")

end = time.time()
print(end - start)