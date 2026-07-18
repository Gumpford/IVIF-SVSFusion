import os
import numpy as np
import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms.functional as ttf
from PIL import Image
import torchvision.transforms as transforms
from imgaug import augmenters as iaa

def make_dataset(root='D:/DataSet/TrainData/MSRS/msrs/', train=True):
    dataset = []

    if train:
        IR = os.path.join(root)
        VIS = os.path.join(root)

    for index in range(400):

        img_ir = 'IR1-1-1/' + '{:04d}.png'.format(index + 1)
        img_vis = 'VIS1-1-1/' + '{:04d}.png'.format(index + 1)

        dataset.append([os.path.join(IR, img_ir), os.path.join(VIS, img_vis)])

    return dataset


class fusiondata(data.Dataset):
    def __init__(self, root, transform=None, train=True, phase='train'):
        self.train = train
        self.phase = phase
        # self._tensor = transforms.ToTensor()
        self._tensor = transforms.Compose([transforms.RandomCrop(128, 128), transforms.ToTensor()])
        if self.train:
            self.train_set_path = make_dataset(root, train)

    def __getitem__(self, index):
        if self.train:

            imgA_path, imgB_path = self.train_set_path[index]

            # imgA = Image.open(imgA_path).convert('RGB')  # RGB图
            imgA = Image.open(imgA_path).convert('L')   #L 将图像变成灰度图
            # imgB = Image.open(imgA_path).convert('RGB')  # RGB图
            imgB = Image.open(imgB_path).convert('L')   #L 将图像变成灰度图

            # irImage = ttf.to_tensor(imgA)  # 将图片转为张量
            # visImage = ttf.to_tensor(imgB)

            irImage = self._tensor(imgA)  # 将图片转为张量
            visImage = self._tensor(imgB)

            [imgA, imgB] = transform_augment([irImage, visImage], phase=self.phase, min_max=(0, 1))
            # 添加batch维度
            imgA = imgA.unsqueeze(0)
            imgB = imgB.unsqueeze(0)

            return imgA, imgB

    def __len__(self):
        if self.train:
            return 400
        else:
            return 361

# 数据增强
hflip = torchvision.transforms.RandomHorizontalFlip()

def transform_augment(imgs, phase='train', min_max=(0, 1)):
    if phase == 'train':
        imgs = torch.cat(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img


# ... existing code ...

if __name__ == '__main__':
    # 初始化数据集
    dataset = fusiondata(root='D:/DataSet/TrainData/MSRS/msrs/', train=True)
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 获取一个batch的数据
    for batch_idx, (imgA, imgB) in enumerate(dataloader):
        print(f'Batch {batch_idx + 1}:')
        print(f'红外图像维度: {imgA.shape}')
        print(f'可见光图像维度: {imgB.shape}')
        print(f'红外图像数值范围: [{imgA.min():.3f}, {imgA.max():.3f}]')
        print(f'可见光图像数值范围: [{imgB.min():.3f}, {imgB.max():.3f}]')
        
        # 只打印第一个batch的信息
        break