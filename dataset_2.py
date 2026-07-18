import os
import numpy as np
import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms.functional as ttf
from PIL import Image
import torchvision.transforms as transforms
from imgaug import augmenters as iaa
sometimes = lambda aug: iaa.Sometimes(0.8, aug)

def make_dataset(root, train=True):
    dataset = []

    if train:
        IR = os.path.join(root)
        VIS = os.path.join(root)

    for index in range(1000):

        # img_ir = 'IR/IR' + str(index + 24) + '.jpg'
        # img_vis = 'VIS/VIS' + str(index + 24) + '.png'

        img_ir = 'IR1-1-1-1/' + '{:04d}.png'.format(index + 1)
        img_vis = 'VIS1-1-1-1/' + '{:04d}.png'.format(index + 1)
        img_vis_E = 'VIS1-1-1-1_E/' + '{:04d}.png'.format(index + 1)

        dataset.append([os.path.join(IR, img_ir), os.path.join(VIS, img_vis), os.path.join(VIS, img_vis_E)])

    return dataset

class fusiondata(data.Dataset):
    def __init__(self, root, transform=None, train=True, phase='train'):
        self.train = train
        self.phase = phase
        # self._tensor = transforms.ToTensor()
        self._tensor = transforms.Compose([transforms.RandomCrop(128, 128), transforms.ToTensor()])
        if self.train:
            self.train_set_path = make_dataset(root, train)

        self.ir_path = ["./DataSet/TestData/MSRS/IR/" + f"{i + 1}.png" for i in range(0,361)]
        self.vi_path = ["./DataSet/TestData/MSRS/VIS/" + f"{i + 1}.png" for i in range(0,361)]

    def __getitem__(self, index):
        if self.train:

            imgA_path, imgB_path, imgB_E_path = self.train_set_path[index]

            imgA = Image.open(imgA_path).convert('L')   #L 将图像变成灰度图
            imgB = Image.open(imgB_path).convert('L')   #L 将图像变成灰度图
            imgB_E = Image.open(imgB_E_path).convert('L')  # L 将图像变成灰度图

            # irImage = ttf.to_tensor(imgA)  # 将图片转为张量
            # visImage = ttf.to_tensor(imgB)

            irImage = self._tensor(imgA)  # 将图片转为张量
            visImage = self._tensor(imgB)
            visImage_E = self._tensor(imgB_E)

            [imgA, imgB, imgB_E] = transform_augment([irImage, visImage, visImage_E], phase=self.phase, min_max=(0, 1))

            imgA = imgA.unsqueeze(0)               # 添加batch维度
            imgB = imgB.unsqueeze(0)
            imgB_E = visImage_E.unsqueeze(0)

            return imgA, imgB, imgB_E
        else:
            imgA1 = Image.open(self.ir_path[index]).convert('L')
            imgA2 = Image.open(self.vi_path[index]).convert('L')
            imgA1 = np.asarray(imgA1)
            imgA2 = np.asarray(imgA2)

            imgA1 = np.atleast_3d(imgA1).transpose(2, 0, 1).astype(float)
            imgA2 = np.atleast_3d(imgA2).transpose(2, 0, 1).astype(float)

            C, Row, Col = imgA1.shape

            imgA1_n = imgA1 / float(255)
            imgA2_n = imgA2 / float(255)

            imgA1 = torch.from_numpy(imgA1_n).float()
            imgA2 = torch.from_numpy(imgA2_n).float()

            imgA1 = imgA1.view(1, 1, Row, Col)
            imgA2 = imgA2.view(1, 1, Row, Col)

            return imgA1, imgA2, imgA1_n, imgA2_n


    def __len__(self):
        if self.train:
            return 600
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