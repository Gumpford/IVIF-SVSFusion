import os
import numpy as np
import torch.utils.data as data
import torch
# import imageio
from PIL import Image
import torchvision.transforms as transforms


def make_dataset(root, train=True):
    dataset = []

    if train:
        IR = os.path.join(root)
        VIS = os.path.join(root)

    for index in range(1000):

        img_ir = 'ir/ir' + '{}.png'.format(index + 1)
        img_vis = 'vi/vi' + '{}.png'.format(index + 1)
        img_vis_E = 'vi_E/vi' + '{}.png'.format(index + 1)
        dataset.append([os.path.join(IR, img_ir), os.path.join(VIS, img_vis), os.path.join(VIS, img_vis_E)])

    return dataset


class fusiondata(data.Dataset):
    def __init__(self, root, transform=None, train=True):
        self.train = train
        # self._tensor = transforms.ToTensor()
        self._tensor = transforms.Compose([transforms.RandomCrop(128, 128), transforms.ToTensor()])
        if self.train:
            self.train_set_path = make_dataset(root, train)

    def __getitem__(self, idx):
        if self.train:

            imgA_path, imgB_path, imgB_E_path = self.train_set_path[idx]

            imgA = Image.open(imgA_path)
            imgA = imgA.convert('L')   #L 将图像变成灰度图
            imgA = self._tensor(imgA)

            imgB = Image.open(imgB_path)
            imgB = imgB.convert('L')
            imgB = self._tensor(imgB)

            imgB_E = Image.open(imgB_E_path)
            imgB_E = imgB_E.convert('L')
            imgB_E = self._tensor(imgB_E)

            return imgA, imgB, imgB_E

    def __len__(self):
        if self.train:
            return 500 #如果要输出16张，就要改成16
        else:
            return 500


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)




