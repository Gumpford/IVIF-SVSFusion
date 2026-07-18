import numpy as np
import math
from scipy.signal import convolve2d
import torch.nn.functional as F
import torch

def sobel_fn(x):
    # Sobel operators
    vtemp = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    htemp = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8

    a, b = htemp.shape
    x_ext = per_extn_im_fn(x, a)
    p, q = x_ext.shape
    gv = np.zeros((p - 2, q - 2))
    gh = np.zeros((p - 2, q - 2))
    gv = convolve2d(x_ext, vtemp, mode='valid')
    gh = convolve2d(x_ext, htemp, mode='valid')
    # for ii in range(1, p - 1):
    #     for jj in range(1, q - 1):
    #         gv[ii - 1, jj - 1] = np.sum(x_ext[ii - 1:ii + 2, jj - 1:jj + 2] * vtemp)
    #         gh[ii - 1, jj - 1] = np.sum(x_ext[ii - 1:ii + 2, jj - 1:jj + 2] * htemp)

    return gv, gh


def per_extn_im_fn(x, wsize):
    """
    Periodic extension of the given image in 4 directions.

    xout_ext = per_extn_im_fn(x, wsize)

    Periodic extension by (wsize-1)/2 on all 4 sides.
    wsize should be odd.

    Example:
        Y = per_extn_im_fn(X, 5);    % Periodically extends 2 rows and 2 columns in all sides.
    """

    hwsize = (wsize - 1) // 2  # Half window size excluding centre pixel.

    p, q = x.shape
    xout_ext = np.zeros((p + wsize - 1, q + wsize - 1))
    xout_ext[hwsize: p + hwsize, hwsize: q + hwsize] = x

    # Row-wise periodic extension.
    if wsize - 1 == hwsize + 1:
        xout_ext[0: hwsize, :] = xout_ext[2, :].reshape(1, -1)
        xout_ext[p + hwsize: p + wsize - 1, :] = xout_ext[-3, :].reshape(1, -1)

    # Column-wise periodic extension.
    xout_ext[:, 0: hwsize] = xout_ext[:, 2].reshape(-1, 1)
    xout_ext[:, q + hwsize: q + wsize - 1] = xout_ext[:, -3].reshape(-1, 1)

    return xout_ext

def get_Qabf(pA, pB, pF):
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5;
    Ta = 0.9879
    ka = -22
    Da = 0.8

    # Sobel Operator Sobel算子
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

    # if y is the response to h1 and x is the response to h3;then the intensity is sqrt(x^2+y^2) and  is arctan(y/x);
    # 如果y对应h1，x对应h2，则强度为sqrt(x^2+y^2)，方向为arctan(y/x)

    strA = pA
    strB = pB
    strF = pF

    # 数组旋转180度
    def flip180(arr):
        return np.flip(arr)

    # 相当于matlab的Conv2
    def convolution(k, data):
        k = flip180(k)
        data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        img_new = convolve2d(data, k, mode='valid')
        return img_new

    def getArray(img):
        SAx = convolution(h3, img)
        SAy = convolution(h1, img)
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        n, m = img.shape
        aA = np.zeros((n, m))
        zero_mask = SAx == 0
        aA[~zero_mask] = np.arctan(SAy[~zero_mask] / SAx[~zero_mask])
        aA[zero_mask] = np.pi / 2
        # for i in range(n):
        #     for j in range(m):
        #         if (SAx[i, j] == 0):
        #             aA[i, j] = math.pi / 2
        #         else:
        #             aA[i, j] = math.atan(SAy[i, j] / SAx[i, j])
        return gA, aA

    # 对strB和strF进行相同的操作
    gA, aA = getArray(strA)
    gB, aB = getArray(strB)
    gF, aF = getArray(strF)

    # the relative strength and orientation value of GAF,GBF and AAF,ABF;
    def getQabf(aA, gA, aF, gF):
        mask = (gA > gF)
        GAF = np.where(mask, gF / gA, np.where(gA == gF, gF, gA / gF))

        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)

        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))

        QAF = QgAF * QaAF
        return QAF

    QAF = getQabf(aA, gA, aF, gF)
    QBF = getQabf(aB, gB, aF, gF)

    # 计算QABF
    deno = np.sum(gA + gB)
    nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
    output = nume / deno
    return output

def get_Qabf_1(pA, pF):
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5;
    Ta = 0.9879
    ka = -22
    Da = 0.8

    # Sobel Operator Sobel算子
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

    # if y is the response to h1 and x is the response to h3;then the intensity is sqrt(x^2+y^2) and  is arctan(y/x);
    # 如果y对应h1，x对应h2，则强度为sqrt(x^2+y^2)，方向为arctan(y/x)

    strA = pA
    strF = pF

    # 数组旋转180度
    def flip180(arr):
        return np.flip(arr)

    # 相当于matlab的Conv2
    def convolution(k, data):
        k = flip180(k)
        data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        img_new = convolve2d(data, k, mode='valid')
        return img_new

    def getArray(img):
        SAx = convolution(h3, img)
        SAy = convolution(h1, img)
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        n, m = img.shape
        aA = np.zeros((n, m))
        zero_mask = SAx == 0
        aA[~zero_mask] = np.arctan(SAy[~zero_mask] / SAx[~zero_mask])
        aA[zero_mask] = np.pi / 2
        # for i in range(n):
        #     for j in range(m):
        #         if (SAx[i, j] == 0):
        #             aA[i, j] = math.pi / 2
        #         else:
        #             aA[i, j] = math.atan(SAy[i, j] / SAx[i, j])
        return gA, aA

    # 对strB和strF进行相同的操作
    gA, aA = getArray(strA)
    gF, aF = getArray(strF)

    # the relative strength and orientation value of GAF,GBF and AAF,ABF;
    def getQabf(aA, gA, aF, gF):
        mask = (gA > gF)
        GAF = np.where(mask, gF / gA, np.where(gA == gF, gF, gA / gF))

        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)

        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))

        QAF = QgAF * QaAF
        return QAF

    QAF = getQabf(aA, gA, aF, gF)

    # 计算QABF
    deno = np.sum(gA)
    nume = np.sum(np.multiply(QAF, gA))
    output = nume / deno
    return output


def get_Qabf_2(strA, strB, strF):
    # Convert input images to tensors
    pA = torch.Tensor(strA)
    pB = torch.Tensor(strB)
    pF = torch.Tensor(strF)

    # Define Sobel operator
    h1 = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    h3 = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Compute Sobel responses and gradients for image A
    SAx = F.conv2d(pA.unsqueeze(0).unsqueeze(0), h3.unsqueeze(0).unsqueeze(0), padding=1)
    SAy = F.conv2d(pA.unsqueeze(0).unsqueeze(0), h1.unsqueeze(0).unsqueeze(0), padding=1)
    gA = torch.sqrt(SAx**2 + SAy**2)
    aA = torch.atan(SAy / (SAx + 1e-6))

    # Compute Sobel responses and gradients for image B
    SBx = F.conv2d(pB.unsqueeze(0).unsqueeze(0), h3.unsqueeze(0).unsqueeze(0), padding=1)
    SBy = F.conv2d(pB.unsqueeze(0).unsqueeze(0), h1.unsqueeze(0).unsqueeze(0), padding=1)
    gB = torch.sqrt(SBx**2 + SBy**2)
    aB = torch.atan(SBy / (SBx + 1e-6))

    # Compute Sobel responses and gradients for fusion result
    SFx = F.conv2d(pF.unsqueeze(0).unsqueeze(0), h3.unsqueeze(0).unsqueeze(0), padding=1)
    SFy = F.conv2d(pF.unsqueeze(0).unsqueeze(0), h1.unsqueeze(0).unsqueeze(0), padding=1)
    gF = torch.sqrt(SFx**2 + SFy**2)
    aF = torch.atan(SFy / (SFx + 1e-6))

    # Compute the relative strength and orientation values
    GAF = torch.where(gA > gF, gF / (gA + 1e-6), gA / (gF + 1e-6))
    AAF = 1 - torch.abs(aA - aF) / (np.pi/2)

    GBF = torch.where(gB > gF, gF / (gB + 1e-6), gB / (gF + 1e-6))
    ABF = 1 - torch.abs(aB - aF) / (np.pi/2)

    # Model parameters
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8

    # Compute the quality measures
    QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))
    QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))
    QAF = QgAF * QaAF

    QgBF = Tg / (1 + torch.exp(kg * (GBF - Dg)))
    QaBF = Ta / (1 + torch.exp(ka * (ABF - Da)))
    QBF = QgBF * QaBF

    # Compute QABF
    deno = gA.sum() + gB.sum()
    nume = (QAF * gA + QBF * gB).sum()
    output = nume / deno

    return output