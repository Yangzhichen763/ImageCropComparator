import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim # pip install scikit-image

import torch
import torch.nn.functional as F

from utils.registry import METRICS_REGISTRY

from .util import paired_reduce, _reduction_modes


__all__ = ['calculate_ssim', 'SSIM']


GAUSSIAN_WINDOW = None  # 用于缓存高斯核


@METRICS_REGISTRY.register()
class SSIM:
    """Structural Similarity Index Measure (SSIM) .

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the SSIM calculation.
    """
    metric_mode = 'FR'

    def __init__(self, reduction='mean', **metrics_kwargs):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.reduction = reduction
        self.metrics_kwargs = metrics_kwargs

    def __call__(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        return calculate_ssim(pred, target)


@paired_reduce
def calculate_ssim(image_1, image_2, **kwargs):
    return get_ssim_tensor(image_1, image_2, **kwargs)


# 计算两个图像（PyTorch 数组）的 SSIM 值
# noinspection SpellCheckingInspection
def get_ssim_tensor(image_1, image_2, crop_border=0, gray_scale=False):
    """
    Calculates the Structural Similarity Index (SSIM) between two images.

    Args:
        image_1 (torch.Tensor): Image A. with shape (C, H, W) or (N, C, H, W)
        image_2 (torch.Tensor): Image B. with shape (C, H, W) or (N, C, H, W)
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the SSIM calculation.
        gray_scale (bool): Whether to convert the images to grayscale before calculating SSIM.

    Returns:
        np.ndarray or float: SSIM value, if input images have 3 dimensions, return a float value; else, return a numpy.ndarray with shape (N,)
    """
    image_1 = torch.round(image_1 * 255) / 255
    image_2 = torch.round(image_2 * 255) / 255

    C = image_1.shape[-3]
    # 如果是 RGB 图像，并且选择先转化为 grayscale 图像再计算，则先转化为 grayscale 图像再计算 ssim
    if C == 3 and gray_scale:
        image_1 = torch.mean(image_1, dim=-3, keepdim=True)
        image_2 = torch.mean(image_2, dim=-3, keepdim=True)

    # 剪裁图像边缘
    if crop_border != 0:
        image_1 = image_1[..., crop_border:-crop_border, crop_border:-crop_border, :]
        image_2 = image_2[..., crop_border:-crop_border, crop_border:-crop_border, :]

    # 计算 SSIM
    ssim_value = ssim_tensor(image_1, image_2)
    return ssim_value


# 计算两个图像的 SSIM 值
# noinspection SpellCheckingInspection,PyPep8Naming
def ssim_tensor(image_1, image_2):
    """
    Calculates the Structural Similarity Index (SSIM) between two images using PyTorch.

    Args:
        image_1 (torch.Tensor): Image A with range [0, 1], and shape (N, C, H, W).
        image_2 (torch.Tensor): Image B with range [0, 1], and shape (N, C, H, W).

    Returns:
        float: SSIM value
    """
    # 常数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ws = 11         # window size
    hws = ws // 2   # half window size
    sigma = 1.5

    input_dim = image_1.dim()
    if input_dim == 3:
        image_1 = image_1.unsqueeze(0)
        image_2 = image_2.unsqueeze(0)
    N, C, H, W = image_1.shape
    device = image_1.device

    # 创建高斯核窗口
    global GAUSSIAN_WINDOW
    if GAUSSIAN_WINDOW is None:
        def create_window(window_size, sigma):
            # 创建高斯核窗口
            x = torch.arange(window_size, dtype=torch.float32) - hws
            y = torch.arange(window_size, dtype=torch.float32) - hws

            # 下面这段代码和这段代码功能一致 x, y = torch.meshgrid(x, y, indexing='ij')
            x = x.view(window_size, 1).repeat(1, window_size)  # (window_size, window_size)
            y = y.view(1, window_size).repeat(window_size, 1)  # (window_size, window_size)

            window = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            window = window / window.sum()  # 归一化

            # 将 kernel 扩展为 4D 张量 (out_channels, in_channels, H, W)
            window = window.view(1, 1, ws, ws).repeat(C, 1, 1, 1)  # expend(C, 1, window_size, window_size).contiguous
            return window
        GAUSSIAN_WINDOW = create_window(ws, sigma)
    kernel = GAUSSIAN_WINDOW.to(device)

    # 计算图像的局部均值
    mu1 = F.conv2d(image_1, kernel, padding=hws, groups=C)
    mu2 = F.conv2d(image_2, kernel, padding=hws, groups=C)

    # 计算均方
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 计算图像的局部方差和协方差
    sigma1_sq = F.conv2d(image_1 ** 2, kernel, padding=hws, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(image_2 ** 2, kernel, padding=hws, groups=C) - mu2_sq
    sigma12 = F.conv2d(image_1 * image_2, kernel, padding=hws, groups=C) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 计算 SSIM 值的均值
    ssim_values = ssim_map.mean(dim=(-3, -2, -1))
    ssim_values = ssim_values.detach().cpu()
    return ssim_values
#endregion