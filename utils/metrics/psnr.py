import numpy as np

import torch

from utils.registry import METRICS_REGISTRY

from .util import paired_reduce, _reduction_modes
from .error_func import get_func


__all__ = ['calculate_psnr', 'PSNR']


error_func_tensor = get_func('mse', "tensor")


@METRICS_REGISTRY.register()
class PSNR:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the PSNR calculation.
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
            pred (Tensor): of shape (B, C, H, W). Predicted tensor.
            target (Tensor): of shape (B, C, H, W). Ground truth tensor.
        """
        return get_psnr_tensor(pred, target)


@paired_reduce
def calculate_psnr(image_1, image_2, **kwargs):
    return get_psnr_tensor(image_1, image_2, **kwargs)


# 计算两个或两组图像（PyTorch 张量）的 PSNR 值
# noinspection SpellCheckingInspection
def get_psnr_tensor(image_1, image_2, crop_border=0):
    """
    Calculate PSNR between two images (PyTorch tensors), with range [0, 1], and shape (C, H, W) or (N, C, H, W)

    Args:
        image_1 (torch.Tensor): Image A, with range [0, 1], and shape (C, H, W) or (N, C, H, W)
        image_2 (torch.Tensor): Image B, with range [0, 1], and shape (C, H, W) or (N, C, H, W)
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the PSNR calculation.
    Returns:
        torch.Tensor or float: PSNR value, if input images have shape (C, H, W), return a float value; else, return a numpy.ndarray with shape (N,)
    """
    # 确保图像数据类型为浮点数
    image_1 = image_1.type(torch.float64)
    image_2 = image_2.type(torch.float64)

    # 剪裁图像边缘
    if crop_border != 0:
        image_1 = image_1[crop_border:-crop_border, crop_border:-crop_border, ...]
        image_2 = image_2[crop_border:-crop_border, crop_border:-crop_border, ...]

    image_1 = torch.round(image_1 * 255) / 255
    image_2 = torch.round(image_2 * 255) / 255

    # 计算 MSE
    error = error_func_tensor(image_1, image_2)
    mse = torch.mean(error, dim=(-3, -2, -1))

    # 计算 PSNR
    # 如果 MSE 为 0，说明两幅图像完全相同，PSNR 为无穷大
    psnr = -20 * torch.log10(torch.sqrt(mse))
    psnr = psnr.detach().cpu()
    return psnr