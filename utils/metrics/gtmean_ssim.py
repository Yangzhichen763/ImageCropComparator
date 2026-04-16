import cv2
import numpy as np

import torch
import torch.nn.functional as F

from utils.registry import METRICS_REGISTRY

from .util import paired_reduce, _reduction_modes

try:
    from .util import gt_mean
    from .ssim import calculate_ssim
except:
    from utils.metrics.util import gt_mean
    from utils.metrics.ssim import calculate_ssim


__all__ = ['calculate_gt_mean_ssim', 'GTMeanSSIM']


GAUSSIAN_WINDOW = None  # 用于缓存高斯核


@METRICS_REGISTRY.register()
class GTMeanSSIM:
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
        return calculate_gt_mean_ssim(pred, target, reduction=self.reduction, **self.metrics_kwargs)


@paired_reduce
def calculate_gt_mean_ssim(image_1, image_2, *args, **kwargs):
    image_1, image_2 = gt_mean(image_1, image_2)
    return calculate_ssim(image_1, image_2, *args, **kwargs)
#endregion