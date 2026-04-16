#region import external lpips
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Directory containing this file, e.g. .../basic/metrics
_this_dir = Path(__file__).resolve().parent

# 1) If "lpips" was previously (incorrectly) loaded from this file, remove it from the import cache
_mod = sys.modules.get("lpips")
if _mod is not None:
    mod_file = getattr(_mod, "__file__", "") or ""
    if Path(mod_file).resolve() == Path(__file__).resolve():
        sys.modules.pop("lpips", None)

# 2) Temporarily remove this directory from sys.path
#    Also handle the empty-string entry which represents the current working directory (cwd)
_removed_items: list[tuple[int, str]] = []

for i in range(len(sys.path) - 1, -1, -1):
    p = sys.path[i]

    # Empty string means "current working directory"
    if not p:
        if Path.cwd().resolve() == _this_dir:
            _removed_items.append((i, sys.path.pop(i)))
        continue

    try:
        if Path(p).resolve() == _this_dir:
            _removed_items.append((i, sys.path.pop(i)))
    except Exception:
        # Skip entries that cannot be resolved
        pass

# 3) Import the external pip package "lpips" and alias it to avoid name conflicts
_lpips = importlib.import_module("lpips")

# 4) Restore the removed sys.path entries back to their original positions
for idx, val in sorted(_removed_items, key=lambda x: x[0]):
    sys.path.insert(idx, val)
#endregion

import torch
import numpy as np
import math

from utils.registry import METRICS_REGISTRY

from .util import paired_reduce, _reduction_modes
from .error_func import get_func



__all__ = ['calculate_lpips', 'LPIPS']


@METRICS_REGISTRY.register()
class LPIPS:
    """
    Calculate MABD(Mean Absolute Brightness Difference) between two videos.

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
        self.net_name = metrics_kwargs.pop('net_name', 'alex')
        self.net = _lpips.LPIPS(net=self.net_name, verbose=False)
        self.metrics_kwargs = metrics_kwargs

        self.metrics_kwargs.update(dict(lpips_net=self.net))


    def __call__(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (B, N, C, H, W). Predicted tensor.
            target (Tensor): of shape (B, N, C, H, W). Ground truth tensor.
        """
        return calculate_lpips(pred, target, device=pred.device, **self.metrics_kwargs)


# [⭐]计算两个图像（numpy 数组）的 LPIPS 值
# noinspection SpellCheckingInspection
@paired_reduce
def calculate_lpips(image_1, image_2, lpips_net, device='cpu'):
    """
    Calculate LPIPS between two images (numpy arrays or PyTorch tensors).

    Args:
        image_1 (torch.Tensor): Image A. A numpy array, with range [0, 1], and shape (H, W, C)
        image_2 (torch.Tensor): Image B. A numpy array, with range [0, 1], and shape (H, W, C)
        lpips_net (torch.nn.Module): The LPIPS network.
        device (str): The device to run the calculations on.

    Returns:
        np.ndarray or float: LPIPS value, if input images have 3 dimensions, return a float value; else, return a numpy.ndarray with shape (N,)
    """
    image_1 = torch.round(image_1 * 255) / 255 * 2 - 1
    image_2 = torch.round(image_2 * 255) / 255 * 2 - 1

    lpips_net.to(device)
    lpips = lpips_net.forward(image_1, image_2)
    return lpips
