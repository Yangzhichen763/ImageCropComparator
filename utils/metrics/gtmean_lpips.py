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

try:
    from .util import gt_mean
    from .lpips import calculate_lpips
except:
    from utils.metrics.util import gt_mean
    from utils.metrics.lpips import calculate_lpips


__all__ = ['calculate_gt_mean_lpips', 'GTMeanLPIPS']


error_func_tensor = get_func('mse', "tensor")
error_func_np = get_func('mse', "np")


@METRICS_REGISTRY.register()
class GTMeanLPIPS:
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
        pred = torch.round(pred * 255) / 255
        target = torch.round(target * 255) / 255

        return calculate_gt_mean_lpips(pred, target, **self.metrics_kwargs, device=pred.device)


# noinspection SpellCheckingInspection
@paired_reduce
def calculate_gt_mean_lpips(image_1, image_2, *args, **kwargs):
    image_1, image_2 = gt_mean(image_1, image_2)
    return calculate_lpips(image_1, image_2, *args, **kwargs)
#endregion