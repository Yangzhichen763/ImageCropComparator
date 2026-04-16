from utils.registry import METRICS_REGISTRY

from .util import paired_reduce, _reduction_modes
from .error_func import get_func

try:
    from .util import gt_mean
    from .psnr import calculate_psnr
except:
    from utils.metrics.util import gt_mean
    from utils.metrics.psnr import calculate_psnr


__all__ = ['calculate_gt_mean_psnr', 'GTMeanPSNR']


error_func_tensor = get_func('mse', "tensor")
error_func_np = get_func('mse', "np")


@METRICS_REGISTRY.register()
class GTMeanPSNR:
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
        return calculate_gt_mean_psnr(pred, target, reduction=self.reduction, **self.metrics_kwargs)


# noinspection SpellCheckingInspection
@paired_reduce
def calculate_gt_mean_psnr(image_1, image_2, *args, **kwargs):
    image_1, image_2 = gt_mean(image_1, image_2)
    return calculate_psnr(image_1, image_2, *args, **kwargs)
#endregion