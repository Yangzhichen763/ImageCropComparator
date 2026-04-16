
from copy import deepcopy
from .psnr import calculate_psnr, PSNR
from .ssim import calculate_ssim, SSIM

from utils.options.options import parse_params, parse_arguments
from utils.registry import METRICS_REGISTRY


from utils.console.log import get_root_logger
logger = get_root_logger()


# 将每个计算指标的模块都导入进来，同时会激活模块中的 register 装饰器，将指标计算函数注册到 MODEL_REGISTRY 中
# automatically scan and import metrics modules
# scan all the files under the 'metrics' folder and collect files
import importlib
import os.path as osp
from utils.console.path import scandir
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(model_folder, suffix='.py')
]
# import all the model modules
_model_modules = []
for file_name in model_filenames:
    try:
        _model_modules.append(importlib.import_module(f'utils.metrics.{file_name}'))
    except Exception as e:
        logger.warning(f'Failed to import {file_name} because of {e}')


def get_metrics_calculator(option):
    """
    Calculate metrics based on the given options.

    Args:
        option (dict): Options for calculating metrics.

    Returns:
        metrics: value of the calculated metrics.
    """
    option = deepcopy(option)  # deepcopy the option to avoid modification

    metrics_type, metrics_params = parse_params(option, other_as_params=False)

    # dynamic instantiation
    metrics_class = METRICS_REGISTRY.get(metrics_type)
    args, kwargs = parse_arguments(metrics_class, metrics_params)
    metrics_func = metrics_class(*args, **kwargs)

    return metrics_func