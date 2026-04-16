import numpy as np
import functools
from torch.nn import functional as F

import torch.nn as nn
import torch
from typing import Optional, Union, Callable, List
from utils.console.log import highlight_diff
from copy import deepcopy



_reduction_modes = ['none', 'mean', 'sum']


def clone_module(
        module: nn.Module,
        n: Optional[int] = None,
        reset_parameters: Union[bool, Callable] = True,
        init_fn: Optional[Callable] = None,
        param_names: Optional[List[str]] = None,
) -> Union[nn.Module, nn.ModuleList]:
    """Deep clone a module with optional parameter re-initialization.

    Args:
        module: The module to clone.
        n: If None or 1, return a single cloned module.
           If >1, return a ModuleList of cloned modules.
        reset_parameters:
           - If True, call module.reset_parameters() if it exists.
           - If False, skip initialization.
           - If a function (e.g., `lambda m: m.weight.data.normal_(0, 0.01)`), use it to initialize.
        init_fn: A function to initialize parameters (deprecated if reset_parameters is a function).
        param_names: Only initialize parameters whose name contains any of these strings (e.g., ['weight']).
    """

    def _deep_copy_module(module: nn.Module, memo=None) -> nn.Module:
        if memo is None:
            memo = {}

        if not isinstance(module, torch.nn.Module):
            return module

        # check if module has already been cloned
        if id(module) in memo:
            return memo[id(module)]

        # create a new module without __init__
        clone = module.__new__(type(module))
        memo[id(module)] = clone

        # shallow copy basic attributes
        clone.__dict__ = {
            k: v for k, v in module.__dict__.items()
        }

        # deep copy parameters, buffers, and child modules
        clone._parameters = {}
        for name, param in module._parameters.items():
            if param is not None:
                param_ptr = param.data_ptr()
                if param_ptr in memo:
                    clone._parameters[name] = memo[param_ptr]
                else:
                    cloned_param = param.clone()
                    clone._parameters[name] = cloned_param
                    memo[param_ptr] = cloned_param

        clone._buffers = {}
        for name, buffer in module._buffers.items():
            if buffer is not None:
                buffer_ptr = buffer.data_ptr()
                if buffer_ptr in memo:
                    clone._buffers[name] = memo[buffer_ptr]
                else:
                    cloned_buffer = buffer.clone()
                    clone._buffers[name] = cloned_buffer
                    memo[buffer_ptr] = cloned_buffer

        clone._modules = {}
        for name, child in module._modules.items():
            if child is not None:
                clone._modules[name] = _deep_copy_module(child, memo)

        clone.training = module.training
        return clone


    def _detach_module(module: nn.Module):
        if not isinstance(module, torch.nn.Module):
            return

        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                module._parameters[param_key] = module._parameters[param_key].detach_()

        for buffer_key in module._buffers:
            if module._buffers[buffer_key] is not None and \
                    module._buffers[buffer_key].requires_grad:
                module._buffers[buffer_key] = module._buffers[buffer_key].detach_()

        for module_key in module._modules:
            _detach_module(module._modules[module_key])


    def _check_shared_params_or_buffers(module1, module2):
        # 检查参数
        for (name1, param1), (name2, param2) in zip(module1.named_parameters(), module2.named_parameters()):
            if param1.data_ptr() == param2.data_ptr():
                print(f"Parameter '{name1}' and '{name2}' share memory (potential reference)!")

        # 检查缓冲区
        for (name1, buf1), (name2, buf2) in zip(module1.named_buffers(), module2.named_buffers()):
            if buf1.data_ptr() == buf2.data_ptr():
                print(f"Buffer '{name1}' and '{name2}' share memory (potential reference)!")

        # 递归检查子模块
        for (name1, child1), (name2, child2) in zip(module1.named_children(), module2.named_children()):
            _check_shared_params_or_buffers(child1, child2)


    def _are_modules_fully_independent(module1, module2):
        # 检查参数是否独立
        for (name1, param1), (name2, param2) in zip(module1.named_parameters(), module2.named_parameters()):
            if param1.data_ptr() == param2.data_ptr():
                return False

        # 检查缓冲区是否独立
        for (name1, buf1), (name2, buf2) in zip(module1.named_buffers(), module2.named_buffers()):
            if buf1.data_ptr() == buf2.data_ptr():
                return False

        # 递归检查子模块
        for (name1, child1), (name2, child2) in zip(module1.named_children(), module2.named_children()):
            if not _are_modules_fully_independent(child1, child2):
                return False

        return True


    def _clone_module(module: nn.Module) -> nn.Module:
        # Plan A: Deep copy the module
        _module = deepcopy(module)
        if module.__repr__() != _module.__repr__():
            print(f"{highlight_diff(module.__repr__(), _module.__repr__())}\n"
                  f"Cloned module has different parameters.")
        _check_shared_params_or_buffers(module, _module)
        if not _are_modules_fully_independent(module, _module):
            print("Cloned module shares parameters or buffers with the original module.")
        # Plan B: Shallow copy the module and manually copy parameters and buffers
        # _module = type(module)(*module.args, **module.kwargs)
        # _module.load_state_dict(deepcopy(module.state_dict()))
        # Plan C: Deep copy the module and detach all parameters and buffers
        # _module = _deep_copy_module(module)
        # _detach_module(_module)

        # Case 1: reset_parameters is a custom function
        if callable(reset_parameters):
            reset_parameters(_module)

        # Case 2: reset_parameters is True and module has .reset_parameters()
        elif reset_parameters and hasattr(_module, 'reset_parameters'):
            _module.reset_parameters()

        # Case 3: Use init_fn if provided (fallback)
        elif init_fn is not None:
            _apply_init_fn(_module, init_fn, param_names)

        return _module


    def _apply_init_fn(
            module: nn.Module,
            init_fn: Callable,
            param_names: Optional[List[str]] = None,
    ) -> None:
        """
        Helper to apply init_fn to parameters recursively.
        """
        for name, param in module.named_parameters(recurse=False):
            if param_names is None or any(p in name for p in param_names):
                init_fn(param)

        # Recursively apply to child modules
        for child in module.children():
            _apply_init_fn(child, init_fn, param_names)

    # Validate n
    if n is not None and n <= 0:
        raise ValueError(f"n must be positive or None, got {n}")

    # Clone single module
    if n is None:
        module = _clone_module(module)
        return module

    # Clone into ModuleList
    module_list = nn.ModuleList([_clone_module(module) for _ in range(n)])
    return module_list


#region ==[Reduction Utils]==
def reduce(tensor, reduction='mean'):
    """Reduce tensor as specified.

    Args:
        tensor (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return tensor
    elif reduction_enum == 1:
        return tensor.mean()
    else:
        return tensor.sum()


def paired_reduce(metrics_func):
    """
    Create a reduction version for metrics function.
    """

    @functools.wraps(metrics_func)
    def wrapper(pred, target, reduction='mean', **kwargs):
        loss = metrics_func(pred, target, **kwargs)
        loss = reduce(loss, reduction)
        return loss

    return wrapper


def unpaired_reduce(metrics_func):
    """
    Create a reduction version for metrics function.
    """

    @functools.wraps(metrics_func)
    def wrapper(pred, reduction='mean', **kwargs):
        loss = metrics_func(pred, **kwargs)
        loss = reduce(loss, reduction)
        return loss

    return wrapper
#endregion


def gt_mean(pred, gt, mode=None):
    """
    Adjusts the brightness of the input predicted image (pred) to match the average brightness of the target image (gt).

    Parameters:
        pred (torch.Tensor): The predicted image, with range [0, 1], and shape (B, C, H, W), and data type float32.
        gt (torch.Tensor): The target image (Ground Truth), with range [0, 1], and shape (B, C, H, W), and data type float32.

    Returns:
        gt (torch.Tensor): The unmodified target image.
        pred (torch.Tensor): The brightness-adjusted predicted image, with the same shape and data type as the input.
    """
    # Compute the mean brightness of the target (gt) and predicted (pred) images
    if mode == 'channel-wise':
        mean_pred = pred.mean(dim=(2, 3), keepdim=True)
        mean_gt = gt.mean(dim=(2, 3), keepdim=True)
    elif mode == 'brightness':
        def get_brightness(rgb):
            r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            return brightness
        mean_pred = get_brightness(pred).mean()
        mean_gt = get_brightness(gt).mean()
    else:
        mean_pred = pred.mean()
        mean_gt = gt.mean()

    # Adjust the brightness of the predicted image by scaling
    scaled_pred = pred * (mean_gt / mean_pred)

    # Ensure the output is within [0, 1] range
    pred = scaled_pred.clamp(0, 1)

    return pred, gt