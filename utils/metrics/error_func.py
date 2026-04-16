import numpy as np
import torch


def get_all_funcs():
    all_funcs = []
    for name, func in globals().items():
        if callable(func) and name not in ['get_all_funcs', 'get_func']:
            all_funcs.append(name)
    return all_funcs


def get_func(name, func_type="tensor"):
    if func_type == "tensor":
        return get_func(f"{name}_tensor", "")
    elif func_type == "numpy":
        return get_func(f"{name}_np", "")

    name_lower = name.lower()
    func = globals().get(name_lower, None)
    if func is None:
        raise ValueError(f"Error function \"{name_lower}\" is not defined.")
    return func


def mae(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        return mae_tensor(y_true, y_pred)
    if isinstance(y_true, np.ndarray):
        return mae_np(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported data type: {type(y_true)} for y_true and {type(y_pred)} for y_pred.")

def mae_np(y_true, y_pred):
    return np.abs(y_true - y_pred)

def mae_tensor(y_true, y_pred):
    return torch.abs(y_true - y_pred)


def mse(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        return mse_tensor(y_true, y_pred)
    if isinstance(y_true, np.ndarray):
        return mse_np(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported data type: {type(y_true)} for y_true and {type(y_pred)} for y_pred.")

def mse_np(y_true, y_pred):
    return (y_true - y_pred) ** 2

def mse_tensor(y_true, y_pred):
    return (y_true - y_pred) ** 2
