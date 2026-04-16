
import enum
import numpy as np
import torch
from torch.nn import functional as F
from contextlib import contextmanager


'''
数据格式的转换，主要将 numpy 转换为 tensor 以及 tensor 转换为 numpy
'''


class Type(enum.Enum):
    TENSOR = enum.auto()
    NUMPY = enum.auto()


# 计算两个图像之间的指标
def apply(
        image_1, image_2,
        np_func, tensor_func,
        device=None,
        **func_kwargs
):
    # 列表类型的输入将其 stack（序列帧类型的输入按照时间维度拼接）
    if isinstance(image_1, list):
        if isinstance(image_1[0], np.ndarray):
            image_1 = np.stack(image_1, axis=0)
            image_2 = np.stack(image_2, axis=0)
        elif isinstance(image_1[0], torch.Tensor):
            image_1 = torch.stack(image_1, dim=0)
            image_2 = torch.stack(image_2, dim=0)

    assert image_1.shape == image_2.shape, \
        f'Input data must have the same dimensions. instead of {image_1.shape} and {image_2.shape}.'

    if isinstance(image_1, np.ndarray):
        return np_func(image_1, image_2, **func_kwargs)
    elif isinstance(image_1, torch.Tensor):
        with torch.no_grad():  # 禁用梯度计算
            if device is not None:
                image_1 = image_1.to(device)
                image_2 = image_2.to(device)
            return tensor_func(image_1, image_2, **func_kwargs)
    else:
        raise TypeError(f'Input data must be a numpy array or a PyTorch tensor. Got {type(image_1)} instead.')


# convert tensor to numpy array
def tensor2numpy(tensor: torch.Tensor, change_range=True, clip_range=True, reverse_channels=True, squeeze_dim=True):
    """ Convert a PyTorch tensor to a numpy array.

    Args:
        tensor(torch.Tensor): Accept shapes:
            1) 4D Tensor of shape (B, 3, H, W) if reverse_channels is True, else (B, C, H, W)
            2) 3D Tensor of shape (3, H, W) if reverse_channels is True, else (C, H, W)
        change_range(bool): If True, change the range of the tensor to uint8 with range [0, 255].
        clip_range(bool): If True, clip the values of the tensor to [0, 1] before changing the range.
        reverse_channels(bool): If True, reverse the order of the channels, e.g. RGB -> BGR or BGR -> RGB.
        squeeze_dim(bool): If True, squeeze the dimensions of the tensor if it has a size of 1 at the beginning.

    Returns:
        The input tensor converted to a numpy array.
    """
    # 下方代码为什么不直接调用下面现成的函数？
    # 为了代码更加直观（代码量不大，无需做多余的规范化工作）

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'Input data must be a PyTorch tensor. Got {type(tensor)} instead.')

    tensor = tensor.detach().cpu()

    if squeeze_dim:
        while tensor.dim() > 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)              # (1, C, H, W)    -> (C, H, W)

    if reverse_channels and tensor.shape[-3] == 3:  # RGB             -> BGR
        if tensor.dim() == 3:
            tensor = tensor[[2, 1, 0], :, :]
        elif tensor.dim() >= 4:
            tensor = tensor[..., [2, 1, 0], :, :]

    dims = tensor.dim()
    if dims >= 3:
        permute_order = list(range(dims - 3)) + [dims - 2, dims - 1, dims - 3]
        tensor = tensor.permute(*permute_order)     # (..., C, H, W)    -> (..., H, W, C)
    nparray = tensor.numpy()                        # tensor            -> numpy

    if clip_range:
        nparray = np.clip(nparray, 0, 1)

    if change_range:
        nparray = ((nparray * 255.0)                # [0, 1]            -> [0, 255]
                   .round().astype(np.uint8))       # float             -> uint8
    return nparray


def tensors2numpys(tensors, change_range=True, clip_range=True, reverse_channels=True, squeeze_dim=True):
    """
    Convert a list of PyTorch tensors to a list of numpy arrays.
    """
    if isinstance(tensors, torch.Tensor):
        return tensor2numpy(tensors, change_range, clip_range, reverse_channels, squeeze_dim)
    elif isinstance(tensors, list) and all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return [tensor2numpy(tensor, change_range, clip_range, reverse_channels, squeeze_dim) for tensor in tensors]
    else:
        raise TypeError(f'Input data must be a list of PyTorch tensors. Got {type(tensors)} instead.')

# | Feature             | `torchvision.transforms.ToTensor()`              | `numpy2tensor`Function                                |
# |---------------------|--------------------------------------------------|-------------------------------------------------------|
# | Input Data Type     | Only supports`PIL.Image`or`numpy.ndarray`        | Only supports`numpy.ndarray`                          |
# | Output Data Type    | `torch.float32`                                  | `torch.float32`                                       |
# | Value Range         | Auto- normalizes to`[0.0, 1.0]`                  | Optional normalization to`[0.0, 1.0]`(`change_range`) |
# | Channel Order       | Auto- converts BGR (OpenCV default) to RGB       | Optional channel reversal (`reverse_channels`)        |
# | Dimension Expansion | Auto- converts`(H, W, C)`to`(C, H, W)`           | Optional expansion to 4D (`expand_dim`)               |
# | Value Clipping      | No clipping functionality                        | Optional clipping to`[0, 1]`(`clip_range`)            |
# | Batch Data Support  | Does not support batch data (only single images) | Supports batch data (4D input`(B, H, W, C)`)          |
# convert numpy array to tensor
def numpy2tensor(nparray: np.ndarray, change_range=True, clip_range=True, reverse_channels=True, expand_dim=True):
    """ Convert a numpy array to a PyTorch tensor.

    Equivalent Implementation of ToTensor():
    ```python
    import torch
    import numpy as np

    def to_tensor(image):
        if isinstance(image, np.ndarray):
            # 转换为 float32 并归一化到 [0, 1]
            tensor = torch.from_numpy(image).float() / 255.0
            # 转换通道顺序 (H, W, C) -> (C, H, W)
            if tensor.ndim == 3:
                tensor = tensor.permute(2, 0, 1)
            return tensor
        else:
            raise TypeError("Input must be a numpy array.")
    ```

    Args:
        nparray(np.ndarray): Accept shapes:
            1) 4D array of shape (B, H, W, C) if expand_dim is True, else (B, H, W, C)
            2) 3D array of shape (H, W, C) if expand_dim is True, else (H, W, C)
        change_range(bool): If True, change the range of the tensor to float32 with range [0, 1].
        clip_range(bool): If True, clip the values of the tensor to [0, 1] after changing the range.
        reverse_channels(bool): If True, reverse the order of the channels, e.g. BGR -> RGB or RGB -> BGR.
        expand_dim(bool): If True, expand the dimensions of the tensor until it has a size of 4.

    Returns:
        The input numpy array converted to a PyTorch tensor.
    """
    # 下方代码为什么不直接调用下面现成的函数？
    # 为了代码更加直观（代码量不大，无需做多余的规范化工作）

    if not isinstance(nparray, np.ndarray):
        raise TypeError(f'Input data must be a numpy array. Got {type(nparray)} instead.')

    tensor = torch.from_numpy(nparray)              # numpy           -> tensor
    dims = nparray.ndim
    if dims >= 3:
        permute_order = list(range(dims - 3)) + [dims - 1, dims - 3, dims - 2]
        tensor = tensor.permute(*permute_order)     # (..., H, W, C)  -> (..., C, H, W)

    if change_range:
        tensor = (tensor.type(torch.float32)        # uint8           -> float
                  / 255.0)                          # [0, 255]        -> [0, 1]

    if clip_range:
        tensor = tensor.clamp(0, 1)

    if reverse_channels:                            # BGR             -> RGB
        if tensor.dim() == 3:
            tensor = tensor[[2, 1, 0], :, :]
        elif tensor.dim() >= 4:
            tensor = tensor[..., [2, 1, 0], :, :]

    if expand_dim:
        while tensor.dim() < 4:
            tensor = tensor.unsqueeze(0)            # (C, H, W)       -> (1, C, H, W)
    return tensor


def numpys2tensors(nparrays, change_range=True, clip_range=True, reverse_channels=True, expand_dim=True):
    """
    Convert a list of numpy arrays to a list of PyTorch tensors.
    """
    if isinstance(nparrays, np.ndarray):
        return numpy2tensor(nparrays, change_range, clip_range, reverse_channels, expand_dim)
    elif isinstance(nparrays, list) and all(isinstance(nparray, np.ndarray) for nparray in nparrays):
        return [numpy2tensor(nparray, change_range, clip_range, reverse_channels, expand_dim) for nparray in nparrays]
    else:
        raise TypeError(f'Input data must be a list of numpy arrays. Got {type(nparrays)} instead.')


# 将图像数据转换为列表
def to_list(value):
    if isinstance(value, list):
        return value
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    elif isinstance(value, (int, float)):
        return [value]
    elif isinstance(value, dict):
        return value
    else:
        raise TypeError(f'Input data must be either a list, numpy array, PyTorch tensor, int, float, or a list of these. Instead of {type(value)}.')


# 将数据转换为单个值
def to_item(value):
    if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
        value = value.item()
    return value


# 将图像数据范围标准化到 [-1, 1] 范围内
def standardize(image, data_range=255):
    """ Normalize the values of a tensor or numpy array to the range [-1, 1].

    e.g.
    image = (image / data_range) * 2 - 1

    Args:
        image: input tensor or numpy array
        data_range: the maximum value of the pixel range (usually 255 for images)

    Returns:
        The input tensor or numpy array with values normalized to the range [-1, 1].
    """
    return image / data_range * 2 - 1

# 将图像数据范围归一化到 min_max 范围内
def min_max_normalize(image, min_max=(0, 1)):
    """ Normalize the values of a tensor or numpy array to the given range.

    e.g.
    image = (image - min_max[0]) / (min_max[1] - min_max[0])

    Args:
        image: input tensor or numpy array
        min_max: the range to normalize the values to

    Returns:
        The input tensor or numpy array with values normalized to the given range.
    """
    image = (image - min_max[0]) / (min_max[1] - min_max[0])
    return image

# 将图像数据范围剪裁到 min_max 范围内
def range_clip(image, min_max=(0, 1)):
    """ Clip the values of a tensor or numpy array to the given range.

    e.g.
    image = np.clip(image, *min_max)

    Args:
        image: input tensor or numpy array
        min_max: the range to clip the values to

    Returns:
        The input tensor or numpy array with values clipped to the given range.
    """
    if isinstance(image, np.ndarray):
        image = np.clip(image, *min_max)
    elif isinstance(image, torch.Tensor):
        image = image.clamp(*min_max)
    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')
    return image

# 转换图像数据类型为 uint8，并将数值乘以 data_range
def to_uint8(image, data_range=255):
    """ Convert a tensor or numpy array to uint8 with the given data range.

    e.g.
    image = (image * data_range).round().astype(np.uint8)

    Args:
        image: input tensor or numpy array
        data_range: the range of the input data (usually 255 for images)

    Returns:
        The input tensor or numpy array converted to uint8 with the given data range.
    """
    if isinstance(image, np.ndarray):
        image = (image * data_range).round().astype(np.uint8)
    elif isinstance(image, torch.Tensor):
        image = (image * data_range).round().type(torch.uint8)
    else:
        raise TypeError(f'Input data must be either numpy arrays or PyTorch tensors, but got {type(image)}.')
    return image

def float32_to_uint8(image, data_range=255):
    """ Try to convert a tensor or numpy array from float32 to uint8.

    e.g.
    image = (image * 255.0).round().astype(np.uint8) if image.dtype == np.float32

    Args:
        image: input tensor or numpy array
        data_range: the range of the input data (usually 255 for images)

    Returns:
        The input tensor or numpy array converted to uint8 if it is not already uint8, otherwise the input tensor or numpy array.
    """
    if isinstance(image, np.ndarray):
        if image.dtype == np.float32:
            image = (image * (255.0 / data_range)).round().astype(np.uint8)
        elif image.dtype == np.uint8:
            pass
        else:
            raise TypeError(f'Input data must be either float32 or uint8, but got {image.dtype}.')
    elif isinstance(image, torch.Tensor):
        if image.dtype == torch.float32:
            image = (image * (255.0 / data_range)).round().type(torch.uint8)
        elif image.dtype == torch.uint8:
            pass
        else:
            raise TypeError(f'Input data must be either float32 or uint8, but got {image.dtype}.')
    else:
        raise TypeError(f'Input data must be either numpy arrays or PyTorch tensors, but got {type(image)}.')
    return image

# 转换图像数据类型为 float32，并将数值除以 data_range
def to_float32(image, data_range=255):
    """ Convert a tensor or numpy array to float32 with the given data range.

    e.g.
    image = image.type(torch.float32) / data_range

    Args:
        image: input tensor or numpy array
        data_range: the range of the input data (usually 255 for images)

    Returns:
        The input tensor or numpy array converted to float32 with the given data range.
    """
    if isinstance(image, np.ndarray):
        image = image.astype(np.float32) / data_range
    elif isinstance(image, torch.Tensor):
        image = image.type(torch.float32) / data_range
    else:
        raise TypeError(f'Input data must be either numpy arrays or PyTorch tensors, but got {type(image)}.')
    return image

def uint8_to_float32(image):
    """ Try to convert a tensor or numpy array from uint8 to float32.

    e.g.
    image = image.type(torch.float32) / 255.0 if image.dtype == np.uint8

    Args:
        image: input tensor or numpy array

    Returns:
        The input tensor or numpy array converted to float32 if it is not already float32, otherwise the input tensor or numpy array.
    """
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.float32:
            pass
        else:
            raise TypeError(f'Input data must be either uint8 or float32, but got {image.dtype}.')
    elif isinstance(image, torch.Tensor):
        if image.dtype == torch.uint8:
            image = image.type(torch.float32) / 255.0
        elif image.dtype == torch.float32:
            pass
        else:
            raise TypeError(f'Input data must be either uint8 or float32, but got {image.dtype}.')
    else:
        raise TypeError(f'Input data must be either numpy arrays or PyTorch tensors, but got {type(image)}.')
    return image

# 反转图像通道（RGB -> BGR, BGR -> RGB）
def reverse_channels(image, dim: int = -1):
    """ Reverse the order of the channels in a tensor or numpy array.

    e.g.
    image = image[[2, 1, 0], :, :] if dims == 3 and dim == 0
    image = image[..., :, :, [2, 1, 0]] if dims >= 4 and dim == -1

    Args:
        image: input tensor or numpy array
        dim: the dimension along which to reverse the channels

    Returns:
        The input tensor or numpy array with reversed channels.
    """
    if isinstance(image, np.ndarray):
        image = np.take(image, [2, 1, 0], axis=dim)
    elif isinstance(image, torch.Tensor):
        image = image.index_select(dim, torch.tensor([2, 1, 0]))
    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')
    return image

# 尝试反转图像通道（RGB -> BGR, BGR -> RGB），如果图像通道数为3，则反转，否则不做任何操作
def try_reverse_channels(image, dim: int = -1):
    """ Try to reverse the order of the channels in a tensor or numpy array, but only if the number of channels is 3.

    e.g.
    image = image[[2, 1, 0], :, :] if dims == 3 and dim == 0
    image = image[..., :, :, [2, 1, 0]] if dims >= 4 and dim == -1
    image = image if dims == 3 and dim == 0 and image.shape[dim] != 3, e.g. grayscale image

    Args:
        image: input tensor or numpy array
        dim: the dimension along which to reverse the channels

    Returns:
        The input tensor or numpy array with reversed channels if the number of channels is 3, otherwise the input tensor or numpy array.
    """
    if image.shape[dim] != 3:
        return image

    image = reverse_channels(image, dim)
    return image

# 维度重排为 CHW 格式（HWC -> CHW）
def transpose_hwc_to_chw(image):
    """ Transpose the dimensions of a tensor or numpy array from HWC to CHW.

    e.g.
    image = image.transpose(2, 0, 1)

    Args:
        image: input tensor or numpy array

    Returns:
        The input tensor or numpy array with dimensions transposed from HWC to CHW.
    """
    if isinstance(image, np.ndarray):
        dims = image.ndim
        permute_order = list(range(dims - 3)) + [dims - 1, dims - 3, dims - 2]
        image = image.transpose(*permute_order)
    elif isinstance(image, torch.Tensor):
        dims = image.dim()
        permute_order = list(range(dims - 3)) + [dims - 1, dims - 3, dims - 2]
        image = image.permute(*permute_order)
    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')
    return image

# 维度重排为 HWC 格式（CHW -> HWC）
def transpose_chw_to_hwc(image):
    """ Transpose the dimensions of a tensor or numpy array from CHW to HWC.

    e.g.
    image = image.transpose(1, 2, 0)

    Args:
        image: input tensor or numpy array

    Returns:
        The input tensor or numpy array with dimensions transposed from CHW to HWC.
    """
    if isinstance(image, np.ndarray):
        dims = image.ndim
        permute_order = list(range(dims - 3)) + [dims - 2, dims - 1, dims - 3]
        image = image.transpose(*permute_order)
    elif isinstance(image, torch.Tensor):
        dims = image.dim()
        permute_order = list(range(dims - 3)) + [dims - 2, dims - 1, dims - 3]
        image = image.permute(*permute_order)
    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')
    return image

# 扩展图像的维度，使其满足指定维度
def expand_dim(image, dims=4):
    if isinstance(image, np.ndarray):
        while image.ndim < dims:
            image = np.expand_dims(image, axis=0)
    elif isinstance(image, torch.Tensor):
        while image.dim() < dims:
            image = image.unsqueeze(0)
    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')
    return image


#region ==[Color Space]==
#region ==[Utils]==
def _max_value(image, dim):
    if isinstance(image, torch.Tensor):
        max_value = torch.max(image, dim=dim).values
    elif isinstance(image, np.ndarray):
        max_value = np.max(image, axis=dim)
    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')

    return max_value

def _min_value(image, dim):
    if isinstance(image, torch.Tensor):
        min_value = torch.min(image, dim=dim).values
    elif isinstance(image, np.ndarray):
        min_value = np.min(image, axis=dim)
    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')

    return min_value

def _zeros_like(image):
    if isinstance(image, torch.Tensor):
        zeros = torch.zeros_like(image)
    elif isinstance(image, np.ndarray):
        zeros = np.zeros_like(image)
    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')

    return zeros

def _stack(image_list, dim):
    if isinstance(image_list[0], torch.Tensor):
        stacked = torch.stack(image_list, dim=dim)
    elif isinstance(image_list[0], np.ndarray):
        stacked = np.stack(image_list, axis=dim)
    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')

    return stacked

def _as_int(image):
    if isinstance(image, torch.Tensor):
        image = image.int()
    elif isinstance(image, np.ndarray):
        image = image.astype(np.int32)
    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')

    return image

def _where(condition, x, y):
    if isinstance(condition, torch.Tensor):
        where = torch.where(condition, x, y)
    elif isinstance(condition, np.ndarray):
        where = np.where(condition, x, y)
    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')

    return where


#endregion
'''
Refers to `colorsys` module in Python standard library.
'''

def rgb_to_hsv(image_rgb):
    """
    Convert an RGB image to HSV image.

    Args:
        image_rgb (torch.Tensor or numpy.ndarray): RGB image with shape (H, W, 3) and range [0, 1].

    Returns:
        hsv (torch.Tensor or numpy.ndarray): HSV image with shape (H, W, 3) and range H: [0, 1], S: [0, 1], V: [0, 1].
    """
    r, g, b = image_rgb[..., 0], image_rgb[..., 1], image_rgb[..., 2]

    maxc = _max_value(image_rgb, dim=-1)
    minc = _min_value(image_rgb, dim=-1)

    # 计算 V（亮度）
    v = maxc

    # 计算 S（饱和度）
    diff = maxc - minc
    s = diff / (maxc + 1e-10)  # 避免除以 0

    # 计算 H（色调）
    rc = (maxc - r) / (diff + 1e-10)
    gc = (maxc - g) / (diff + 1e-10)
    bc = (maxc - b) / (diff + 1e-10)

    h = _zeros_like(v)
    h[r == maxc] = bc[r == maxc] - gc[r == maxc]
    h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
    h[b == maxc] = 4.0 + gc[b == maxc] - rc[b == maxc]

    h = (h / 6.0) % 1.0

    hsv = _stack([h, s, v], dim=-1)
    return hsv


def hsv_to_rgb(image_hsv):
    """
    Convert an HSV image to RGB image.

    Args:
        image_hsv (torch.Tensor or numpy.ndarray): HSV image with shape (H, W, 3) and range H: [0, 1], S: [0, 1], V: [0, 1].

    Returns:
        rgb (torch.Tensor or numpy.ndarray): RGB image with shape (H, W, 3) and range [0, 1].
    """
    h, s, v = image_hsv[..., 0], image_hsv[..., 1], image_hsv[..., 2]

    r = _zeros_like(h)
    g = _zeros_like(h)
    b = _zeros_like(h)

    # 当饱和度为 0 时，直接返回灰度值
    mask = s == 0.0
    r[mask] = v[mask]
    g[mask] = v[mask]
    b[mask] = v[mask]

    # 计算 RGB 值
    i = _as_int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i = i % 6

    # 根据 i 的值计算 RGB
    r[i == 0] = v[i == 0]
    g[i == 0] = t[i == 0]
    b[i == 0] = p[i == 0]

    r[i == 1] = q[i == 1]
    g[i == 1] = v[i == 1]
    b[i == 1] = p[i == 1]

    r[i == 2] = p[i == 2]
    g[i == 2] = v[i == 2]
    b[i == 2] = t[i == 2]

    r[i == 3] = p[i == 3]
    g[i == 3] = q[i == 3]
    b[i == 3] = v[i == 3]

    r[i == 4] = t[i == 4]
    g[i == 4] = p[i == 4]
    b[i == 4] = v[i == 4]

    r[i == 5] = v[i == 5]
    g[i == 5] = p[i == 5]
    b[i == 5] = q[i == 5]

    return _stack([r, g, b], dim=-1)


def rgb_to_hls(image_rgb):
    """
    Convert an RGB image to HLS image.

    Args:
        image_rgb (torch.Tensor or numpy.ndarray): RGB image with shape (H, W, 3) and range [0, 255].

    Returns:
        hls (torch.Tensor or numpy.ndarray): HLS image with shape (H, W, 3) and range H: [0, 1], L: [0, 1], S: [0, 1].
    """
    r, g, b = image_rgb[..., 0], image_rgb[..., 1], image_rgb[..., 2]

    maxc = _max_value(image_rgb, dim=-1)
    minc = _min_value(image_rgb, dim=-1)

    # 计算亮度 (L)
    l = (minc + maxc) / 2.0

    # 当最大值和最小值相等时，返回 H=0, S=0
    mask = minc == maxc
    h = _zeros_like(r)
    s = _zeros_like(r)

    # 计算饱和度 (S)
    s[~mask] = _where(
        l[~mask] <= 0.5,
        (maxc[~mask] - minc[~mask]) / (maxc[~mask] + minc[~mask]),
        (maxc[~mask] - minc[~mask]) / (2.0 - maxc[~mask] - minc[~mask])
    )

    # 计算色相 (H)
    rc = (maxc - r) / (maxc - minc + 1e-10)  # 避免除零
    gc = (maxc - g) / (maxc - minc + 1e-10)
    bc = (maxc - b) / (maxc - minc + 1e-10)

    h = _where(
        r == maxc,
        bc - gc,
        _where(
            g == maxc,
            2.0 + rc - bc,
            4.0 + gc - rc
        )
    )
    h = (h / 6.0) % 1.0

    # 处理最大值和最小值相等的情况
    h[mask] = 0.0
    s[mask] = 0.0

    return _stack([h, l, s], dim=-1)


def _v(m1, m2, hue):
    hue = hue % 1.0
    return _where(
        hue < 1.0 / 6.0,
        m1 + (m2 - m1) * hue * 6.0,
        _where(
            hue < 0.5,
            m2,
            _where(
                hue < 2.0 / 3.0,
                m1 + (m2 - m1) * (2.0 / 3.0 - hue) * 6.0,
                m1
            )
        )
    )


def hls_to_rgb(image_hls):
    """
    Convert an HLS image to RGB image.

    Args:
        image_hls (torch.Tensor or numpy.ndarray): HLS image with shape (H, W, 3) and range H: [0, 1], L: [0, 1], S: [0, 1].

    Returns:
        rgb (torch.Tensor or numpy.ndarray): RGB image with shape (H, W, 3) and range [0, 1].
    """
    h, l, s = image_hls[..., 0], image_hls[..., 1], image_hls[..., 2]

    # 当饱和度为0时，返回灰度值
    mask = s == 0.0
    r = _zeros_like(h)
    g = _zeros_like(h)
    b = _zeros_like(h)
    r[mask] = l[mask]
    g[mask] = l[mask]
    b[mask] = l[mask]

    # 计算中间值 m1 和 m2
    m2 = _where(
        l <= 0.5,
        l * (1.0 + s),
        l + s - (l * s)
    )
    m1 = 2.0 * l - m2

    # 计算 RGB 值
    r[~mask] = _v(m1[~mask], m2[~mask], h[~mask] + 1.0 / 3.0)
    g[~mask] = _v(m1[~mask], m2[~mask], h[~mask])
    b[~mask] = _v(m1[~mask], m2[~mask], h[~mask] - 1.0 / 3.0)

    return _stack([r, g, b], dim=-1)


def rgb_to_yuv(image_rgb, normalize=False):
    """
    Convert an RGB image to YUV image.

    Args:
        image_rgb (torch.Tensor): RGB image with shape (H, W, 3) and range [0, 1].
        normalize (bool): Whether to normalize YUV values to [0, 1].

    Returns:
        yuv (torch.Tensor): YUV image with shape (H, W, 3) and range Y: [0, 1], U: [-0.436, 0.436], V: [-0.615, 0.615].
    """
    if isinstance(image_rgb, torch.Tensor):
        # Conversion matrix from RGB to YUV
        transform_matrix = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001]
        ], device=image_rgb.device, dtype=image_rgb.dtype)

        # Apply transformation
        yuv = torch.einsum('...c,hc->...h', image_rgb, transform_matrix)

    elif isinstance(image_rgb, np.ndarray):
        # Conversion matrix from RGB to YUV
        transform_matrix = np.array([
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001]
        ])

        # Apply transformation
        yuv = np.dot(image_rgb, transform_matrix.T)

    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')

    if normalize:
        # Normalize YUV values to [0, 1]
        yuv[..., 0] = yuv[..., 0]                                # Y
        yuv[..., 1] = (yuv[..., 1] + 0.436) * (1 / (2 * 0.436))  # U
        yuv[..., 2] = (yuv[..., 2] + 0.615) * (1 / (2 * 0.615))  # V

    return yuv


def yuv_to_rgb(image_yuv, denormalize=False):
    """
    Convert a YUV image to RGB image.

    Args:
        image_yuv (torch.Tensor): YUV image with shape (H, W, 3) and range Y: [0, 1], U: [-0.436, 0.436], V: [-0.615, 0.615].

    Returns:
        rgb (torch.Tensor): RGB image with shape (H, W, 3) and range [0, 1].
    """
    if denormalize:
        # Denormalize YUV values from [0, 1]
        image_yuv[..., 0] = image_yuv[..., 0]                      # Y
        image_yuv[..., 1] = image_yuv[..., 1] * 2 * 0.436 - 0.436  # U
        image_yuv[..., 2] = image_yuv[..., 2] * 2 * 0.615 - 0.615  # V

    if isinstance(image_yuv, torch.Tensor):
        # Inverse transformation matrix from YUV to RGB
        transform_matrix = torch.tensor([
            [1.0, 0.0, 1.13983],
            [1.0, -0.39465, -0.58060],
            [1.0, 2.03211, 0.0]
        ], device=image_yuv.device, dtype=image_yuv.dtype)

        # Apply transformation
        rgb = torch.einsum('...c,hc->...h', image_yuv, transform_matrix)
        rgb = torch.clamp(rgb, 0.0, 1.0)

    elif isinstance(image_yuv, np.ndarray):
        # Inverse transformation matrix from YUV to RGB
        transform_matrix = np.array([
            [1.0, 0.0, 1.13983],
            [1.0, -0.39465, -0.58060],
            [1.0, 2.03211, 0.0]
        ])

        # Apply transformation
        rgb = np.dot(image_yuv, transform_matrix.T)
        rgb = np.clip(rgb, 0.0, 1.0)

    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')

    return rgb


def rgb_to_ycbcr(image_rgb):
    """
    Convert an RGB image to YCbCr image.

    Args:
        image_rgb (torch.Tensor): RGB image with shape (H, W, 3) and range [0, 1].

    Returns:
        ycbcr (torch.Tensor): YCbCr image with shape (H, W, 3) and range Y: [0, 1], Cb: [0, 1], Cr: [0, 1].
    """
    if isinstance(image_rgb, torch.Tensor):
        # Conversion matrix from RGB to YCbCr
        transform_matrix = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ], device=image_rgb.device, dtype=image_rgb.dtype)

        # Apply transformation
        ycbcr = torch.einsum('...c,hc->...h', image_rgb, transform_matrix)

    elif isinstance(image_rgb, np.ndarray):
        # Conversion matrix from RGB to YCbCr
        transform_matrix = np.array([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ])

        # Apply transformation
        ycbcr = np.dot(image_rgb, transform_matrix.T)

    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')

    # Shift Cb and Cr to [0, 1]
    ycbcr[..., 1:] += 0.5
    return ycbcr


def ycbcr_to_rgb(image_ycbcr):
    """
    Convert a YCbCr image to RGB image.

    Args:
        image_ycbcr (torch.Tensor): YCbCr image with shape (H, W, 3) and range Y: [0, 1], Cb: [0, 1], Cr: [0, 1].

    Returns:
        rgb (torch.Tensor): RGB image with shape (H, W, 3) and range [0, 1].
    """
    if isinstance(image_ycbcr, torch.Tensor):
        # Shift Cb and Cr back to [-0.5, 0.5]
        image_ycbcr = image_ycbcr.clone()
        image_ycbcr[..., 1:] -= 0.5

        # Inverse transformation matrix from YCbCr to RGB
        transform_matrix = torch.tensor([
            [1.0, 0.0, 1.402],
            [1.0, -0.344136, -0.714136],
            [1.0, 1.772, 0.0]
        ], device=image_ycbcr.device, dtype=image_ycbcr.dtype)

        # Apply transformation
        rgb = torch.einsum('...c,hc->...h', image_ycbcr, transform_matrix)

        # Clip values to [0, 1]
        rgb = torch.clamp(rgb, 0.0, 1.0)

    elif isinstance(image_ycbcr, np.ndarray):
        # Shift Cb and Cr back to [-0.5, 0.5]
        image_ycbcr = image_ycbcr.copy()
        image_ycbcr[..., 1:] -= 0.5

        # Inverse transformation matrix from YCbCr to RGB
        transform_matrix = np.array([
            [1.0, 0.0, 1.402],
            [1.0, -0.344136, -0.714136],
            [1.0, 1.772, 0.0]
        ])

        # Apply transformation
        rgb = np.dot(image_ycbcr, transform_matrix.T)

        # Clip values to [0, 1]
        rgb = np.clip(rgb, 0.0, 1.0)

    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')

    return rgb
#enddregion


#region ==[Paddings]==
# Padding in case images are not multiples of 4
def padding_image(image, mul=4, mode='reflect', centering=True):
    """
    Padding images to multiples of <mul>

    e.g.
    padding_image(image[32, 31], 4) => image[32, 32]

    Args:
        image: if image is tensor, with shape (B, C, H, W) or (C, H, W), and range [0, 1]
                if image is numpy array, with shape (H, W, C) or (H, W), and range [0, 255]
        mul: the multiple to pad to.
        mode: padding mode, see torch.nn.functional.pad or numpy.pad.
    """
    h, w = image.shape[-2:]
    new_h = (h + mul - 1) // mul * mul
    new_w = (w + mul - 1) // mul * mul
    if centering:
        lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
        lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    else:
        lh, uh = 0, new_h - h
        lw, uw = 0, new_w - w

    if uh == 0 and uw == 0:
        return image, (0, 0, 0, 0)

    pad_size = (lw, uw, lh, uh)

    if isinstance(image, torch.Tensor):
        image = F.pad(image, pad_size, mode=mode)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:     # (H, W)
            padding = ((lh, uh), (lw, uw))
        elif image.ndim == 3:   # (H, W, C)
            padding = ((lh, uh), (lw, uw), (0, 0))
        else:
            raise ValueError(f"Unsupported numpy array shape: {image.shape}")

        image = np.pad(image, padding, mode=mode)
    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')
    return image, pad_size


# Unpadding images to original dimensions
# noinspection SpellCheckingInspection
def unpadding_image(image, size, centering=True):
    """
    Unpadding images to original dimensions

    e.g.
    unpadding_image(image[32, 32], (32, 31)) => image[32, 31]

    Args:
        image: if image is tensor, with shape (B, C, H, W) or (C, H, W), and range [0, 1]
                if image is numpy array, with shape (H, W, C) or (H, W), and range [0, 255]
        size: the original size of the image.
    """
    lw, uw, lh, uh = size
    if isinstance(image, torch.Tensor):
        if uh == 0 and lw == 0:
            image = image[..., lh:, lw:]
        elif uh == 0:
            image = image[..., lh:, lw:-uw]
        elif lw == 0:
            image = image[..., lh:-uh, lw:]
        else:
            image = image[..., lh:-uh, lw:-uw]
    elif isinstance(image, np.ndarray):
        if uh == 0 and lw == 0:
            image = image[..., lh:, lw:, :]
        elif uh == 0:
            image = image[..., lh:, lw:-uw, :]
        elif lw == 0:
            image = image[..., lh:-uh, lw:, :]
        else:
            image = image[..., lh:-uh, lw:-uw, :]
    else:
        raise TypeError('Input data must be either numpy arrays or PyTorch tensors.')

    return image


@contextmanager
def padding(image, mul=4, mode='reflect', centering=True):
    if mul <= 1:
        yield image, lambda x: x
    else:
        image, pad_size = padding_image(image, mul=mul, mode=mode, centering=centering)
        yield image, lambda x: unpadding_image(x, pad_size, centering=centering)
#endregion

