import os
import os.path as osp


import glob
import numpy as np
import cv2
from PIL import Image
from natsort import natsorted
from collections import defaultdict


IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'ppm', 'bmp']


#region ==[图像文件操作]==
# 读取图像，可支持 npy 和图像输入格式，代码修改自 FastLLVE：https://github.com/Wenhao-Li-777/FastLLVE/blob/main/data/util.py 的 read_img 和 read_img2 函数
def read_image_as_numpy(path, read_mode=cv2.IMREAD_UNCHANGED):
    """
    Read image from file path.

    Args:
        path (str): file path of the image.
        read_mode (int): cv2 read mode.

    Returns:
        image (np.ndarray): image array with shape [H, W, C], with range [0, 255].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: \"{path}\"')

    if path.endswith('.npy'):
        image = np.load(path)
    elif any(path.endswith(extension.lower()) or path.endswith(extension.upper()) for extension in IMG_EXTENSIONS):
        image = cv2.imread(path, read_mode)
    else:
        raise ValueError(f'Unsupported image format: \"{path}\"')

    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)   # grayscale to RGB: (H, W) -> (H, W, 1)
    elif image.shape[2] > 3:
        image = image[:, :, :3]                 # RGBA to RGB: (H, W, 4) -> (H, W, 3)

    return image


def read_image_as_pil(path, mode='RGB'):
    """
    Read image from file path.

    Args:
        path (str): file path of the image.
        mode (str): PIL image mode.

    Returns:
        image (PIL.Image.Image): image object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: \"{path}\"')

    if any(path.endswith(extension.lower()) or path.endswith(extension.upper()) for extension in IMG_EXTENSIONS):
        with Image.open(path) as img:   # Close the file automatically
            image = img.convert(mode=mode)
    else:
        raise ValueError(f'Unsupported image format: \"{path}\"')

    return image


def read_images_as_numpy(*paths, read_mode=cv2.IMREAD_UNCHANGED):
    """
    Read image from file path.

    Args:
        paths (str | list[str]): file path of the image.
        read_mode (int): cv2 read mode.

    Returns:
        image (np.ndarray): image array with shape [H, W, C], with range [0, 255].
    """
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        image = read_image_as_numpy(path, read_mode)
        images.append(image)

    return images if len(images) > 1 else images[0]


def read_images_as_pil(*paths, mode='RGB'):
    """
    Read image from file path.

    Args:
        paths (str | list[str]): file path of the image.
        mode (str): PIL image mode.

    Returns:
        image (PIL.Image.Image): image object.
    """
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        image = read_image_as_pil(path, mode)
        images.append(image)

    return images if len(images) > 1 else images[0]


# 保存图像，可支持 npy 和图像输出格式
def save_image(image, path, auto_mkdir=True):
    """
    Save image to file path.

    Args:
        image (np.ndarray): image array.
        path (str): file path to save the image.
        auto_mkdir (bool): whether to create the directory if it does not exist.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(path))
        os.makedirs(dir_name, exist_ok=True)

    if path.endswith('.npy'):
        np.save(path, image)
    elif any(path.endswith(extension.lower()) or path.endswith(extension.upper()) for extension in IMG_EXTENSIONS):
        if isinstance(image, np.ndarray):
            cv2.imwrite(path, image)
        elif isinstance(image, Image.Image):
            image.save(path)
        else:
            raise ValueError(f'Unsupported image format: {type(image)}')
    else:
        raise ValueError(f'Unsupported path format: {path}')
#endregion


#region ==[tensor文件操作]==
def read_feature(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: \"{path}\"')

    if path.endswith('.npy') or path.endswith('.npz'):
        features = np.load(path, allow_pickle=True)
        return features
    else:
        raise ValueError(f'Unsupported path format: {path}')

def save_feature(features, path, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(path))
        os.makedirs(dir_name, exist_ok=True)

    if path.endswith('.npy'):
        np.save(path, features)
    elif path.endswith('.npz'):
        np.savez(path, **features)
    else:
        raise ValueError(f'Unsupported path format: {path}')
#endregion


class PathHandler:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def get_vanilla_path(path):
        return path

    @staticmethod
    # path="/home/alice/Documents/project/file.txt"
    # return: "file.txt"
    def get_basename(path):
        return os.path.basename(path)

    @staticmethod
    # path="/home/alice/Documents/project/file.txt", extension=".txt"
    # return: "/home/alice/Documents/project/file"
    def remove_extension(path):
        """
        Get the path with the extension removed.
        """
        filename, extension = os.path.splitext(path)
        return filename

    # path="/home/alice/Documents/project/file.txt", dir_name="/home/alice"
    # return: "Documents/project/file.txt"
    def get_dir_removed_path(self, path):
        """
        Get the path with the root directory removed.
        """
        return os.path.relpath(path, self.dirname)


#region ==[文件路径操作]==
def get_paths_common_prefix(paths, to_string=False):
    """
    Args:
        paths (list[str]): A list of paths.
        to_string (bool): Whether to return a string representation of the common prefix and the relative paths.
    Returns:
        str: A string representing the common prefix of the paths.
    """
    common_prefix = os.path.commonprefix(paths)
    paths = [os.path.relpath(path, common_prefix) for path in paths]
    if to_string:
        return f"{common_prefix}:[{','.join(paths)}]"
    else:
        return common_prefix, paths
#endregion


#region ==[glob 文件路径操作]==
# 从给定的文件夹路径中读取图像文件路径列表
# 代码修改自 FastLLVE：https://github.com/Wenhao-Li-777/FastLLVE/blob/main/data/util.py 的 glob_file_list 函数
'''
# Original:
def glob_file_list(root):
    return sorted(glob.glob(os.path.join(root, '*')))
'''
def glob_file_list(pattern, path_handler=PathHandler.get_vanilla_path, sort=True, recursive=False):
    """
    Glob all files in the given root directory.
    Args:
        pattern (str): a pattern to match file paths. e.g. "/path/to/images/*.jpg"
        path_handler (function): a function to process the file paths.
        sort (bool): whether to sort the file paths by natural order.
        recursive (bool): whether to search recursively.
    Returns:
        file_list (list[str]): a list of file paths. sorted by natural order. e.g. [1.jpg, 2.jpg, 11.jpg] instead of [1.jpg, 11.jpg, 2.jpg]
    """
    # 读取所有匹配的文件路径
    path_list = glob.glob(pattern, recursive=recursive)

    path_list = [path_handler(os.path.normpath(path)) for path in path_list]
    return natsorted(path_list) if sort else path_list


# 使用 glob 递归查找所有匹配的文件，并按照自然顺序排序
def glob_single_files(directory, file_extensions, path_handler=PathHandler.get_vanilla_path):
    """
    Glob all files with the given extension in the given directory.

    example:
    for image_path in glob_single_files(images_load_path, 'png'):
        print(image_path)

    Args:
        directory (str): the directory to search.
        file_extensions (str or list[str]): the file extension to match. (without the dot)
    Returns:
        file_paths (list[str]): a list of file paths. sorted by natural order.
    """
    if isinstance(file_extensions, str):
        file_extensions = [file_extensions]

    file_paths = []
    for file_extension in file_extensions:
        pattern = os.path.join(directory, f"**/*.{file_extension}")
        file_paths += natsorted(glob.glob(pattern, recursive=True))
    file_paths = [path_handler(os.path.normpath(path)) for path in file_paths]
    return file_paths


# 使用 glob 递归查找所有匹配的文件，按照自然顺序排序，并按子目录分组
def glob_packed_files(directory, file_extensions, path_handler=PathHandler.get_vanilla_path):
    """
    Glob all files with the given extension in the given directory, and group them by subdirectory.
    Args:
        directory (str): the directory to search.
        file_extensions (str): the file extension to match.
    Returns:
        grouped_files (list[list[str]]): a list of lists of file paths. each sublist contains files in the same subdirectory.
    """
    file_paths = glob_single_files(directory, file_extensions, path_handler)

    # 使用字典按子目录分组
    grouped_files = defaultdict(list)
    for file_path in file_paths:
        subdir = os.path.dirname(file_path)
        grouped_files[subdir].append(file_path)

    # 将字典的值（子数组）转换为列表
    return list(grouped_files.values())
#endregion