import os
import os.path as osp

import random

import torch
import torch.utils.data as data

import utils.io as io
from utils.convert import numpy2tensor
from utils.registry import DATASET_REGISTRY
from utils.general import try_fill_default_dict, obsolete

from .base_dataset import TransformsDatasetBase
from .util import generate_random_indices_in_a_sequence


def read_image_to_tensor(file_path, transform=None, seed=None):
    """
    Reads image file to tensor.

    The illustration below shows the order of the random state applied:
    if seed is not None:
        gt:[b], ..., [a], [c]
        lq:[b], ..., [a], [c]
        (a, b, c represents the different transforms applied to the image)
    """
    if seed is not None:
        random.seed(seed)           # apply this seed to img transforms
        torch.manual_seed(seed)     # needed for torchvision 0.7

    if transform is not None:
        image_pil = io.read_image_as_pil(file_path)
        image_tensor = transform(image_pil)
    else:
        image_np = io.read_image_as_numpy(file_path)
        image_tensor = numpy2tensor(image_np, expand_dim=False)
    return image_tensor


def read_images_to_tensor(file_paths, transform=None, seed=None):
    """
    Reads image sequence to tensor.

    The illustration below shows the order of the random state applied:
    if seed is not None:
        seq_gt:[b] -> [e] -> [a] -> [d] -> [c], ..., [i] -> [g] -> [j] -> [f] -> [h]
        seq_lq:[b] -> [e] -> [a] -> [d] -> [c], ..., [i] -> [g] -> [j] -> [f] -> [h]
        (a, b, ..., j represents the different transforms applied to the image)
    """
    if seed is not None:
        random.seed(seed)           # apply this seed to img transforms
        torch.manual_seed(seed)     # needed for torchvision 0.7

    tensors = []
    for file_path in file_paths:
        image_tensor = read_image_to_tensor(file_path, transform)
        tensors.append(image_tensor)
    return torch.stack(tensors)


def read_seq_images_to_tensor(file_paths, transform=None, seed=None):
    """
    Reads image sequence to tensor.

    The illustration below shows the order of the random state applied:
    if seed is not None:
        seq_gt:[a] -> [a] -> [a], ..., [c] -> [c] -> [c], [b] -> [b] -> [b]
        seq_lq:[a] -> [a] -> [a], ..., [c] -> [c] -> [c], [b] -> [b] -> [b]
        (a, b, c represents the different transforms applied to the image)
    """
    tensors = []
    for file_path in file_paths:
        image_tensor = read_image_to_tensor(file_path, transform, seed)
        tensors.append(image_tensor)
    return torch.stack(tensors)


# noinspection SpellCheckingInspection
@obsolete
class GlobSingleDatasetBase(data.Dataset):
    def __init__(self, files_glob_func=None):
        super(GlobSingleDatasetBase, self).__init__()
        self.files_glob_func = files_glob_func
        self.names = []
        self.common_dirs = {}

    def get_all_file_paths(self, **kwargs):
        dir_names = []
        attr_names = []
        name_prefixs = []
        if "dataroot" in kwargs:
            # dataroot, xx_dir, xx_ext
            for key, value in kwargs.items():
                if key.endswith("_dir") and value is not None:
                    name_prefix = key[:-len("_dir")]

                    dir_name = f"{name_prefix}_dir"
                    ext_name = f"{name_prefix}_ext"
                    attr_name = f"{name_prefix}_file_paths"
                    if ext_name not in kwargs:
                        ext_name = None
                    self.dir_attr(attr_name, dir_name, ext_name, **kwargs)

                    dir_names.append(dir_name)
                    attr_names.append(attr_name)
                    name_prefixs.append(name_prefix)
        else:
            # dataroot_xx
            for key, value in kwargs.items():
                if key == "dataroot":
                    continue
                if key.startswith("dataroot_") and value is not None:
                    name_prefix = key[len("dataroot_"):]

                    dir_name = f"dataroot_{name_prefix}"  # 所有图片的文件夹路径
                    attr_name = f"{name_prefix}_file_paths"  # 在 Dataset 类中的属性名称，用来记录所有图片的文件路径
                    self.dir_attr(attr_name, dir_name, **kwargs)

                    dir_names.append(dir_name)
                    attr_names.append(attr_name)
                    name_prefixs.append(name_prefix)

        # 检查文件对应性
        if len(attr_names) > 0:
            def get_log_str(_attr_names, _dir_names):
                return ', '.join([f"{attr_name}({dir_name})" for attr_name, dir_name in zip(_attr_names, _dir_names)])
            # 检查文件数据量是否为 0
            if any(len(getattr(self, attr_name)) == 0 for attr_name in attr_names):
                empty_attr_names = [attr_name for attr_name in attr_names if len(getattr(self, attr_name)) == 0]
                empty_dir_names = [dir_name for dir_name in dir_names if dir_name in empty_attr_names]
                raise ValueError(f"No image files found in {get_log_str(empty_attr_names, empty_dir_names)}")
            # 检查文件数量是否不同
            if len({len(getattr(self, attr_name)) for attr_name in attr_names}) != 1:
                raise ValueError(f"The number of image files in {get_log_str(attr_names, dir_names)} are not the same")
            # 检查所有的 self.xx_file_paths 中的图片是否都一一对应
            attr_lists = [getattr(self, n) for n in attr_names]
            for i, group in enumerate(zip(*attr_lists)):
                basenames = [osp.basename(p) for p in group]
                if len(set(basenames)) != 1:
                    raise ValueError(f"Files do not correspond at index {i}: {basenames}")


        # 保存属性名称
        self.names = name_prefixs
        self.check_file_paths()

        self.common_dirs = {
            name: self.get_file_paths_common_dir(name)
            for name in self.names
        }

    def dir_attr(self, attr_name, dir_key, ext_key=None, **kwargs):
        if dir_key not in kwargs or (ext_key is not None and ext_key not in kwargs):
            return

        root_dir = ""
        if 'dataroot' in kwargs:
            root_dir = kwargs['dataroot']   # 如果在配置文件中有指定文件的根目录，就使用根目录

        dir_path = osp.join(root_dir, kwargs[dir_key])
        assert isinstance(dir_path, str), f"{dir_key} should be a string, but got {type(dir_path)}"
        # 确保路径是有效的
        if not osp.exists(dir_path):
            any_found = False

            if not any_found:
                # simple prefix check
                for prefix in ["/mnt/", "/data/", "/home/"]:
                    if dir_path.startswith(prefix):
                        for _prefix in ["/mnt/", "/data/", "/home/"]:
                            if prefix == _prefix:
                                continue

                            _dir_path = dir_path.replace(prefix, _prefix, 1)
                            if osp.exists(_dir_path):

                                from utils.console.log import get_root_logger
                                logger = get_root_logger()
                                logger.warning(f"The directory {dir_path} does not exist, but found {_dir_path} instead.")
                                dir_path = _dir_path

                                any_found = True
                                break
                        break

            if not any_found:
                # advanced prefix check with user name
                pre_names = ["xr", "yzc"]
                all_prefixes = [
                    f"{prefix}{name}"
                    for name in pre_names
                    for prefix in ["/mnt/", "/data/", "/home/"]
                ]
                for prefix in all_prefixes:
                    if dir_path.startswith(prefix):
                        for _prefix in all_prefixes:
                            if prefix == _prefix:
                                continue

                            _dir_path = dir_path.replace(prefix, _prefix, 1)
                            if osp.exists(_dir_path):

                                from utils.console.log import get_root_logger
                                logger = get_root_logger()
                                logger.warning(f"The directory {dir_path} does not exist, but found {_dir_path} instead.")
                                dir_path = _dir_path

                                any_found = True
                                break

                            any_found = False
                            _dir_path = dir_path.replace(prefix, _prefix, 1)
                            test_stages = ["/eval/", "/test/"]
                            for test_stage in test_stages:
                                if test_stage in _dir_path:
                                    for _test_stage in test_stages:
                                        if _test_stage == test_stage:
                                            continue

                                        if osp.exists(_dir_path.replace(test_stage, _test_stage, 1)):
                                            __dir_path = dir_path
                                            dir_path = _dir_path.replace(test_stage, _test_stage, 1)

                                            from utils.console.log import get_root_logger
                                            logger = get_root_logger()
                                            logger.warning(f"The directory {__dir_path} does not exist, but found {dir_path} instead.")

                                            any_found = True
                                            break
                                    break
                            if any_found:
                                break
                        break

        assert osp.exists(dir_path), f"{dir_path} does not exist"
        assert osp.isdir(dir_path), f"{dir_path} is not a directory"
        self.__setattr__(dir_key, dir_path)

        exts = kwargs[ext_key] if ext_key is not None in kwargs else io.IMG_EXTENSIONS
        if isinstance(exts, str):
            exts = [exts]
        elif not isinstance(exts, (list, tuple)):
            raise ValueError(f"{ext_key} should be a string or a list or tuple of extensions")
        if isinstance(exts, (list, tuple)) and len(exts) <= 0:
            raise ValueError(f"{ext_key} should be a non-empty list or tuple of extensions")

        dirs = self.files_glob_func(dir_path, exts) # 所有图片的文件路径
        self.__setattr__(attr_name, dirs)


    def get_file_paths(self, name, without_notified=True):
        """
        Returns:
            list[str]: A list of file paths.
        """
        file_paths_name = f"{name}_file_paths"
        if not without_notified:
            if not hasattr(self, file_paths_name):
                raise ValueError(f"{file_paths_name} is not a valid attribute of {self.__class__.__name__}. It means {name}_dir and {name}_ext shoule be provided.")
        return getattr(self, file_paths_name)


    def get_file_paths_case_insensitive(self, name, without_notified=True):
        names = [name.lower(), name.upper(), name.capitalize()]
        for name in names:
            file_paths_name = f"{name}_file_paths"
            if hasattr(self, file_paths_name):
                return getattr(self, file_paths_name)
        if not without_notified:
            raise ValueError(f"{name} is not a valid attribute of {self.__class__.__name__}. It means {name}_dir and {name}_ext shoule be provided.")


    def check_file_paths(self):
        if len(self.names) == 0:
            raise ValueError(
                "No dataset found in the config. Please add ('dataroot', 'xx_dir', 'xx_ext') or ('dataroot_xx') to the config. "
                "Or check the path of the dataset."
            )


    def get_file_paths_common_dir(self, name, without_notified=True):
        file_paths = self.get_file_paths(name, without_notified)
        if len(file_paths) == 0:
            return "/"

        def flatten(_list):
            result = []
            for item in _list:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(item)
            return result

        common_dir = osp.commonpath(flatten(file_paths))
        return common_dir


# noinspection SpellCheckingInspection
class GlobDatasetBase(data.Dataset):
    _SIMPLE_PREFIXES = ("/mnt/", "/data/", "/home/")
    _USER_NAMES = ("xr", "yzc")
    _TEST_STAGES = ("/eval/", "/test/")

    def __init__(self, files_glob_func=None):
        super(GlobDatasetBase, self).__init__()
        self.files_glob_func = files_glob_func
        self.names = []
        self._file_paths_attr_by_name = {}

    @staticmethod
    def _to_relative_path(path, base_dir_norm, base_dir_norm_len):
        norm_path = osp.normpath(path)
        # Keep legacy behavior: remove base_dir as a plain prefix, then strip a leading separator.
        # This supports historical matching patterns like low00001.png <-> normal00001.png.
        if norm_path.startswith(base_dir_norm):
            return norm_path[base_dir_norm_len:].lstrip(os.sep)
        return osp.relpath(norm_path, base_dir_norm)

    @classmethod
    def _to_relative_path_structure(cls, path_data, base_dir_norm, base_dir_norm_len):
        if isinstance(path_data, list):
            return [
                cls._to_relative_path_structure(item, base_dir_norm, base_dir_norm_len)
                for item in path_data
            ]
        if isinstance(path_data, tuple):
            return tuple(
                cls._to_relative_path_structure(item, base_dir_norm, base_dir_norm_len)
                for item in path_data
            )
        return cls._to_relative_path(path_data, base_dir_norm, base_dir_norm_len)

    @staticmethod
    def _build_name_lookup(names):
        lookup = {}
        for name in names:
            attr_name = f"{name}_file_paths"
            lookup[name.lower()] = attr_name
        return lookup

    @staticmethod
    def is_dataset_dir_valid(path):
        if osp.exists(path):
            return osp.isdir(path)

        _dir = osp.dirname(path)
        if osp.exists(_dir):
            return osp.isdir(_dir)

    def get_all_file_paths(self, **kwargs):
        dir_names = []
        attr_names = []
        name_prefixs = []
        if "dataroot" in kwargs:
            # dataroot, xx_dir, xx_ext
            for key, value in kwargs.items():
                if key.endswith("_dir") and value is not None:
                    name_prefix = key[:-len("_dir")]    # name_prefix is the part before "_dir", e.g., "gt" for "gt_dir"

                    dir_name = f"{name_prefix}_dir"
                    ext_name = f"{name_prefix}_ext"
                    attr_name = f"{name_prefix}_file_paths"
                    if ext_name not in kwargs:
                        ext_name = None
                    self.dir_attr(attr_name, dir_name, ext_name, **kwargs)

                    dir_names.append(dir_name)
                    attr_names.append(attr_name)
                    name_prefixs.append(name_prefix)
        else:
            # dataroot_xx
            for key, value in kwargs.items():
                if key == "dataroot":
                    continue
                if key.startswith("dataroot_") and value is not None:
                    name_prefix = key[len("dataroot_"):]

                    dir_name = f"dataroot_{name_prefix}"  # 所有图片的文件夹路径
                    attr_name = f"{name_prefix}_file_paths"  # 在 Dataset 类中的属性名称，用来记录所有图片的文件路径
                    self.dir_attr(attr_name, dir_name, **kwargs)

                    dir_names.append(dir_name)
                    attr_names.append(attr_name)
                    name_prefixs.append(name_prefix)

        # 检查文件对应性
        if len(attr_names) > 0:
            def get_log_str(_attr_names, _dir_names):
                return ', '.join([f"{attr_name}({dir_name})" for attr_name, dir_name in zip(_attr_names, _dir_names)])

            attr_lists = [getattr(self, attr_name) for attr_name in attr_names]
            lengths = [len(paths) for paths in attr_lists]

            # 检查文件数据量是否为 0
            if any(length == 0 for length in lengths):
                empty_attr_names = [attr_name for attr_name, length in zip(attr_names, lengths) if length == 0]
                empty_dir_names = [dir_name for dir_name, length in zip(dir_names, lengths) if length == 0]
                raise ValueError(f"No image files found in {get_log_str(empty_attr_names, empty_dir_names)}")
            # 检查文件数量是否不同
            if len(set(lengths)) != 1:
                raise ValueError(f"The number of image files in {get_log_str(attr_names, dir_names)} are not the same")

            # 检查所有的 self.xx_file_paths 中的图片是否都一一对应
            def move_first_dim_to_last(data, index_prefix=()):
                """
                data: shape [k, a, b, ...]
                yield: (index, [obj0, obj1, ..., objk])
                """
                if not isinstance(data[0], list):
                    yield index_prefix, data
                    return
                for i, sub_items in enumerate(zip(*data)):
                    yield from move_first_dim_to_last(
                        list(sub_items),
                        index_prefix + (i,)
                    )

            def get_common_suffix(strings):
                if not strings:
                    return ""

                reversed_strings = [s[::-1] for s in strings]
                suffix_chars = []

                for chars in zip(*reversed_strings):
                    if len(set(chars)) == 1:
                        suffix_chars.append(chars[0])
                    else:
                        break

                return "".join(reversed(suffix_chars))

            # Use precomputed relative paths to avoid repeated normpath/dict lookups in large datasets.
            rel_attr_lists = [getattr(self, f"{attr_name}_rel_paths", paths) for attr_name, paths in
                              zip(attr_names, attr_lists)]

            for idx, path_group in move_first_dim_to_last(rel_attr_lists):
                path_group = [p.replace("\\", "/") for p in path_group]

                common_suffix = get_common_suffix(path_group)

                if common_suffix not in path_group:
                    raise ValueError(
                        f"The common suffix is not an existing path at index {idx}: "
                        f"common_suffix={common_suffix}, paths={path_group}"
                    )

        # 保存属性名称
        self.names = name_prefixs
        self._file_paths_attr_by_name = self._build_name_lookup(self.names)
        self.check_file_paths()

    def dir_attr(self, attr_name, dir_name, ext_name=None, **kwargs):
        if dir_name not in kwargs or (ext_name is not None and ext_name not in kwargs):
            return

        root_dir = ""
        if 'dataroot' in kwargs:
            root_dir = kwargs['dataroot']  # 如果在配置文件中有指定文件的根目录，就使用根目录

        # ==[多数据集支持]==
        if isinstance(kwargs[dir_name], list):
            dir_base_paths = kwargs[dir_name]
        else:
            dir_base_paths = [kwargs[dir_name]]

        dir_paths = []
        all_file_paths = []
        all_file_rel_paths = []
        all_file_dir_dict = {}
        for dir_base_path in dir_base_paths:
            dir_path = osp.join(root_dir, dir_base_path)
            assert isinstance(dir_path, str), f"{dir_name} should be a string, but got {type(dir_path)}"
            # 确保路径是有效的
            if not osp.exists(dir_path):
                any_found = self.is_dataset_dir_valid(dir_path)

                # simple prefix check
                if not any_found:
                    for prefix in self._SIMPLE_PREFIXES:
                        if dir_path.startswith(prefix):
                            for _prefix in self._SIMPLE_PREFIXES:
                                if prefix == _prefix:
                                    continue

                                _dir_path = dir_path.replace(prefix, _prefix, 1)
                                if self.is_dataset_dir_valid(_dir_path):
                                    from utils.console.log import get_root_logger
                                    logger = get_root_logger()
                                    logger.warning(f"The directory {dir_path} does not exist, but found {_dir_path} instead.")
                                    dir_path = _dir_path

                                    any_found = True
                                    break
                            break

                if not any_found:
                    # advanced prefix check with user name
                    all_prefixes = [
                        f"{prefix}{name}"
                        for name in self._USER_NAMES
                        for prefix in self._SIMPLE_PREFIXES
                    ]
                    for prefix in all_prefixes:
                        if dir_path.startswith(prefix):
                            for _prefix in all_prefixes:
                                if prefix == _prefix:
                                    continue

                                _dir_path = dir_path.replace(prefix, _prefix, 1)
                                if self.is_dataset_dir_valid(_dir_path):
                                    from utils.console.log import get_root_logger
                                    logger = get_root_logger()
                                    logger.warning(f"The directory {dir_path} does not exist, but found {_dir_path} instead.")
                                    dir_path = _dir_path

                                    any_found = True
                                    break

                                any_found = False
                                _dir_path = dir_path.replace(prefix, _prefix, 1)
                                for test_stage in self._TEST_STAGES:
                                    if test_stage in _dir_path:
                                        for _test_stage in self._TEST_STAGES:
                                            if _test_stage == test_stage:
                                                continue

                                            if self.is_dataset_dir_valid(_dir_path.replace(test_stage, _test_stage, 1)):
                                                __dir_path = dir_path
                                                dir_path = _dir_path.replace(test_stage, _test_stage, 1)

                                                from utils.console.log import get_root_logger
                                                logger = get_root_logger()
                                                logger.warning(
                                                    f"The directory {__dir_path} does not exist, but found {dir_path} instead.")

                                                any_found = True
                                                break
                                        break
                                if any_found:
                                    break
                            break

            assert self.is_dataset_dir_valid(dir_path), f"{dir_path} does not exist"
            dir_paths.append(dir_path)

            exts = kwargs[ext_name] if ext_name is not None in kwargs else io.IMG_EXTENSIONS
            if isinstance(exts, str):
                exts = [exts]
            elif not isinstance(exts, (list, tuple)):
                raise ValueError(f"{ext_name} should be a string or a list or tuple of extensions")
            if isinstance(exts, (list, tuple)) and len(exts) <= 0:
                raise ValueError(f"{ext_name} should be a non-empty list or tuple of extensions")

            file_paths = self.files_glob_func(dir_path, exts)  # 所有图片的文件路径
            all_file_paths.extend(file_paths)

            dir_path_norm = osp.normpath(dir_path)
            dir_path_norm_len = len(dir_path_norm)
            all_file_rel_paths.extend(
                self._to_relative_path_structure(file_path, dir_path_norm, dir_path_norm_len)
                for file_path in file_paths
            )

            for file_path in file_paths:
                all_file_dir_dict[file_path] = dir_path

        self.__setattr__(dir_name, dir_paths)                           # 'xx_dir' attribute, list of dataset directories
        self.__setattr__(attr_name, all_file_paths)                     # 'xx_file_paths' attribute, list of all image file paths in the dataset directories
        self.__setattr__(f"{attr_name}_rel_paths", all_file_rel_paths) # 'xx_file_paths_rel_paths' attribute, relative paths used for fast correspondence checks
        self.__setattr__(f"{attr_name}_dir_dict", all_file_dir_dict)    # 'xx_file_paths_dir_dict' attribute, dict mapping each image file path to its dataset directory

    def get_file_paths(self, name, without_notified=True):
        """
        Returns:
            list[str]: A list of file paths.
        """
        file_paths_name = f"{name}_file_paths"
        if not without_notified:
            if not hasattr(self, file_paths_name):
                raise ValueError(
                    f"{file_paths_name} is not a valid attribute of {self.__class__.__name__}. It means {name}_dir and {name}_ext shoule be provided.")
        return getattr(self, file_paths_name)

    def get_file_paths_case_insensitive(self, name, without_notified=True):
        file_paths_name = self._file_paths_attr_by_name.get(name.lower())
        if file_paths_name is not None and hasattr(self, file_paths_name):
            return getattr(self, file_paths_name)
        if not without_notified:
            raise ValueError(
                f"{name} is not a valid attribute of {self.__class__.__name__}. It means {name}_dir and {name}_ext shoule be provided.")

    def check_file_paths(self):
        if len(self.names) == 0:
            raise ValueError(
                "No dataset found in the config. Please add ('dataroot', 'xx_dir', 'xx_ext') or ('dataroot_xx') to the config. "
                "Or check the path of the dataset."
            )

    def get_file_paths_common_dir(self, name, without_notified=True):
        file_paths = self.get_file_paths(name, without_notified)
        if len(file_paths) == 0:
            return "/"

        def flatten(_list):
            result = []
            for item in _list:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(item)
            return result

        common_dir = osp.commonpath(flatten(file_paths))
        return common_dir

    def get_file_paths_dir_dict(self, name, without_notified=True):
        file_paths = self.get_file_paths(name, without_notified)
        file_paths_dir_dict = getattr(self, f"{name}_file_paths_dir_dict", None)
        return file_paths_dir_dict

    @staticmethod
    def _get_first_file_path(path_data):
        if isinstance(path_data, (list, tuple)):
            if len(path_data) == 0:
                return None
            return GlobDatasetBase._get_first_file_path(path_data[0])
        return path_data

    def get_data_dir(self, name, path_data, without_notified=True):
        file_path = self._get_first_file_path(path_data)
        if file_path is None:
            return "/"

        file_paths_dir_dict = self.get_file_paths_dir_dict(name, without_notified)
        if file_paths_dir_dict is None:
            return self.get_file_paths_common_dir(name, without_notified)
        return file_paths_dir_dict[file_path]


# noinspection SpellCheckingInspection
@DATASET_REGISTRY.register()
class ImageDataset(GlobDatasetBase, TransformsDatasetBase):
    def __init__(self, **option):
        """
        Dataset for image-to-image tasks.

        Args:
            option: Config for train datasets. It contains the following keys:
            (xx can be filled with 'gt', 'lq', etc., but not allows multiple values)
                - dataroot (str): root directory of the dataset.
                - xx_dir (str): directory of the xx images.
                - [Optional] xx_ext (str or list or tuple): extensions of the xx images.

                or

                - dataroot_xx (str): root directory of the xx images
        """
        GlobDatasetBase.__init__(self, io.glob_single_files)
        TransformsDatasetBase.__init__(self, **option)
        self.get_all_file_paths(**option)


    def __len__(self):
        return len(self.get_file_paths_case_insensitive(self.names[0], without_notified=False))


    def __getitem__(self, index):
        datas = {}
        for name in self.names: # 'lq', 'gt', etc.
            file_paths = self.get_file_paths_case_insensitive(name)[index]
            file = read_image_to_tensor(file_paths, self.transforms)
            datas[name] = {
                "image": file,
                "path": file_paths,
                "common_dir": self.get_data_dir(name, file_paths),
            }
            return datas

        raise ValueError("No dataset found in the config. Please add ('dataroot', 'xx_dir', 'xx_ext') or ('dataroot_xx') to the config. Or check the path of the dataset.")


# noinspection SpellCheckingInspection
@DATASET_REGISTRY.register()
class PairedImageDataset(ImageDataset):
    def __init__(self, **option):
        """
        Dataset for image-to-image tasks.

        Args:
            option: Config for train datasets. It contains the following keys:
            (xx can be filled with 'gt', 'lq', etc., and allows multiple values)
                - dataroot (str): root directory of the dataset.
                - xx_dir (str): directory of the xx images.
                - [Optional] xx_ext (str or list or tuple): extensions of the xx images.

                or

                - dataroot_xx (str): root directory of the xx images
        """
        super(PairedImageDataset, self).__init__(**option)

    def __getitem__(self, index):
        """
        dataset = PairedImageDataset(dataroot_pred=folder1, dataroot_gt=folder2)
        for data i dataset:
            pred, gt = data['pred']['image'], data['gt']['image']
            path = data['pred']['path'], data['gt']['path']
            common_dir = data['pred']['common_dir'], data['gt']['common_dir']
        """
        seed = random.randint(1, 2**32) # ensure the same transform for paired images

        datas = {}
        for name in self.names: # 'lq', 'gt', etc.
            file_paths = self.get_file_paths_case_insensitive(name)[index]
            file = read_image_to_tensor(file_paths, self.transforms, seed)
            datas[name] = {
                "image": file,
                "path": file_paths,
                "common_dir": self.get_file_paths_dir_dict(name)[file_paths],
            }
        return datas


# noinspection SpellCheckingInspection
@DATASET_REGISTRY.register()
class VideoDataset(GlobDatasetBase, TransformsDatasetBase):
    def __init__(self, pad_s=True, **option):
        """
        Dataset for video-to-video tasks.

        Args:
            option: Config for train datasets. It contains the following keys:
            (xx can be filled with 'gt', 'lq', etc., and allows multiple values)
                - dataroot (str): root directory of the dataset.
                - xx_dir (str): directory of the xx images.
                - [Optional] xx_ext (str or list or tuple): extensions of the xx images.

                or

                - dataroot_xx (str): root directory of the xx images
        """
        GlobDatasetBase.__init__(self, io.glob_packed_files)
        TransformsDatasetBase.__init__(self, **option)
        self.get_all_file_paths(**option)
        self.random_clip = try_fill_default_dict(
            option.get('random_clip', None),
            seq_length=30,
        )

        self.pad_s = pad_s


    def __len__(self):
        return len(self.get_file_paths_case_insensitive(self.names[0], without_notified=False))

    def __getitem__(self, index):
        total_seq_length = len(self.get_file_paths_case_insensitive(self.names[0])[index])
        if self.random_clip:
            if isinstance(self.random_clip['seq_length'], tuple):
                seq_length = random.randint(*self.random_clip['seq_length'])
            elif isinstance(self.random_clip['seq_length'], list):
                seq_length = random.choice(self.random_clip['seq_length'])
            else:
                seq_length = self.random_clip['seq_length']
            padding_mode = self.random_clip.get('padding_mode', 'reflect')
            indices = generate_random_indices_in_a_sequence(total_seq_length, seq_length, padding_mode=padding_mode)
        else:
            indices = list(range(total_seq_length))

        datas = {}
        for name in self.names: # 'lq', 'gt', etc.
            key_name = name
            if self.pad_s:
                key_name = f"{name}s"   # 'lqs', 'gts', etc.

            frame_paths = self.get_file_paths_case_insensitive(name)[index]
            frame_paths = [frame_paths[i] for i in indices]
            frames = read_seq_images_to_tensor(frame_paths, self.transforms)
            datas[key_name] = {
                "images": frames,
                "paths": frame_paths,
                "common_dir": self.get_data_dir(name, frame_paths),
            }
            return data

        raise ValueError("No dataset found in the config. Please add ('dataroot', 'xx_dir', 'xx_ext') or ('dataroot_xx') to the config.")

    def reprepare_data(self):
        pass


# noinspection SpellCheckingInspection
@DATASET_REGISTRY.register()
class PairedVideoDataset(VideoDataset):
    def __init__(self, pad_s=True, **option):
        """
        Dataset for video-to-video tasks.

        Args:
            option: Config for train datasets. It contains the following keys:
            (xx can be filled with 'gt', 'lq', etc., and allows multiple values)
                - dataroot (str): root directory of the dataset.
                - xx_dir (str): directory of the xx images.
                - [Optional] xx_ext (str or list or tuple): extensions of the xx images.

                or

                - dataroot_xx (str): root directory of the xx images
        """
        super(PairedVideoDataset, self).__init__(pad_s=pad_s, **option)

    def __getitem__(self, index):
        seed = random.randint(1, 2**32) # ensure the same transform for paired images

        total_seq_length = len(self.get_file_paths_case_insensitive(self.names[0])[index])
        if self.random_clip:
            if isinstance(self.random_clip['seq_length'], tuple):
                seq_length = random.randint(*self.random_clip['seq_length'])
            elif isinstance(self.random_clip['seq_length'], list):
                seq_length = random.choice(self.random_clip['seq_length'])
            else:
                seq_length = self.random_clip['seq_length']
            padding_mode = self.random_clip.get('padding_mode', 'reflect')
            indices = generate_random_indices_in_a_sequence(total_seq_length, seq_length, padding_mode=padding_mode)
        else:
            indices = list(range(total_seq_length))

        datas = {}
        for name in self.names: # 'lq', 'gt', etc.
            key_name = name
            if self.pad_s:
                key_name = f"{name}s"   # 'lqs', 'gts', etc.

            frame_paths = self.get_file_paths_case_insensitive(name)[index]
            frame_paths = [frame_paths[i] for i in indices]
            frames = read_seq_images_to_tensor(frame_paths, self.transforms, seed)
            datas[key_name] = {
                "images": frames,   # [N, C, H, W]
                "paths": frame_paths,
                "common_dir": self.get_data_dir(name, frame_paths),
            }
        return datas


# noinspection SpellCheckingInspection
@DATASET_REGISTRY.register()
class DynamicVideoDataset(GlobDatasetBase, TransformsDatasetBase):
    def __init__(self, **option):
        """
        Dataset for video-to-video tasks. Instead of loading all frames into memory, it loads only one frame at a time.

        Args:
            option: Config for train datasets. It contains the following keys:
            (xx can be filled with 'gt', 'lq', etc., and allows multiple values)
                - dataroot (str): root directory of the dataset.
                - xx_dir (str): directory of the xx images.
                - [Optional] xx_ext (str or list or tuple): extensions of the xx images.

                or

                - dataroot_xx (str): root directory of the xx images
        """
        GlobDatasetBase.__init__(self, io.glob_packed_files)
        TransformsDatasetBase.__init__(self, **option)
        self.get_all_file_paths(**option)

        ### preparation
        frames_paths = None
        for name in self.names: # 'lq', 'gt', etc.
            frames_paths = self.get_file_paths_case_insensitive(name)
            break
        assert frames_paths is not None, (
            "No dataset found in the config. Please add ('dataroot', 'xx_dir', 'xx_ext') or ('dataroot_xx') to the config. "
            "Or check the path of the dataset."
        )
        self.frames_paths = frames_paths

        self.video_count = len(self.frames_paths)
        self.frame_counts = [len(frame_paths) for frame_paths in self.frames_paths]
        self.all_frame_count = sum(len(frame_paths) for frame_paths in self.frames_paths)

        self.seeds = []
        self.reprepare_data()

    def __len__(self):
        return self.all_frame_count

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            raise ValueError(
                "Index should be a tuple of (video_index, frame_index, end)."
                "Or use VideoClipSampler in the dataloader."
                f"Got {index} with type {type(index)}."
            )

        video_index, frame_index, end = index
        seed = self.seeds[video_index] # ensure the same transform for paired and video-same frames

        datas = {}
        for name in self.names: # 'lq', 'gt', etc.
            frame_path = self.get_file_paths_case_insensitive(name)[video_index][frame_index]
            frame = read_image_to_tensor(frame_path, self.transforms, seed)
            datas[name] = {
                "image": frame,
                "path": frame_path,
                "common_dir": self.get_data_dir(name, frame_path),
            }
            datas["end"] = end,
            return datas
        raise ValueError("No dataset found in the config. Please add ('dataroot', 'xx_dir', 'xx_ext') or ('dataroot_xx') to the config.")

    def reprepare_data(self):
        ### the seeds for each video (used for random transform)
        self.seeds = [random.randint(1, 2 ** 32) for _ in range(len(self.frames_paths))]


# noinspection SpellCheckingInspection
@DATASET_REGISTRY.register()
class PairedDynamicVideoDataset(DynamicVideoDataset):
    def __init__(self, **option):
        """
        Dataset for video-to-video tasks. Instead of loading all frames into memory, it loads only one frame at a time.

        Args:
            option: Config for train datasets. It contains the following keys:
            (xx can be filled with 'gt', 'lq', etc., and allows multiple values)
                - dataroot (str): root directory of the dataset.
                - xx_dir (str): directory of the xx images.
                - [Optional] xx_ext (str or list or tuple): extensions of the xx images.

                or

                - dataroot_xx (str): root directory of the xx images
        """
        super(PairedDynamicVideoDataset, self).__init__(**option)

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            raise ValueError(
                f"Index should be a tuple of (video_index, frame_index, end)."
                f"Or use VideoClipSampler in the dataloader."
                f"Got {index} with type {type(index)}."
            )

        video_index, frame_index, end = index
        seed = self.seeds[video_index] # ensure the same transform for paired and video-same frames

        datas = {}
        for name in self.names: # 'lq', 'gt', etc.
            frame_path = self.get_file_paths_case_insensitive(name)[video_index][frame_index]
            frame = read_image_to_tensor(frame_path, self.transforms, seed)
            datas[name] = {
                "image": frame,
                "path": frame_path,
                "common_dir": self.get_data_dir(name, frame_path),
            }
        datas["end"] = end,
        return datas



if __name__ == '__main__':
    image_dataset = PairedImageDataset(dataroot_gt='/path/to/Datasets/LLIE/LOL-v1/eval15/high', dataroot_lq='/path/to/Datasets/LLIE/LOL-v1/eval15/low')
    print(len(image_dataset))

    image_dataset = PairedImageDataset(dataroot='/path/to/Datasets/LLIE/LOL-v1/eval15', gt_dir='high', lq_dir='low')
    print(len(image_dataset))

    image_dataset = ImageDataset(dataroot_gt='/path/to/Datasets/LLIE/LOL-v1/eval15/high')
    print(len(image_dataset))

    image_dataset = ImageDataset(dataroot='/path/to/Datasets/LLIE/LOL-v1/eval15', gt_dir='high')
    print(len(image_dataset))