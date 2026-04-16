import re
import os.path as osp
from copy import deepcopy
from collections import OrderedDict
import inspect
import importlib
import enum

from utils.console.path import root
from utils.general import NoneDict
from utils.console.log import get_root_logger, dict_to_str

logger = get_root_logger()

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


# 确保键值对的顺序和 YAML 文件中的顺序是一致的
def OrderedYaml():
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    # noinspection SpellCheckingInspection
    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)

    def _import_object(dotted_path: str):
        """Import an object from a dotted path.

        Examples:
            - torchvision.transforms.InterpolationMode
            - torch.nn.functional.interpolate
        """
        if not isinstance(dotted_path, str) or not dotted_path.strip():
            raise ValueError(f"Invalid import path: {dotted_path!r}")

        dotted_path = dotted_path.strip()
        if ':' in dotted_path:
            module_name, attr_path = dotted_path.split(':', 1)
            module = importlib.import_module(module_name)
            obj = module
            for attr in filter(None, attr_path.split('.')):
                obj = getattr(obj, attr)
            return obj

        parts = dotted_path.split('.')
        last_exc = None
        for i in range(len(parts), 0, -1):
            module_name = '.'.join(parts[:i])
            try:
                module = importlib.import_module(module_name)
                obj = module
                for attr in parts[i:]:
                    obj = getattr(obj, attr)
                return obj
            except Exception as exc:  # ImportError or AttributeError
                last_exc = exc
                continue
        raise ValueError(f"Failed to import '{dotted_path}': {last_exc}")

    def _construct_from_type(loader_obj, target, node):
        """Construct a Python object from a resolved target + YAML node."""
        # Scalar node: Enum member / attribute / ctor(single-arg) / raw type
        if isinstance(node, yaml.ScalarNode):
            value = loader_obj.construct_scalar(node)
            # allow empty scalar to mean "the type itself"
            if value is None or (isinstance(value, str) and value.strip() == ''):
                return target

            if isinstance(target, type) and issubclass(target, enum.Enum):
                if isinstance(value, str) and value in target.__members__:
                    return target[value]
                return target(value)

            if isinstance(value, str) and hasattr(target, value):
                return getattr(target, value)

            if callable(target):
                try:
                    return target(value)
                except Exception as exc:
                    raise ValueError(
                        f"Cannot construct '{getattr(target, '__name__', str(target))}' from scalar {value!r}: {exc}"
                    )

            return target

        # Sequence node: treat as *args
        if isinstance(node, yaml.SequenceNode):
            args = loader_obj.construct_sequence(node, deep=True)
            if callable(target):
                return target(*args)
            raise ValueError(f"Target {target!r} is not callable; cannot apply args {args!r}")

        # Mapping node: treat as **kwargs
        if isinstance(node, yaml.MappingNode):
            kwargs = loader_obj.construct_mapping(node, deep=True)
            if callable(target):
                return target(**kwargs)
            raise ValueError(f"Target {target!r} is not callable; cannot apply kwargs {kwargs!r}")

        raise ValueError(f"Unsupported YAML node type: {type(node)}")

    def _construct_type(loader_obj, node):
        """YAML tag !!type: resolve scalar into a Python object (typically a type/class)."""
        path = loader_obj.construct_scalar(node)
        return _import_object(path)

    def _construct_undefined(loader_obj, node):
        """Handle custom YAML tags.

        Supported:
            - !!type "some.module.Class" -> returns the imported object
            - !!some.module.Enum "MEMBER" -> returns Enum member
            - !!some.module.Class {kwargs} -> returns instantiated object
            - !!some.module.func [args] -> returns func(*args)
        """
        tag = getattr(node, 'tag', '') or ''
        # !!foo maps to tag:yaml.org,2002:foo
        if tag.startswith('tag:yaml.org,2002:'):
            suffix = tag.split('tag:yaml.org,2002:', 1)[1]
        elif tag.startswith('!'):
            suffix = tag[1:]
        else:
            suffix = tag

        if suffix == 'type':
            return _construct_type(loader_obj, node)

        target = _import_object(suffix)
        return _construct_from_type(loader_obj, target, node)

    # Support both !!type and !type
    Loader.add_constructor('tag:yaml.org,2002:type', _construct_type)
    Loader.add_constructor('!type', _construct_type)
    # Fallback for other custom tags like !!torchvision.transforms.InterpolationMode
    Loader.add_constructor(None, _construct_undefined)
    return Loader, Dumper


loader, dumper = OrderedYaml()


'''
Modified from Retinexformer(https://github.com/caiyuanhao1998/Retinexformer/blob/master/basicsr/utils/options.py)
'''


# 解析 config 中 ${key} 形式的占位符
def resolve_placeholders(config, context=None):
    """
    Recursively resolve placeholders (e.g., ${key}) in config.

    Args:
        config (dict): Config dict.
        context (dict): Context dict. Default: None.

    Returns:
        dict: Resolved config dict.
    """
    if context is None:
        context = dict()

    if isinstance(config, dict):
        for key, value in config.items():
            # context only contains the key-value pairs in the current level or the parent level
            # if the key is not in the current level, it will be searched in the parent level
            context_next = deepcopy(config)
            for k, v in context.items():
                # k == key means:
                # if k is in the parent level, and key is in the current level,
                # the value in the parent level will be used instead of the value in the current level
                if k not in context_next or k == key:
                    context_next[k] = v
            config[key] = resolve_placeholders(value, context_next)
    elif isinstance(config, list):
        for i, item in enumerate(config):
            config[i] = resolve_placeholders(item, context)
    elif isinstance(config, str):
        match = re.match(r"\$\{(.*)}", config)  # match placeholder like ${key}
        if match:
            expr = match.group(1)
            try:
                value = eval(expr, {}, context)         # evaluate expression in context
                return value
            except Exception as e:
                raise ValueError(f"Failed to evaluate expression '{expr}': {e}")
    return config


# noinspection SpellCheckingInspection
def parse(opt_path, is_train=True, alias=None):
    with open(opt_path, mode='r',encoding='utf-8') as f:
        opt = yaml.load(f, Loader=loader)

    # [resolve placeholders] 解析 config 中 ${key} 形式的占位符
    opt = resolve_placeholders(opt)

    # [set default values] 设置一些默认值
    opt['is_train'] = is_train
    if 'name' not in opt:
        basename = osp.basename(opt_path)
        opt_name = '.'.join(basename.split('.')[:-1]) if '.' in basename else basename
        opt['name'] = opt_name  # 文件名即为 option 名称
    if alias is not None:
        opt['name'] = f'{opt["name"]}_{alias}'

    # [metrics]
    if 'val' in opt and 'metrics' in opt['val']:
        for metric in opt['val']['metrics']:
            if '↑' in metric['type']:
                metric['type'] = metric['type'].replace('↑', '').strip()
                metric['better'] = 'higher'
            elif '↓' in metric['type']:
                metric['type'] = metric['type'].replace('↓', '').strip()
                metric['better'] = 'lower'

    # [datasets]
    if 'datasets' in opt:
        for phase, dataset in opt['datasets'].items():
            # for several datasets, e.g., test_1, test_2
            phase = phase.split('_')[0]
            dataset['phase'] = phase
            if 'scale' in opt:
                dataset['scale'] = opt['scale']
            # 将带有 dataroot_ 前缀的路径展开
            for key, path in dataset.items():
                if 'dataroot_' in key:
                    dataset[key] = osp.expanduser(path)

    # [path]
    if 'path' in opt:
        path_opt = opt['path']
        def expand_path(path_opt):
            if isinstance(path_opt, dict):
                for key, path in path_opt.items():
                    if isinstance(path, str) and 'strict_load' not in key:
                        path_opt[key] = osp.expanduser(path)
                    elif isinstance(path, (dict, list)):
                        expand_path(path)
            elif isinstance(path_opt, list):
                for i, item in enumerate(path_opt):
                    expand_path(item)
        if 'root' not in path_opt:
            path_opt['root'] = root
        if is_train:
            experiments_root: str = osp.join(path_opt['root'], 'experiments', opt['name'])
            path_opt['experiments_root'] = experiments_root
            path_opt['models'] = osp.join(experiments_root, 'models')
            path_opt['training_state'] = osp.join(experiments_root, 'training_state')
            path_opt['log'] = experiments_root
            path_opt['tensorboard_log'] = osp.join(experiments_root, 'tensorboard_log')
            path_opt['val_images'] = osp.join(experiments_root, 'val_images')

            # change some options for debug mode
            if 'debug' in opt['name']:
                opt['train']['val_freq'] = 10
                opt['logger']['print_freq'] = 1
                opt['logger']['save_checkpoint_freq'] = 10
        else:  # test
            results_root: str = osp.join(path_opt['root'], 'results', opt['name'])
            path_opt['results_root'] = results_root
            path_opt['log'] = results_root
            path_opt['test_images'] = osp.join(results_root, 'test_images')

    return opt


def parse_params(option, other_as_params=True, type_default=None):
    assert isinstance(option, dict), f"The option must be a dict, but got {type(option)}."
    assert 'type' in option, f"The option must contain the key 'type', but got {dict_to_str(option, max_depth=1)}."
    type = option.pop('type', type_default)
    if 'params' in option:
        params = option.pop('params')
    elif other_as_params:
        params = option
    else:
        params = {}
    return type, params


def parse_arguments(_class, params):
    """
    Solving the "*dims" problem in the argument of class.
    """
    init_signature = inspect.signature(_class.__init__)
    var_pos_name = None
    for name, param in init_signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            var_pos_name = name
            break

    if var_pos_name:
        args = params.pop(var_pos_name, [])
        kwargs = params
    else:
        args = []
        kwargs = params

    return args, kwargs


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def check_resume(config, net_label, resume_iter):
    """
    Check resume states and pretrain_model paths
    """
    def get_model_name(state_name):
        if net_label is None:
            model_name = f'{state_name}.pth'
        else:
            model_name = f'{state_name}_{net_label}.pth'
        return model_name

    # 如果存在 resume_state，则 pretrain_model_G 路径设置为 resume_state 中指定的模型路径
    path_config = config['path']
    if 'resume_state' in path_config:
        resume_state_path = path_config['resume_state']
        old_pretrain_model_G = path_config.get('pretrain_model_G', None)
        old_pretrain_model_D = path_config.get('pretrain_model_D', None)

        training_state_dir = osp.dirname(resume_state_path)
        # 如果 training_state 路径和 config 名对应，则按照 config 中的默认 models 读取 pretrain_model_G
        # 否则按照 training_state 读取 pretrain_model_G
        if path_config['training_state'] == training_state_dir:
            model_dir = path_config['models']
        else:
            model_dir = osp.join(osp.dirname(training_state_dir), osp.basename(path_config['models']))

        # 优先读取有带 iter 的模型路径，再读取和 state 名字对应的模型路径
        state_name = resume_iter
        path_config['pretrain_model_G'] = osp.join(model_dir, get_model_name(state_name))
        if old_pretrain_model_G is not None and path_config['pretrain_model_G'] != old_pretrain_model_G:
            logger.warning('pretrain_model_G path will be ignored when resuming training.')
        if not osp.exists(path_config['pretrain_model_G']):
            state_name = osp.splitext(osp.basename(path_config['resume_state']))[0]
            path_config['pretrain_model_G'] = osp.join(model_dir, get_model_name(state_name))
        logger.info('Set [pretrain_model_G] to ' + path_config['pretrain_model_G'])

        # 如果是 GAN，则同时修改判别模型的路径
        if 'gan' in config['model']:
            path_config['pretrain_model_D'] = osp.join(model_dir, get_model_name(state_name))
            if old_pretrain_model_D is not None and path_config['pretrain_model_D'] != old_pretrain_model_D:
                logger.warning('pretrain_model_D path will be ignored when resuming training.')

            logger.info('Set [pretrain_model_D] to ' + path_config['pretrain_model_D'])
