import re
from datetime import datetime
from contextlib import contextmanager
import difflib

'''
和调试相关的工具函数
'''


#region ==[字体颜色]==
# 颜色代码
class Font:
    # 前景色（字体颜色）
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    DEFAULT = "\033[39m"  # 默认颜色

    # 背景色
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"
    BG_DEFAULT = "\033[49m"  # 默认背景色

    # 样式
    BOLD = "\033[1m"        # 加粗
    DIM = "\033[2m"         # 暗淡
    ITALIC = "\033[3m"      # 斜体（部分终端不支持）
    UNDERLINE = "\033[4m"   # 下划线
    BLINK = "\033[5m"       # 闪烁
    REVERSE = "\033[7m"     # 反色（前景色和背景色互换）
    HIDDEN = "\033[8m"      # 隐藏（文字不可见）

    # 重置所有样式和颜色
    RESET = "\033[0m"

    # 亮色（高亮前景色）
    LIGHT_BLACK = "\033[90m"
    LIGHT_RED = "\033[91m"
    LIGHT_GREEN = "\033[92m"
    LIGHT_YELLOW = "\033[93m"
    LIGHT_BLUE = "\033[94m"
    LIGHT_MAGENTA = "\033[95m"
    LIGHT_CYAN = "\033[96m"
    LIGHT_WHITE = "\033[97m"

    # 亮色背景
    BG_LIGHT_BLACK = "\033[100m"
    BG_LIGHT_RED = "\033[101m"
    BG_LIGHT_GREEN = "\033[102m"
    BG_LIGHT_YELLOW = "\033[103m"
    BG_LIGHT_BLUE = "\033[104m"
    BG_LIGHT_MAGENTA = "\033[105m"
    BG_LIGHT_CYAN = "\033[106m"
    BG_LIGHT_WHITE = "\033[107m"

    # 暗色
    DIM_BLACK = "\033[2m\033[30m"
    DIM_RED = "\033[2m\033[31m"
    DIM_GREEN = "\033[2m\033[32m"
    DIM_YELLOW = "\033[2m\033[33m"
    DIM_BLUE = "\033[2m\033[34m"
    DIM_MAGENTA = "\033[2m\033[35m"
    DIM_CYAN = "\033[2m\033[36m"
    DIM_WHITE = "\033[2m\033[37m"

    # 暗色背景
    BG_DIM_BLACK = "\033[2m\033[40m"
    BG_DIM_RED = "\033[2m\033[41m"
    BG_DIM_GREEN = "\033[2m\033[42m"
    BG_DIM_YELLOW = "\033[2m\033[43m"
    BG_DIM_BLUE = "\033[2m\033[44m"
    BG_DIM_MAGENTA = "\033[2m\033[45m"
    BG_DIM_CYAN = "\033[2m\033[46m"
    BG_DIM_WHITE = "\033[2m\033[47m"


    @staticmethod
    def _debug_color():
        print(f"{Font.BLACK}BLACK{Font.RESET} "
              f"{Font.DEFAULT}DEFAULT{Font.RESET} "
              f"{Font.MAGENTA}MAGENTA{Font.RESET} "
              f"{Font.RED}RED{Font.RESET} "
              f"{Font.YELLOW}YELLOW{Font.RESET} "
              f"{Font.GREEN}GREEN{Font.RESET} "
              f"{Font.BLUE}BLUE{Font.RESET} "
              f"{Font.CYAN}CYAN{Font.RESET} "
              f"{Font.WHITE}WHITE{Font.RESET} "
              f"{Font.LIGHT_BLACK}LIGHT_BLACK{Font.RESET} "
              f"{Font.LIGHT_RED}LIGHT_RED{Font.RESET} "
              f"{Font.LIGHT_GREEN}LIGHT_GREEN{Font.RESET} "
              f"{Font.LIGHT_YELLOW}LIGHT_YELLOW{Font.RESET} "
              f"{Font.LIGHT_BLUE}LIGHT_BLUE{Font.RESET} "
              f"{Font.LIGHT_MAGENTA}LIGHT_MAGENTA{Font.RESET} "
              f"{Font.LIGHT_CYAN}LIGHT_CYAN{Font.RESET} "
              f"{Font.LIGHT_WHITE}LIGHT_WHITE{Font.RESET} "
              f"{Font.DIM_BLACK}DIM_BLACK{Font.RESET} "
              f"{Font.DIM_RED}DIM_RED{Font.RESET} "
              f"{Font.DIM_GREEN}DIM_GREEN{Font.RESET} "
              f"{Font.DIM_YELLOW}DIM_YELLOW{Font.RESET} "
              f"{Font.DIM_BLUE}DIM_BLUE{Font.RESET} "
              f"{Font.DIM_MAGENTA}DIM_MAGENTA{Font.RESET} "
              f"{Font.DIM_CYAN}DIM_CYAN{Font.RESET} "
              f"{Font.DIM_WHITE}DIM_WHITE{Font.RESET} "
              
              f"{Font.BOLD}BOLD{Font.RESET} "
              f"{Font.DIM}DIM{Font.RESET} "
              f"{Font.ITALIC}ITALIC{Font.RESET} "
              f"{Font.UNDERLINE}UNDERLINE{Font.RESET} "
              f"{Font.BLINK}BLINK{Font.RESET} "
              f"{Font.REVERSE}REVERSE{Font.RESET} "
              f"{Font.HIDDEN}HIDDEN{Font.RESET} "
              
              f"{Font.BG_BLACK}BG_BLACK{Font.RESET} "
              f"{Font.BG_RED}BG_RED{Font.RESET} "
              f"{Font.BG_GREEN}BG_GREEN{Font.RESET} "
              f"{Font.BG_YELLOW}BG_YELLOW{Font.RESET} "
              f"{Font.BG_BLUE}BG_BLUE{Font.RESET} "
              f"{Font.BG_MAGENTA}BG_MAGENTA{Font.RESET} "
              f"{Font.BG_CYAN}BG_CYAN{Font.RESET} "
              f"{Font.BG_WHITE}BG_WHITE{Font.RESET} "
              f"{Font.BG_DEFAULT}BG_DEFAULT{Font.RESET} "
              f"{Font.BG_LIGHT_BLACK}BG_LIGHT_BLACK{Font.RESET} "
              f"{Font.BG_LIGHT_RED}BG_LIGHT_RED{Font.RESET} "
              f"{Font.BG_LIGHT_GREEN}BG_LIGHT_GREEN{Font.RESET} "
              f"{Font.BG_LIGHT_YELLOW}BG_LIGHT_YELLOW{Font.RESET} "
              f"{Font.BG_LIGHT_BLUE}BG_LIGHT_BLUE{Font.RESET} "
              f"{Font.BG_LIGHT_MAGENTA}BG_LIGHT_MAGENTA{Font.RESET} "
              f"{Font.BG_LIGHT_CYAN}BG_LIGHT_CYAN{Font.RESET} "
              f"{Font.BG_LIGHT_WHITE}BG_LIGHT_WHITE{Font.RESET} "
              f"{Font.RESET}RESET{Font.RESET}")

    @staticmethod
    def get_256_color(code):
        return f"\033[38;5;{code}m"

# 颜色函数
def color_text(text, color_code):
    return f"{color_code}{text}{Font.RESET}"


#region --[预制函数]--
class ColorPrefeb:
    @staticmethod
    def number(text, format=None):
        if isinstance(text, float):
            if format is None:
                return color_text(auto_number_to_str(text), Font.LIGHT_MAGENTA)
            else:
                return color_text(f"{text:{format}}", Font.LIGHT_MAGENTA)
        else:
            return color_text(text, Font.LIGHT_MAGENTA)

    @staticmethod
    def best_number(text, format=None):
        if isinstance(text, float):
            if format is None:
                return color_text(auto_number_to_str(text), Font.YELLOW)
            else:
                return color_text(f"{text:{format}}", Font.YELLOW)
        else:
            return color_text(text, Font.YELLOW)

    @staticmethod
    def integer_number(text):
        return color_text(text, Font.LIGHT_BLUE)

    @staticmethod
    def number_scaled(value, e_notation=1, format=None):
        """
        Format a number with a scaled exponent.

        e.g.
            number_scaled(0.00196, 1e-3) -> 1.96(1e-3)

        Args:
            value: The number to format.
            e_notation: The exponent notation to use.
            format: The format string for the coefficient.

        Returns:
            The formatted number.
        """
        return ColorPrefeb.number(number_scaled_to_str(value, e_notation, format))

    @staticmethod
    def keyword(text):
        return color_text(text, Font.DIM_MAGENTA)

    @staticmethod
    def path(text):
        return color_text(text, Font.BLUE)

    @staticmethod
    def command(text):
        return color_text(text, Font.UNDERLINE)

    @staticmethod
    def bool(bool_value: bool, text_on_true: str = None, text_on_false: str = None):
        if text_on_true is None or text_on_false is None:
            if bool_value and text_on_true is not None:
                return color_text(text_on_true, Font.LIGHT_GREEN)
            elif not bool_value and text_on_false is not None:
                return color_text(text_on_false, Font.LIGHT_RED)
            elif not (text_on_true is None and text_on_false is None):
                raise ValueError("If bool_value is True, text_on_true must be provided, and if bool_value is False, text_on_false must be provided.")

            if bool_value:
                return color_text("True", Font.LIGHT_GREEN)
            else:
                return color_text("False", Font.LIGHT_RED)
        else:
            if bool_value:
                return color_text(text_on_true, Font.LIGHT_GREEN)
            else:
                return color_text(text_on_false, Font.LIGHT_RED)


    @staticmethod
    def frozen(text):
        return color_text(text, Font.LIGHT_BLUE)

    @staticmethod
    def active(text):
        return color_text(text, Font.LIGHT_RED)

    @staticmethod
    def yes(text):
        return color_text(text, Font.LIGHT_GREEN)
    
    @staticmethod
    def true(text):
        return color_text(text, Font.LIGHT_GREEN)

    @staticmethod
    def no(text):
        return color_text(text, Font.LIGHT_RED)

    @staticmethod
    def false(text):
        return color_text(text, Font.LIGHT_RED)

    @staticmethod
    def dim(text):
        return color_text(text, Font.DIM)

    @staticmethod
    def error(text):
        return color_text(text, Font.RED)

    @staticmethod
    def warning(text):
        return color_text(text, Font.YELLOW)

    @staticmethod
    def success(text):
        return color_text(text, Font.LIGHT_GREEN)

    @staticmethod
    def fail(text):
        return color_text(text, Font.LIGHT_RED)

#endregion
#endregion


#region ==[to_str 函数]==
def number_scaled_to_str(value, e_notation=1, format=None):
    """
    Format a number with a scaled exponent.

    e.g.
        number_scaled(0.00196, 1e-3) -> 1.96(1e-3)

    Args:
        value: The number to format.
        e_notation: The exponent notation to use.
        format: The format string for the coefficient.

    Returns:
        The formatted number.
    """
    # 去除指数部分的前导零
    def exponential_format(number):
        formatted = "{:e}".format(number)
        coefficient, exponent = formatted.split('e')
        sign = exponent[0]
        exponent = exponent[1:].lstrip('0')
        if exponent == '':
            exponent = '0'
        return f"{float(coefficient):.0f}e{sign}{exponent}"

    if e_notation == 1:
        if format is None:
            return f"{value}"
        else:
            return f"{value:{format}}"
    else:
        if format is None:
            return f'{f"{value / e_notation}"}(×{exponential_format(e_notation)})'
        else:
            return f'{f"{value / e_notation:{format}}"}(×{exponential_format(e_notation)})'


def auto_number_to_str(value):
    if value < 1e-2:
        value = f"{value * 1000:.4f}(1e-3)"
    elif value >= 100:
        value = f"{value / 1000:.4f}(1e3)"
    else:
        value = f"{value:.4f}"
    return value


def dict_to_str(yaml_options, indent_length=1, max_depth=None):
    """
    Format a dictionary(e.g. yaml options) as a string.

    Args:
        yaml_options: The dictionary to format.
        indent_length: The number of spaces to indent each level.

    Returns:
        The formatted string.
    """
    if not isinstance(yaml_options, dict):
        raise TypeError(f"yaml_options must be a dictionary. But got {type(yaml_options)}.")

    indent = ' ' * (indent_length * 2)

    if max_depth is not None:
        max_depth -= 1
        if max_depth < 0:
            return f'\n{indent}...'

    text = ''
    for key, value in yaml_options.items():
        if isinstance(value, dict):
            text += (f"\n{indent}{key}: {{"
                     f"{dict_to_str(value, indent_length + 1, max_depth)}"
                     f"\n{indent}}}")
        elif isinstance(value, list):
            text += f"\n{indent}{key}: ["
            for item in value:
                if isinstance(item, dict):
                    text += f"{dict_to_str(item, indent_length + 1, max_depth)}, "
                else:
                    text += f"{item}, "
            text = text.rstrip(", ") + f"]"  # Remove the trailing comma and space
        else:
            text += f"\n{indent}{key}: {value}"
    return text


def get_time_str(with_color=True, path_friendly=False):
    """
    Get the current time string in the format of "YYYY-MM-DD HH:MM:SS.SSS".
    """
    from datetime import datetime
    try:
        import pytz

        china_tz = pytz.timezone('Asia/Shanghai')
        now = datetime.now(china_tz)
    except ImportError:
        now = datetime.now()

    if path_friendly:
        timestamp_text = now.strftime('%Y%m%d_%H%M%S_%f')[:-3]
    else:
        timestamp_text = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    if with_color:
        timestamp_text = color_text(f"[{timestamp_text}]", Font.LIGHT_BLACK)
    return timestamp_text

def get_striped_time_str():
    return get_time_str(with_color=False, path_friendly=True)


def get_env_str(with_color=True):
    import torch
    import torchvision
    text = r"""
     __    __   ______   __    __   __       __       __    __ ______    
    /\ "-./  \ /\  ___\ /\ "-./  \ /\ \     /\ \     /\ \  / //\  ___\   
    \ \ \-./\ \\ \  __\ \ \ \-./\ \\ \ \____\ \ \____\ \ \/ / \ \  __\   
     \ \_\ \ \_\\ \_____\\ \_\ \ \_\\ \_____\\ \_____\\ \__/   \ \_____\ 
      \/_/  \/_/ \/_____/ \/_/  \/_/ \/_____/ \/_____/ \/_/     \/_____/ 
    """
    if with_color:
        text = color_text(text, Font.LIGHT_CYAN)
    text += ('\nVersion Information: '
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}')
    return text


def pad_blank(text, fill_char=' '):
    import shutil
    width = shutil.get_terminal_size().columns
    text_without_color = re.sub(r'\x1b\[[0-9;]*m', '', text)
    fill_width = width - len(text_without_color)
    return text + fill_char * fill_width
#endregion


#region ==[其他工具函数]==
def highlight_diff(str1, str2):
    differ = difflib.Differ()
    diff = list(differ.compare(str1.splitlines(), str2.splitlines()))

    result = []
    for line in diff:
        if line.startswith('  '):       # 无差异
            result.append(line[2:])
        elif line.startswith('- '):     # 第一个字符串有
            result.append(color_text(line[2:], Font.BG_RED))
        elif line.startswith('+ '):     # 第二个字符串有
            result.append(color_text(line[2:], Font.BG_GREEN))
    return '\n'.join(result)
#endregion


#region ==[logger]==
import os
from os import path as osp
import shutil
import logging


initialized_logger = {}


def get_root_logger(
        logger_name='main', log_level=logging.INFO,
        log_root=None, log_file_name_prefix=None,
        force_set_info=False
):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default, a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
        log_root (str): The root directory to save log files. If not specified,
            the log will be saved to the current working directory.
        log_file_name_prefix (str): The prefix of the log file name. If not
            specified, the log will be saved to `log.log`.
        force_set_info (bool): Whether to force set the log level to INFO. This
            will also enable the console logging.

    Returns:
        logging.Logger: The root logger.
    """
    # [get file name]
    if log_root is not None and log_file_name_prefix is not None:
        if not osp.exists(log_root):
            os.makedirs(log_root, exist_ok=True)
        log_file = os.path.join(log_root, f'{log_file_name_prefix}_{get_time_str(with_color=False)}.log')
    else:
        log_file = None

    # [pre-]
    logger = logging.getLogger(logger_name)
    if force_set_info:
        logger.setLevel(logging.INFO)
    # if the logger has been initialized, just return it
    if log_file is None and logger_name in initialized_logger:
        return logger

    # [add screen handler]
    if ((logger_name != 'metrics' and 'test_' not in logger_name and 'fileonly_' not in logger_name)
            and logger_name not in initialized_logger):   # can only be initialized once
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(CustomFormatter())
        logger.addHandler(stream_handler)
        logger.propagate = False

    # [add file handler]
    if log_file is not None:
        logger.setLevel(log_level)

        # remove old file handler, and move old log file to new log file path
        if (any([isinstance(l, logging.StreamHandler) for l in logger.handlers]) and len(logger.handlers) > 1) or len(logger.handlers) >= 1:
            for handler in reversed(logger.handlers):  # newer first
                if isinstance(handler, logging.FileHandler):
                    old_log_file_path = handler.baseFilename
                    handler.close()
                    logger.removeHandler(handler)

                    # move old log file to new log file path
                    new_log_file_path = log_file
                    if os.path.exists(old_log_file_path):
                        shutil.move(old_log_file_path, new_log_file_path)

                    logger.info(f'Log file moved from {ColorPrefeb.path(old_log_file_path)} to {ColorPrefeb.path(new_log_file_path)}')

        # add new file handler
        if logger_name != 'metric':
            file_handler = logging.FileHandler(log_file, 'a')
        else:
            file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(StripedCustomFormatter())
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)        # can add multiple file handlers
    initialized_logger[logger_name] = dict(
        log_root=log_root
    )
    return logger


def auto_path(path: str, logger_name='main'):
    if logger_name not in initialized_logger:
        raise ValueError(f"Logger {logger_name} has not been initialized. Please initialize it first.")

    root = initialized_logger[logger_name]['log_root']
    if root is None:
        root = "./.plotlogs"

    if '.' in os.path.basename(path):
        fig_name, suffix = '.'.join(path.split('.')[:-1]), path.split('.')[-1]
        suffix = f".{suffix}"
    else:
        fig_name, suffix = path, ""
    if fig_name is not None:
        if fig_name.endswith('+'):
            fig_name = f"{fig_name[:-1]}_{get_striped_time_str()}{suffix}"
        else:
            fig_name = f"{fig_name}{suffix}"
    else:
        fig_name = f"{get_striped_time_str()}{suffix}"

    file_path = osp.join(root, fig_name)
    dir_path = osp.dirname(file_path)
    if not osp.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    return file_path


@contextmanager
def logger_set_level(logger, level, ignore_user_warning=True):
    """
    Set the logging level of the logger within the context.
    """
    import warnings

    old_level = logger.level
    if not isinstance(old_level, int):
        old_level = logging.INFO

    if isinstance(level, str):
        level = level.upper()
    logger.setLevel(level)
    if ignore_user_warning:
        warnings.filterwarnings("ignore", category=UserWarning)

    try:
        yield
    finally:
        logger.setLevel(old_level)
        if ignore_user_warning:
            warnings.filterwarnings("default", category=UserWarning)


class CustomFormatter(logging.Formatter):
    def format(self, record):
        # if the log level is INFO, use the default format (without level name)
        if record.levelno == logging.INFO:
            self._style._fmt = f'{color_text("[%(asctime)s]", Font.LIGHT_BLACK)} %(message)s'
        else:
            # for other levels, use the format with level name
            self._style._fmt = f'{color_text("[%(asctime)s]", Font.LIGHT_BLACK)} %(levelname)s: %(message)s'

        return super().format(record)


class StripedCustomFormatter(logging.Formatter):
    def format(self, record):
        original_msg = record.msg
        striped_msg = re.sub(r'\x1b\[[0-9;]*m', '', original_msg)
        record.msg = striped_msg

        # if the log level is INFO, use the default format (without level name)
        if record.levelno == logging.INFO:
            self._style._fmt = f'[%(asctime)s] %(message)s'
        else:
            # for other levels, use the format with level name
            self._style._fmt = f'[%(asctime)s] %(levelname)s: %(message)s'

        return super().format(record)


# 用来包装 logger 的日志打印器，使得在分布式计算或其他状况下时，不会报错（也不会显示），logger 还可以手动关闭（不会引起大批量代码的修改）
# noinspection SpellCheckingInspection
class Logger:
    def __init__(self, logger=None, **kwargs):
        if not kwargs.pop('init', True):
            self.logger = None
        else:
            self.logger = logger

    def __getattr__(self, name):
        if self.logger is not None:
            if hasattr(self.logger, name):
                def method(*args, **kwargs):
                    return getattr(self.logger, name)(*args, **kwargs)
                return method
            raise AttributeError(f"'Logger' object has no attribute '{name}'")
        else:
            return self.do_nothing

    def do_nothing(self, *args, **kwargs):
        pass

    @staticmethod
    def get_empty_logger(logger=None):
        return Logger(logger, init=False)


@contextmanager
def log_context(logger_factory, show_if=True, start_msg=None, end_msg=None, **kwargs) -> Logger:
    """
    A context manager to log the start and end of a process.

    Args:
        logger_factory (function): A function that returns a logger.
        show_if (bool): Whether to show the log messages.
        start_msg (str): The message to log at the start of the process.
        end_msg (str): The message to log at the end of the process.
    """
    _logger = get_log_if(logger_factory, condition=show_if, **kwargs)
    if start_msg is not None:
        _logger.info(start_msg)
    try:
        yield _logger
    except Exception as e:
        _logger.exception(e)
        raise e
    finally:
        if end_msg is not None:
            _logger.info(end_msg)


def get_log_if(logger_factory, condition=True, **kwargs) -> Logger:
    _logger = Logger(logger_factory(), **kwargs) if condition else Logger.get_empty_logger()
    return _logger


def logger_init_from_config(conf, log_file_name_prefix="train", rename=False, mkdir=False, log_info=False):
    """
    Initialize logger and create log directory.

    Args:
        conf (dict): Configuration.
        log_file_name_prefix (str): The prefix of the log file name.
        rename (bool): Whether to rename the log directory.
        mkdir (bool): Whether to create the log directory.
        log_info (bool): Whether to log the environment and configuration information.
    """
    # 如果要输出炫酷的版本信息，则 log_info=True
    # 在模型的 resume 权重加载之前的配置是 rename=False, mkdir=True, log_info=False
    # 在模型的 resume 权重加载之后的配置是 rename=True, mkdir=resume_state is None, log_info=True
    if mkdir:
        try:
            from utils.path import make_exp_dirs
            make_exp_dirs(conf, rename=rename)
        except ImportError:
            pass
    logger = get_root_logger(log_root=conf['path']['log'], log_file_name_prefix=log_file_name_prefix)
    file_logger = get_root_logger(logger_name="fileonly_main", log_root=conf['path']['log'], log_file_name_prefix=f"{log_file_name_prefix}_config")
    if log_info:
        try:
            from utils.console.torch_version import get_torch_version_str
            file_logger.info(get_torch_version_str())
        except ImportError:
            pass
        file_logger.info(dict_to_str(conf))
    return logger
#endregion


def obsolete(replacement=None):
    """
    如果是函数，可以直接使用：
        import warnings
        warnings.warn(
            f"{obj.__name__} is obsolete, use {replacement} instead",
            DeprecationWarning,
            stacklevel=2,
        )
    """
    import warnings
    def decorator(obj):
        msg = f"{obj.__name__} is obsolete"
        if replacement:
            msg += f", use {replacement} instead"

        if isinstance(obj, type):  # class
            class Wrapped(obj):
                def __init__(self, *args, **kwargs):
                    warnings.warn(msg, DeprecationWarning, stacklevel=2)
                    super().__init__(*args, **kwargs)

            return Wrapped
        else:                       # function
            def wrapped(*args, **kwargs):
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                return obj(*args, **kwargs)

            return wrapped

    return decorator


#region debug
def get_stats(x):
    import torch
    return {
        'max': f"{torch.max(x).item():.4f}",
        'min': f"{torch.min(x).item():.4e}",
        'avg': f"{torch.mean(x).item():.4f}",
        'std': f"{torch.std(x).item():.4f}",
    }
#endregion