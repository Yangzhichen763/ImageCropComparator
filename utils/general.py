import time
from contextlib import contextmanager

import warnings
from functools import wraps


try:
    from utils.console.log import ColorPrefeb as CP
except:
    class CPType(type):
        def __getattr__(cls, item):
            return lambda x: x  # 返回恒等函数

    class CP(metaclass=CPType):
        pass


#region ==[Retry]==
class Signal:
    def __init__(self):
        self.success = False


@contextmanager
def retry(max_retries=3, delay=1, backoff=2, task_name="Task", on_failure=None):
    """
    Retry a task with exponential backoff.

    e.g.
    ```python
    with retry(max_retries=3, task_name="Saving model") as signal:
        # save model code here
        signal.success = True
    ```

    Args:
        max_retries (int, optional): Maximum number of retries. Defaults to 3.
        delay (int, optional): Initial delay in seconds. Defaults to 1.
        backoff (int, optional): Backoff multiplier. Defaults to 2.
        task_name (str, optional): Name of the task being retried. Defaults to "Saving model".
        on_failure (function, optional): Function to call when all retries have been exhausted. Defaults to None.

    Yields:
        Signal: Signal object to indicate success or failure.
    """
    signal = Signal()
    retries = 0
    while not signal.success and retries < max_retries:
        try:
            yield signal
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                if on_failure is not None:
                    on_failure(e)
                else:
                    print(f"{CP.keyword(task_name)} failed. Giving up after {CP.keyword(max_retries)} attempts.")
                    raise e
            else:
                print(f"{CP.keyword(task_name)} failed. Retrying in {CP.keyword(delay)} seconds... (attempt {CP.keyword(retries)}/{CP.keyword(max_retries)})")
                signal.success = False
                time.sleep(delay)
                delay *= backoff
#endregion


def try_parse(value, to_type=float, default=None):
    try:
        return to_type(value)
    except (ValueError, TypeError):
        return default


@contextmanager
def try_or_ignore():
    try:
        yield
    except:
        pass


class NoneDict(dict):
    def __missing__(self, key):
        return None


class DefaultDict(dict):
    def __init__(self, default_factory=None, **kwargs):
        super().__init__(**kwargs)
        self.default_factory = default_factory

    def __missing__(self, key):
        return self.default_factory()


def try_fill_default_dict(dict_obj, **kwargs):
    if not isinstance(dict_obj, dict):
        return dict_obj

    for key, value in kwargs.items():
        if key not in dict_obj:
            dict_obj[key] = value
    return dict_obj


def obsolete(cls):
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        warnings.warn(
            f"{CP.keyword(cls.__name__)} is obsolete, will be removed in future versions.",
            DeprecationWarning,
            stacklevel=2
        )
        return original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    cls.__is_obsolete__ = True  # 添加标记属性

    return cls


def future(cls):
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        warnings.warn(
            f"{CP.keyword(cls.__name__)} is an upcoming feature and will be available in future versions.",
            FutureWarning,
            stacklevel=2
        )
        return original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    cls.__is_future__ = True  # 标记属性

    return cls


import inspect

def get_original_callable(func):
    if hasattr(func, '__self__') and hasattr(func, '__func__'):
        instance = func.__self__
        unwrapped_func = inspect.unwrap(func.__func__)

        def wrapper(*args, **kwargs):
            return unwrapped_func(instance, *args, **kwargs)

        return wrapper

    return inspect.unwrap(func)