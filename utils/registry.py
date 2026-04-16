

try:
    from utils.console.log import ColorPrefeb as CP
except:
    class CPType(type):
        def __getattr__(cls, item):
            return lambda x: x  # 返回恒等函数

    class CP(metaclass=CPType):
        pass


# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py  # noqa: E501
class Registry():
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        # print(f"Registering {CP.keyword(name)} to '{CP.keyword(self._name)}' registry.")
        # assert name not in self._obj_map, f"{CP.keyword(name)} was already registered in '{CP.keyword(self._name)}' registry!"
        if name in self._obj_map:
            print(f"Warning: {CP.keyword(name)} was already registered in '{CP.keyword(self._name)}' registry! It will be overwritten.")
        self._obj_map[name] = obj

    def register(self, obj=None):
        """
        Register the given object under the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise NotImplementedError(f'{self._name.capitalize()} {CP.keyword(name)} is not implemented yet. '
                                      f'Please choose one from {list(self._obj_map.keys())}.'
                                      f'Or decorate a function (or class) with @{CP.keyword(self._name.upper())}_REGISTRY.register().')
        return ret
    
    def try_get(self, name):
        ret = self._obj_map.get(name)
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


ARCH_REGISTRY = Registry('arch')
MODULE_REGISTRY = Registry('module')
MODEL_REGISTRY = Registry('model')
SCHEDULER_REGISTRY = Registry('scheduler')
LOSS_REGISTRY = Registry('loss')
DATASET_REGISTRY = Registry('dataset')
TRANSFORM_REGISTRY = Registry('transform')
SAMPLER_REGISTRY = Registry('sampler')
METRICS_REGISTRY = Registry('metrics')
