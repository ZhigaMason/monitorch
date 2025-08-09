"""
    An inspector for neural networks that implement PyTorch nn.Module
"""
from torch.nn import Module
from typing_extensions import Self

from monitorch.lens import AbstractLens
from monitorch.preprocessor import ExplicitCall
from monitorch.visualizer import _vizualizer_dict, AbstractVisualizer

class PyTorchInspector:

    def __init__(
            self,
            lenses : list[AbstractLens], *,
            visualizer : str|AbstractVisualizer = 'matplotlib',
            module : None|Module = None,
            depth : int = -1,
            module_name_prefix : str = '.',
            train_loss_str = 'train_loss',
            non_train_loss_str = 'val_loss'
    ):
        """ Initializes all lenses and hooks to a module if one is given """
        self._lenses = lenses
        self._call_preprocessor = ExplicitCall(train_loss_str, non_train_loss_str)

        self.epoch_counter = 0

        if isinstance(visualizer, str):
            if visualizer not in _vizualizer_dict:
                raise AttributeError(f"Unknown vizualizer, string defined vizualizer must be one of {list(_vizualizer_dict.keys())} ")
            self.visualizer = _vizualizer_dict[visualizer]()
        else:
            self.visualizer : AbstractVisualizer = visualizer

        for lens in self._lenses:
            lens.register_foreign_preprocessor(self._call_preprocessor)
            lens.introduce_tags(self.visualizer)
        if module is not None:
            self.attach(module, depth, module_name_prefix)

    def attach(self, module : Module, depth : int, module_name_prefix='.') -> Self:
        module_names = PyTorchInspector._module_leaves(module, depth, module_name_prefix)
        for module, name in module_names:
            for lens in self._lenses:
                lens.register_module(module, name)
        return self

    def detach(self) -> Self:
        for lens in self._lenses:
            lens.detach_from_module()
        return self

    def push_metric(self, name : str, value : float, *, running : bool=True):
        if running:
            self._call_preprocessor.push_running(name, value)
        else:
            self._call_preprocessor.push_memory(name, value)

    def push_loss(self, value : float, *, train : bool, running : bool = True):
        self._call_preprocessor.push_loss(value, train=train, running=running)

    def tick_epoch(self, epoch : int|None=None):
        if epoch is not None:
            self.epoch_counter = epoch
        for lens in self._lenses:
            lens.finalize_epoch()
            lens.vizualize(self.visualizer, self.epoch_counter)
            lens.reset_epoch()
        self._call_preprocessor.reset()
        self.epoch_counter += 1

    @staticmethod
    def _decide_prefix(prefix : str, grand_name : str):
        return prefix if grand_name else ''

    @staticmethod
    def _module_leaves(module : Module, depth : int = -1, prefix : str = '.') -> list[tuple[Module, str]]:
        assert depth >= -1, "Depth of leaves must be non-negative or -1 (maximal depth)"
        if depth == -1:
            return PyTorchInspector._module_deep_leaves(module, prefix=prefix)
        if depth == 0:
            return [(module, '')]

        ret = []
        for name, child in module.named_children():
            ret += [(module, name + PyTorchInspector._decide_prefix(prefix, grand_name) + grand_name) for module, grand_name in PyTorchInspector._module_leaves(child, depth - 1)]
        return ret

    @staticmethod
    def _module_deep_leaves(module : Module, prefix : str) -> list[tuple[Module, str]]:
        ret = []
        for name, child in module.named_children():
            ret += [(module, name + PyTorchInspector._decide_prefix(prefix, grand_name) + grand_name) for module, grand_name in PyTorchInspector._module_deep_leaves(child, prefix=prefix)]
        if ret == []:
            return [(module, '')]
        return ret
