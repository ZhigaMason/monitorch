"""
    An inspector for neural networks that implement PyTorch nn.Module
"""
from dataclasses import dataclass

from torch.nn import Module as Module
from typing_extensions import Self
from monitorch.lens import AbstractLens
from monitorch.gatherer import FeedForwardGatherer, BackwardGatherer
from monitorch.preprocessor import AbstractForwardPreprocessor, AbstractBackwardPreprocessor, ExplicitCall
from monitorch.vizualizer import _vizualizer_dict

class PyTorchInspector:

    def __init__(self, lenses : list[AbstractLens], *, vizualization = 'matplotlib', module = None, level = -1):
        """ Initializes all lenses and hooks to a module if one is given """
        self._lenses = lenses
        self._vizualizer = _vizualizer_dict[vizualization]

        self._fw_preprocessors, self._bw_preprocessors = self._define_preprocessors()
        self._call_preprocessor = ExplicitCall()

        self._fw_handles = {}
        self._bw_handles = {}

        if module:
            self.attach(module, level)

    def attach(self, module, level = -1) -> Self:
        """ Hooks inspector to given module """
        named_children = PyTorchInspector._module_leaves(module, level)

        for child, name in named_children:
            fw = [ prp for prp in self._fw_preprocessors if prp.is_preprocessing(child) ]
            bw = [ prp for prp in self._bw_preprocessors if prp.is_preprocessing(child) ]
            fw_handle = child.register_forward_hook(
                FeedForwardGatherer(fw, name)
            )
            bw_handle = child.register_full_backward_hook(
                BackwardGatherer(bw, name)
            )
            self._fw_handles[name] = fw_handle
            self._bw_handles[name] = bw_handle

        return self

    def detach(self) -> Self:
        """ Detaches inspector from its module """
        for handle in self._fw_handles:
            handle.remove()
        for handle in self._bw_handles:
            handle.remove()
        return self

    def tick_epoch(self) -> None:
        """ Draws information obtained during the epoch and resets internal state """
        for lens in self._lenses:
            lens.vizualize(self._vizualizer)

        for preprocessor in self._fw_preprocessors:
            preprocessor.reset()
        for preprocessor in self._bw_preprocessors:
            preprocessor.reset()

    def push_loss(self, value, *, train : bool, loss_type : str = 'val') -> None:
        name = ('train' if train else loss_type) + '_loss'
        self._call_preprocessor.push_mean_var(name, float(value))

    def push_named_value(self, name : str, value) -> None:
        self._call_preprocessor.push_remember(name, value)

    def _define_preprocessors(self) -> tuple[list[AbstractForwardPreprocessor], list[AbstractBackwardPreprocessor]]:
        fw_classes = set()
        bw_classes = set()
        for lens in self._lenses:
            fw_classes.update(lens.requires_forward())
            bw_classes.update(lens.requires_backward())

        preprocessors = {
            cls : cls() for cls in (fw_classes | bw_classes)
        }

        for lens in self._lenses:
            lens.register_preprocessors(
                { cls : preprocessors[cls] for cls in (lens.requires_forward() | lens.requires_backward())}
            )

        fw_preprocessors = [ prp for prp in preprocessors.values() if isinstance(prp, AbstractForwardPreprocessor)]
        bw_preprocessors = [ prp for prp in preprocessors.values() if isinstance(prp, AbstractBackwardPreprocessor)]

        return (fw_preprocessors, bw_preprocessors)


    @staticmethod
    def _decide_prefix(prefix : str, grand_name : str):
        return prefix if grand_name else ''

    @staticmethod
    def _module_leaves(module : Module, depth : int = -1, prefix : str = '.') -> list[tuple[Module, str]]:
        assert depth >= -1, "Depth of leaves must be non-negative of -1 (maximal depth)"
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
