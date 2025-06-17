import torch.nn as nn
from dataclasses import dataclass

@dataclass
class NamedModule():
    module : nn.Module
    name : str = ''

    def __iter__(self):
        return iter((self.module, self.name))

def module_leaves(module : nn.Module, depth : int = -1, prefix : str = '.') -> list[NamedModule]:
    assert depth >= -1, "Depth of leaves must be non-negative of -1 (maximal depth)"
    if depth == -1:
        return module_deep_leaves(module, prefix=prefix)
    if depth == 0:
        return [NamedModule(module=module)]

    ret = []
    for name, child in module.named_children():
        ret += [NamedModule(module, name + decide_prefix(prefix, grand_name) + grand_name) for module, grand_name in module_leaves(child, depth - 1)]
    return ret

def module_deep_leaves(module : nn.Module, prefix : str) -> list[NamedModule]:
    ret = []
    for name, child in module.named_children():
        ret += [NamedModule(module, name + decide_prefix(prefix, grand_name) + grand_name) for module, grand_name in module_deep_leaves(child, prefix=prefix)]
    if ret == []:
        return [NamedModule(module, '')]
    return ret

def decide_prefix(prefix : str, grand_name : str):
    return prefix if grand_name else ''