from torch import is_grad_enabled


def make_train_switch(using_grad):
    return _is_grad_training if using_grad else _is_module_training


def _is_grad_training(m):
    return is_grad_enabled()


def _is_module_training(m):
    return m.training
