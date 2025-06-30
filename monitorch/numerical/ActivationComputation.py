from torch import Tensor


def reduce_activation_to_activation_rates(act_tensor : Tensor, batch : bool) -> Tensor:
    act_tensor = reduce_spatial_(act_tensor, batch)
    if batch:
        if len(act_tensor.shape) > 2:
            spatial_dim = act_tensor.shape[-1]
            batch_dim = act_tensor.shape[0]
            act_tensor = act_tensor.float().sum(dim=(0,-1)) / (spatial_dim * batch_dim)
        else:
            act_tensor = act_tensor.float().mean(dim=0)

    else:
        if len(act_tensor.shape) > 1:
            act_tensor = act_tensor.float().mean(dim=-1)
    return act_tensor

def reduce_spatial_(act_tensor : Tensor, batch : bool) -> Tensor:
    if batch:
        if len(act_tensor.shape) > 2:
            return act_tensor.flatten(2, -1)
    else:
        if len(act_tensor.shape) > 1:
            return act_tensor.flatten(1, -1)
    return act_tensor
