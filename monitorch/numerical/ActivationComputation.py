from torch import Tensor


def reduce_activation_to_activation_rates(act_tensor : Tensor, batch : bool) -> Tensor:
    act_tensor = reduce_spatial_(act_tensor, batch)
    print('after reduce', act_tensor)
    if batch:
        if len(act_tensor.shape) > 2:
            print('aaa')
            spatial_dim = act_tensor.shape[-1]
            batch_dim = act_tensor.shape[0]
            act_tensor = act_tensor.float().sum(dim=(0,-1)) / (spatial_dim * batch_dim)
        else:
            act_tensor = act_tensor.float().mean(dim=0)

    else:
        act_tensor = act_tensor.float().mean(dim=-1)
    print('after aggregation', act_tensor)
    return act_tensor

def reduce_spatial_(act_tensor : Tensor, batch : bool) -> Tensor:
    if batch:
        if len(act_tensor.shape) > 2:
            return act_tensor.flatten(2, -1)
    else:
        if len(act_tensor.shape) > 1:
            return act_tensor.flatten(1, -1)
    return act_tensor
