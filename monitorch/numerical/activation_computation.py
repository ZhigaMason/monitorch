from torch import Tensor


def reduce_activation_to_activation_rates(act_tensor : Tensor, batch : bool) -> Tensor:
    """
    Reduces (boolean) activation tensor to activation rates per channel/neuron.

    Flattens spatial dimensions. For batch input computes mean over batch and spatial dimension,
    for non-batch input computes mean only over spatial dimensions.

    Parameters
    ----------
    act_tensor : torch.Tensor
        Tensor to compute activation rates from. Can have any dtype, as it is converted to float in the function if mean is computed.
    batch : bool
        Flag indicating whether `act_tensor` is a batch of data.

    Returns
    -------
    torch.Tensor
        If input tensor is one dimensional returns itself, otherwise computes mean accross spatial and batch dimensions,
        if the last is present. In case of multidimensional input tensor, output's dtype is float.
    """
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
    """
    Flattens spatial dimensions of tensor.

    Flattens rows and columns in image, or time in 1D signals.
    Assumes PyTorch convention (channel-first) and coalesce all dimensions after the channel.
    Channel dimension is thought to be the first if data is not batched and the second in batch.

    Parameters
    ----------
    act_tensor : torch.Tensor
        Activation tensor for which spatial dimensions will be reduced
    batch : bool
        Flag indicating whether data is in a batch

    Returns
    -------
    torch.Tensor
        Tensor with flattened spatial dimensions. If input activation tensor has no spatial dimensions, returns itself.
        For batch input tensor is at most 3D, for non-batch input tensor is at most 2D.
    """
    if batch:
        if len(act_tensor.shape) > 2:
            return act_tensor.flatten(2, -1)
    else:
        if len(act_tensor.shape) > 1:
            return act_tensor.flatten(1, -1)
    return act_tensor

def reduce_non_channels(tensor : Tensor, channel_dim : int) -> Tensor:
    """
    Flattens all dimension except for channel.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be reduced.
    channel_dim : int
        Number of dimension corresponding to channels.

    Returns
    -------
    3D Tensor, where second dimension is channels and has as many channels as input tensor had.
    """
    ret = tensor
    orig_shape = ret.shape
    if channel_dim > 0:
        ret = ret.flatten(0, channel_dim - 1)
    if channel_dim < (len(orig_shape) - 1):
        ret = ret.flatten(channel_dim - 1, -1)
    else:
        ret = ret.reshape(*ret.shape, 1)
    if channel_dim == 0:
        ret = ret.reshape(1, *ret.shape)
    return ret

