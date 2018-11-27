__author__ = 'max'

# from torch.autograd import Variable
import torch

def assign_tensor(tensor, val):
    """
    copy val to tensor
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        val: an n-dimensional torch.Tensor to fill the tensor with

    Returns:

    """
    # if isinstance(tensor, Variable):
    #     assign_tensor(tensor.data, val)
    #     return tensor
    with torch.no_grad():
        return tensor.copy_(val)
