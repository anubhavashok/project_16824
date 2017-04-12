import torch
from torch.autograd import Variable
from sklearn.feature_extraction import image
import numpy as np

def breakIntoPatches(input):
    inp = input.numpy()
    h, w = inp.shape[2:4]
    tl = inp[:, :, 0:h/2, 0:w/2]
    tr = inp[:, :, 0:h/2, w/2:w]
    bl = inp[:, :, h/2:h, 0:w/2]
    br = inp[:, :, h/2:h, w/2:w]
    # stack these together and convert back into pytorch tensor
    out = np.concatenate([tl, tr, bl, br], 1)
    return torch.from_numpy(out)

def reassemblePatches(input):
    inp = input.numpy()
    tl = inp[:, 0:6, :, :]
    tr = inp[:, 6:12, :, :]
    bl = inp[:, 12:18, :, :]
    br = inp[:, 18:24, :, :]
    out = np.zeros([inp.shape[0], inp.shape[1]/4, inp.shape[2]*2, inp.shape[3]*2], )
    h, w = out.shape[2:4]
    out[:, :, 0:h/2, 0:w/2] = tl
    out[:, :, 0:h/2, w/2:w] = tr
    out[:, :, h/2:h, 0:w/2] = bl
    out[:, :, h/2:h, w/2:w] = br
    return torch.from_numpy(out)


class SpatialPatching(torch.autograd.Function):
  """
    If kernel width or height is -1, we just break the feature map into n parts
    If stride is -1, but kW, kH is not -1, we randomize the patches chosen such that n patches are picked
  """
  def __init__(self, n=4, kW=-1, kH=-1, sW=-1, sH=-1):
    super(SpatialPatching, self).__init__()
    self.n=n
    self.kW=kW
    self.kH=kH
    self.sW=sW
    self.sH=sH
    
  """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """
  def forward(self, input):
    """
    In the forward pass we receive a Tensor containing the input and return a
    Tensor containing the output. You can cache arbitrary Tensors for use in the
    backward pass using the save_for_backward method.
    """
    output = breakIntoPatches(input)
    assert(output.size()[0] == input.size()[0])
    assert(output.size()[1] == input.size()[1]* 4)
    assert(output.size()[2] == input.size()[2]/2)
    assert(output.size()[3] == input.size()[3]/2)
    return output

  def backward(self, grad_output):
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    # rearrange gradients for all 4 patches
    return reassemblePatches(grad_output).float()
