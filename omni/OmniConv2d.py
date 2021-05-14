import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import os

from .OmniGridGenerator import OmniGridGenerator, MaskGenerator
from .utils import generateStrides

class OmniConv2d(nn.Conv2d):
  """
  kernel_size: (H, W)
  """

  def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3),
               stride=1, padding=0, dilation=1,
               groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
    super(OmniConv2d, self).__init__(
      in_channels, out_channels, kernel_size,
      stride, padding, dilation, groups, bias, padding_mode)

    self.convert_option = "mollweide"
    self.strides = None

    self.grid = None
    self.mask = None

    # self.reset_parameters()  # sphere_cnn의 경우 무조건 하게 됨!

  def reset_parameters(self):
    nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
    if self.bias is not None:
      self.bias.data.zero_()

  def genSamplingPattern(self, h, w):
    self.strides = generateStrides(h, w, self.convert_option)

    save_file = 'template/grid_{}_{}_{}_{}_{}_{}_{}.npy'.format(
      h, w, self.kernel_size[0], self.kernel_size[1], self.stride[0], self.stride[1], self.convert_option)
    if os.path.isfile(save_file):
      grid = np.load(save_file).copy()
    else:
      gridGenerator = OmniGridGenerator(
        h, w, self.strides, self.kernel_size, self.stride)
      LonLatSamplingPattern = gridGenerator.createSamplingPattern()

      # generate grid to use `F.grid_sample`
      lat_grid = (LonLatSamplingPattern[:, :, :, 0] / h) * 2 - 1
      lon_grid = (LonLatSamplingPattern[:, :, :, 1] / w) * 2 - 1

      grid = np.stack((lon_grid, lat_grid), axis=-1)

      np.save('template/grid_{}_{}_{}_{}_{}_{}_{}'.format(
        h, w, self.kernel_size[0], self.kernel_size[1], self.stride[0], self.stride[1], self.convert_option),
        grid)

    with torch.no_grad():
      self.grid = torch.FloatTensor(grid)
      self.grid.requires_grad = False

  def forward(self, x):
    # Generate Sampling Pattern
    B, C, H, W = x.shape

    if self.strides is None:
      self.genSamplingPattern(H, W)

    with torch.no_grad():
      grid = self.grid.repeat((B, 1, 1, 1)).to(x.device)  # (B, H*Kh, W*Kw, 2)
      grid.requires_grad = False

    x = F.grid_sample(x, grid, align_corners=True, mode='bilinear')  # (B, in_c, H*Kh, W*Kw)

    # self.weight -> (out_c, in_c, Kh, Kw)
    x = F.conv2d(x, self.weight, self.bias, stride=self.kernel_size)

    """
    Masking
    """
    B, C, out_H, out_W = x.shape

    if self.mask is None:
      print("generate mask!")
      mask = MaskGenerator(out_H, out_W, self.strides).createMaks()

      with torch.no_grad():
        self.mask = torch.FloatTensor(mask)
        self.mask.requires_grad = False

    with torch.no_grad():
      mask = self.mask.repeat((B, C, 1, 1)).to(x.device)
      mask.requires_grad = False

    x = mask * x

    return x  # (B, out_c, H/stride_h, W/stride_w)
