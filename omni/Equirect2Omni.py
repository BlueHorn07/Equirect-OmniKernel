import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .utils import generateStrides


class Equirect2Omni(nn.Module):
  def __init__(self):
    super(Equirect2Omni, self).__init__()
    self.grid_shape = None
    self.grid = None

    self.mask_shape = None
    self.mask = None

  def generateGrid(self, h, w):
    """
    generate sampling grid
    :return: grid (H, W, 2) = (H, W, (lat, lon)
    """
    # generate stride patterns
    strides = generateStrides(h, w)

    # generate sampling grid
    grid = np.full((h, w, 2), w)  # (H, W, 2) = (H, W, (lat, lon)
    center = w // 2

    for i in range(h):
      grid[i, center] = [i, center]
      idx = 1
      while True:
        position = center + strides[i] * idx
        if w <= position or w <= center + idx:
          break
        grid[i, center + idx] = [i, position]
        idx += 1

      idx = 1
      while True:
        position = center - strides[i] * idx
        if position < 0 or center - idx < 0:
          break
        grid[i, center - idx] = [i, position]
        idx += 1
    grid = np.expand_dims(grid, 0)
    return grid  # (1, H, W, 2)

  def genSamplingPattern(self, h, w):
    LonLatSamplingPattern = self.generateGrid(h, w)

    # generate grid to use `F.grid_sample`
    lat_grid = (LonLatSamplingPattern[:, :, :, 0] / h) * 2 - 1
    lon_grid = (LonLatSamplingPattern[:, :, :, 1] / w) * 2 - 1

    grid = np.stack((lon_grid, lat_grid), axis=-1)

    with torch.no_grad():
      self.grid = torch.FloatTensor(grid)
      self.grid.requires_grad = False

  def forward(self, img):
    B, C, H, W = img.shape

    if (self.grid_shape is None) or (self.grid_shape != (H, W)):
      self.grid_shape = (H, W)
      self.genSamplingPattern(H, W)

    with torch.no_grad():
      grid = self.grid.repeat((B, 1, 1, 1)).to(img.device)  # (B, H*Kh, W*Kw, 2)
      grid.requires_grad = False

    img = F.grid_sample(
      img, grid,
      align_corners=True, mode='bilinear', padding_mode='zeros'
    )  # (B, in_c, H*Kh, W*Kw)
    img.requires_grad = False

    # B, C, out_H, out_W = img.shape
    #
    # if (self.mask_shape is None) or (self.mask_shape != (out_H, out_W)):
    #   mask = MaskGenerator(out_H, out_W).createMaks()
    #   self.mask_shape = (out_H, out_W)
    #
    #   with torch.no_grad():
    #     self.mask = torch.FloatTensor(mask)
    #     self.mask.requires_grad = False
    #
    # with torch.no_grad():
    #   mask = self.mask.repeat((B, C, 1, 1)).to(img.device)
    #   mask.requires_grad = False
    #
    # img = mask * img
    # img.requires_grad = False

    return img


if __name__ == '__main__':
  # generate demo image
  h, w = (200, 400)
  image = np.ones((h, w, 3))
  for r in range(h):
    for c in range(w):
      image[r, c, 0] = image[r, c, 0] - r / h
      image[r, c, 1] = image[r, c, 1] - c / w

  image = image.transpose((2, 0, 1))
  image = np.expand_dims(image, 0)
  image = torch.from_numpy(image).float()

  out = Equirect2Omni().forward(image)
  out = np.squeeze(out.numpy(), 0).transpose([1, 2, 0])

  plt.imsave("equirect2mollweide.png", out)
