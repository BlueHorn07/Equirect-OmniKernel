import numpy as np


def generateStrides(height, width):
  """
  :return: array of the size of strides (H, )
  """
  delta_lon = 2 * np.pi / width
  nu = delta_lon  # arctan(rho)
  tan_nu = np.tan(nu)

  h_range = np.arange(0, height)  # [0, h]
  lat_range = ((h_range / height) - 0.5) * np.pi  # [-π/2, π/2]
  center = width // 2

  next_steps = np.arctan(tan_nu / np.cos(lat_range))
  next_steps = 0.5 + (next_steps / (2 * np.pi))
  # next_steps = np.round(next_steps * width)  # <- discretization
  next_steps = next_steps * width  # remove `np.round()`!
  strides = next_steps - center
  return strides  # (H, )
