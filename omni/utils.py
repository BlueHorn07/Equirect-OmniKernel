import numpy as np
import os

def generateStrides(height, width, convert_option="mollweide"):
  """
  :return: array of the size of strides (H, )
  """

  save_file = 'template/strides_{}_{}_{}.npy'.format(height, width, convert_option)
  if os.path.isfile(save_file):
    tmplate = np.load(save_file).copy()
    return tmplate

  delta_lon = 2 * np.pi / width
  nu = delta_lon  # arctan(rho)
  tan_nu = np.tan(nu)

  h_range = np.arange(0, height)  # [0, h]
  lat_range = ((h_range / height) - 0.5) * np.pi  # [-π/2, π/2]
  center = width // 2

  # arctan((next_step * sin_nu) / (rho * cos(lat) * cos_nu))
  next_steps = np.arctan(tan_nu / np.cos(lat_range))
  next_steps = 0.5 + (next_steps / (2 * np.pi))

  if convert_option == "mollweide":
    next_steps = next_steps * width  # no discretization!
  elif convert_option == "adaptive":
    next_steps = np.round(next_steps * width)  # discretization!

  strides = next_steps - center
  np.save('template/strides_{}_{}_{}'.format(height, width, convert_option), strides)
  return strides  # (H, )

