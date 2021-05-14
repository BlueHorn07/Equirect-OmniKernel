import numpy as np


class MaskGenerator:
  def __init__(self, h, w, strides):
    self.height = h
    self.width = w
    self.center = self.width // 2

    self.counts = (self.width // strides) // 2  # (H, )
    distance = np.arange(0, self.width) - self.center
    self.ratios = [distance / cnt for cnt in self.counts]  # (H, W)

  def createMaks(self):
    """
    Mask indicates the sampling region!
    """
    mask = np.zeros((self.height, self.width))  # (H, W)
    center = self.width // 2

    for lat in range(self.height):
      count = int(self.counts[lat])
      # print(lat, count)
      # print(center - count, center, center + count)
      mask[lat][center: center + count] = 1
      mask[lat][center - count: center] = 1

    return mask  # (H, W)


class OmniGridGenerator:
  def __init__(self, h, w, strides, kernel_size=3, stride=1):
    self.height = h
    self.width = w
    self.center = self.width // 2

    self.kernel_size = kernel_size
    if isinstance(kernel_size, int):
      self.kernel_size = (kernel_size, kernel_size)
    self.stride = stride
    if isinstance(stride, int):
      self.stride = (stride, stride)

    self.counts = (self.width // strides) // 2  # (H, )
    distance = np.arange(0, self.width) - self.center
    self.ratios = [distance / cnt for cnt in self.counts]  # (H, W)

    Kh, Kw = self.kernel_size
    self.range_x = np.arange(-(Kw // 2), Kw // 2 + 1)
    if not Kw % 2:
      self.range_x = np.delete(self.range_x, Kw // 2)

    self.range_y = np.arange(-(Kh // 2), Kh // 2 + 1)
    if not Kh % 2:
      self.range_y = np.delete(self.range_y, Kh // 2)

  def createSamplingPattern(self):
    Kh, Kw = self.kernel_size
    sh, sw = self.stride
    grid = np.zeros((self.height // sh, self.width // sw, Kh, Kw, 2))  # (H, W, Kh, Kw, 2)

    for i in range(self.height // sh):
      for j in range(self.width // sw):
        grid[i, j] = self.createKernel((i * sh, j * sw))

    grid = grid.transpose((0, 2, 1, 3, 4))  # (H, Kh, W, Kw, (lat, lon))

    H, Kh, W, Kw, d = grid.shape
    grid = grid.reshape((1, H * Kh, W * Kw, d))  # (1, H*Kh, W*Kw, 2)

    return grid  # (1, H*Kh, W*Kw, 2)

  def createKernel(self, LatLon):
    """
    :param LatLon: location (lat, lon) to create kernel
    :return: (Kh, Kw, 2) kernel
    """
    lat, lon = LatLon
    Kh, Kw = self.kernel_size
    center = self.width // 2

    range_x = self.range_x
    range_y = self.range_y

    range_lat = (lat + range_y)
    range_lat = np.where(range_lat < 0, 0, range_lat)  # refine values
    range_lat = np.where(self.height <= range_lat, self.height - 1, range_lat)  # refine values

    counts = self.counts[range_lat]  # (Kh, )
    ratio = self.ratios[lat][lon]  # scalar

    if (ratio < -1.0) or (ratio >= 1.0):  # given position is over the omni area
      return np.full((Kh, Kw, 2), self.width)  # (H, W, Kh, Kw, 2)

    lons = center + np.round(ratio * counts)  # (Kw, )

    kernel_lon = []
    for i in range(Kh):
      tmp = lons[i] + range_x
      diff1 = (center - tmp) - counts[i]
      diff2 = (tmp - center) - counts[i]
      tmp = np.where((center - tmp) > counts[i], center + counts[i] - diff1, tmp)
      tmp = np.where((tmp - center) >= counts[i], center - counts[i] + diff2, tmp)
      kernel_lon.append(tmp)

    # kernel_lon = np.array([lons[i] + range_x for i in range(Kh)])
    kernel_lon = np.array(kernel_lon)
    kernel_lat = np.array([np.full(Kw, lat + idx) for idx in range_y])

    LatLon = np.stack((kernel_lat, kernel_lon))  # (2, Kh, Kw)
    LatLon = LatLon.transpose((1, 2, 0))  # (Kh, Kw, 2)
    return LatLon
