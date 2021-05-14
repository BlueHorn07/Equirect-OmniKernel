import numpy as np
import cv2
import torch

from omni import Equirect2Omni

"""
np.round()를 제거하면서 부수적으로 얻은 ERP2Mollweide의 demo 코드이다.
"""

if __name__ == '__main__':
  print("=== ERP2Mollweide ===")

  # open source image
  src_path = "./images/ERP-1.png"
  src_image = cv2.imread(src_path)

  src_image = src_image.transpose((2, 0, 1))
  src_image = np.expand_dims(src_image, 0)
  src_image = torch.from_numpy(src_image).float()

  out = Equirect2Omni().forward(src_image)
  out = np.squeeze(out.numpy(), 0).transpose([1, 2, 0])

  cv2.imwrite("./images/ERP2Mollweide.png", out)


"""
다만, 이렇게 되면 AdpativeStride에서 그나마 plain을 유지하던 부분까지 모두 일그러지게 되면서, 
더이상 kernel을 최적화시켜줄 부분이 없게 된다.
물론 아직 최적화를 안 했으니, memory 관점에서는 전혀 바뀐게 없을 것이다 ㅋㅋ

게다가 이로 인해 e2a 뿐만 아니라 adpativeConv/Pool 역시 이것에 맞춰 잘 동작하는지도 확인해봐야 한다.
"""

