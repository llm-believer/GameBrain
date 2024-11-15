import numpy as np
import skimage


def compress(original, output_shape):
  return (255 * skimage.transform.resize(original, output_shape)).astype(np.uint8)
