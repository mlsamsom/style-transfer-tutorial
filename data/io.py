from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import tensorflow as tf
import numpy as np


def load_img(p, max_dim=512):
    """Load image and convert to 4-D array

    :param p: path to image
    :type p: str
    :param max_dim: size of longest axis, defaults to 512
    :param max_dim: int, optional
    :return: image array
    :rtype: ndarray
    """

    img = Image.open(p)

    # scale image
    long_axis = max(img.size)
    scale = max_dim / long_axis
    img = img.resize(
        (int(round(img.size[0]*scale)), int(round(img.size[1]*scale))),
        Image.ANTIALIAS,
    )

    # convert to ndarray
    img = tf.keras.preprocessing.image.img_to_array(
        img,
    )
    img = np.expand_dims(img, axis=0)

    return img
