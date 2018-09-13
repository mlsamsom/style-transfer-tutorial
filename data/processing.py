
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def processVGG19(img):
    """Wraps the vgg19 preprocessor

    :param img: Image array
    :type img: ndarray
    :return: Processed image array
    :rtype: ndarray
    """

    proc = tf.keras.applications.vgg19.preprocess_input(img)
    # return tf.convert_to_tensor(proc, tf.float32)
    return proc


def deprocessVGG19(img):
    """Deprocess image for visualization

    :param img: Image array
    :type img: ndarray
    :raises ValueError: image needs to have 3 axes
    :return: deprocessed image
    :rtype: ndarray
    """

    x = img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)

    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x
