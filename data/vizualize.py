from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def imshow(img, title=None):
    # Remove the batch dimension
    out = np.squeeze(img, axis=0)
    # Normalize for display
    out = out.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)


def show_content_style(content, style):
    """Plots the content and style images

    :param content: Content image
    :type content: ndarray
    :param style: Style image
    :type style: ndarray
    """

    plt.figure(figsize(10, 10))

    plt.subplot(1, 2, 1)
    imshow(content, "Content Image")

    plt.subplot(1, 2, 2)
    imshow(style, "Style Image")

    plt.show()


def show_results(best_img, content, style):
    # content and style images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')

    # final image
    plt.figure(figsize=(10, 10))
    plt.imshow(best_img)
    plt.title('Output Image')
    plt.show()