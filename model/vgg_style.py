from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))

# Constants
# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer we are interested in
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def get_model():
    """Creates our model with access to intermediate layers

    :return: a keras model that takes image inputs and outputs
        the style and content intermediate layers
    :rtype: keras.Model
    """

    # Load the base model from keras model zoo
    vgg = tf.keras.applications.vgg19.VGG19(
        include_top=False,
        weights='imagenet',
    )
    vgg.trainable = False

    # Get our style and content layers
    style_out = [vgg.get_layer(name).output for name in style_layers]
    content_out = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_out + content_out

    # Build and return
    return tf.keras.models.Model(vgg.input, model_outputs)


def get_content_loss(base_content, target):
    """Content loss function

    :param base_content: Content tensors
    :param target: target image tensor
    :return: loss
    :rtype: loss value tensor
    """

    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
    """Gram matrix calculator for style loss

    The style loss is the distance between the gram matrices
    of the input image and the style image.

    :param input_tensor: filter representation
    :type input_tensor: tensor
    :return: gram matrix
    :rtype: tensor
    """

    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    """Style loss

    The distance between the gram matrices of the
    style image and target image

    :return: [description]
    :rtype: [type]
    """

    # We scale the loss at a given layer by the
    # size of the feature map and the number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))


def get_feature_representations(model, content_image, style_image):
    """Helper function to get feature maps

    :param model: base model with appropriate feature outputs
    :param content_path: content image
    :param style_path: style image
    :return: style and content features
    """

    # batch compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Get the style and content feature representations from our model
    style_features = [
        style_layer[0]
        for style_layer in style_outputs[:num_style_layers]
    ]
    content_features = [
        content_layer[0]
        for content_layer in content_outputs[num_style_layers:]
    ]
    return style_features, content_features


def compute_loss(model,
                 loss_weights,
                 init_image,
                 gram_style_features,
                 content_features):
    """Compute total loss

    :param model: model with appropriate intermediate layers
    :param loss_weights: weights for each loss func
    :param init_image: initial base image
    :param gram_syle_features: gram matrices for style features
    :param gram_content_features: gram matrices for content features
    :return: total loss
    """

    style_weight, content_weight = loss_weights

    # Feed our init image through our model.
    # This will give us the content and
    # style representations at our desired layers.
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features,
                                        style_output_features):
        style_score += weight_per_style_layer * get_style_loss(
            comb_style[0], target_style)

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features,
                                            content_output_features):
        content_score += weight_per_content_layer * get_content_loss(
            comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score


def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    # Compute gradients wrt input image
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss


def run_style_transfer(content_image,
                       style_image,
                       num_iterations=1000,
                       content_weight=1e3,
                       style_weight=1e-2):

    # Set initial image
    init_image = content_image
    init_image = tfe.Variable(init_image, dtype=tf.float32)
    
    # Get model
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    # Get the style and content feature representations
    style_features, content_features = get_feature_representations(
        model,
        content_image,
        style_image,
    )
    gram_style_features = [
        gram_matrix(style_feature) for style_feature in style_features
    ]

    # Create our optimizer
    opt = tf.train.AdamOptimizer(
        learning_rate=5, beta1=0.99, epsilon=1e-1)

    # Store our best result
    best_loss, best_img = float('inf'), None

    # Create a nice config
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    # For displaying
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations/(num_rows*num_cols)
    start_time = time.time()
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        end_time = time.time()

        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = init_image.numpy()

        print('Iter: {} of {}'.format(i, num_iterations))
        print('Total time: {:.4f}s'.format(time.time() - global_start))

    return best_img, best_loss
