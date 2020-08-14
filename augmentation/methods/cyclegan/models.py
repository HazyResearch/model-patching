import tensorflow as tf
import numpy as np
from tensorflow_examples.models.pix2pix.pix2pix import upsample, downsample, InstanceNormalization


def unet_generator(output_channels, input_shape=(256, 256, 3), norm_type='batchnorm', output_init=0.02,
                   residual_output=False):
    """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

    Args:
      output_channels: Output channels
      norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

    Returns:
      Generator model
    """
    assert input_shape[0] <= 256 and input_shape[1] <= 256, 'Input shape must be less than (256, 256, 3).'
    assert input_shape[0] == input_shape[1], 'Modify padding to handle this.'

    ceil_pow2 = int(2 ** np.ceil(np.log2(input_shape[0])))

    if ceil_pow2 == 256:
        down_stack = [
            downsample(64, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
            downsample(128, 4, norm_type),  # (bs, 64, 64, 128)
            downsample(256, 4, norm_type),  # (bs, 32, 32, 256)
            downsample(512, 4, norm_type),  # (bs, 16, 16, 512)
            downsample(512, 4, norm_type),  # (bs, 8, 8, 512)
            downsample(512, 4, norm_type),  # (bs, 4, 4, 512)
            downsample(512, 4, norm_type),  # (bs, 2, 2, 512)
            downsample(512, 4, norm_type),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
            upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
            upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
            upsample(512, 4, norm_type),  # (bs, 16, 16, 1024)
            upsample(256, 4, norm_type),  # (bs, 32, 32, 512)
            upsample(128, 4, norm_type),  # (bs, 64, 64, 256)
            upsample(64, 4, norm_type),  # (bs, 128, 128, 128)
        ]
    elif ceil_pow2 == 128:
        down_stack = [
            downsample(64, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
            downsample(128, 4, norm_type),  # (bs, 64, 64, 128)
            downsample(256, 4, norm_type),  # (bs, 32, 32, 256)
            downsample(512, 4, norm_type),  # (bs, 16, 16, 512)
            downsample(512, 4, norm_type),  # (bs, 8, 8, 512)
            downsample(512, 4, norm_type),  # (bs, 4, 4, 512)
            downsample(512, 4, norm_type),  # (bs, 2, 2, 512)
        ]

        up_stack = [
            upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
            upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
            upsample(512, 4, norm_type),  # (bs, 16, 16, 1024)
            upsample(256, 4, norm_type),  # (bs, 32, 32, 512)
            upsample(128, 4, norm_type),  # (bs, 64, 64, 256)
            upsample(64, 4, norm_type),  # (bs, 128, 128, 128)
        ]
    else:
        raise NotImplementedError

    initializer = tf.random_normal_initializer(0., output_init)
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4, strides=2,
        padding='same', kernel_initializer=initializer,
        activation='tanh')  # (bs, 256, 256, 3)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    padding = int((ceil_pow2 - input_shape[0]) // 2)
    if padding > 0:
        x = tf.keras.layers.ZeroPadding2D(padding=padding)(x)

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)
    if padding > 0:
        x = tf.keras.layers.Cropping2D(cropping=padding)(x)

    outputs = x
    if residual_output:
        # inputs is in [-1, 1], so offset should be in [-2, 2] to flip a pixel completely
        outputs = 2 * x + inputs

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def mnist_discriminator(norm_type='batchnorm', target=True):
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).

    Args:
      norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
      target: Bool, indicating whether target image is an input or not.

    Returns:
      Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[28, 28, 1], name='input_image')
    x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[28, 28, 1], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer,
        use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)


def mnist_unet_generator(norm_type='batchnorm'):
    """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

    Args:
    output_channels: Output channels
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

    Returns:
    Generator model
    """

    down_stack = [
        downsample(32, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
        downsample(64, 4, norm_type),  # (bs, 64, 64, 128)
        downsample(128, 4, norm_type),  # (bs, 32, 32, 256)
        downsample(256, 4, norm_type),  # (bs, 16, 16, 512)
        downsample(512, 4, norm_type),  # (bs, 16, 16, 512)
    ]

    up_stack = [
        upsample(256, 4, norm_type),  # (bs, 16, 16, 1024)
        upsample(128, 4, norm_type),  # (bs, 16, 16, 1024)
        upsample(64, 4, norm_type),  # (bs, 32, 32, 512)
        upsample(32, 4, norm_type),  # (bs, 64, 64, 256)
        upsample(16, 4, norm_type),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        1, 4, strides=2,
        padding='same', kernel_initializer=initializer,
        activation='tanh')  # (bs, 256, 256, 3)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[28, 28, 1])
    x = inputs
    x = tf.keras.layers.ZeroPadding2D(padding=2)(x)

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)
    x = tf.keras.layers.Cropping2D(cropping=2)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
