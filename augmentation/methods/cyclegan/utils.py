import datetime

import tensorflow as tf
import random
import wandb
from tensorflow_examples.models.pix2pix import pix2pix

from augmentation.dataflows.utils import create_paired_direct_dataflow, \
    create_paired_parallel_dataflow_via_numpy
from augmentation.methods.cyclegan.models import mnist_unet_generator, mnist_discriminator, unet_generator
from augmentation.utilities.optim import build_lr_scheduler
from augmentation.utilities.visualize import gallery


# Other places to look for training GANs
# https://github.com/eriklindernoren/Keras-GAN

def gradient_penalty(f, real, fake, mode, scale=10.0):
    # https://github.com/LynnHo/CycleGAN-Tensorflow-2/blob/master/tf2gan/loss.py
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is None:  # interpolation in DRAGAN
                beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
                b = a + 0.5 * tf.math.reduce_std(a) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = tf.reduce_mean(tf.reshape(f(x), [tf.shape(real)[0], -1]), axis=1)
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.) ** 2)

        return gp

    if mode == 'none':
        gp = tf.constant(0, dtype=real.dtype)
    elif mode == 'dragan':
        gp = _gradient_penalty(f, real)
    elif mode == 'wgan-gp':
        gp = _gradient_penalty(f, real, fake)
    else:
        raise NotImplementedError

    return gp * scale


class ReplayBuffer(object):
    """
    Adapted from https://github.com/tensorflow/models/blob/master/research/pcl_rl/replay_buffer.py
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self.cur_size = 0
        self.buffer = {}
        self.oldest_idx = 0
        self.init_length = 0

    def __len__(self):
        return self.cur_size

    def add(self, images):
        idx = 0
        while self.cur_size < self.max_size and idx < len(images):
            self.buffer[self.cur_size] = images[idx]
            self.cur_size += 1
            idx += 1

        if idx < len(images):
            remove_idxs = self.remove_n(len(images) - idx)
            for remove_idx in remove_idxs:
                self.buffer[remove_idx] = images[idx]
                idx += 1

        assert len(self.buffer) == self.cur_size

    def remove_n(self, n):
        return random.sample(range(self.init_length, self.cur_size), n)

    def get_batch(self, n):
        idxs = random.sample(range(self.cur_size), n)
        return [self.buffer[idx] for idx in idxs]

    def get_tf_batch(self, n):
        idxs = random.sample(range(self.cur_size), n)
        return tf.convert_to_tensor([self.buffer[idx] for idx in idxs])


def wgan_loss(targets, predictions):
    return tf.reduce_mean((-2 * targets + 1.) * predictions)


def build_gan_loss_fn(loss_name):
    if loss_name == 'bce':
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)
    elif loss_name == 'lsgan':
        return tf.keras.losses.MeanSquaredError()
    elif loss_name == 'wgan':
        return wgan_loss
    else:
        raise NotImplementedError


def discriminator_loss(real, generated, loss_fn):
    # Classification loss for the discriminator, maximize log-prob of the real example
    real_loss = loss_fn(tf.ones_like(real), real)
    generated_loss = loss_fn(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


def generator_loss(generated, loss_fn):
    # The discriminator's probability (generated) for realness is maximized
    return loss_fn(tf.ones_like(generated), generated)


def cycle_loss(real_image, cycled_image, scale):
    # Cycle-consistency using an L! loss
    return scale * tf.reduce_mean(tf.abs(real_image - cycled_image))


def identity_loss(real_image, same_image, scale):
    # Map the image to itself and compute the L1 loss
    return scale * 0.5 * tf.reduce_mean(tf.abs(real_image - same_image))


def build_cyclegan_models(n_channels, norm_type):
    assert norm_type in ['instancenorm', 'batchnorm']
    generator_g = pix2pix.unet_generator(n_channels, norm_type=norm_type)
    generator_f = pix2pix.unet_generator(n_channels, norm_type=norm_type)

    discriminator_x = pix2pix.discriminator(norm_type=norm_type, target=False)
    discriminator_y = pix2pix.discriminator(norm_type=norm_type, target=False)

    return generator_g, generator_f, discriminator_x, discriminator_y


def build_mnist_cyclegan_models(norm_type):
    assert norm_type in ['instancenorm', 'batchnorm']
    generator_g = mnist_unet_generator(norm_type=norm_type)
    generator_f = mnist_unet_generator(norm_type=norm_type)

    discriminator_x = mnist_discriminator(norm_type=norm_type, target=False)
    discriminator_y = mnist_discriminator(norm_type=norm_type, target=False)

    return generator_g, generator_f, discriminator_x, discriminator_y


def get_models_from_input_shape(input_shape, norm_type, output_init=0.02, residual_output=False):
    if input_shape == (28, 28, 1):
        # MNIST-like data
        return mnist_unet_generator(norm_type=norm_type), \
               mnist_discriminator(norm_type=norm_type, target=False)
    elif input_shape == (256, 256, 3):
        # TODO: just use our unet_generator fn
        if residual_output is True or output_init != 0.02:
            raise NotImplementedError
        return pix2pix.unet_generator(output_channels=3, norm_type=norm_type), \
               pix2pix.discriminator(norm_type=norm_type, target=False)
    else:
        return unet_generator(output_channels=3, input_shape=input_shape, norm_type=norm_type,
                              output_init=output_init, residual_output=residual_output), \
               pix2pix.discriminator(norm_type=norm_type, target=False)


def build_models(source_input_shape, target_input_shape, norm_type, output_init=0.02, residual_output=False):
    assert norm_type in ['instancenorm', 'batchnorm']
    generator_s_to_t, discriminator_s = get_models_from_input_shape(source_input_shape, norm_type, output_init, residual_output)
    generator_t_to_s, discriminator_t = get_models_from_input_shape(target_input_shape, norm_type, output_init, residual_output)

    return generator_s_to_t, generator_t_to_s, discriminator_s, discriminator_t


def build_optimizers(lr_gen=2e-4, lr_disc=2e-4,
                     beta_1_gen=0.5, beta_1_disc=0.5,
                     lr_scheduler='constant', lr_decay_steps=None):
    generator_g_optimizer = tf.keras.optimizers.Adam(build_lr_scheduler(lr_scheduler, 0, 0, lr_gen,
                                                                        lr_decay_steps=lr_decay_steps),
                                                     beta_1=beta_1_gen)
    generator_f_optimizer = tf.keras.optimizers.Adam(build_lr_scheduler(lr_scheduler, 0, 0, lr_gen,
                                                                        lr_decay_steps=lr_decay_steps),
                                                     beta_1=beta_1_gen)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(build_lr_scheduler(lr_scheduler, 0, 0, lr_disc,
                                                                            lr_decay_steps=lr_decay_steps),
                                                         beta_1=beta_1_disc)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(build_lr_scheduler(lr_scheduler, 0, 0, lr_disc,
                                                                            lr_decay_steps=lr_decay_steps),
                                                         beta_1=beta_1_disc)

    return generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer, discriminator_y_optimizer


def create_cyclegan_data_generator(source_dataset, target_dataset, batch_size, augmentations,
                                   dataflow, cache_dir):
    if dataflow == 'disk_cached':
        cache_dir = cache_dir + datetime.datetime.now().strftime('%d_%m_%y__%H_%M_%S')
        # Shuffle hangs sometimes (e.g. for horse2zebra)
        return create_paired_direct_dataflow(source_dataset, target_dataset, batch_size,
                                             augmentations, x_only=True,
                                             cache_dir1=cache_dir + '1',
                                             cache_dir2=cache_dir + '2',
                                             shuffle=True)
    elif dataflow == 'in_memory':
        return create_paired_parallel_dataflow_via_numpy(source_dataset, target_dataset,
                                                         batch_size, augmentations, x_only=True)
    else:
        raise NotImplementedError


def generate_and_log_one_image_batch(data_generator,
                                     generator_g,
                                     generator_f,
                                     step):
    # Grab a batch from the dataset
    for real_x, real_y in data_generator:
        # Convert to tensors
        real_x, real_y = tf.convert_to_tensor(real_x), tf.convert_to_tensor(real_y)

        # Compute the fake examples
        fake_y = generator_g(real_x, training=True)
        fake_x = generator_f(real_y, training=True)

        # Cycle the fake examples
        cycled_x = generator_f(fake_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # Compute the identity examples
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        # Log everything to Weights and Biases
        wandb.log({'test/real_x': wandb.Image(gallery(real_x.numpy() * 0.5 + 0.5)),
                   'test/fake_x': wandb.Image(gallery(fake_x.numpy() * 0.5 + 0.5)),
                   'test/cycled_x': wandb.Image(gallery(cycled_x.numpy() * 0.5 + 0.5)),
                   'test/same_x': wandb.Image(gallery(same_x.numpy() * 0.5 + 0.5)),
                   'test/real_y': wandb.Image(gallery(real_y.numpy() * 0.5 + 0.5)),
                   'test/fake_y': wandb.Image(gallery(fake_y.numpy() * 0.5 + 0.5)),
                   'test/cycled_y': wandb.Image(gallery(cycled_y.numpy() * 0.5 + 0.5)),
                   'test/same_y': wandb.Image(gallery(same_y.numpy() * 0.5 + 0.5))}, step=step)

        # Break after a single batch: note, this will not run if you remove the break due to wandb reasons (ask Karan)
        break


if __name__ == '__main__':
    buffer = ReplayBuffer(1)
    buffer.add([1])
    buffer.add([2])
    buffer.add([3])
    print(buffer.get_batch(1))
    print(buffer.get_batch(1))
    print(buffer.get_batch(1))
    buffer.add([4])
    print(buffer.get_batch(1))
    print(buffer.buffer)

    buffer = ReplayBuffer(1)
    buffer.add(tf.convert_to_tensor([1]))
    buffer.add(tf.convert_to_tensor([2]))
    buffer.add(tf.convert_to_tensor([3]))
    print(tf.convert_to_tensor(buffer.get_batch(1)))
    print(buffer.get_batch(1))
    print(buffer.get_batch(1))
    buffer.add(tf.convert_to_tensor([4]))
    print(buffer.get_batch(1))
    print(buffer.buffer)
