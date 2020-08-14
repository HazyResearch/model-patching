import tensorflow as tf
import numpy as np
import wandb
from types import SimpleNamespace


def set_global_seeds(seed):
    """
    Set all the random seeds.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)


def basic_setup(seed, logical_gpu_memory_limits=(4096, 10240)):
    """
    Function for setting up basic options.
    """
    # Set seeds
    set_global_seeds(seed)

    # Set print options
    np.set_printoptions(precision=2, suppress=True)

    # Set GPU growth in Tensorflow: disable for the moment
    # set_gpu_growth()

    # Create logical GPUs
    logical_gpus = create_logical_gpus(logical_gpu_memory_limits)

    # Figure out the devices we can put things on
    device_0 = tf.device(logical_gpus[0].name)
    device_1 = tf.device(logical_gpus[0].name) if len(logical_gpus) == 1 else tf.device(logical_gpus[1].name)
    devices = [device_0, device_1]

    return SimpleNamespace(logical_gpus=logical_gpus, devices=devices)


def create_logical_gpus(memory_limits=(4096, 10240)):
    """
    Create logical GPUs that split the physical GPU into separate devices.
    One use case is when we want to put models on separate logical GPUs to manage memory allocation on the GPU.
    """
    # Get a list of GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Create 2 virtual GPUs with 1GB memory each
        try:
            virtual_devices = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=lim)
                               for lim in memory_limits]
            tf.config.experimental.set_virtual_device_configuration(gpus[0], virtual_devices)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
            return logical_gpus
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def set_gpu_growth():
    """
    Set the GPU growth in Tensorflow so that GPU memory is not a bottleneck.
    """
    # Get a list of GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def checkpoint(model, path):
    model.save(filepath=f'{wandb.run.dir}/{path}_model.h5', include_optimizer=True)
