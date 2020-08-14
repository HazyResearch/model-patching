import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # https://github.com/LynnHo/CycleGAN-Tensorflow-2/blob/master/module.py
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate *
                            (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate


def build_optimizer(optimizer, learning_rate_fn, momentum=0.9):
    if optimizer == 'Adam':
        return keras.optimizers.Adam(learning_rate=learning_rate_fn)
    elif optimizer == 'SGD':
        return keras.optimizers.SGD(learning_rate=learning_rate_fn,
                                    momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError


def build_lr_scheduler(scheduler,
                       steps_per_epoch,
                       n_epochs,
                       lr_start,
                       lr_decay_steps=None,
                       lr_end=None):
    decay_steps = lr_decay_steps if lr_decay_steps is not None else n_epochs * steps_per_epoch
    if scheduler == 'linear':
        lr_end = lr_end if lr_end is not None else 0.
        learning_rate_fn = keras.optimizers.schedules.PolynomialDecay(lr_start,
                                                                      decay_steps,
                                                                      lr_end,
                                                                      power=1.0)
    elif scheduler == 'linear_decay_0.5':
        # Used in CycleGAN
        learning_rate_fn = LinearDecay(lr_start, decay_steps, decay_steps//2)
    elif scheduler == 'constant':
        # assert lr_decay_steps is None and lr_end is None, 'No decay for constant learning rate.'
        learning_rate_fn = lr_start
    elif scheduler == 'cosine':
        learning_rate_fn = keras.experimental.CosineDecay(initial_learning_rate=lr_start,
                                                          decay_steps=decay_steps,
                                                          alpha=0.0)

    elif scheduler == 'piecewise_quarter':
        # Splits decay_steps into 4 equal phases, and reduces learning rate by 10. in each phase
        learning_rate_fn = keras.optimizers.schedules.\
            PiecewiseConstantDecay(boundaries=list(np.linspace(0, decay_steps, 5)[1:-1]),
                                   values=[lr_start / div for div in [1., 10., 100., 1000.]])

    elif scheduler == 'piecewise_custom_1':
        # Splits decay_steps into 3 phases (50%, 25%, 25%) and reduces learning rate by 10. in each phase
        learning_rate_fn = keras.optimizers.schedules.\
            PiecewiseConstantDecay(boundaries=[decay_steps//2, (decay_steps * 3)//4],
                                   values=[lr_start / div for div in [1., 10., 100.]])

    elif scheduler == 'piecewise_cifar10':
        # These are the numbers in the ResNet paper
        learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[30000, 60000, 90000, 120000],
                                                                             values=[0.1, 0.01, 0.001, 0.0001, 0.00005])
    else:
        raise NotImplementedError

    return learning_rate_fn
