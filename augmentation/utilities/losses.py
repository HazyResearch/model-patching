import tensorflow as tf
import tensorflow.keras as keras


def decay_weights(model, weight_decay_rate):
    """Calculates the loss for l2 weight decay and returns it."""

    # @tf.function
    def _decay_weights(weights, weight_decay_rate):
        reg_loss = 0.
        for var in weights:
            reg_loss = reg_loss + tf.nn.l2_loss(var)
        reg_loss = weight_decay_rate * reg_loss
        return reg_loss

    return _decay_weights(model.trainable_weights, weight_decay_rate)


def create_loss_fn(loss_name):
    if loss_name == 'sparse_categorical_crossentropy':
        return keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    elif loss_name == 'binary_crossentropy':
        return keras.losses.BinaryCrossentropy()
    else:
        raise NotImplementedError
