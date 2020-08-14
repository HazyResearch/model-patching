from typing import List
import tensorflow as tf
import tensorflow.keras as keras
from augmentation.utilities.metrics import reset_metrics, update_metrics


def evaluate_model(model: keras.Model,
                   generator,
                   metrics: List[keras.metrics.Metric],
                   aggregate=None,
                   dtype=tf.float32) -> List[keras.metrics.Metric]:
    """
    Evaluate a model on a dataset by measuring performance on some Keras metrics.
    :param model: A model of type keras.Model.
    :param generator: A data generator that can be iterated through.
    :param metrics: A list of keras.metrics.Metric objects.
    :param aggregate: A list of keras.metrics.Metric objects representing aggregate metrics
    if this method is called multiple times.
    :return: Performance on metrics.
    """

    # Reset the metrics
    reset_metrics(metrics)
    # Loop over the data
    for batch, targets in generator:
        # Convert to tensors
        batch, targets = tf.convert_to_tensor(batch, dtype=dtype), tf.convert_to_tensor(targets, dtype=dtype)
        # Make predictions
        predictions = model(batch, training=False)
        # Update the metrics
        update_metrics(metrics, targets, predictions)
        # Update the aggregate metrics if any
        if aggregate is not None:
            update_metrics(aggregate, targets, predictions)

    return metrics
