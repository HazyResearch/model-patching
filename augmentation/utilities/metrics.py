import tensorflow as tf
import tensorflow.keras as keras
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.python.keras import backend as K
from augmentation.methods.robust.utils import irm_penalty_explicit


class ConfusionMatrix(keras.metrics.Metric):

    def __init__(self, n_classes, name='confusion_matrix', **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.confusion_matrix = self.add_weight(name='cm', initializer='zeros',
                                                shape=(n_classes, n_classes), dtype=tf.int32)
        self.n_classes = n_classes

    def reset_states(self):
        K.batch_set_value([(self.variables[0], tf.zeros((self.n_classes, self.n_classes)))])

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.confusion_matrix.assign_add(tf.math.confusion_matrix(labels=y_true,
                                                                  predictions=y_pred,
                                                                  weights=sample_weight,
                                                                  num_classes=self.n_classes))

    def result(self):
        return self.confusion_matrix

    def log_wandb(self, step, prefix='metrics/'):
        cm = self.result().numpy()
        wandb.run.summary[f'{prefix}confusion_matrix'] = cm
        wandb.log({f'{prefix}confusion_matrix': [wandb.Image(sns.heatmap(cm))]}, step=step, commit=False)
        plt.clf()


class MultiLabelAUC(keras.metrics.Metric):

    def __init__(self, n_outputs, output_labels, num_thresholds=200, curve='ROC', summation_method='interpolation'):
        super(MultiLabelAUC, self).__init__(name='multi_label_auc')
        self.AUCs = [keras.metrics.AUC(num_thresholds, curve, summation_method) for _ in range(n_outputs)]
        self.n_outputs = n_outputs
        self.output_labels = output_labels

    def reset_states(self):
        [auc.reset_states() for auc in self.AUCs]

    def update_state(self, y_true, y_pred):
        assert y_true.shape[-1] == y_pred.shape[-1] == self.n_outputs, 'Number of outputs must match shapes.'
        assert len(y_true.shape) == 2, 'Shape of y_true and y_pred must be 2.'

        for i, auc in enumerate(self.AUCs):
            auc.update_state(y_true[:, i], y_pred[:, i])

    def result(self):
        return tf.convert_to_tensor([auc.result() for auc in self.AUCs])

    def log_wandb(self, step, prefix='metrics/'):
        aucs = self.result().numpy()
        wandb.log({f'{prefix}multi_label_auc_{self.output_labels[i].lower()}': aucs[i] for i in range(self.n_outputs)},
                  step=step, commit=False)


class AUC(keras.metrics.Metric):

    def __init__(self, num_thresholds=200, curve='ROC', summation_method='interpolation'):
        super(AUC, self).__init__(name='auc')
        self.auc = keras.metrics.AUC(num_thresholds=num_thresholds, curve=curve, summation_method=summation_method)

    def update_state(self, y_true, y_pred):
        self.auc.update_state(y_true, y_pred)

    def reset_states(self):
        self.auc.reset_states()

    def result(self):
        return self.auc.result()

    def log_wandb(self, step, prefix='metrics/'):
        auc = self.result().numpy()
        wandb.run.summary[f'{prefix}{self.name}'] = auc
        wandb.log({f'{prefix}{self.name}': auc}, step=step, commit=False)


class MultiLabelRecall(keras.metrics.Metric):

    def __init__(self, n_outputs, output_labels, thresholds=np.linspace(0, 1, 11, dtype=np.float32).tolist()):
        super(MultiLabelRecall, self).__init__(name='multi_label_recall')
        self.recalls = [keras.metrics.Recall(thresholds) for _ in range(n_outputs)]
        self.thresholds = thresholds
        self.n_outputs = n_outputs
        self.output_labels = output_labels

    def reset_states(self):
        [recall.reset_states() for recall in self.recalls]

    def update_state(self, y_true, y_pred):
        assert y_true.shape[-1] == y_pred.shape[-1] == self.n_outputs, 'Number of outputs must match shapes.'
        assert len(y_true.shape) == 2, 'Shape of y_true and y_pred must be 2.'
        for i, recall in enumerate(self.recalls):
            recall.update_state(y_true[:, i], y_pred[:, i])

    def result(self):
        return tf.convert_to_tensor([recall.result() for recall in self.recalls])

    def log_wandb(self, step, prefix='metrics/'):
        recalls = self.result().numpy()
        for i in range(self.n_outputs):
            for j, rec in enumerate(recalls[i]):
                wandb.log({f'{prefix}multi_label_recall_{self.output_labels[i].lower()}@{self.thresholds[j]}': rec},
                          step=step, commit=False)


class Recall(keras.metrics.Metric):

    def __init__(self, thresholds=np.linspace(0, 1, 11, dtype=np.float32).tolist()):
        super(Recall, self).__init__(name='recall')
        self.recall = keras.metrics.Recall(thresholds=thresholds)
        self.thresholds = thresholds

    def update_state(self, y_true, y_pred):
        self.recall.update_state(y_true, y_pred)

    def reset_states(self):
        self.recall.reset_states()

    def result(self):
        return self.recall.result()

    def log_wandb(self, step, prefix='metrics/'):
        recall = self.result().numpy()
        for i, rec in enumerate(recall):
            wandb.log({f'{prefix}recall@{self.thresholds[i]:.2f}': rec}, step=step, commit=False)


class IRMPenalty(keras.metrics.Metric):

    def __init__(self):
        super(IRMPenalty, self).__init__(name='irm_penalty')
        self.loss = self.add_weight(name='irm', initializer='zeros', shape=(1,), dtype=tf.float32)
        self.count = self.add_weight(name='count', initializer='zeros', shape=1, dtype=tf.int32)

    def reset_states(self):
        K.set_value(self.loss, tf.zeros(1))
        K.set_value(self.count, [0])

    def update_state(self, y_true, y_pred):
        # Compute the IRM penalty
        y_pred_logits = tf.math.log(y_pred + 1e-6)
        self.loss.assign_add(tf.convert_to_tensor([irm_penalty_explicit(y_true, y_pred_logits, penalty_weight=1.0)]))
        # Update the total count
        self.count.assign_add([y_true.shape[0]])

    def result(self):
        if self.count > 0:
            return self.loss / tf.cast(self.count, tf.float32)
        else:
            return self.loss

    def log_wandb(self, step, prefix='metrics/'):
        loss = self.result().numpy()
        wandb.log({f'{prefix}irm_penalty': loss}, step=step, commit=False)


class Accuracy(keras.metrics.Accuracy):

    def __init__(self):
        super(Accuracy, self).__init__(name='accuracy', dtype=None)

    def log_wandb(self, step, prefix='metrics/'):
        acc = self.result().numpy()
        wandb.run.summary[f'{prefix}{self.name}'] = acc
        wandb.log({f'{prefix}{self.name}': acc}, step=step, commit=False)


class BinaryCrossentropy(keras.metrics.BinaryCrossentropy):

    def __init__(self, from_logits=False, label_smoothing=0.):
        super(BinaryCrossentropy, self).__init__(from_logits=from_logits, label_smoothing=label_smoothing)

    def log_wandb(self, step, prefix='metrics/'):
        bce = self.result().numpy()
        wandb.run.summary[f'{prefix}{self.name}'] = bce
        wandb.log({f'{prefix}{self.name}': bce}, step=step, commit=False)


class SparseCategoricalCrossentropy(keras.metrics.SparseCategoricalCrossentropy):

    def __init__(self, from_logits=False):
        super(SparseCategoricalCrossentropy, self).__init__(from_logits=from_logits)

    def log_wandb(self, step, prefix='metrics/'):
        cce = self.result().numpy()
        wandb.run.summary[f'{prefix}{self.name}'] = cce
        wandb.log({f'{prefix}{self.name}': cce}, step=step, commit=False)


class MultiLabelBinaryAccuracy(keras.metrics.Metric):

    def __init__(self, n_outputs, output_labels, threshold=0.5, name='multi_label_binary_accuracy', **kwargs):
        super(MultiLabelBinaryAccuracy, self).__init__(name=name, **kwargs)
        self.accuracies = self.add_weight(name='mob_acc', initializer='zeros', shape=n_outputs, dtype=tf.int32)
        self.count = self.add_weight(name='count', initializer='zeros', shape=1, dtype=tf.int32)
        self.n_outputs = n_outputs
        self.output_labels = output_labels
        self.threshold = threshold

    def reset_states(self):
        K.batch_set_value([(self.variables[0], tf.zeros(self.n_outputs))])
        K.set_value(self.count, [0])

    def update_state(self, y_true, y_pred):
        assert y_true.shape[-1] == y_pred.shape[-1] == self.n_outputs, 'Number of outputs must match shapes.'
        assert len(y_true.shape) == 2, 'Shape of y_true and y_pred must be 2.'

        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred > self.threshold, tf.bool)
        # Update the total count
        self.count.assign_add([y_true.shape[0]])
        # Add in the number of correctly predicted targets for each output
        correct_or_not = tf.math.reduce_sum(tf.cast(y_true == y_pred, tf.int32), 0)
        self.accuracies.assign_add(correct_or_not)

    def result(self):
        if self.count > 0:
            return self.accuracies / self.count
        else:
            return self.accuracies

    def log_wandb(self, step, prefix='metrics/'):
        accuracies = self.result().numpy()
        wandb.log({f'{prefix}binary_accuracy_{self.output_labels[i].lower()}': accuracies[i]
                   for i in range(self.n_outputs)}, step=step, commit=False)


def create_metrics(metric_names, n_classes, output_labels):
    metrics = []
    for metric_name in metric_names:
        if metric_name == 'accuracy':
            metrics.append(keras.metrics.Accuracy())
        elif metric_name == 'recall':
            metrics.append(Recall())
        elif metric_name == 'auc':
            metrics.append(AUC())
        elif metric_name == 'confusion_matrix':
            metrics.append(ConfusionMatrix(n_classes=n_classes))
        elif metric_name == 'multi_label_binary_accuracy':
            metrics.append(MultiLabelBinaryAccuracy(n_outputs=n_classes, output_labels=output_labels))
        elif metric_name == 'binary_crossentropy':
            metrics.append(BinaryCrossentropy())
        elif metric_name == 'sparse_categorical_crossentropy':
            metrics.append(SparseCategoricalCrossentropy())
        elif metric_name == 'multi_label_auc':
            metrics.append(MultiLabelAUC(n_outputs=n_classes, output_labels=output_labels))
        elif metric_name == 'multi_label_recall':
            metrics.append(MultiLabelRecall(n_outputs=n_classes, output_labels=output_labels))
        elif metric_name == 'irm_penalty':
            metrics.append(IRMPenalty())
        else:
            raise NotImplementedError
    return metrics


def reset_metrics(list_of_metrics):
    """
    Reset each metric in a list of Keras metrics
    """
    [metric.reset_states() for metric in list_of_metrics]


def log_metric_to_wandb(metric, step, prefix='metrics/'):
    """
    Manually log a Keras metric to wandb.
    """
    wandb.log({f'{prefix}{metric.name}': metric.result().numpy()}, step=step, commit=False)


def log_metrics_to_wandb(list_of_metrics, step, prefix='metrics/'):
    """
    Log a list of Keras Metrics to wandb.
    """
    for metric in list_of_metrics:
        try:
            metric.log_wandb(step, prefix)
        except AttributeError:
            log_metric_to_wandb(metric, step, prefix)


def update_metrics(list_of_metrics, targets, predictions):
    for metric in list_of_metrics:
        if metric.name in ['accuracy', 'confusion_matrix']:
            # Compatible with Softmax at the output
            metric.update_state(targets, tf.argmax(predictions, axis=-1))
        elif metric.name in ['auc', 'recall']:
            # Compatible with Softmax at the output
            metric.update_state(targets, predictions[..., 1])
        elif metric.name in [
            'multi_label_binary_accuracy',
            'binary_crossentropy',
            'sparse_categorical_crossentropy',
            'multi_label_auc',
            'multi_label_recall',
            'irm_penalty',
        ]:
            # Compatible with multiple Sigmoids at the output
            metric.update_state(targets, predictions)
        else:
            raise NotImplementedError


def test_auc():
    auc = AUC()

    auc.reset_states()

    y_true = tf.convert_to_tensor([0, 1, 1, 0])
    y_pred = tf.convert_to_tensor([[0.3, 0.7], [0.2, 0.8], [0.3, 0.7], [0.3, 0.7]])
    print(tf.argmax(y_pred, axis=-1))

    auc.update_state(y_true, y_pred[:, 1])
    print(auc.result())


def test_recall():
    recall = Recall()

    recall.reset_states()

    y_true = tf.convert_to_tensor([0, 1, 1, 0])
    y_pred = tf.convert_to_tensor([[0.3, 0.7], [0.2, 0.8], [0.3, 0.7], [0.3, 0.7]])
    print(tf.argmax(y_pred, axis=-1))

    recall.update_state(y_true, y_pred[..., 1])
    print(recall.result())


def test_mlba():
    mlba = MultiLabelBinaryAccuracy(3, range(3))

    mlba.reset_states()

    y_true = tf.convert_to_tensor([[0, 1, 0.], [1, 0, 0]])
    y_pred = tf.convert_to_tensor([[1, 0.49, .99], [1, 0, 0]])

    mlba.update_state(y_true, y_pred)

    print(mlba.result())
    print(mlba.count)
    assert np.allclose(mlba.result().numpy(), [0.5, 0.5, 0.5])

    y_true = tf.convert_to_tensor([[0, 1, 0.], [1, 0, 0]])
    y_pred = tf.convert_to_tensor([[0, 1, 0.], [0, 0, 0]])

    mlba.update_state(y_true, y_pred)
    print(mlba.result())
    print(mlba.count)
    assert np.allclose(mlba.result().numpy(), [0.5, 0.75, 0.75])

    mlba.reset_states()

    y_true = tf.convert_to_tensor([[0, 1, 0.], [1, 0, 0]])
    y_pred = tf.convert_to_tensor([[1, 0, 1.], [1, 0, 0]])

    mlba.update_state(y_true, y_pred)

    print(mlba.result())
    print(mlba.count)
    assert np.allclose(mlba.result().numpy(), [0.5, 0.5, 0.5])

    y_true = tf.convert_to_tensor([[0, 1, 0.], [1, 0, 0]])
    y_pred = tf.convert_to_tensor([[0, 1, 0.], [0, 0, 0]])

    mlba.update_state(y_true, y_pred)
    print(mlba.result())
    print(mlba.count)
    assert np.allclose(mlba.result().numpy(), [0.5, 0.75, 0.75])


def test_bce():
    y_true = tf.convert_to_tensor([[0, 1, 0.], [1, 0, 0]])
    y_pred = tf.convert_to_tensor([[0, 1, 0.], [0, 0, 0]])

    bce = BinaryCrossentropy()
    bce.update_state(y_true, y_pred)
    print(bce.result())


def test_irm():
    y_true = tf.convert_to_tensor([1, 2, 0, 1])
    y_pred = tf.convert_to_tensor([[0, 1, 0.], [0.5, 0.4, 0.1], [0.3, 0.3, 0.4], [0.9, 0, 0.1]])

    irm = IRMPenalty()
    irm.update_state(y_true, y_pred)
    print(irm.name in ['irm_penalty'])
    print(irm.result())  # 215.75023


def test_mauc():
    mauc = MultiLabelAUC(3, range(3), num_thresholds=3)
    y_true = tf.convert_to_tensor([[0, 1, 0.], [0, 0, 0], [1, 0, 0], [1, 0, 1]])
    y_pred = tf.convert_to_tensor([[0, 1, 0.], [0.5, 0, 0], [0.3, 0, 0], [0.9, 0, 0.6]])

    mauc.update_state(y_true, y_pred)
    print(mauc.result())

    y_true = tf.convert_to_tensor([[0, 1, 0.]])
    y_pred = tf.convert_to_tensor([[0.3, 0.1, 0.1]])
    mauc.update_state(y_true, y_pred)
    print(mauc.result())

    mauc.reset_states()


if __name__ == '__main__':
    import numpy as np

    test_mlba()
    test_mauc()
    test_irm()
    test_auc()
    test_recall()
