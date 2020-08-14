import tensorflow as tf
import wandb
import yaml
import subprocess
from augmentation.utilities.visualize import gallery
from augmentation.utilities.wandb import *
from augmentation.utilities.checkpoint import load_tf_optimizer_state


def rewrite_config_for_resumption(config):
    config.prev_wandb_entity = config.wandb_entity
    config.prev_wandb_project = config.wandb_project
    config.prev_wandb_run_id = wandb.run.id
    config.resume = True
    yaml.dump(config.__dict__, open(config._config_path, 'w'))

    # Push the change for this config
    for cmd in [['git', 'add', config._config_path],
                ['git', 'commit', '-m', f'cfg_update_{wandb.run.id}'],
                ['git', 'pull'],
                ['git', 'push']]:
        subprocess.run(cmd)

    return config


def reload_run(model,
               optimizer,
               robust_loss_calc,
               wandb_run_id,
               wandb_project,
               wandb_entity,
               wandb_ckpt_path,
               resume_epoch=-1,
               continue_training=True):
    # By default, we start at the beginning
    start_epoch, start_step = 0, 0

    # Load up the previous run
    prev_run = load_wandb_run(wandb_run_id, wandb_project, wandb_entity)
    step_extractor = particular_checkpoint_step_extractor(resume_epoch,
                                                          lambda fname: fname.split(".")[-2].split("_")[-1])
    # If the previous run crashed, wandb_ckpt_path should be '': this is the typical use case
    # but this should be changed in the future
    _, loaded_epoch = load_most_recent_keras_model_weights(model, prev_run,
                                                           model_name='ckpt',
                                                           exclude='generator',
                                                           step_extractor=step_extractor,
                                                           wandb_ckpt_path=wandb_ckpt_path)

    # If we're continuing training AND if we reloaded a model
    # - load up the optimizer and DRO state
    # - set the start epoch and start step
    if continue_training and loaded_epoch is not None:
        start_epoch = loaded_epoch
        for line in prev_run.history():
            if 'epochs' in line and line['epochs'] == start_epoch:
                start_step = line['train_step/step']
                break

        # Reloading the optimizer states from that epoch
        opt_ckpt = get_most_recent_model_file(prev_run,
                                              wandb_ckpt_path=wandb_ckpt_path,
                                              model_name='optimizer',
                                              step_extractor=particular_checkpoint_step_extractor(start_epoch))
        load_tf_optimizer_state(optimizer, opt_ckpt.name)

        # Reloading the state of GDRO from that epoch
        gdro_ckpt = get_most_recent_model_file(prev_run,
                                               wandb_ckpt_path=wandb_ckpt_path,
                                               model_name='gdro',
                                               step_extractor=particular_checkpoint_step_extractor(start_epoch))
        robust_loss_calc._adv_prob_logits = tf.convert_to_tensor(np.load(gdro_ckpt.name))

    print(f"Loaded epoch {loaded_epoch} from {wandb_run_id}. Starting from step {start_step} and epoch {start_epoch}.",
          flush=True)

    return start_epoch, start_step


def log_robust_train_step_to_wandb(group_aliases, group_batches, group_targets, group_predictions, group_losses,
                                   robust_loss, consistency_loss, consistency_penalty_weight,
                                   irm_losses, irm_penalty_weight,
                                   gradients, model, optimizer,
                                   robust_loss_calc, step, log_images=False, log_weights_and_grads=False):
    # Loop over the data from each group
    # for i, (batch, targets, predictions, loss) in enumerate(zip(group_batches, group_targets,
    for (alias, batch, targets, predictions, loss, irm) in zip(group_aliases, group_batches, group_targets,
                                                               group_predictions, group_losses, irm_losses):
        # Log data generated in this train step
        wandb.log({f'train_step/{alias}/targets': targets.numpy(),
                   f'train_step/{alias}/predictions': wandb.Histogram(predictions.numpy()),
                   f'train_step/{alias}/argmax_predictions': tf.argmax(predictions, axis=-1).numpy(),
                   f'train_step/{alias}/loss': loss.numpy(),
                   f'train_step/{alias}/irm': irm.numpy()},
                  step=step)

        # Optionally, log the minibatch of images
        if log_images:
            wandb.log({f'train_step/{alias}/images': wandb.Image(gallery(batch.numpy()))}, step=step)

    # Log all the gradients and weights: every 50 steps
    if log_weights_and_grads:
        wandb.log({f'gradients/{v.name}': g.numpy() for v, g in zip(model.trainable_variables, gradients)}, step=step)
        wandb.log({f'weights/{v.name}': v.numpy() for v in model.trainable_variables}, step=step)

    for prob, alias in zip(tf.nn.softmax(robust_loss_calc._adv_prob_logits, axis=-1).numpy().reshape(-1),
                           robust_loss_calc._aliases):
        wandb.log({f'train_step/gdro_adv_prob.{alias}': prob}, step=step)

    wandb.log({'train_step/irm_penalty_weight': irm_penalty_weight,
               'train_step/consistency_penalty_weight': consistency_penalty_weight,
               # 'train_step/gdro_adv_probs': tf.nn.softmax(robust_loss_calc._adv_prob_logits, axis=-1).numpy(),
               'train_step/robust_loss': robust_loss.numpy(),
               'train_step/consistency_loss': consistency_loss.numpy(),
               'train_step/global_gradient_norm': tf.linalg.global_norm(gradients).numpy(),
               'train_step/learning_rate': optimizer._decayed_lr(tf.float32).numpy(),
               'train_step/step': step}, step=step)


def consistency_penalty(predictions_orig, predictions_1, predictions_2, consistency_type, scale=1.0):
    # CAMEL consistency: JS-Divergence of augmentations, plus KL between original and average augmentation
    if consistency_type == 'camel':
        avg_predictions = (predictions_1 + predictions_2) / 2.0
        return tf.reduce_mean((tf.keras.losses.KLD(predictions_orig, avg_predictions) * 0.5 +
                               tf.keras.losses.KLD(predictions_1, avg_predictions) * 0.25 +
                               tf.keras.losses.KLD(predictions_2, avg_predictions) * 0.25)) * scale
    # JS-Divergence between original and both augmentations (as in AugMix)
    elif consistency_type == 'triplet-js':
        avg_predictions = (predictions_orig + predictions_1 + predictions_2) / 3.0
        return tf.reduce_mean((tf.keras.losses.KLD(predictions_orig, avg_predictions) +
                               tf.keras.losses.KLD(predictions_1, avg_predictions) +
                               tf.keras.losses.KLD(predictions_2, avg_predictions)) / 3.0) * scale
    # KL divergence between original and each augmentation
    elif consistency_type == 'kl':
        return tf.reduce_mean((tf.keras.losses.KLD(predictions_orig, predictions_1) +
                               tf.keras.losses.KLD(predictions_orig, predictions_2)) * scale * 0.5)
    elif consistency_type == 'reverse-kl':
        return tf.reduce_mean((tf.keras.losses.KLD(predictions_1, predictions_orig) +
                               tf.keras.losses.KLD(predictions_2, predictions_orig)) * scale * 0.5)
    elif consistency_type == 'none':
        return tf.convert_to_tensor(0.)
    else:
        assert False, f'consistency_type {consistency_type} not supported'


def irm_penalty_explicit(targets, pred_logits, penalty_weight):
    """ Computes the IRM penalty grad_{w} |_{w=1.0} crossent(targets, w*logits) explicitly """
    if penalty_weight == 0.:
        return tf.convert_to_tensor(0.)
    xent = tf.keras.losses.sparse_categorical_crossentropy(targets, pred_logits, from_logits=True)
    sparse_logit = xent + tf.reduce_logsumexp(pred_logits,
                                              axis=-1)  # equivalent to grabbing the logit indexed by target
    grad = sparse_logit - tf.reduce_sum(pred_logits * tf.nn.softmax(pred_logits, axis=-1), axis=-1)
    return tf.reduce_sum(grad ** 2) * penalty_weight


def irm_penalty_gradient(targets, pred_logits, penalty_weight, tape):
    """ Computes IRM penalty as formulated in the paper
    Currently does not work: tf does not support second order gradients of cross entropy
    """
    if penalty_weight == 0.:
        return 0.
    # Taken from https://github.com/facebookresearch/InvariantRiskMinimization/blob/6aad47e689913b9bdad05880833530a5edac389e/code/colored_mnist/main.py#L107
    scale = tf.convert_to_tensor(1.)
    tape.watch(scale)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(targets, pred_logits * scale)
    grad = tape.gradient(loss, scale)
    return tf.reduce_sum(grad ** 2) * penalty_weight


def consistency_penalty_scheduler(step, n_anneal_steps, base_penalty_weight):
    """
    Schedule the consistency penalty.
    """
    if base_penalty_weight == 0:
        return 0.
    if step >= n_anneal_steps:
        return base_penalty_weight
    return 0.0


def irm_penalty_scheduler(step, n_anneal_steps=100, base_penalty_weight=10000.):
    """
    Schedule the IRM penalty weight using a step function as done by
    https://github.com/facebookresearch/InvariantRiskMinimization
    If the penalty weight is 0. (IRM disabled), just return 0.
    """
    if base_penalty_weight == 0.:
        return 0.
    if step >= n_anneal_steps:
        return base_penalty_weight
    # return 1.0
    return 0.0  # train with no irm at first


def irm_loss_rescale(total_loss, irm_penalty_weight):
    """
    Rescale the total loss by the IRM penalty weight as done by
    https://github.com/facebookresearch/InvariantRiskMinimization
    """
    if irm_penalty_weight > 1.0:
        return total_loss / irm_penalty_weight
    return total_loss


class GDROLoss:

    def __init__(self, group_aliases, group_counts, superclass_ids, adj_coef, step_size):
        """
        group_counts: list of integer sizes of the groups
        adj_coef: scalar coefficient of the generalization gap penalty
        step_size: robust learning rate for the "mixture of expert" probabilities
        """
        assert len(group_aliases) == len(group_counts) == len(superclass_ids)

        group_counts = tf.cast(tf.stack(group_counts), tf.float32)
        print(f"GDROLoss: Group counts {group_counts}")
        self._adj = adj_coef * 1. / tf.math.sqrt(group_counts)
        print("adj_coef", adj_coef)
        print("total adjustment", self._adj)
        self._step_size = step_size

        self._adv_probs = tf.ones(len(group_counts)) / len(group_counts)
        # _adv_prob_logits must exist, being logged by wandb now
        self._adv_prob_logits = tf.zeros_like(group_counts)
        self._aliases = group_aliases

        # For now, assume superclass_ids are 0, 1, -1
        superclass_idxs_ = {}
        for i in set(superclass_ids):
            superclass_idxs_[i] = [idx for idx, j in enumerate(superclass_ids) if j == i]
        superclass_freqs_ = {i: len(idxs) / len(group_aliases) for i, idxs in superclass_idxs_.items()}

        self.superclass_idxs = superclass_idxs_.values()
        self.superclass_freqs = superclass_freqs_.values()
        print("GDROLoss: superclass indices, freqs", self.superclass_idxs, self.superclass_freqs)

    def compute_loss(self, losses):
        """ losses: list of losses (scalars) """
        if len(losses) == 0: return tf.convert_to_tensor(0.0)

        losses = tf.stack(losses, axis=-1) + self._adj
        self._adv_prob_logits += self._step_size * losses

        loss = tf.convert_to_tensor(0.)
        for idxs, freq in zip(self.superclass_idxs, self.superclass_freqs):
            adv_probs = tf.nn.softmax(tf.gather(self._adv_prob_logits, idxs), axis=-1)
            loss = loss + tf.reduce_sum(adv_probs * tf.gather(losses, idxs), axis=-1) * freq
        return loss
