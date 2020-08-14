import argparse
import os
import yaml
import subprocess
import glob
import functools
from augmentation.augment.utils import create_multiple_train_eval_augmentation_pipelines
from augmentation.augment.static import create_multiple_train_eval_static_augmentation_pipelines
from augmentation.datasets.utils import *
from augmentation.methods.robust.utils import *
from augmentation.models.models import *
from augmentation.utilities.config import recursively_create_config_simple_namespace
from augmentation.utilities.eval import evaluate_model
from augmentation.utilities.losses import create_loss_fn, decay_weights
from augmentation.utilities.metrics import create_metrics, update_metrics, reset_metrics, log_metrics_to_wandb
from augmentation.utilities.optim import build_optimizer, build_lr_scheduler
from augmentation.utilities.utils import basic_setup
from augmentation.utilities.checkpoint import *
from augmentation.utilities.wandb import *
import tempfile


def train_robust_model(config):
    # Do basic setup
    # assert len(config.logical_gpus) > 1, 'Must specify at least 2 logical GPUs for training robust models.'
    basic_setup_info = basic_setup(seed=config.seed, logical_gpu_memory_limits=config.logical_gpus)
    logical_gpus, devices = basic_setup_info.logical_gpus, basic_setup_info.devices

    # Calculate how many folds we're looping over
    n_folds = 1 if not config.cross_validation else 1. // config.validation_frac

    # Training loop
    for fold in range(n_folds):
        # Set up weights and biases
        while True:
            try:
                if not config.resume:
                    # Start a new Weights and Biases run
                    wandb.init(entity=config.wandb_entity,
                               project=config.wandb_project,
                               group=config.wandb_group,
                               job_type=config.wandb_job_type,
                               reinit=True,
                               config=config,
                               dir=tempfile.mkdtemp(dir=os.getcwd()))
                else:
                    # Resume a previous Weights and Biases run
                    wandb.init(entity=config.prev_wandb_entity,
                               project=config.prev_wandb_project,
                               id=config.prev_wandb_run_id,
                               reinit=True,
                               resume=True)
                os.makedirs(f'{wandb.run.dir}/{config.checkpoint_path}', exist_ok=True)
                break
            except:
                continue

        # Setup the augmentation pipeline we'll be using: if only a single augmentation was passed, it will be applied
        # to all the datasets
        with devices[0]:
            train_augmentations_pipelines, eval_augmentations_pipelines = \
                create_multiple_train_eval_augmentation_pipelines(
                    train_augmentation_pipelines=config.train_augmentation_pipelines,
                    train_augmentation_pipelines_args=config.train_augmentation_pipelines_args,
                    eval_augmentation_pipelines=config.eval_augmentation_pipelines,
                    eval_augmentation_pipelines_args=config.eval_augmentation_pipelines_args,
                    broadcast_train_to=len(config.train_datasets),
                    broadcast_eval_to=len(config.eval_datasets))

            train_gpu_augmentations_pipelines, eval_gpu_augmentations_pipelines = \
                create_multiple_train_eval_augmentation_pipelines(
                    train_augmentation_pipelines=config.train_gpu_augmentation_pipelines,
                    train_augmentation_pipelines_args=config.train_gpu_augmentation_pipelines_args,
                    eval_augmentation_pipelines=config.eval_gpu_augmentation_pipelines,
                    eval_augmentation_pipelines_args=config.eval_gpu_augmentation_pipelines_args,
                    broadcast_train_to=len(config.train_datasets),
                    broadcast_eval_to=len(config.eval_datasets))

            train_static_augmentations_pipelines, eval_static_augmentations_pipelines = \
                create_multiple_train_eval_static_augmentation_pipelines(
                    train_augmentation_pipelines=config.train_static_augmentation_pipelines,
                    train_augmentation_pipelines_args=config.train_static_augmentation_pipelines_args,
                    eval_augmentation_pipelines=config.eval_static_augmentation_pipelines,
                    eval_augmentation_pipelines_args=config.eval_static_augmentation_pipelines_args,
                    broadcast_train_to=len(config.train_datasets),
                    broadcast_eval_to=len(config.eval_datasets))

        # Get the dataset generators
        (train_generators, val_generators, test_generators), \
        (input_shape, n_classes, classes, n_training_examples, group_training_examples), \
        (train_dataset_aliases, val_dataset_aliases, test_dataset_aliases) = \
            fetch_list_of_data_generators_for_trainer(train_dataset_names=config.train_datasets,
                                                      train_dataset_versions=config.train_dataset_versions,
                                                      train_datadirs=config.train_datadirs,
                                                      train_dataset_aliases=config.train_dataset_aliases,
                                                      eval_dataset_names=config.eval_datasets,
                                                      eval_dataset_versions=config.eval_dataset_versions,
                                                      eval_datadirs=config.eval_datadirs,
                                                      eval_dataset_aliases=config.eval_dataset_aliases,
                                                      train_augmentations=train_augmentations_pipelines,
                                                      train_gpu_augmentations=train_gpu_augmentations_pipelines,
                                                      train_static_augmentations=train_static_augmentations_pipelines,
                                                      eval_augmentations=eval_augmentations_pipelines,
                                                      eval_gpu_augmentations=eval_gpu_augmentations_pipelines,
                                                      eval_static_augmentations=eval_static_augmentations_pipelines,
                                                      cache_dir=os.path.join(config.cache_dir, wandb.run.id),
                                                      validation_frac=config.validation_frac,
                                                      batch_size=config.batch_size,
                                                      dataflow=config.dataflow,
                                                      repeat=True,
                                                      shuffle_before_repeat=config.shuffle_before_repeat,
                                                      max_shuffle_buffer=config.max_shuffle_buffer,
                                                      train_shuffle_seeds=config.train_shuffle_seeds,
                                                      cross_validation=config.cross_validation,
                                                      fold=fold)

        with devices[1]:
            # Create the model
            model = create_keras_classification_model(config.model_source,
                                                      config.architecture,
                                                      input_shape,
                                                      n_classes,
                                                      config.pretrained)

        # Set things to float32
        tf.keras.backend.set_floatx(config.dtype)

        # Create a scheduler for the learning rate
        steps_per_epoch = n_training_examples // config.baseline_batch_size
        print(f"Number of total training examples: {n_training_examples}\nSteps per epoch: {steps_per_epoch}")
        # Recalculate batch size per group
        learning_rate_fn = build_lr_scheduler(scheduler=config.lr_scheduler,
                                              steps_per_epoch=steps_per_epoch,
                                              n_epochs=config.n_epochs,
                                              lr_start=config.lr_start,
                                              lr_decay_steps=config.lr_decay_steps,
                                              lr_end=config.lr_end)

        # Set up the optimizer
        optimizer = build_optimizer(config.optimizer, learning_rate_fn, config.momentum)

        # Compile the model
        compile_keras_models([model], [optimizer])

        # Set up the loss function and append it to the metrics
        loss_fn = create_loss_fn(config.loss_name)

        # Set up more specific loss info: the consistency training info, and GDRO loss
        consistency_triplets, training_groups_mask, robust_loss_calc = get_loss_info(train_dataset_aliases,
                                                                                     config.augmentation_training,
                                                                                     group_training_examples,
                                                                                     config.gdro_adj_coef,
                                                                                     config.gdro_lr,
                                                                                     config.gdro_mixed,
                                                                                     )

        # Set up the metrics being tracked
        aggregate_metrics = create_metrics(config.metric_names, n_classes, output_labels=classes)
        metrics_by_group = [create_metrics(config.metric_names, n_classes, output_labels=classes)
                            for _ in range(len(train_generators))]
        eval_metrics_by_group = [create_metrics(config.metric_names, n_classes, output_labels=classes)
                                 for _ in range(len(test_generators))]

        # By default, assume we're starting training from scratch
        start_epoch, start_step = 0, 0

        # Resume a run from Weights and Biases
        # This could be for continuing training, or reloading the model for testing its invariance
        if config.resume:
            start_epoch, start_step = reload_run(model=model,
                                                 optimizer=optimizer,
                                                 robust_loss_calc=robust_loss_calc,
                                                 wandb_run_id=config.prev_wandb_run_id,
                                                 wandb_project=config.prev_wandb_project,
                                                 wandb_entity=config.prev_wandb_entity,
                                                 wandb_ckpt_path=config.prev_ckpt_path,
                                                 resume_epoch=config.prev_ckpt_epoch,
                                                 continue_training=config.resume  # only if we're continuing training
                                                 )

        with devices[0]:
            # Train the end model
            _train_robust_model(train_generators=train_generators,
                                val_generators=val_generators,
                                test_generators=test_generators,
                                train_dataset_aliases=train_dataset_aliases,
                                val_dataset_aliases=val_dataset_aliases,
                                test_dataset_aliases=test_dataset_aliases,
                                model=model,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                aggregate_metrics=aggregate_metrics,
                                metrics_by_group=metrics_by_group,
                                eval_metrics_by_group=eval_metrics_by_group,
                                n_epochs=config.n_epochs,
                                steps_per_epoch=steps_per_epoch,
                                max_global_grad_norm=config.max_global_grad_norm,
                                weight_decay_rate=config.weight_decay_rate,
                                irm_anneal_steps=config.irm_anneal_steps,
                                irm_penalty_weight=config.irm_penalty_weight,
                                robust_loss_calc=robust_loss_calc,
                                training_groups_mask=training_groups_mask,
                                consistency_triplets=consistency_triplets,
                                consistency_type=config.consistency_type,
                                consistency_penalty_weight=config.consistency_penalty_weight,
                                checkpoint_path=config.checkpoint_path,
                                checkpoint_freq=config.checkpoint_freq,
                                devices=devices,
                                start_step=start_step,
                                start_epoch=start_epoch,
                                dtype=config.dtype)

        # Clean up by removing all the data caches
        for cache in glob.glob(os.path.join(config.cache_dir, wandb.run.id) + '*'):
            os.remove(cache)


def get_loss_info(train_dataset_aliases, augmentation_training, group_training_examples, gdro_adj_coef, gdro_lr,
                  gdro_mixed):
    consistency_triplets = []
    # TODO not general. We're grabbing subsets based on an assumed alias convention
    af = {alias[:-5]: i for i, alias in enumerate(train_dataset_aliases) if alias[-5:] == '(A-F)'}
    ag = {alias[:-5]: i for i, alias in enumerate(train_dataset_aliases) if alias[-5:] == '(A-G)'}
    a = {alias: i for i, alias in enumerate(train_dataset_aliases) if alias in af}
    for alias in af:
        if alias not in a:
            a[alias] = -1
    assert a.keys() == af.keys() == ag.keys()
    consistency_triplets = list(zip(a.values(), af.values(), ag.values()))

    # Create mask over datasets indicating which ones are to be used for the main training loss
    training_groups_mask = [True] * len(train_dataset_aliases)
    if augmentation_training == 'original':
        for orig_idx, f_idx, g_idx in consistency_triplets:
            training_groups_mask[f_idx] = False
            training_groups_mask[g_idx] = False
    elif augmentation_training == 'augmented':
        for orig_idx, f_idx, g_idx in consistency_triplets:
            training_groups_mask[orig_idx] = False
    elif augmentation_training == 'both':
        pass
    else:
        assert False, f"augmentation_training value {augmentation_training} should be 'original', 'augmented', or 'both'"

    assert sum(
        training_groups_mask) > 0, \
        "No training datasets are used for main loss calculation! Check config.augmentation_training and augmentation flags"  # TODO it's conceivable that the user may want this, but have to check that some other loss is used (e.g. consistency)

    print("Consistency triplets: ", consistency_triplets)
    print("Dataset training mask: ", training_groups_mask)
    print(flush=True)

    # Set up the GDRO loss calculator
    group_training_examples_used = [n for n, use in zip(group_training_examples, training_groups_mask) if use]
    group_training_aliases_used = [a for a, use in zip(train_dataset_aliases, training_groups_mask) if use]

    if gdro_mixed:
        def extract(alias):
            if '(Y=0)' in alias:
                return 0
            elif '(Y=1)' in alias:
                return 1
            else:
                return -1

        superclass_ids = [extract(alias) for alias in group_training_aliases_used]
    else:
        superclass_ids = [0] * len(group_training_aliases_used)
    robust_loss_calc = GDROLoss(group_training_aliases_used, group_training_examples_used, superclass_ids,
                                gdro_adj_coef, gdro_lr)

    return consistency_triplets, training_groups_mask, robust_loss_calc


# IRM Train Loop
def _train_robust_model(train_generators,
                        val_generators,
                        test_generators,
                        train_dataset_aliases,
                        val_dataset_aliases,
                        test_dataset_aliases,
                        model,
                        optimizer,
                        loss_fn,
                        aggregate_metrics,
                        metrics_by_group,
                        eval_metrics_by_group,
                        # batch_size,
                        n_epochs,
                        steps_per_epoch,
                        max_global_grad_norm,
                        weight_decay_rate,
                        irm_anneal_steps,
                        irm_penalty_weight,
                        robust_loss_calc,
                        # augmentation_training,
                        training_groups_mask,
                        consistency_triplets,
                        consistency_type,
                        consistency_penalty_weight,
                        checkpoint_path,
                        checkpoint_freq,
                        devices,
                        start_step=0, start_epoch=0, dtype=tf.float32):
    def eval_and_log(split_name, model, generators, dataset_aliases, aggregate_metrics, eval_metrics_by_group, step):
        # Evaluate the model on each evaluation set and log to weights and biases
        reset_metrics(aggregate_metrics)
        for i, generator in enumerate(generators):
            log_metrics_to_wandb(evaluate_model(model, generator, eval_metrics_by_group[i], aggregate_metrics),
                                 step=step, prefix=f'{split_name}_metrics/{dataset_aliases[i]}/')
        log_metrics_to_wandb(aggregate_metrics, step=step, prefix=f'{split_name}_metrics/aggregate/')

    # Keep track of how many gradient steps we've taken
    # For the robust train loop, we track steps instead of epochs
    # This is because each group in the dataset is processed at different rates
    step = start_step

    # Convert generators to iterators
    # NOTE group DRO uses sampling with replacement to ensure each group has enough data
    # this can be emulated by ensuring that the dataset modifiers have the form
    # .repeat(-1).shuffle(N).batch(bs) for a value of N large relative to the dataset size
    train_iterators = list(map(iter, train_generators))

    # Function to create floatX inputs to the model
    make_floatx_tensor = functools.partial(tf.convert_to_tensor, dtype=dtype)

    if step == 0:
        with devices[1]:
            eval_and_log('validation', model, val_generators, val_dataset_aliases, aggregate_metrics,
                         eval_metrics_by_group, step)
    # Run over the epochs
    for epoch in range(start_epoch, n_epochs):

        # Reset the metrics for each epoch
        reset_metrics(aggregate_metrics)
        for metrics in metrics_by_group:
            reset_metrics(metrics)

        for _ in range(steps_per_epoch):
            # Get batches of data from each group's training iterator
            group_batches, group_targets = tuple(zip(*[tuple(map(make_floatx_tensor, next(it)))
                                                       for it in train_iterators]))

            # Compute the IRM penalty weight
            step_irm_penalty_weight = irm_penalty_scheduler(step, irm_anneal_steps, irm_penalty_weight)
            step_consistency_penalty_weight = consistency_penalty_scheduler(step, 0,
                                                                            consistency_penalty_weight)

            with devices[1]:
                # Train using these group's batches of data
                robust_loss, consistency_loss, irm_losses, group_losses, group_predictions, gradients = \
                    train_step_robust(model=model,
                                      loss_fn=loss_fn,
                                      group_batches=group_batches,
                                      group_targets=group_targets,
                                      optimizer=optimizer,
                                      max_global_grad_norm=max_global_grad_norm,
                                      weight_decay_rate=weight_decay_rate,
                                      irm_penalty_weight=step_irm_penalty_weight,
                                      robust_loss_calc=robust_loss_calc,
                                      training_groups_mask=training_groups_mask,
                                      consistency_type=consistency_type,
                                      consistency_triplets=consistency_triplets,
                                      consistency_penalty_weight=step_consistency_penalty_weight,
                                      )

            # Update the metrics
            for targets, predictions, metrics in zip(group_targets, group_predictions, metrics_by_group):
                update_metrics(aggregate_metrics, targets, predictions)
                update_metrics(metrics, targets, predictions)

            # Update the step counter
            step += 1

            # Log to weights and biases
            log_robust_train_step_to_wandb(train_dataset_aliases, group_batches, group_targets, group_predictions,
                                           group_losses, robust_loss, consistency_loss, step_consistency_penalty_weight,
                                           irm_losses, step_irm_penalty_weight, gradients, model, optimizer,
                                           robust_loss_calc, step,
                                           log_images=(epoch == 0 and step < 10),
                                           log_weights_and_grads=False)

            del robust_loss, consistency_loss, group_losses, group_predictions, gradients

        # Log the training metrics to weights and biases
        log_metrics_to_wandb(aggregate_metrics, step, prefix='training_metrics/aggregate/')
        for i, metrics in enumerate(metrics_by_group):
            log_metrics_to_wandb(metrics, step, prefix=f'training_metrics/{train_dataset_aliases[i]}/')

        with devices[1]:
            # Evaluate the model on each validation set and log to weights and biases
            eval_and_log('validation', model, val_generators, val_dataset_aliases, aggregate_metrics,
                         eval_metrics_by_group, step)
            eval_and_log('test', model, test_generators, test_dataset_aliases, aggregate_metrics,
                         eval_metrics_by_group, step)

        # End of epoch, log to weights and biases
        wandb.log({'epochs': epoch + 1}, step=step)

        # Store the model every few epochs
        if (epoch + 1) % checkpoint_freq == 0:
            model.save(f'{wandb.run.dir}/{checkpoint_path}/ckpt_{epoch + 1}.h5')
            save_tf_optimizer_state(optimizer, f'{wandb.run.dir}/{checkpoint_path}/optimizer_{epoch + 1}.pkl')
            np.save(f'{wandb.run.dir}/{checkpoint_path}/gdro_{epoch + 1}.npy',
                    robust_loss_calc._adv_prob_logits.numpy())
            # Sync the model to the cloud
            wandb.save(f'{wandb.run.dir}/{checkpoint_path}/ckpt_{epoch + 1}.h5')
            wandb.save(f'{wandb.run.dir}/{checkpoint_path}/optimizer_{epoch + 1}.pkl')
            wandb.save(f'{wandb.run.dir}/{checkpoint_path}/gdro_{epoch + 1}.npy')

    with devices[1]:  # TODO: add flags to make the with device optional
        # Evaluate the model on each test set and log to weights and biases
        reset_metrics(aggregate_metrics)
        for i, test_generator in enumerate(test_generators):
            log_metrics_to_wandb(evaluate_model(model, test_generator, eval_metrics_by_group[i], aggregate_metrics),
                                 step=step, prefix=f'test_metrics/{test_dataset_aliases[i]}/')
        log_metrics_to_wandb(aggregate_metrics, step=step, prefix=f'test_metrics/aggregate/')

    # Commits everything
    wandb.log({})


# Robust Train Step
def train_step_robust(model,
                      loss_fn,
                      group_batches,
                      group_targets,
                      optimizer,
                      max_global_grad_norm,
                      weight_decay_rate,
                      irm_penalty_weight,
                      robust_loss_calc,
                      training_groups_mask,  # indices of groups for which to take the loss of
                      consistency_type,
                      consistency_triplets,
                      consistency_penalty_weight,
                      ):
    def _train_step_robust(_group_batches, _group_targets):
        group_losses, group_predictions = [], []
        irm_losses = []
        gdro_losses = []

        loss_idxs = list(range(len(training_groups_mask)))

        loss_batches = [_group_batches[i] for i in loss_idxs]

        # Compute the batch sizes of each group's batch
        loss_batch_sizes = [e.shape[0] for e in loss_batches]
        # Concatenate into one tensor
        concat_batch = tf.concat(loss_batches, axis=0)

        # Open up the gradient tape
        with tf.GradientTape() as tape:
            # Pass through the model
            loss_predictions = model(concat_batch, training=True)
            # Split up the predictions
            loss_group_predictions = tf.split(loss_predictions, loss_batch_sizes, axis=0)

            # Scatter loss back to one list
            group_predictions = [None] * len(group_batches)
            for i, idx in enumerate(loss_idxs):
                group_predictions[idx] = loss_group_predictions[i]

            for (_use_for_training, targets, predictions) in zip(training_groups_mask, _group_targets,
                                                                 group_predictions):
                if _use_for_training:
                    # Compute the loss
                    loss = loss_fn(targets, predictions)

                    # Compute the IRM penalty
                    irm_penalty = irm_penalty_explicit(targets, tf.math.log(predictions + 1e-6), irm_penalty_weight)
                    loss = loss + irm_penalty
                    irm_losses.append(irm_penalty)

                    # Rescale the loss
                    # loss = irm_loss_rescale(loss, irm_penalty_weight)

                    gdro_losses.append(loss)
                    group_losses.append(loss)
                else:
                    # Trick to ensure that all losses can be logged
                    group_losses.append(tf.convert_to_tensor(0.))
                    irm_losses.append(tf.convert_to_tensor(0.))

            # Compute the robust loss
            robust_loss = robust_loss_calc.compute_loss(gdro_losses)

            # Compute the l2 regularizer
            robust_loss = robust_loss + decay_weights(model, weight_decay_rate)

            # Compute the consistency loss
            consistency_loss = tf.convert_to_tensor(0.)
            if consistency_triplets is not None:
                for orig, f, g in consistency_triplets:
                    consistency_loss += consistency_penalty(group_predictions[orig],
                                                            group_predictions[f],
                                                            group_predictions[g],
                                                            consistency_type,
                                                            consistency_penalty_weight)

            robust_loss = robust_loss + consistency_loss

        # Compute gradients
        gradients = tape.gradient(robust_loss, model.trainable_weights)

        # Clip the gradients
        if max_global_grad_norm > 0.:
            gradients, _ = tf.clip_by_global_norm(gradients, max_global_grad_norm)

        # Apply the gradients to the model
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        # Delete the tape
        del tape

        # Return the group losses, group predictions and gradients
        return robust_loss, consistency_loss, irm_losses, group_losses, group_predictions, gradients

    return _train_step_robust(group_batches, group_targets)


def setup_and_train_robust_model(args):
    # Load up the config
    config = recursively_create_config_simple_namespace(args.config, args.template_config)

    # Train the end model
    train_robust_model(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--template_config', type=str, default='augmentation/configs/template_robust_training.yaml')

    # Set up the configuration and train the end model
    setup_and_train_robust_model(parser.parse_args())
