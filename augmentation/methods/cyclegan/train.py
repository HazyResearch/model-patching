import argparse
import os
import functools
import time
import subprocess
from augmentation.utilities.config import *
from augmentation.utilities.metrics import *
from augmentation.datasets.utils import get_processed_dataset_info, apply_modifier_to_dataset_payload, load_dataset
from augmentation.dataflows.utils import dataflow_len
from augmentation.augment.utils import create_augmentation_pipelines
from augmentation.models.models import *
from augmentation.methods.cyclegan.utils import *
from augmentation.utilities.checkpoint import *
from augmentation.utilities.visualize import *
from augmentation.utilities.wandb import load_wandb_run, load_most_recent_keras_model_weights, \
    get_most_recent_model_file, particular_checkpoint_step_extractor
from augmentation.utilities.utils import basic_setup


def train_cyclegan(config):
    # Do basic setup
    basic_setup(seed=config.seed, logical_gpu_memory_limits=(14336,))

    # Set up the source dataset
    source_dataset_payload = load_dataset(config.source_dataset,
                                          config.source_dataset_version,
                                          config.datadir,
                                          config.validation_frac)

    target_dataset_payload = load_dataset(config.target_dataset,
                                          config.target_dataset_version,
                                          config.datadir,
                                          config.validation_frac)

    # Get some dataset information
    source_proc_dataset_info = get_processed_dataset_info(source_dataset_payload.dataset_info,
                                                          config.validation_frac, config.batch_size)

    target_proc_dataset_info = get_processed_dataset_info(target_dataset_payload.dataset_info,
                                                          config.validation_frac, config.batch_size)

    source_input_shape = source_proc_dataset_info.input_shape
    target_input_shape = target_proc_dataset_info.input_shape

    # Do selection on each dataset
    source_dataset_payload = apply_modifier_to_dataset_payload(source_dataset_payload, config.source_dataset_modifier)
    target_dataset_payload = apply_modifier_to_dataset_payload(target_dataset_payload, config.target_dataset_modifier)

    # Setup the augmentation pipeline we'll be using
    train_augmentations, val_augmentations, test_augmentations = \
        create_augmentation_pipelines(config.train_daug_pipeline, config.train_daug_pipeline_args,
                                      config.val_daug_pipeline, config.val_daug_pipeline_args,
                                      config.test_daug_pipeline, config.test_daug_pipeline_args)

    # Create the data generators
    train_generator = create_cyclegan_data_generator(source_dataset_payload.train_dataset,
                                                     target_dataset_payload.train_dataset,
                                                     config.batch_size,
                                                     train_augmentations,
                                                     config.dataflow,
                                                     config.cache_dir + 'train')

    test_generator = create_cyclegan_data_generator(source_dataset_payload.test_dataset,
                                                    target_dataset_payload.test_dataset,
                                                    config.batch_size,
                                                    test_augmentations,
                                                    config.dataflow,
                                                    config.cache_dir + 'test')

    # Create the models
    generator_g, generator_f, discriminator_x, discriminator_y = \
        build_models(source_input_shape, target_input_shape, config.norm_type, config.output_init, config.residual_outputs)

    generator_g.summary()
    generator_f.summary()
    discriminator_x.summary()
    discriminator_y.summary()

    # Set up the optimizers
    generator_optimizer, _, discriminator_optimizer, _ = build_optimizers(lr_gen=config.lr_gen,
                                                                          lr_disc=config.lr_disc,
                                                                          beta_1_gen=config.beta_1_gen,
                                                                          beta_1_disc=config.beta_1_disc,
                                                                          lr_scheduler=config.lr_scheduler,
                                                                          lr_decay_steps=config.n_epochs *
                                                                                         dataflow_len(train_generator))

    # Compile the models
    compile_keras_models([generator_g, generator_f, discriminator_x, discriminator_y],
                         [generator_optimizer, generator_optimizer, discriminator_optimizer, discriminator_optimizer])

    # Create the replay buffers for the discriminators
    disc_x_replay, disc_y_replay = ReplayBuffer(config.replay_buffer_size), ReplayBuffer(config.replay_buffer_size)

    # Define the loss function to pass to the generator and discriminator
    gan_loss_fn = build_gan_loss_fn(config.gan_loss)

    # By default, assume we're starting training from scratch
    start_epoch, start_step = 0, 0
    if config.resume:
        # If we're resuming a run
        prev_run = load_wandb_run(config.prev_wandb_run_id, config.prev_wandb_project, config.prev_wandb_entity)
        # If the previous run crashed, wandb_ckpt_path should be '': this is the typical use case
        # but this should be changed in the future
        step_extraction_fn = lambda fname: fname.split("_")[1].split(".")[0]
        _, gen_g_ep = load_most_recent_keras_model_weights(generator_g, prev_run, model_name='generator_g',
                                                           wandb_ckpt_path=config.prev_ckpt_path,
                                                           step_extractor=step_extraction_fn)
        _, gen_f_ep = load_most_recent_keras_model_weights(generator_f, prev_run, model_name='generator_f',
                                                           wandb_ckpt_path=config.prev_ckpt_path,
                                                           step_extractor=step_extraction_fn)
        _, disc_x_ep = load_most_recent_keras_model_weights(discriminator_x, prev_run, model_name='discriminator_x',
                                                            wandb_ckpt_path=config.prev_ckpt_path,
                                                            step_extractor=step_extraction_fn)
        _, disc_y_ep = load_most_recent_keras_model_weights(discriminator_y, prev_run, model_name='discriminator_y',
                                                            wandb_ckpt_path=config.prev_ckpt_path,
                                                            step_extractor=step_extraction_fn)
        assert gen_g_ep == gen_f_ep == disc_x_ep == disc_y_ep, 'All restored models should be from the same epoch.'

        if gen_g_ep is not None:
            start_epoch, start_step = gen_g_ep, 0
            for line in prev_run.history():
                if 'epochs' in line and line['epochs'] == start_epoch:
                    start_step = line['steps']
                    break

            # Reloading the optimizer states from that epoch
            step_extraction_fn = lambda fname: fname.split(".")[0].split("_")[-1]
            gen_opt_ckpt = get_most_recent_model_file(prev_run,
                                                      wandb_ckpt_path=config.prev_ckpt_path,
                                                      model_name='generator_optimizer',
                                                      step_extractor=
                                                      particular_checkpoint_step_extractor(start_epoch,
                                                                                           step_extractor=
                                                                                           step_extraction_fn))
            load_tf_optimizer_state(generator_optimizer, gen_opt_ckpt.name)
            disc_opt_ckpt = get_most_recent_model_file(prev_run,
                                                       wandb_ckpt_path=config.prev_ckpt_path,
                                                       model_name='discriminator_optimizer',
                                                       step_extractor=
                                                       particular_checkpoint_step_extractor(start_epoch,
                                                                                            step_extractor=
                                                                                            step_extraction_fn))
            load_tf_optimizer_state(discriminator_optimizer, disc_opt_ckpt.name)

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
                           config=config)
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

    _train_cyclegan(train_data_generator=train_generator,
                    val_data_generator=test_generator,
                    generator_g=generator_g,
                    generator_f=generator_f,
                    discriminator_x=discriminator_x,
                    discriminator_y=discriminator_y,
                    generator_optimizer=generator_optimizer,
                    discriminator_optimizer=discriminator_optimizer,
                    discriminator_x_replay=disc_x_replay,
                    discriminator_y_replay=disc_y_replay,
                    metrics=None,
                    batch_size=None,
                    n_epochs=config.n_epochs,
                    gan_loss_fn=gan_loss_fn,
                    cycle_loss_scale_x=config.cycle_loss_scale,
                    cycle_loss_scale_y=config.cycle_loss_scale * (1 - config.source_cycle_loss_only),
                    identity_loss_scale=config.identity_loss_scale,
                    grad_penalty=config.grad_penalty,
                    grad_penalty_scale=config.grad_penalty_scale,
                    checkpoint_path=config.checkpoint_path,
                    checkpoint_freq=config.checkpoint_freq,
                    image_log_freq=config.image_log_freq,
                    start_step=start_step, start_epoch=start_epoch)


def _train_cyclegan(train_data_generator, val_data_generator,
                    generator_g, generator_f,
                    discriminator_x, discriminator_y,
                    generator_optimizer, discriminator_optimizer,
                    discriminator_x_replay, discriminator_y_replay,
                    metrics, batch_size, n_epochs,
                    gan_loss_fn, cycle_loss_scale_x, cycle_loss_scale_y, identity_loss_scale,
                    grad_penalty, grad_penalty_scale,
                    checkpoint_path, checkpoint_freq,
                    image_log_freq=50,
                    start_step=0, start_epoch=0):
    # Keep track of how many gradient steps we've taken
    step = start_step

    # Multiple training epochs
    for epoch in range(start_epoch, n_epochs):
        # Iterate over the dataset
        for batch_x, batch_y in train_data_generator:
            # Convert to tensors
            batch_x, batch_y = tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_y)

            # Train using this batch of data
            gen_losses, gen_predictions, gen_gradients = train_step_generator(generator_g, generator_f,
                                                                              discriminator_x, discriminator_y,
                                                                              gan_loss_fn,
                                                                              batch_x, batch_y,
                                                                              generator_optimizer,
                                                                              discriminator_x_replay,
                                                                              discriminator_y_replay,
                                                                              cycle_loss_scale_x, cycle_loss_scale_y,
                                                                              identity_loss_scale)

            disc_losses, disc_predictions, disc_gradients = train_step_discriminator(discriminator_x, discriminator_y,
                                                                                     gan_loss_fn,
                                                                                     batch_x, batch_y,
                                                                                     discriminator_optimizer,
                                                                                     discriminator_x_replay,
                                                                                     discriminator_y_replay,
                                                                                     grad_penalty,
                                                                                     grad_penalty_scale)

            # Update the step counter
            step += 1

            # Unpack and log to weights and biases
            (gen_g_loss, gen_f_loss, cycle_loss_x, cycle_loss_y, identity_loss_x, identity_loss_y) = gen_losses
            ((same_x, fake_x, cycled_x, disc_fake_x),
             (same_y, fake_y, cycled_y, disc_fake_y)) = gen_predictions

            (disc_x_loss, disc_y_loss, disc_x_gp, disc_y_gp) = disc_losses
            ((disc_real_x, disc_sampled_fake_x),
             (disc_real_y, disc_sampled_fake_y)) = disc_predictions

            wandb.log({'training_metrics/gen_g_loss': gen_g_loss.numpy(),
                       'training_metrics/gen_f_loss': gen_f_loss.numpy(),
                       'training_metrics/cycle_loss_x': cycle_loss_x.numpy(),
                       'training_metrics/cycle_loss_y': cycle_loss_y.numpy(),
                       'training_metrics/identity_loss_x': identity_loss_x.numpy(),
                       'training_metrics/identity_loss_y': identity_loss_y.numpy(),

                       'training_metrics/disc_x_loss': disc_x_loss.numpy(),
                       'training_metrics/disc_y_loss': disc_y_loss.numpy(),
                       'training_metrics/disc_x_gp': disc_x_gp.numpy(),
                       'training_metrics/disc_y_gp': disc_y_gp.numpy(),

                       'predictions/disc_real_x': wandb.Histogram(disc_real_x.numpy()),
                       'predictions/disc_real_y': wandb.Histogram(disc_real_y.numpy()),
                       'predictions/disc_fake_x': wandb.Histogram(disc_fake_x.numpy()),
                       'predictions/disc_fake_y': wandb.Histogram(disc_fake_y.numpy()),
                       'predictions/disc_sampled_fake_x': wandb.Histogram(disc_sampled_fake_x.numpy()),
                       'predictions/disc_sampled_fake_y': wandb.Histogram(disc_sampled_fake_y.numpy()),

                       'gradient_norms/generators': tf.linalg.global_norm(gen_gradients).numpy(),
                       'gradient_norms/discriminators': tf.linalg.global_norm(disc_gradients).numpy(),

                       'learning_rates/generators': generator_optimizer._decayed_lr(tf.float32).numpy(),
                       'learning_rates/discriminators': discriminator_optimizer._decayed_lr(tf.float32).numpy(),
                       'steps': step},
                      step=step)

            # Log images frequently to admire
            if step % image_log_freq == 0:
                # Use a (* 0.5 + 0.5) offset before visualizing since the data lies in [-1, 1]
                wandb.log({'real_x': wandb.Image(gallery(batch_x.numpy() * 0.5 + 0.5)),
                           'fake_x': wandb.Image(gallery(fake_x.numpy() * 0.5 + 0.5)),
                           'cycled_x': wandb.Image(gallery(cycled_x.numpy() * 0.5 + 0.5)),
                           'same_x': wandb.Image(gallery(same_x.numpy() * 0.5 + 0.5)),
                           'real_y': wandb.Image(gallery(batch_y.numpy() * 0.5 + 0.5)),
                           'fake_y': wandb.Image(gallery(fake_y.numpy() * 0.5 + 0.5)),
                           'cycled_y': wandb.Image(gallery(cycled_y.numpy() * 0.5 + 0.5)),
                           'same_y': wandb.Image(gallery(same_y.numpy() * 0.5 + 0.5))}, step=step)

                # Visualize a batch of validation data every epoch
                generate_and_log_one_image_batch(val_data_generator, generator_g, generator_f, step)

            del gen_losses, disc_losses, gen_predictions, disc_predictions, gen_gradients, disc_gradients

        # End of epoch, log to weights and biases
        wandb.log({'epochs': epoch + 1}, step=step)

        # Checkpoint every few epochs
        if (epoch + 1) % checkpoint_freq == 0:
            # Store the models
            generator_g.save_weights(f'{wandb.run.dir}/{checkpoint_path}/ckpt_{epoch + 1}_generator_g.h5')
            generator_f.save_weights(f'{wandb.run.dir}/{checkpoint_path}/ckpt_{epoch + 1}_generator_f.h5')
            discriminator_x.save_weights(f'{wandb.run.dir}/{checkpoint_path}/ckpt_{epoch + 1}_discriminator_x.h5')
            discriminator_y.save_weights(f'{wandb.run.dir}/{checkpoint_path}/ckpt_{epoch + 1}_discriminator_y.h5')
            # Store the optimizers
            save_tf_optimizer_state(generator_optimizer,
                                    f'{wandb.run.dir}/{checkpoint_path}/generator_optimizer_{epoch + 1}.pkl')
            save_tf_optimizer_state(discriminator_optimizer,
                                    f'{wandb.run.dir}/{checkpoint_path}/discriminator_optimizer_{epoch + 1}.pkl')
            # Save to Weights and Biases
            wandb.save(f'{wandb.run.dir}/{checkpoint_path}/ckpt_{epoch + 1}_generator_g.h5')
            wandb.save(f'{wandb.run.dir}/{checkpoint_path}/ckpt_{epoch + 1}_generator_f.h5')
            wandb.save(f'{wandb.run.dir}/{checkpoint_path}/ckpt_{epoch + 1}_discriminator_x.h5')
            wandb.save(f'{wandb.run.dir}/{checkpoint_path}/ckpt_{epoch + 1}_discriminator_y.h5')
            wandb.save(f'{wandb.run.dir}/{checkpoint_path}/generator_optimizer_{epoch + 1}.pkl')
            wandb.save(f'{wandb.run.dir}/{checkpoint_path}/discriminator_optimizer_{epoch + 1}.pkl')

    return generator_g, generator_f, discriminator_x, discriminator_y


def train_step_generator(generator_g, generator_f,
                         discriminator_x, discriminator_y,
                         loss_fn,
                         batch_x, batch_y,
                         generator_optimizer,
                         discriminator_x_replay, discriminator_y_replay,
                         cycle_loss_scale_x, cycle_loss_scale_y, identity_loss_scale):
    def _train_step_generator(real_x, real_y):
        with tf.GradientTape() as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = generator_g(real_x, training=True)
            cycled_x = generator_f(fake_y, training=True)

            fake_x = generator_f(real_y, training=True)
            cycled_y = generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = generator_f(real_x, training=True)
            same_y = generator_g(real_y, training=True)

            disc_fake_x = discriminator_x(fake_x, training=True)
            disc_fake_y = discriminator_y(fake_y, training=True)

            # Calculate all the losses
            gen_g_loss = generator_loss(disc_fake_y, loss_fn)
            gen_f_loss = generator_loss(disc_fake_x, loss_fn)

            cycle_loss_x = cycle_loss(real_x, cycled_x, cycle_loss_scale_x)
            cycle_loss_y = cycle_loss(real_y, cycled_y, cycle_loss_scale_y)

            identity_loss_x = identity_loss(real_x, same_x, identity_loss_scale)
            identity_loss_y = identity_loss(real_y, same_y, identity_loss_scale)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_loss = gen_g_loss + gen_f_loss + cycle_loss_x + cycle_loss_y + identity_loss_x + identity_loss_y

        # Update the discriminator replay buffers
        discriminator_x_replay.add(fake_x)
        discriminator_y_replay.add(fake_y)

        # Calculate the gradients for generator and discriminator
        generator_gradients = tape.gradient(total_gen_loss,
                                            generator_g.trainable_variables + generator_f.trainable_variables)

        # Apply the gradients to the optimizer
        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator_g.trainable_variables + generator_f.trainable_variables))

        del tape

        return (gen_g_loss, gen_f_loss, cycle_loss_x, cycle_loss_y, identity_loss_x, identity_loss_y), \
               ((same_x, fake_x, cycled_x, disc_fake_x),
                (same_y, fake_y, cycled_y, disc_fake_y)), generator_gradients

    return _train_step_generator(batch_x, batch_y)


def train_step_discriminator(discriminator_x, discriminator_y,
                             loss_fn,
                             batch_x, batch_y,
                             discriminator_optimizer,
                             discriminator_x_replay, discriminator_y_replay,
                             grad_penalty, grad_penalty_scale):
    def _train_step_discriminator(real_x, real_y):
        # Sample fake_x and fake_y from the replay buffers
        sampled_fake_x = discriminator_x_replay.get_tf_batch(real_x.shape[0])
        sampled_fake_y = discriminator_y_replay.get_tf_batch(real_y.shape[0])

        with tf.GradientTape() as tape:
            disc_real_x = discriminator_x(real_x, training=True)
            disc_real_y = discriminator_y(real_y, training=True)

            disc_fake_x = discriminator_x(sampled_fake_x, training=True)
            disc_fake_y = discriminator_y(sampled_fake_y, training=True)

            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x, loss_fn)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y, loss_fn)

            disc_x_gp = gradient_penalty(functools.partial(discriminator_x, training=True),
                                         real_x, sampled_fake_x, mode=grad_penalty, scale=grad_penalty_scale)
            disc_y_gp = gradient_penalty(functools.partial(discriminator_y, training=True),
                                         real_y, sampled_fake_y, mode=grad_penalty, scale=grad_penalty_scale)

            total_disc_loss = disc_x_loss + disc_y_loss + disc_x_gp + disc_y_gp

        # Calculate the gradients for generator and discriminator
        discriminator_gradients = tape.gradient(total_disc_loss,
                                                discriminator_x.trainable_variables +
                                                discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator_x.trainable_variables +
                                                    discriminator_y.trainable_variables))

        del tape

        return (disc_x_loss, disc_y_loss, disc_x_gp, disc_y_gp), \
               ((disc_real_x, disc_fake_x),
                (disc_real_y, disc_fake_y)), discriminator_gradients

    return _train_step_discriminator(batch_x, batch_y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file.')
    parser.add_argument('--template', type=str, default='augmentation/configs/template_cyclegan_training.yaml')

    args = parser.parse_args()

    # Load up the config files
    config = create_config_simple_namespace(config_path=args.config, template_config_path=args.template)

    # Train the end model
    train_cyclegan(config)
