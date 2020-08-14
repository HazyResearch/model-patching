import wandb
import json
import time
import numpy as np
from collections import namedtuple
from augmentation.methods.cyclegan.models import mnist_unet_generator, unet_generator
from augmentation.models.models import create_keras_classification_model

WandbRun = namedtuple('WandbRun', 'path id name history files cfg url')


def particular_checkpoint_step_extractor(checkpoint, step_extractor=lambda fname: fname.split("_")[1].split(".")[0]):
    def particular_checkpoint_step_extractor_(filename):
        step = int(step_extractor(filename))
        if step == checkpoint:
            return step
        else:
            return 0

    if checkpoint > 0:
        return particular_checkpoint_step_extractor_
    return step_extractor


def fetch_all_wandb_run_ids(wandb_project, wandb_entity='hazy-research', wandb_api=None):
    if wandb_api is None:
        wandb_api = wandb.Api()
    wandb_path = f'{wandb_entity}/{wandb_project}/*'
    runs = wandb_api.runs(wandb_path)
    return [run.id for run in runs]


def load_wandb_run(wandb_run_id, wandb_project, wandb_entity='hazy-research', wandb_api=None):
    if wandb_api is None:
        wandb_api = wandb.Api()
    wandb_path = f'{wandb_entity}/{wandb_project}/{wandb_run_id}'
    run = wandb_api.run(wandb_path)
    return WandbRun(path=wandb_path, id=run.id, name=run.name, history=run.scan_history,
                    files=run.files(per_page=10000), cfg=json.loads(run.json_config), url=run.url)


def get_most_recent_model_file(wandb_run: WandbRun, wandb_ckpt_path='checkpoints/',
                               model_name='', exclude=None,
                               step_extractor=lambda fname: fname.split("_")[1].split(".")[0]):
    # Find checkpoints
    checkpoints = [file for file in wandb_run.files if file.name.startswith(wandb_ckpt_path.lstrip("/"))]
    relevant_checkpoints = [e for e in checkpoints if model_name in e.name]
    if exclude:
        relevant_checkpoints = [e for e in relevant_checkpoints if exclude not in e.name]
    # Grab the latest checkpoint
    latest_checkpoint = relevant_checkpoints[np.argmax([int(step_extractor(e.name)) for e in relevant_checkpoints])]
    print(f"Retrieved checkpoint {latest_checkpoint.name}.")
    # Restore the model
    model_file = wandb.restore(latest_checkpoint.name, run_path=wandb_run.path, replace=True)

    return model_file


def load_most_recent_keras_model_weights(keras_model,
                                         wandb_run,
                                         wandb_ckpt_path='checkpoints/',
                                         model_name='',
                                         exclude=None,
                                         step_extractor=None):
    # Make sure the step extractor is set to a reasonable default
    if step_extractor is None:
        step_extractor = lambda fname: fname.split(".")[-2].split("_")[-1]

    # Get the most recent model file and load weights from it
    try:
        model_file = get_most_recent_model_file(wandb_run, wandb_ckpt_path, model_name, exclude, step_extractor)
        time.sleep(3)
        keras_model.load_weights(model_file.name)
        print('load_most_recent_keras_model_weights: file ', model_file.name)
        try:
            return model_file.name, int(step_extractor(model_file.name))
        except ValueError:
            return model_file.name, int(step_extractor(model_file.name.split("/")[-1]))
    except ValueError:
        print("No model file found. Continuing without loading..")

    return None, None


def load_pretrained_keras_model_from_wandb(wandb_run_id, wandb_project, wandb_entity,
                                           keras_model_creation_fn, keras_model_creation_fn_args,
                                           model_name, step_extractor,
                                           wandb_ckpt_path='checkpoints/'):
    # Load the run
    wandb_run = load_wandb_run(wandb_run_id, wandb_project, wandb_entity)

    # Create the model architecture
    keras_model = globals()[keras_model_creation_fn](**keras_model_creation_fn_args)

    # Load up the model weights
    if step_extractor is None:
        load_file, load_step = load_most_recent_keras_model_weights(keras_model, wandb_run,
                                                                    model_name=model_name,
                                                                    wandb_ckpt_path=wandb_ckpt_path)
    else:
        load_file, load_step = load_most_recent_keras_model_weights(keras_model, wandb_run,
                                                                    model_name=model_name,
                                                                    step_extractor=step_extractor,
                                                                    wandb_ckpt_path=wandb_ckpt_path)

    return keras_model, (load_file, load_step)


def load_pretrained_keras_classification_model(source, architecture, input_shape, n_classes, imagenet_pretrained,
                                               pretraining_source, pretraining_info, checkpoint_path):
    # Create the model
    model = create_keras_classification_model(source, architecture, input_shape, n_classes, imagenet_pretrained)

    if pretraining_source == 'wandb':
        # Extract the Weights and Biases run information
        run_id, project, entity = pretraining_info.split(":")
        # Load up the relevant run
        wandb_run = load_wandb_run(run_id, project, entity)
        # Load up the most recent checkpoint from that run
        load_most_recent_keras_model_weights(model, wandb_run, checkpoint_path)

    return model
