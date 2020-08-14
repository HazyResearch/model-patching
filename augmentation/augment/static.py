from augmentation.augment.utils import WandbModelPseudoLabelingPipeline, BinaryMNISTWandbModelPseudoLabelingPipeline, \
    PretrainedMNISTCycleGANAugmentationPipeline, ResizeImage
from augmentation.utilities.wandb import load_pretrained_keras_model_from_wandb, particular_checkpoint_step_extractor, \
    load_wandb_run, get_most_recent_model_file
from augmentation.datasets.custom.tfrecords import image_label_to_tfrecord, read_image_label_tfrecord
import augmentation.datasets.utils

import numpy as np
import tensorflow as tf
import os
from copy import copy


def split_batch_size(total_batch_size, n_groups):
    return [total_batch_size // n_groups] * (n_groups - 1) + [
        total_batch_size // n_groups + total_batch_size % n_groups]


def compose_static_augmentations(static_augmentation_pipelines, datasets, aliases, identifiers, dataset_lens,
                                 batch_sizes,
                                 keep_datasets=False):
    print(
        f"Composing static augmentations:\ndatasets - {datasets}\naliases - {aliases}\nlengths - {dataset_lens}\nbatch sizes - {batch_sizes}\nidentifiers - {identifiers}\nstatic_aug_pipelines - {static_augmentation_pipelines}",
        flush=True)
    assert len(static_augmentation_pipelines) == len(datasets) == len(aliases) == len(dataset_lens) == len(
        batch_sizes) == len(identifiers), "compose_static_augmentations: lengths of arguments should be equal"

    datasets = list(datasets)
    original_idx = list(range(len(datasets)))

    all_datasets = []
    all_aliases = []
    all_dataset_lens = []
    all_batch_sizes = []
    all_original_idx = []

    # Loop over all the datasets and their corresponding static augmentations
    for dataset, alias, ident, dlen, batch_size, idx, aug_pipeline \
            in zip(datasets, aliases, identifiers, dataset_lens, batch_sizes, original_idx,
                   static_augmentation_pipelines):

        dataset, alias, dlen, batch_size, idx = [dataset], [alias], [dlen], [batch_size], [idx]

        # Run the dataset through the augmentation pipeline
        for augmentation in aug_pipeline:
            # Append the augmented datasets
            if keep_datasets:
                all_datasets += list(dataset)
                all_aliases += alias
                all_dataset_lens += dlen
                all_batch_sizes += batch_size
                all_original_idx += idx

            # Call the static augmentation
            dataset, alias, dlen, batch_size, idx = augmentation(dataset, alias, dlen, batch_size, idx,
                                                                 **{'dataset_identifier': ident})

        all_datasets += list(dataset)
        all_aliases += alias
        all_dataset_lens += dlen
        all_batch_sizes += batch_size
        all_original_idx += idx

    return all_datasets, all_aliases, all_dataset_lens, all_batch_sizes, all_original_idx


def create_static_augmentation_pipeline(daug_pipeline, daug_pipeline_args, broadcast_to=1):
    """Takes as input an augmentation pipeline: a list of strings where each string is an augmentation. Their
    corresponding arguments are in daug_pipeline_args."""
    # Setup the augmentation pipeline we'll be using
    if broadcast_to > 1:
        # If broadcasting, return a list of augmentation pipelines (rather than a single augmentation pipeline)
        # by replication
        return [[globals()[daug](*daug_args) for daug, daug_args in zip(daug_pipeline, daug_pipeline_args)]] \
               * broadcast_to

    # By default, just return a single augmentation pipeline
    return [globals()[daug](*daug_args) for daug, daug_args in zip(daug_pipeline, daug_pipeline_args)]


def create_static_augmentation_pipelines(train_daug_pipeline, train_daug_pipeline_args,
                                         val_daug_pipeline, val_daug_pipeline_args,
                                         test_daug_pipeline, test_daug_pipeline_args):
    # Setup the augmentation pipeline we'll be using
    train_augmentations = create_static_augmentation_pipeline(train_daug_pipeline, train_daug_pipeline_args)
    val_augmentations = create_static_augmentation_pipeline(val_daug_pipeline, val_daug_pipeline_args)
    test_augmentations = create_static_augmentation_pipeline(test_daug_pipeline, test_daug_pipeline_args)

    return train_augmentations, val_augmentations, test_augmentations


def create_multiple_train_eval_static_augmentation_pipelines(train_augmentation_pipelines,
                                                             train_augmentation_pipelines_args,
                                                             eval_augmentation_pipelines,
                                                             eval_augmentation_pipelines_args,
                                                             broadcast_train_to=1,
                                                             broadcast_eval_to=1):
    assert len(train_augmentation_pipelines) == len(train_augmentation_pipelines_args) and \
           len(eval_augmentation_pipelines) == len(eval_augmentation_pipelines_args), \
        'Number of pipelines and args must be the same.'

    # Find the number of pipelines
    n_train_pipelines = len(train_augmentation_pipelines)
    n_eval_pipelines = len(eval_augmentation_pipelines)

    if n_train_pipelines == 0:
        # No train augmentation, push in an empty list to handle this properly
        train_augmentation_pipelines, train_augmentation_pipelines_args = [[]], [[]]

    if n_eval_pipelines == 0:
        # No eval augmentation, push in an empty list to handle this properly
        eval_augmentation_pipelines, eval_augmentation_pipelines_args = [[]], [[]]

    # 'Broadcast' the single pipeline and replicate it broadcast_to times (otherwise don't)
    broadcast_train_to = broadcast_train_to if (n_train_pipelines <= 1 and broadcast_train_to > 1) else 1
    broadcast_eval_to = broadcast_eval_to if (n_eval_pipelines <= 1 and broadcast_eval_to > 1) else 1

    # Standard stuff, just create the pipelines and return them
    train_augmentations = [
        (create_static_augmentation_pipeline(*z))
        for z in zip(train_augmentation_pipelines * broadcast_train_to,
                     train_augmentation_pipelines_args * broadcast_train_to)
    ]

    eval_augmentations = [
        (create_static_augmentation_pipeline(*z))
        for z in zip(eval_augmentation_pipelines * broadcast_eval_to,
                     eval_augmentation_pipelines_args * broadcast_eval_to)
    ]
    return train_augmentations, eval_augmentations


def create_multiple_static_augmentation_pipelines(train_daug_pipelines, train_daug_pipelines_args,
                                                  val_daug_pipelines, val_daug_pipelines_args,
                                                  test_daug_pipelines, test_daug_pipelines_args,
                                                  broadcast_to=1):
    """
    Same as create_augmentation_pipelines but takes list of pipelines each
    and returns lists of same length.
    'Broadcast' to pass in a single pipeline and get k replicates.
    """

    assert len(train_daug_pipelines) == len(train_daug_pipelines_args) and \
           len(val_daug_pipelines) == len(val_daug_pipelines_args) and \
           len(test_daug_pipelines) == len(test_daug_pipelines_args), 'Number of pipelines and args must be the same.'

    # Find the number of pipelines
    n_train_pipelines = len(train_daug_pipelines)
    n_val_pipelines = len(val_daug_pipelines)
    n_test_pipelines = len(test_daug_pipelines)

    if n_train_pipelines == 0:
        # No augmentation, push in an empty list to handle this properly
        train_daug_pipelines, train_daug_pipelines_args = [[]], [[]]
        val_daug_pipelines, val_daug_pipelines_args = [[]], [[]]
        test_daug_pipelines, test_daug_pipelines_args = [[]], [[]]

    # 'Broadcast' the single pipeline and replicate it broadcast_to times (otherwise don't)
    broadcast_train_to = broadcast_to if (n_train_pipelines <= 1 and broadcast_to > 1) else 1
    broadcast_val_to = broadcast_to if (n_val_pipelines <= 1 and broadcast_to > 1) else 1
    broadcast_test_to = broadcast_to if (n_test_pipelines <= 1 and broadcast_to > 1) else 1

    # Standard stuff, just create the pipelines and return them
    augmentations = [
        create_static_augmentation_pipelines(*z)
        for z in zip(train_daug_pipelines * broadcast_train_to,
                     train_daug_pipelines_args * broadcast_train_to,
                     val_daug_pipelines * broadcast_val_to,
                     val_daug_pipelines_args * broadcast_val_to,
                     test_daug_pipelines * broadcast_test_to,
                     test_daug_pipelines_args * broadcast_test_to)
    ]
    return tuple(zip(*augmentations))


class StaticAugmentation:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, datasets, aliases, dataset_lens, batch_sizes, original_idx, *args, **kwargs):
        """
        Returns list of lists, one copy for each pseudolabel.
        original_idx: for each dataset, specifies which of the "original" datasets (in the config)
        batch_sizes: batch size for each dataset
          it was generated from. Useful for broadcasting other arguments
        """

        updated_datasets, updated_aliases, updated_lens, updated_batch_sizes, updated_idx = [], [], [], [], []
        for dataset, alias, dataset_len, batch_size, idx \
                in zip(datasets, aliases, dataset_lens, batch_sizes, original_idx):
            # Apply the transform to the dataset
            datasets_, aliases_, lens_, batch_sizes_ = self.transform(dataset, alias, dataset_len, batch_size,
                                                                      *args, **kwargs)

            # Keep track of things
            updated_datasets += datasets_
            updated_aliases += aliases_
            updated_lens += lens_
            updated_batch_sizes += batch_sizes_
            k = len(datasets_)
            updated_idx += [idx] * k

        return updated_datasets, updated_aliases, updated_lens, updated_batch_sizes, updated_idx

    def transform(self, dataset, alias, dataset_len, batch_size, *args, **kwargs):
        """ Takes a dataset and returns a list of datasets and alias [suffixes] """
        # """ This must take a list of datasets and aliases and returns updated lists """
        raise NotImplementedError

class PretrainedExternalGANStaticAugmentationTFRecordPipeline(StaticAugmentation):

    def __init__(self,
                 store_path,
                 gan_name,
                 version,
                 relabel=False,
                 keep_original=False,
                 shard_size=10240,
                 overwrite=False,
                 *args, **kwargs):
        super(PretrainedExternalGANStaticAugmentationTFRecordPipeline, self).__init__(*args, **kwargs)

        assert relabel is False, 'Relabeling not supported.'

        # Base path for location of the TFRecords
        self.base_store_path = store_path
        # Name of the GAN model that was used to dump augmentations
        self.gan_name = gan_name
        # Version of the dump from the GAN model that we're using
        self.version = version
        # Prefix for the folder where the TFRecords will be stored
        self.filename_prefix = f'gan[{gan_name}].version[{version}]'
        # Size of the TFRecord shard
        self.shard_size = shard_size
        # Keep the original dataset
        self.keep_original = keep_original
        # If the TFRecords were previously dumped, overwrite them
        assert not overwrite, 'Overwriting is not yet implemented.'
        self.overwrite = overwrite

        # Set the batch size to 1: a batch size larger than 1 is not supported for this class
        self.batch_size = 1

    def transform(self, dataset, alias, dataset_len, batch_size, *args, **kwargs):
        assert 'dataset_identifier' in kwargs, 'Please pass in a unique identifier for the dataset.'
        # Specific paths for the TFRecords
        dataset_identifier = kwargs['dataset_identifier'].replace("/", ".")
        gen_f_store_path = os.path.join(self.base_store_path, dataset_identifier,
                                        self.filename_prefix + f'.model[gen_f]')
        gen_g_store_path = os.path.join(self.base_store_path, dataset_identifier,
                                        self.filename_prefix + f'.model[gen_g]')

        # Specific paths from which we're loading the pre-cached outputs of the external GAN that was already run
        gen_f_dump_path = os.path.join(self.base_store_path, 'external', self.gan_name,
                                       dataset_identifier, f'gen_f_v{self.version}.npy')
        gen_g_dump_path = os.path.join(self.base_store_path, 'external', self.gan_name,
                                       dataset_identifier, f'gen_g_v{self.version}.npy')

        print(os.path.exists(gen_f_dump_path), os.path.exists(gen_g_dump_path))
        print(os.path.exists(gen_f_store_path), os.path.exists(gen_g_store_path))

        if not os.path.exists(gen_f_store_path) and not os.path.exists(gen_g_store_path):
            # Write the TF Records to disk
            os.makedirs(gen_f_store_path, exist_ok=True)
            os.makedirs(gen_g_store_path, exist_ok=True)
            lockfile = os.path.join(gen_f_store_path, 'writer.lock')

            # Try to procure a lock to dump TF Records: if it already exists, wait until it is released to continue
            someone_has_lock = False
            while True:
                if not os.path.exists(lockfile) and not someone_has_lock:
                    # Nobody has the lock: create the lock
                    open(lockfile, 'w')
                    # Write the TFRecords
                    self.dump_tf_records(dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE),
                                         gen_f_store_path, gen_g_store_path,
                                         gen_f_dump_path, gen_g_dump_path)
                    # Release the lock
                    os.remove(lockfile)
                    # Break out
                    break
                elif not os.path.exists(lockfile) and someone_has_lock:
                    # The lock was released and the TFRecords are available to read
                    break
                elif os.path.exists(lockfile):
                    # Another run is writing the TFRecords, so wait around until they're done
                    someone_has_lock = True

        # Load up the TF Records datasets
        dataset_f, dataset_g = self.build_tf_datasets(gen_f_store_path, gen_g_store_path)

        alias_f = alias + '(A-F)'
        alias_g = alias + '(A-G)'
        if self.keep_original:
            return [dataset, dataset_f, dataset_g], [alias, alias_f, alias_g], [dataset_len] * 3, [batch_size] * 3
        else:
            return [dataset_f, dataset_g], [alias_f, alias_g], [dataset_len] * 2, [batch_size] * 2

    def dump_tf_records(self, dataset, gen_f_store_path, gen_g_store_path, gen_f_dump_path, gen_g_dump_path):
        # Take a dataset and write TFRecords after loading the pre-dumped outputs from an external CycleGAN-like model
        print(f"Writing TFRecords with shard size {self.shard_size} at \n{gen_f_store_path}\nand\n{gen_g_store_path}.")

        curr_shard = -1
        shard_progress = 10 ** 10
        assert self.shard_size < 10 ** 10

        # Load up the dataset from disk
        gen_f_images = np.load(gen_f_dump_path)
        gen_g_images = np.load(gen_g_dump_path)

        # Initialize the TFRecords file writers
        f_out_file, g_out_file = None, None

        # Assume dataset contains (image, label) pairs and iterate over it
        for i, (image, label) in enumerate(dataset):
            # Check if the current shard needs to be incremented
            if shard_progress >= self.shard_size:
                # Update the current shard
                curr_shard += 1
                shard_progress = 0
                # Get the new filenames
                f_filename = os.path.join(gen_f_store_path, "{:02d}-{}.tfrec".format(curr_shard, self.shard_size))
                g_filename = os.path.join(gen_g_store_path, "{:02d}-{}.tfrec".format(curr_shard, self.shard_size))
                # Open up the new files
                f_out_file = tf.io.TFRecordWriter(f_filename)
                g_out_file = tf.io.TFRecordWriter(g_filename)
                print(f"Opened files {f_filename} and {g_filename}.")

            # Grab the batch size for the current batch: this will be 1
            batch_size = image.numpy().shape[0]

            # Run the image batch through the generators
            f_image = gen_f_images[i:i + 1]
            g_image = gen_g_images[i:i + 1]

            # Encode the images to JPEG for storage
            f_image = tf.convert_to_tensor(
                [tf.image.encode_jpeg(im, optimize_size=True, chroma_downsampling=False) for im in f_image])
            g_image = tf.convert_to_tensor(
                [tf.image.encode_jpeg(im, optimize_size=True, chroma_downsampling=False) for im in g_image])

            # Iterate over the batch of data
            for i in range(batch_size):
                if isinstance(label, tuple):
                    ith_labels = nested_map(lambda e: int(e.numpy()), list(zip(*label))[i])
                else:
                    ith_labels = int(label.numpy()[i])

                try:
                    f_example = image_label_to_tfrecord(f_image.numpy()[i], ith_labels)
                    g_example = image_label_to_tfrecord(g_image.numpy()[i], ith_labels)
                except IndexError:
                    continue
                f_out_file.write(f_example.SerializeToString())
                g_out_file.write(g_example.SerializeToString())

            print(f"\tShard progress: {shard_progress}/{self.shard_size}")
            shard_progress += batch_size

    def build_tf_datasets(self, gen_f_store_path, gen_g_store_path):
        # Load up the files for the CycleGAN-ed dataset
        gen_f_store_path = gen_f_store_path.replace("[", "\[").replace("]", "\]").replace("*", "\*")
        gen_g_store_path = gen_g_store_path.replace("[", "\[").replace("]", "\]").replace("*", "\*")

        gen_f_dataset = tf.data.Dataset.list_files(os.path.join(gen_f_store_path, '*.tfrec'), shuffle=False)
        gen_g_dataset = tf.data.Dataset.list_files(os.path.join(gen_g_store_path, '*.tfrec'), shuffle=False)

        # Load up the TFRecords datasets
        gen_f_dataset = augmentation.datasets.utils. \
            get_dataset_from_list_files_dataset(gen_f_dataset, proc_batch=128,
                                                tfrecord_example_reader=read_image_label_tfrecord,
                                                sequential=True).unbatch()
        gen_g_dataset = augmentation.datasets.utils. \
            get_dataset_from_list_files_dataset(gen_g_dataset, proc_batch=128,
                                                tfrecord_example_reader=read_image_label_tfrecord,
                                                sequential=True).unbatch()

        gen_f_dataset, gen_g_dataset = self.remap_tfdataset(gen_f_dataset, gen_g_dataset)

        return gen_f_dataset, gen_g_dataset

    def remap_tfdataset(self, gen_f_dataset, gen_g_dataset):
        for img, label in gen_f_dataset:
            if not isinstance(label, tuple):
                if len(label.shape) == 0:
                    return gen_f_dataset, gen_g_dataset
                else:
                    # TODO: generalize to more than 2 labels
                    return gen_f_dataset.map(lambda im, lab: (im, (lab[0], lab[1]))), \
                           gen_g_dataset.map(lambda im, lab: (im, (lab[0], lab[1])))
            else:
                return gen_f_dataset, gen_g_dataset


class PretrainedCycleGANStaticAugmentationTFRecordPipeline(StaticAugmentation):

    def __init__(self,
                 store_path,
                 wandb_entity,
                 wandb_project,
                 wandb_run_id,
                 run_in_eval_mode=False,
                 input_shape=(256, 256, 3),
                 norm_type='batchnorm',
                 checkpoint_step=-1,
                 relabel=False,
                 wandb_ckpt_path='checkpoints/',
                 batch_size=1,
                 keep_original=False,
                 shard_size=10240,
                 overwrite=False,
                 *args, **kwargs):
        super(PretrainedCycleGANStaticAugmentationTFRecordPipeline, self).__init__(*args, **kwargs)

        assert relabel is False, 'Relabeling not supported.'

        # Since checkpoint_step can be -1, figure out the actual load steps: this is separately checked
        # to avoid instantiating the CycleGAN models unless they're needed for dumping the data.
        f_load_step, g_load_step = self.get_load_epochs(wandb_run_id,
                                                        wandb_project,
                                                        wandb_entity,
                                                        wandb_ckpt_path,
                                                        checkpoint_step)

        # Instantiate the CycleGAN pipeline but don't load the models yet
        self.cyclegan = PretrainedDefaultCycleGANStaticAugmentationPipeline(
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_run_id=wandb_run_id,
            run_in_eval_mode=run_in_eval_mode,
            input_shape=input_shape,
            norm_type=norm_type,
            checkpoint_step=checkpoint_step,
            relabel=relabel,
            wandb_ckpt_path=wandb_ckpt_path,
            batch_size=batch_size,
            keep_original=keep_original,
            load_immediately=False,  # don't load the models
            *args, **kwargs)

        # Base path for location of the TFRecords
        self.base_store_path = store_path
        # Prefix for the folder where the TFRecords will be stored
        self.filename_prefix = f'wandb[{wandb_entity}:{wandb_project}:{wandb_run_id}].' \
                               f'f_epoch[{f_load_step}].g_epoch[{g_load_step}].' \
                               f'mode[{run_in_eval_mode}].batch[{batch_size}]'
        # Size of the TFRecord shard
        self.shard_size = shard_size
        # Batch size for the CycleGAN augmentation (important for speed, batchnorm's behavior)
        self.batch_size = batch_size
        # Keep the original dataset
        self.keep_original = keep_original
        # If the TFRecords were previously dumped, overwrite them
        assert not overwrite, 'Overwriting is not yet implemented.'
        self.overwrite = overwrite

    def get_load_epochs(self, wandb_run_id, wandb_project, wandb_entity, wandb_ckpt_path, checkpoint_step):
        # Create a function for doing step extraction for CycleGAN generator models
        step_extractor = particular_checkpoint_step_extractor(checkpoint_step)
        f_model_file = get_most_recent_model_file(wandb_run=load_wandb_run(wandb_run_id, wandb_project, wandb_entity),
                                                  wandb_ckpt_path=wandb_ckpt_path,
                                                  model_name='generator_f',
                                                  step_extractor=step_extractor)

        g_model_file = get_most_recent_model_file(wandb_run=load_wandb_run(wandb_run_id, wandb_project, wandb_entity),
                                                  wandb_ckpt_path=wandb_ckpt_path,
                                                  model_name='generator_g',
                                                  step_extractor=step_extractor)

        return step_extractor(f_model_file.name.split("/")[-1]), \
               step_extractor(g_model_file.name.split("/")[-1])

    def transform(self, dataset, alias, dataset_len, batch_size, *args, **kwargs):
        assert 'dataset_identifier' in kwargs, 'Please pass in a unique identifier for the dataset.'
        # Specific paths for the TFRecords
        dataset_identifier = kwargs['dataset_identifier'].replace("/", ".")
        gen_f_store_path = os.path.join(self.base_store_path, dataset_identifier,
                                        self.filename_prefix + f'.model[gen_f]')
        gen_g_store_path = os.path.join(self.base_store_path, dataset_identifier,
                                        self.filename_prefix + f'.model[gen_g]')

        print(os.path.exists(gen_f_store_path), os.path.exists(gen_g_store_path))

        if not os.path.exists(gen_f_store_path) and not os.path.exists(gen_g_store_path):
            # Write the TF Records to disk
            os.makedirs(gen_f_store_path, exist_ok=True)
            os.makedirs(gen_g_store_path, exist_ok=True)
            lockfile = os.path.join(gen_f_store_path, 'writer.lock')

            # Try to procure a lock to dump TF Records: if it already exists, wait until it is released to continue
            someone_has_lock = False
            while True:
                if not os.path.exists(lockfile) and not someone_has_lock:
                    # Nobody has the lock: create the lock
                    open(lockfile, 'w')
                    # Write the TFRecords
                    self.dump_tf_records(dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE),
                                         gen_f_store_path, gen_g_store_path)
                    # Release the lock
                    os.remove(lockfile)
                    # Break out
                    break
                elif not os.path.exists(lockfile) and someone_has_lock:
                    # The lock was released and the TFRecords are available to read
                    break
                elif os.path.exists(lockfile):
                    # Another run is writing the TFRecords, so wait around until they're done
                    someone_has_lock = True

        # Don't delete the CycleGAN model if you're doing this on the val set, because the test set will reuse it
        if not 'val' in dataset_identifier:
            del self.cyclegan
        # Load up the TF Records datasets
        dataset_f, dataset_g = self.build_tf_datasets(gen_f_store_path, gen_g_store_path)

        alias_f = alias + '(A-F)'
        alias_g = alias + '(A-G)'
        if self.keep_original:
            return [dataset, dataset_f, dataset_g], [alias, alias_f, alias_g], [dataset_len] * 3, [batch_size] * 3
        else:
            return [dataset_f, dataset_g], [alias_f, alias_g], [dataset_len] * 2, [batch_size] * 2

    def dump_tf_records(self, dataset, gen_f_store_path, gen_g_store_path):
        # Load up the CycleGAN models
        self.cyclegan.load_models()

        # Take a dataset and write TFRecords after passing through both CycleGAN generators
        print(f"Writing TFRecords with shard size {self.shard_size} at \n{gen_f_store_path}\nand\n{gen_g_store_path}.")

        curr_shard = -1
        shard_progress = 10 ** 10
        assert self.shard_size < 10 ** 10

        # Initialize the TFRecords file writers
        f_out_file, g_out_file = None, None

        # Assume dataset contains (image, label) pairs and iterate over it
        for image, label in dataset:
            # Check if the current shard needs to be incremented
            if shard_progress >= self.shard_size:
                # Update the current shard
                curr_shard += 1
                shard_progress = 0
                # Get the new filenames
                f_filename = os.path.join(gen_f_store_path, "{:02d}-{}.tfrec".format(curr_shard, self.shard_size))
                g_filename = os.path.join(gen_g_store_path, "{:02d}-{}.tfrec".format(curr_shard, self.shard_size))
                # Open up the new files
                f_out_file = tf.io.TFRecordWriter(f_filename)
                g_out_file = tf.io.TFRecordWriter(g_filename)
                print(f"Opened files {f_filename} and {g_filename}.")

            # Grab the batch size for the current batch
            batch_size = image.numpy().shape[0]

            # Run the image batch through the generators
            f_image, _ = self.cyclegan.map_fn_f(image=image, label=None)
            g_image, _ = self.cyclegan.map_fn_g(image=image, label=None)

            # Encode the images to JPEG for storage
            f_image = tf.convert_to_tensor(
                [tf.image.encode_jpeg(im, optimize_size=True, chroma_downsampling=False) for im in f_image])
            g_image = tf.convert_to_tensor(
                [tf.image.encode_jpeg(im, optimize_size=True, chroma_downsampling=False) for im in g_image])

            # Iterate over the batch of data
            for i in range(batch_size):
                try:
                    f_example = image_label_to_tfrecord(f_image.numpy()[i], label.numpy()[i])
                    g_example = image_label_to_tfrecord(g_image.numpy()[i], label.numpy()[i])
                except IndexError:
                    continue
                f_out_file.write(f_example.SerializeToString())
                g_out_file.write(g_example.SerializeToString())

            print(f"\tShard progress: {shard_progress}/{self.shard_size}")
            shard_progress += batch_size

    def build_tf_datasets(self, gen_f_store_path, gen_g_store_path):
        # Load up the files for the CycleGAN-ed dataset
        gen_f_store_path = gen_f_store_path.replace("[", "\[").replace("]", "\]").replace("*", "\*")
        gen_g_store_path = gen_g_store_path.replace("[", "\[").replace("]", "\]").replace("*", "\*")

        gen_f_dataset = tf.data.Dataset.list_files(os.path.join(gen_f_store_path, '*.tfrec'), shuffle=False)
        gen_g_dataset = tf.data.Dataset.list_files(os.path.join(gen_g_store_path, '*.tfrec'), shuffle=False)

        # Load up the TFRecords datasets
        gen_f_dataset = augmentation.datasets.utils. \
            get_dataset_from_list_files_dataset(gen_f_dataset, proc_batch=128,
                                                tfrecord_example_reader=read_image_label_tfrecord,
                                                sequential=True).unbatch()
        gen_g_dataset = augmentation.datasets.utils. \
            get_dataset_from_list_files_dataset(gen_g_dataset, proc_batch=128,
                                                tfrecord_example_reader=read_image_label_tfrecord,
                                                sequential=True).unbatch()

        return gen_f_dataset, gen_g_dataset


class PretrainedCycleGANStaticAugmentationPipeline(StaticAugmentation):

    def __init__(self,
                 wandb_entity,
                 wandb_project,
                 wandb_run_id,
                 keras_model_creation_fn,
                 keras_model_creation_fn_args,
                 step_extractor=None,
                 run_in_eval_mode=True,
                 relabel=False,
                 wandb_ckpt_path='checkpoints/',
                 batch_size=1,
                 keep_original=False,
                 load_immediately=True,
                 *args, **kwargs):
        super(PretrainedCycleGANStaticAugmentationPipeline, self).__init__(*args, **kwargs)

        # Store the parameters passed in
        self.wandb_entity, self.wandb_project, self.wandb_run_id = wandb_entity, wandb_project, wandb_run_id
        self.wandb_ckpt_path = wandb_ckpt_path
        self.keras_model_creation_fn, self.keras_model_creation_fn_args = \
            keras_model_creation_fn, keras_model_creation_fn_args
        self.step_extractor = step_extractor
        self.run_in_eval_mode = run_in_eval_mode
        self.training = not run_in_eval_mode
        self.relabel = relabel
        self.batch_size = batch_size
        self.keep_original = keep_original

        self.models_loaded = False
        if load_immediately:
            self.load_models()

    def load_models(self):
        self.generator_f, (_, self.f_load_step) = load_pretrained_keras_model_from_wandb(
            wandb_run_id=self.wandb_run_id,
            wandb_project=self.wandb_project,
            wandb_entity=self.wandb_entity,
            keras_model_creation_fn=self.keras_model_creation_fn,
            keras_model_creation_fn_args=self.keras_model_creation_fn_args,
            model_name='generator_f',
            step_extractor=self.step_extractor,
            wandb_ckpt_path=self.wandb_ckpt_path)

        self.generator_g, (_, self.g_load_step) = load_pretrained_keras_model_from_wandb(
            wandb_run_id=self.wandb_run_id,
            wandb_project=self.wandb_project,
            wandb_entity=self.wandb_entity,
            keras_model_creation_fn=self.keras_model_creation_fn,
            keras_model_creation_fn_args=self.keras_model_creation_fn_args,
            model_name='generator_g',
            step_extractor=self.step_extractor,
            wandb_ckpt_path=self.wandb_ckpt_path)

        self.models_loaded = True
        print("Done building CycleGAN models.")

    def map_fn_f(self, image, label):
        # Rescale the data
        image = (tf.cast(image, tf.float32) / 127.5) - 1.
        # Pass it through the generator
        image = self.generator_f(image, training=self.training)
        # Rescale output to [0, 255]
        image = tf.cast(255 * (image * 0.5 + 0.5), tf.uint8)
        if self.relabel:
            return image, tf.zeros_like(label)
        else:
            return image, label

    def map_fn_g(self, image, label):
        # Rescale the data
        image = (tf.cast(image, tf.float32) / 127.5) - 1.
        # Pass it through the generator
        image = self.generator_g(image, training=self.training)
        # Rescale output to [0, 255]
        image = tf.cast(255 * (image * 0.5 + 0.5), tf.uint8)
        if self.relabel:
            return image, tf.ones_like(label)
        else:
            return image, label

    def transform(self, dataset, alias, dataset_len, batch_size, *args, **kwargs):
        dataset_f = dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE).map(self.map_fn_f,
                                                                                               num_parallel_calls=16).unbatch()
        dataset_g = dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE).map(self.map_fn_g,
                                                                                               num_parallel_calls=16).unbatch()
        alias_f = alias + '(A-F)'
        alias_g = alias + '(A-G)'
        if self.keep_original:
            return [dataset, dataset_f, dataset_g], [alias, alias_f, alias_g], [dataset_len] * 3, [batch_size] * 3
        else:
            return [dataset_f, dataset_g], [alias_f, alias_g], [dataset_len] * 2, [batch_size] * 2


class PretrainedDefaultCycleGANStaticAugmentationPipeline(PretrainedCycleGANStaticAugmentationPipeline):

    def __init__(self,
                 wandb_entity,
                 wandb_project,
                 wandb_run_id,
                 run_in_eval_mode=False,
                 input_shape=(256, 256, 3),
                 norm_type='batchnorm',
                 checkpoint_step=-1,
                 relabel=False,
                 wandb_ckpt_path='checkpoints/',
                 batch_size=1,
                 keep_original=False,
                 load_immediately=True,
                 *args, **kwargs):
        super(PretrainedDefaultCycleGANStaticAugmentationPipeline,
              self).__init__(wandb_entity=wandb_entity,
                             wandb_project=wandb_project,
                             wandb_run_id=wandb_run_id,
                             keras_model_creation_fn='unet_generator',
                             keras_model_creation_fn_args={'output_channels': 3,
                                                           'input_shape': input_shape,
                                                           'norm_type': norm_type},
                             step_extractor=particular_checkpoint_step_extractor(checkpoint_step),
                             run_in_eval_mode=run_in_eval_mode,
                             relabel=relabel,
                             wandb_ckpt_path=wandb_ckpt_path,
                             batch_size=batch_size,
                             keep_original=keep_original,
                             load_immediately=load_immediately,
                             *args, **kwargs)


class PretrainedMNISTCycleGANStaticAugmentationPipeline(PretrainedCycleGANStaticAugmentationPipeline):

    def __init__(self,
                 wandb_entity,
                 wandb_project,
                 wandb_run_id,
                 run_in_eval_mode=False,
                 norm_type='batchnorm',
                 checkpoint_step=-1,
                 relabel=False,
                 batch_size=128,
                 keep_original=False,
                 *args, **kwargs):
        super(PretrainedMNISTCycleGANStaticAugmentationPipeline,
              self).__init__(wandb_entity=wandb_entity,
                             wandb_project=wandb_project,
                             wandb_run_id=wandb_run_id,
                             keras_model_creation_fn='mnist_unet_generator',
                             keras_model_creation_fn_args={'norm_type': norm_type},
                             step_extractor=particular_checkpoint_step_extractor(checkpoint_step),
                             run_in_eval_mode=run_in_eval_mode,
                             relabel=relabel,
                             batch_size=batch_size,
                             keep_original=keep_original,
                             *args, **kwargs)


class BinaryMNISTWandbModelPseudolabelPartition(StaticAugmentation):

    def __init__(self,
                 wandb_entity,
                 wandb_project,
                 wandb_run_id,
                 partition=True,
                 relabel=False,
                 batch_size=128,
                 *args, **kwargs):

        super(BinaryMNISTWandbModelPseudolabelPartition, self).__init__(*args, **kwargs)
        self.pseudolabeler = BinaryMNISTWandbModelPseudoLabelingPipeline(wandb_entity, wandb_project, wandb_run_id)
        self.batch_size = batch_size
        self.partition = partition
        self.relabel = relabel
        assert partition or relabel, "BinaryMNISTWandbModelPseudolabelPartition is a no-op if both partition and relabel are False."

    def transform(self, tf_dataset, alias, dataset_len, batch_size, *args, **kwargs):
        """
        Splits a dataset into multiple datasets based on the labels generated by a helper

        There is no way to handle shuffling before pseudolabeling since once a .shuffle is applied,
        the dataset is (by default) reshuffled *every batch*. There's no easy way to undo this shuffling.
        """
        # Batch up the dataset
        try:
            tf_dataset = tf_dataset.unbatch().batch(self.batch_size)
        except:
            tf_dataset = tf_dataset.batch(self.batch_size)

        pseudolabels = []
        # Generate pseudolabels for the entire dataset: tf_dataset must serve up (image_batch, label_batch) pairs
        for batch in tf_dataset:
            # Pseudolabel the data batch
            data, labels = batch
            plab_batch = self.pseudolabeler((data.numpy(), labels.numpy()))
            pseudolabels.append(plab_batch[1])
            # pseudolabels.append(plab_batch)

        # Concatenate data from all the batches
        pseudolabels = np.concatenate(pseudolabels, axis=0)

        # Unbatch the dataset again so we can line up the labels properly
        tf_dataset = tf_dataset.unbatch()

        # Zip the pseudolabels with the tf_dataset
        pseudolabels_tf_dataset = tf.data.Dataset.from_tensor_slices(pseudolabels)
        pseudolabels_tf_dataset = tf.data.Dataset.zip((tf_dataset, pseudolabels_tf_dataset))

        if self.relabel:
            pseudolabels_tf_dataset = pseudolabels_tf_dataset.map(
                lambda xy, z: ((xy[0], tf.reshape(z, xy[1].shape)), z))
        if self.partition:
            # Partition the dataset according to pseudolabels and remove pseudolabels
            z_classes = list(set(pseudolabels))
            aliases = [alias + f'P{z}' for z in z_classes]
            k = len(z_classes)
            batch_sizes = [batch_size // k] * (k - 1) + [batch_size // k + batch_size % k]
            datasets = [pseudolabels_tf_dataset.filter(lambda _, z_: z_ == z) for z in z_classes]
            # Compute the lengths of the split datasets
            dataset_lens = [np.sum(pseudolabels == z) for z in z_classes]
        else:
            aliases = [alias]
            batch_sizes = [batch_size]
            datasets = [pseudolabels_tf_dataset]
            dataset_lens = [dataset_len]
        # Remove auxiliary z label
        datasets = [dataset.map(lambda x, z: x) for dataset in datasets]

        return datasets, aliases, dataset_lens, batch_sizes


class ConcatenateStaticAugmentation(StaticAugmentation):
    """ Takes a list of datasets and concatenates it into one dataset.

    Useful for training pipelines that expect only one dataset (e.g. multihead methods (MTL, DAT)) instead of multiple datasets (e.g. GDRO),
    yet wants to apply other 'partitioning' static augmentations
    """

    def __call__(self, datasets, aliases, dataset_lens, batch_sizes, original_idx, *args, **kwargs):
        """
        Returns list of lists, one copy for each pseudolabel.
        original_idx: for each dataset, specifies which of the "original" datasets (in the config)
          it was generated from. Useful for broadcasting other arguments
        """

        concat_datasets, concat_aliases, concat_batch_sizes, concat_dataset_lens = {}, {}, {}, {}
        for dataset, alias, dataset_len, batch_size, idx in zip(datasets, aliases, dataset_lens, batch_sizes,
                                                                original_idx):
            if idx in concat_datasets:
                concat_datasets[idx] = concat_datasets[idx].concatenate(dataset)
                concat_aliases[idx] += '+' + alias
                concat_batch_sizes[idx] += batch_size
                concat_dataset_lens[idx] += dataset_len
            else:
                concat_datasets[idx] = dataset
                concat_aliases[idx] = alias
                concat_batch_sizes[idx] = batch_size
                concat_dataset_lens[idx] = dataset_len

        updated_datasets = list(concat_datasets.values())
        updated_dataset_lens = list(concat_dataset_lens.values())
        updated_aliases = list(concat_aliases.values())
        updated_batch_sizes = list(concat_batch_sizes.values())
        updated_idx = list(concat_datasets.keys())
        return updated_datasets, updated_aliases, updated_dataset_lens, updated_batch_sizes, updated_idx
