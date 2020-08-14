from augmentation.dataflows.utils import create_parallel_dataflow_via_numpy, create_direct_dataflow
import augmentation.augment.static
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from types import SimpleNamespace
import datetime

import augmentation.datasets.custom.mnist
import augmentation.datasets.custom.mnist_correlation
import augmentation.datasets.custom.waterbirds
import augmentation.datasets.custom.celeba_128


def dataset_len(tf_dataset, verbose=False):
    """
    Compute the length of a TF dataset.
    """
    tot = 0
    for data, _ in tf_dataset.batch(128):
        tot += data.shape[0]
        if verbose: print('.', end='')
    if verbose: print(flush=True)
    return tot


def get_dataset_from_list_files_dataset(list_files_dataset, proc_batch, tfrecord_example_reader, sequential=False):
    num_parallel_reads = cpu_count()
    if sequential:
        num_parallel_reads = 1
    # Load up the TFRecord dataset with parallel reads: this will interleave unless sequential is set to True
    dataset = tf.data.TFRecordDataset(list_files_dataset, num_parallel_reads=num_parallel_reads)
    # Batch up the TFRecord examples
    dataset = dataset.batch(proc_batch)
    # Decode the examples in parallel for batches: this produces a deterministic ordering always
    dataset = dataset.map(tfrecord_example_reader, num_parallel_calls=cpu_count())
    return dataset


def apply_modifier_to_dataset(dataset, dataset_payload, modifier, modifier_args):
    """
    Apply a modifier to a tf.data.Dataset.
    """

    if modifier == 'class':
        # Filter out only the examples that belong to the given class label.
        class_label = int(modifier_args)
        dataset = dataset.filter(lambda image, label: label == class_label)
    elif modifier == 'shuffle':
        # Shuffle the dataset
        buffer_size, seed = [int(e) for e in modifier_args.split(",")]
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
    elif modifier == 'take':
        # Take a few examples from the dataset
        n_examples = int(modifier_args)
        dataset = dataset.take(n_examples)
    elif modifier == 'repeat':
        # Repeat the dataset a few times
        n_repeats = int(modifier_args)
        dataset = dataset.repeat(n_repeats)
    elif modifier == 'noval':
        assert dataset_payload is not None, 'dataset_payload must be a namespace with train_dataset and val_dataset.'
        # Put the validation dataset into the training data and set the validation dataset to the test data
        dataset_payload.train_dataset = dataset_payload.train_dataset.concatenate(dataset_payload.val_dataset)
        dataset_payload.val_dataset = dataset_payload.test_dataset
    elif modifier == '':
        # Apply no modifier, return as is
        pass
    else:
        raise NotImplementedError
    return dataset, dataset_payload


def apply_modifier_command_to_dataset(dataset, dataset_payload, modifier_command):
    """
    A modifier command is represented as a string
    <modifier_1>[:modifier_1_args]/<modifier_2>[:modifier_2_args]/...
    and is applied to a tf.data.Dataset.
    """

    # Split up the command to grab the list of modifiers
    list_of_modifiers = modifier_command.split("/")

    # Apply the modifiers in sequence
    for modifier in list_of_modifiers:
        try:
            modifier_, modifier_args = modifier.split(":")
        except ValueError:
            # If there's no argument, set it to None
            modifier_, modifier_args = modifier.split(":")[0], None

        # Apply the modifier
        dataset, dataset_payload = apply_modifier_to_dataset(dataset, dataset_payload, modifier_, modifier_args)

    return dataset, dataset_payload


def apply_modifier_to_dataset_payload(dataset_payload, train_dataset_modifier, eval_dataset_modifier=None):
    """
    Take a dataset_payload namespace that contains train_dataset, val_dataset and test_dataset (all tf.data.Datasets)
    and applies modifiers to each dataset.
    """
    # Apply the modifier commands to each of the datasets in the dataset payload
    dataset_payload.train_dataset, dataset_payload = apply_modifier_command_to_dataset(dataset_payload.train_dataset,
                                                                                       dataset_payload,
                                                                                       train_dataset_modifier)

    # If we didn't specify an eval_dataset_modifier, just use ''
    eval_dataset_modifier = '' if eval_dataset_modifier is None else eval_dataset_modifier

    dataset_payload.val_dataset, dataset_payload = apply_modifier_command_to_dataset(dataset_payload.val_dataset,
                                                                                     dataset_payload,
                                                                                     eval_dataset_modifier)

    dataset_payload.test_dataset, dataset_payload = apply_modifier_command_to_dataset(dataset_payload.test_dataset,
                                                                                      dataset_payload,
                                                                                      eval_dataset_modifier)

    return dataset_payload


def get_processed_dataset_info(dataset_info, validation_frac, batch_size):
    n_classes = dataset_info.features['label'].num_classes
    try:
        classes = dataset_info.classes
    except:
        classes = [f'Class {i}' for i in range(n_classes)]
    try:
        n_domains = dataset_info.features['label'].num_domains
        domains = dataset_info.domains
    except:
        n_domains = 0
        domains = []
    input_shape = dataset_info.features['image'].shape
    try:
        n_training_examples = int(dataset_info.splits['train'].num_examples * (1 - validation_frac))
    except:
        # Sometimes the train split isn't exactly called 'train' for a dataset
        n_training_examples = 0
    n_batches = int(np.ceil(n_training_examples / float(batch_size)))

    return SimpleNamespace(n_classes=n_classes,
                           classes=classes,
                           n_domains=n_domains,
                           domains=domains,
                           input_shape=input_shape,
                           n_training_examples=n_training_examples,
                           n_batches=n_batches)


def load_dataset_using_tfds(dataset_name, dataset_version, data_dir) -> tfds.core.DatasetBuilder:
    """
    Load up a dataset using Tensorflow Datasets.

    :param dataset_name: Name of the dataset, e.g. 'cifar10'
    :param dataset_version: Dataset version, e.g. '3.*.*'
    :param data_dir: Path to where data should be downloaded
    :return: dataset builder of type tfds.core.DatasetBuilder
    """

    # Use tensorflow datasets to load up the dataset
    dataset_builder = tfds.builder(name=f'{dataset_name}:{dataset_version}', data_dir=data_dir)
    dataset_builder.download_and_prepare(download_dir=data_dir + '/downloads')

    return dataset_builder


def show_dataset_info(dataset_builder, plot=False):
    """
    Get information about the dataset using the tfds.core.DatasetBuilder.
    """
    # Get information about the dataset
    dataset_info = dataset_builder.info

    if plot:
        # Grab 9 examples from the training data and display them
        dataset = dataset_builder.as_dataset(split='train[:9]')
        viz = tfds.show_examples(dataset_info, dataset)
        plt.show(viz)

    return dataset_info


def generate_dataset_split(validation_frac, cross_validation=False, fold=None):
    """
    Generate splits conforming to the tf.data split API.
    """
    assert 100 * validation_frac % 5 == 0, 'Use validation fractions that are multiples of 0.05!'
    assert 0 <= validation_frac <= 1., 'Validation fractions must be in [0, 1].'

    # Convert the validation fraction to the range [0, 100]
    validation_frac = int(validation_frac * 100)

    if not cross_validation:
        if validation_frac > 0.:
            # Simply grab the last x% of the training data as the validation set
            split = [f'train[:{100 - validation_frac}%]',
                     f'train[{100 - validation_frac}%:]',
                     'test']
        else:
            # We're training on the full dataset, monitor performance on the test dataset
            split = ['train',
                     'test',
                     'test']
    else:
        # Check if the fold is correctly specified
        max_folds = 100 // validation_frac
        assert 0 <= fold < max_folds, 'Cross-validation fold is out of range.'
        # Find the location of the fold in the training data and slice out the validation data
        fold_loc = fold * validation_frac
        split = [f'train[:{fold_loc}%]+train[{fold_loc + validation_frac}%:]',
                 f'train[{fold_loc}%:{fold_loc + validation_frac}%]',
                 'test']

    assert len(split) == 3, 'Split must contain descriptions of the train, validation and test datasets.'

    return split


CUSTOM_DATASET_PREFIXES = ['mnist_spurious',
                           'mnist_combined',
                           'mnist_correlation',
                           'celeb_a',
                           'waterbirds',
                           ]


def load_dataset(dataset_name, dataset_version, data_dir, validation_frac, cross_validation=False, fold=None):
    """
    The main entry point to load any dataset.
    """

    # For a custom dataset, call the custom dataset loader
    if np.any([dataset_name.startswith(e) for e in CUSTOM_DATASET_PREFIXES]):
        assert cross_validation is False, 'Cross-validation is not supported for the custom datasets.'
        return load_custom_dataset(dataset_name, dataset_version, data_dir, validation_frac)

    # Set up the dataset
    dataset_builder = load_dataset_using_tfds(dataset_name=dataset_name,
                                              dataset_version=dataset_version,
                                              data_dir=data_dir)

    # Get some dataset information
    dataset_info = show_dataset_info(dataset_builder=dataset_builder, plot=False)

    # Generate the dataset
    dataset_split = generate_dataset_split(validation_frac, cross_validation, fold)

    # Put the dataset into memory and return a training, validation and test dataset
    train_dataset, val_dataset, test_dataset = dataset_builder.as_dataset(split=dataset_split, as_supervised=True)

    return SimpleNamespace(dataset_builder=dataset_builder,
                           dataset_info=dataset_info,
                           dataset_split=dataset_split,
                           train_dataset=train_dataset,
                           val_dataset=val_dataset,
                           test_dataset=test_dataset)


def decode_raw_image(raw_bytes):
    return tf.image.decode_jpeg(raw_bytes, channels=3)


def fetch_datasets_for_trainer(dataset,
                               dataset_version,
                               # train_dataset_modifier,
                               # eval_dataset_modifier,
                               datadir,
                               validation_frac,
                               batch_size,
                               cross_validation=False,
                               fold=None):
    # Load the dataset payload
    dataset_payload = load_dataset(dataset, dataset_version, datadir, validation_frac, cross_validation, fold)

    # Apply modifiers on the datasets
    # dataset_payload = apply_modifier_to_dataset_payload(dataset_payload, train_dataset_modifier, eval_dataset_modifier)

    # Get some dataset information
    proc_dataset_info = get_processed_dataset_info(dataset_payload.dataset_info, validation_frac, batch_size)

    return (dataset_payload.train_dataset, dataset_payload.val_dataset, dataset_payload.test_dataset), \
           (proc_dataset_info.input_shape, proc_dataset_info.n_classes,
            proc_dataset_info.classes, proc_dataset_info.n_training_examples)


def fetch_list_of_datasets(datasets,
                           dataset_versions,
                           datadirs,
                           validation_frac,
                           batch_size,
                           cross_validation=False,
                           fold=None):
    dataset_splits, training_examples_by_dataset = [], []
    input_shape, n_classes, classes = None, None, None

    # Loop over all the datasets
    for dataset, dataset_version, datadir in zip(datasets, dataset_versions, datadirs):
        # Fetch the dataset
        print(f"Fetching dataset {dataset} from {datadir}.")
        splits, (input_shape_, n_classes_, classes_, n_training_examples_) \
            = fetch_datasets_for_trainer(dataset,
                                         dataset_version,
                                         datadir,
                                         validation_frac,
                                         batch_size,
                                         cross_validation,
                                         fold)
        dataset_splits.append(splits)
        if input_shape is None:
            input_shape, n_classes, classes = input_shape_, n_classes_, classes_
        else:
            # All the datasets should have the same schema
            assert input_shape == input_shape_ and n_classes == n_classes_ and classes == classes_

        # Update the n_training_examples
        training_examples_by_dataset.append(n_training_examples_)

    return dataset_splits, training_examples_by_dataset, input_shape, n_classes, classes


def fetch_list_of_train_datasets(train_datasets,
                                 train_dataset_versions,
                                 train_datadirs,
                                 validation_frac,
                                 batch_size,
                                 cross_validation=False,
                                 fold=None):
    # Fetch the list of training datasets
    dataset_splits, training_examples_by_dataset, input_shape, n_classes, classes = \
        fetch_list_of_datasets(datasets=train_datasets,
                               dataset_versions=train_dataset_versions,
                               datadirs=train_datadirs,
                               validation_frac=validation_frac,
                               batch_size=batch_size,
                               cross_validation=cross_validation,
                               fold=fold)

    # Grab the train datasets
    train_datasets, _, _ = zip(*dataset_splits)

    # Total number of training examples
    n_training_examples = np.sum(training_examples_by_dataset)

    return train_datasets, \
           (training_examples_by_dataset, n_training_examples, input_shape, n_classes, classes)


def fetch_list_of_eval_datasets(eval_datasets,
                                eval_dataset_versions,
                                eval_datadirs,
                                validation_frac,
                                batch_size,
                                cross_validation=False,
                                fold=None):
    # Fetch the list of training datasets
    dataset_splits, _, input_shape, n_classes, classes = \
        fetch_list_of_datasets(datasets=eval_datasets,
                               dataset_versions=eval_dataset_versions,
                               datadirs=eval_datadirs,
                               validation_frac=validation_frac,
                               batch_size=batch_size,
                               cross_validation=cross_validation,
                               fold=fold)

    # Grab the train datasets
    _, val_datasets, test_datasets = zip(*dataset_splits)

    return (val_datasets, test_datasets), \
           (input_shape, n_classes, classes)


def fetch_list_of_data_generators_for_trainer(train_dataset_names,
                                              train_dataset_versions,
                                              train_datadirs,
                                              train_dataset_aliases,
                                              eval_dataset_names,
                                              eval_dataset_versions,
                                              eval_datadirs,
                                              eval_dataset_aliases,
                                              # train_dataset_modifier,
                                              # eval_dataset_modifier,
                                              train_augmentations, train_gpu_augmentations, train_static_augmentations,
                                              eval_augmentations, eval_gpu_augmentations, eval_static_augmentations,
                                              cache_dir,
                                              validation_frac,
                                              batch_size,
                                              dataflow,
                                              max_shuffle_buffer=0,
                                              train_shuffle_seeds=None,
                                              repeat=False,
                                              shuffle_before_repeat=False,
                                              cross_validation=False,
                                              fold=None):
    # Fetch the list of training datasets
    print("Fetching training datasets.", flush=True)
    train_datasets, (training_examples_by_dataset, n_training_examples,
                     train_input_shape, train_n_classes, train_classes) = \
        fetch_list_of_train_datasets(train_datasets=train_dataset_names,
                                     train_dataset_versions=train_dataset_versions,
                                     train_datadirs=train_datadirs,
                                     validation_frac=validation_frac,
                                     batch_size=batch_size,
                                     cross_validation=cross_validation,
                                     fold=fold)

    # Fetch the list of evaluation datasets
    print("Fetching evaluation datasets.", flush=True)
    (val_datasets, test_datasets), (eval_input_shape, eval_n_classes, eval_classes) = \
        fetch_list_of_eval_datasets(eval_datasets=eval_dataset_names,
                                    eval_dataset_versions=eval_dataset_versions,
                                    eval_datadirs=eval_datadirs,
                                    validation_frac=validation_frac,
                                    batch_size=batch_size,
                                    cross_validation=cross_validation,
                                    fold=fold)

    assert train_input_shape == eval_input_shape and train_n_classes == eval_n_classes \
           and train_classes == eval_classes, \
        'Train and eval sets must have the same schema (input_shape, n_classes, classes).'
    assert train_shuffle_seeds is None or len(train_shuffle_seeds) == len(train_datasets), \
        'Either set train_shuffle_seeds to None or specify one seed per training dataset.'

    print("Applying static augmentations.", flush=True)
    train_dataset_identifiers = [f'[{name}].[{version}].train.{validation_frac:.3f}' for name, version in
                                 zip(train_dataset_names, train_dataset_versions)]
    val_dataset_identifiers = [f'[{name}].[{version}].val.{validation_frac:.3f}' for name, version in
                               zip(eval_dataset_names, eval_dataset_versions)]
    test_dataset_identifiers = [f'[{name}].[{version}].test' for name, version in
                                zip(eval_dataset_names, eval_dataset_versions)]

    train_datasets, train_dataset_aliases, training_examples_by_dataset, train_batch_sizes, train_original_idx = \
        augmentation.augment.static.compose_static_augmentations(
            static_augmentation_pipelines=train_static_augmentations,
            datasets=train_datasets,
            aliases=train_dataset_aliases,
            identifiers=train_dataset_identifiers,
            dataset_lens=training_examples_by_dataset,
            batch_sizes=augmentation.augment.static.split_batch_size(batch_size, len(train_datasets)),
            keep_datasets=False)

    val_datasets, val_dataset_aliases, _, val_batch_sizes, val_original_idx = \
        augmentation.augment.static.compose_static_augmentations(
            static_augmentation_pipelines=eval_static_augmentations,
            datasets=val_datasets,
            aliases=eval_dataset_aliases,
            identifiers=val_dataset_identifiers,
            dataset_lens=[0] * len(val_datasets),
            batch_sizes=[batch_size] * len(val_datasets),
            keep_datasets=True)

    test_datasets, test_dataset_aliases, _, test_batch_sizes, test_original_idx = \
        augmentation.augment.static.compose_static_augmentations(
            static_augmentation_pipelines=eval_static_augmentations,
            datasets=test_datasets,
            aliases=eval_dataset_aliases,
            identifiers=test_dataset_identifiers,
            dataset_lens=[0] * len(test_datasets),
            batch_sizes=[batch_size] * len(test_datasets),
            keep_datasets=True)

    # TODO: fix this assert with np.all
    assert val_dataset_aliases == test_dataset_aliases and val_batch_sizes == test_batch_sizes and \
           val_original_idx == test_original_idx, \
        'Currently, evaluation datasets must have the same aliases, batch_sizes and variants.'

    # Make sure augmentations on the fly are applied to the appropriate datasets
    train_augmentations = [train_augmentations[i] for i in train_original_idx]
    train_gpu_augmentations = [train_gpu_augmentations[i] for i in train_original_idx]
    val_augmentations = [eval_augmentations[i] for i in val_original_idx]
    val_gpu_augmentations = [eval_gpu_augmentations[i] for i in val_original_idx]
    test_augmentations = [eval_augmentations[i] for i in test_original_idx]
    test_gpu_augmentations = [eval_gpu_augmentations[i] for i in test_original_idx]

    print("Diagnostics...")
    print("--------------------------------------------------------------")
    print("Train datasets:", train_datasets)
    print("Train dataset aliases:", train_dataset_aliases)
    print("Train examples per group:", training_examples_by_dataset)
    print("Train batch sizes:", train_batch_sizes)
    print("Train original indices per new dataset:", train_original_idx)
    print("--------------------------------------------------------------")
    print("Val datasets:", val_datasets)
    print("Val dataset aliases:", val_dataset_aliases)
    print("Val batch sizes:", val_batch_sizes)
    print("Val original indices per new dataset:", val_original_idx)
    print("--------------------------------------------------------------")
    print("Test datasets:", test_datasets)
    print("Test dataset aliases:", test_dataset_aliases)
    print("Test batch sizes:", test_batch_sizes)
    print("Test original indices per new dataset:", test_original_idx)
    print("--------------------------------------------------------------", flush=True)


    # Create the generators
    if max_shuffle_buffer < 0:
        train_shuffle_buffers = training_examples_by_dataset
    else:
        train_shuffle_buffers = min(training_examples_by_dataset,
                                    [max_shuffle_buffer] * len(training_examples_by_dataset))

    if train_shuffle_seeds is None:
        train_shuffle_seeds = train_original_idx
    else:
        train_shuffle_seeds = [train_shuffle_seeds[i] for i in train_original_idx]

    train_generators = create_multiple_data_generators(datasets=train_datasets,
                                                       dataset_aliases=train_dataset_aliases,
                                                       augmentations_by_dataset=train_augmentations,
                                                       gpu_augmentations_by_dataset=train_gpu_augmentations,
                                                       label_augmentations_by_dataset=[()] * len(train_dataset_aliases),
                                                       batch_sizes=train_batch_sizes,
                                                       shuffle_buffers=train_shuffle_buffers,
                                                       shuffle_seeds=train_shuffle_seeds,
                                                       dataflow=dataflow,
                                                       repeat=repeat,
                                                       shuffle_before_repeat=shuffle_before_repeat,
                                                       cache_dir=cache_dir,
                                                       cache_dir_postfix='__train')
    val_generators = create_multiple_data_generators(datasets=val_datasets,
                                                     dataset_aliases=val_dataset_aliases,
                                                     augmentations_by_dataset=val_augmentations,
                                                     gpu_augmentations_by_dataset=val_gpu_augmentations,
                                                     label_augmentations_by_dataset=[()] * len(val_dataset_aliases),
                                                     batch_sizes=val_batch_sizes,
                                                     shuffle_buffers=[0] * len(val_dataset_aliases),
                                                     shuffle_seeds=[0] * len(val_dataset_aliases),
                                                     dataflow=dataflow,
                                                     repeat=False,
                                                     shuffle_before_repeat=shuffle_before_repeat,
                                                     cache_dir=cache_dir,
                                                     cache_dir_postfix='__val')
    test_generators = create_multiple_data_generators(datasets=test_datasets,
                                                      dataset_aliases=test_dataset_aliases,
                                                      augmentations_by_dataset=test_augmentations,
                                                      gpu_augmentations_by_dataset=test_gpu_augmentations,
                                                      label_augmentations_by_dataset=[()] * len(test_dataset_aliases),
                                                      batch_sizes=test_batch_sizes,
                                                      shuffle_buffers=[0] * len(test_dataset_aliases),
                                                      shuffle_seeds=[0] * len(test_dataset_aliases),
                                                      dataflow=dataflow,
                                                      repeat=False,
                                                      shuffle_before_repeat=shuffle_before_repeat,
                                                      cache_dir=cache_dir,
                                                      cache_dir_postfix='__test')

    print("Generator lengths:", len(train_generators), len(val_generators), len(test_generators))
    print("Done with creating generators for train and eval.", flush=True)
    return (train_generators, val_generators, test_generators), \
           (train_input_shape, train_n_classes, train_classes, n_training_examples, training_examples_by_dataset), \
           (train_dataset_aliases, val_dataset_aliases, test_dataset_aliases)


def create_data_generator(dataset,
                          augmentations,
                          gpu_augmentations,
                          label_augmentations,
                          batch_size,
                          dataflow,
                          repeat=False,
                          shuffle_buffer=0,
                          shuffle_seed=0,
                          shuffle_before_repeat=False,
                          cache_dir=None,
                          cache_dir_postfix=''):
    """Given a single tf_dataset, construct a generator that applies augmentations to that dataset."""
    if dataflow == 'in_memory':
        generator = create_parallel_dataflow_via_numpy(tf_dataset=dataset,
                                                       batch_size=batch_size,
                                                       augmentations=augmentations,
                                                       gpu_augmentations=gpu_augmentations)

    elif dataflow == 'disk_cached':
        assert cache_dir is not None, 'You must specify a cache directory when using disk_cached.'
        cache_dir = cache_dir + '_' + datetime.datetime.now().strftime('%d_%m_%y__%H_%M_%S') + cache_dir_postfix

        # Cache the dataset first
        dataset = dataset.cache(cache_dir)
        # You can't test the dataflow unless you manually cache, otherwise
        # concurrent caching iterators will be generated (leading to a TF error).
        dataset_len(dataset, verbose=True)

        if shuffle_before_repeat:
            if shuffle_buffer > 0:
                dataset = dataset.shuffle(shuffle_buffer, seed=shuffle_seed)
            if repeat:
                dataset = dataset.repeat(-1)
        else:
            if repeat:
                dataset = dataset.repeat(-1)
            if shuffle_buffer > 0:
                dataset = dataset.shuffle(shuffle_buffer, seed=shuffle_seed)

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        try:
            # Unbatch it
            dataset = dataset.unbatch()
        except ValueError:
            pass

        generator = create_direct_dataflow(tf_dataset=dataset,
                                           batch_size=batch_size,
                                           augmentations=augmentations,
                                           gpu_augmentations=gpu_augmentations,
                                           label_augmentations=label_augmentations,
                                           test_flow=False,  # (not repeat),
                                           )

    else:
        raise NotImplementedError

    return generator


def create_multiple_data_generators(datasets,
                                    dataset_aliases,
                                    augmentations_by_dataset,
                                    gpu_augmentations_by_dataset,
                                    label_augmentations_by_dataset,
                                    batch_sizes,
                                    shuffle_buffers,
                                    shuffle_seeds,
                                    dataflow,
                                    repeat=False,
                                    shuffle_before_repeat=False,
                                    cache_dir=None,
                                    cache_dir_postfix=''):
    """Given a list of tf_datasets, construct a list of generators, one per dataset.

    batch_sizes: list of batch sizes, one for each dataset
    """

    assert len(datasets) == \
           len(dataset_aliases) == \
           len(augmentations_by_dataset) == \
           len(gpu_augmentations_by_dataset) == \
           len(label_augmentations_by_dataset) == \
           len(batch_sizes) == \
           len(shuffle_buffers) == \
           len(shuffle_seeds), \
        "All lengths passed in must be identical."

    generators = []
    for i, (dataset, alias, augmentations,
            gpu_augmentations, label_augmentations, batch_size,
            shuffle_buffer, shuffle_seed) in enumerate(zip(datasets,
                                                           dataset_aliases,
                                                           augmentations_by_dataset,
                                                           gpu_augmentations_by_dataset,
                                                           label_augmentations_by_dataset,
                                                           batch_sizes,
                                                           shuffle_buffers, shuffle_seeds)):
        # Create a data generator for this dataset
        print(f"Creating {alias} data generator: shuffle with {shuffle_buffer}, {shuffle_seed}")
        generators.append(create_data_generator(dataset=dataset,
                                                augmentations=augmentations,
                                                gpu_augmentations=gpu_augmentations,
                                                label_augmentations=label_augmentations,
                                                batch_size=batch_size,
                                                dataflow=dataflow,
                                                repeat=repeat,
                                                shuffle_buffer=shuffle_buffer,
                                                shuffle_seed=shuffle_seed,
                                                shuffle_before_repeat=shuffle_before_repeat,
                                                cache_dir=cache_dir,
                                                cache_dir_postfix=cache_dir_postfix + "_" + alias.replace("/", "")
                                                                  + str(i)))

    return generators


def get_dataset_aliases(dataset_aliases, datasets):
    if len(dataset_aliases) == len(datasets):
        return dataset_aliases
    else:
        return datasets

def load_custom_dataset(dataset_name, dataset_version, data_dir, validation_frac):
    """
    Load up a custom dataset.
    """
    assert np.any([dataset_name.startswith(e) for e in CUSTOM_DATASET_PREFIXES]), 'Dataset specified is not supported.'
    if dataset_name.startswith('mnist_spurious'):
        return augmentation.datasets.custom.mnist.load_mnist_spurious(dataset_name,
                                                                      dataset_version,
                                                                      data_dir,
                                                                      validation_frac)
    elif dataset_name.startswith('mnist_combined'):
        return augmentation.datasets.custom.mnist.load_mnist_combined(dataset_name,
                                                                      dataset_version,
                                                                      data_dir,
                                                                      validation_frac)
    elif dataset_name.startswith('mnist_correlation_yz_multihead'):
        return augmentation.datasets.custom.mnist_correlation.load_mnist_correlation_yz_multihead(dataset_name,
                                                                                                  dataset_version,
                                                                                                  data_dir,
                                                                                                  validation_frac)
    elif dataset_name.startswith('mnist_correlation_yz'):
        return augmentation.datasets.custom.mnist_correlation.load_mnist_correlation_yz(dataset_name,
                                                                                        dataset_version,
                                                                                        data_dir,
                                                                                        validation_frac)
    elif dataset_name.startswith('mnist_correlation_y'):
        return augmentation.datasets.custom.mnist_correlation.load_mnist_correlation_y(dataset_name,
                                                                                       dataset_version,
                                                                                       data_dir,
                                                                                       validation_frac)
    elif dataset_name.startswith('mnist_correlation_partial'):
        return augmentation.datasets.custom.mnist_correlation.load_mnist_correlation_partial(dataset_name,
                                                                                             dataset_version,
                                                                                             data_dir,
                                                                                             validation_frac)
    elif dataset_name.startswith('mnist_correlation_multihead'):
        return augmentation.datasets.custom.mnist_correlation.load_mnist_correlation_multihead(dataset_name,
                                                                                               dataset_version,
                                                                                               data_dir,
                                                                                               validation_frac)
    elif dataset_name.startswith('mnist_correlation_'):
        return augmentation.datasets.custom.mnist_correlation.load_mnist_correlation_(dataset_name,
                                                                                      dataset_version,
                                                                                      data_dir,
                                                                                      validation_frac)
    elif dataset_name.startswith('mnist_correlation'):
        return augmentation.datasets.custom.mnist_correlation.load_mnist_correlation(dataset_name,
                                                                                     dataset_version,
                                                                                     data_dir,
                                                                                     validation_frac)
    elif dataset_name.startswith('waterbirds'):
        return augmentation.datasets.custom.waterbirds.load_waterbirds(dataset_name,
                                                                       dataset_version,
                                                                       data_dir)
    elif dataset_name.startswith('celeb_a_128'):
        return augmentation.datasets.custom.celeba_128.load_celeba_128(dataset_name,
                                                                       dataset_version,
                                                                       data_dir)
    else:
        raise NotImplementedError

