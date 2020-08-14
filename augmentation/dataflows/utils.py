import tensorflow as tf
import dataflow as D
import time
import numpy as np
import datetime
from multiprocessing import cpu_count
from augmentation.augment.utils import compose_augmentations


def benchmark(dataflow, num_epochs=2, sleep=0.):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        s = time.time()
        counter = 0
        for _ in dataflow:
            # Performing a training step
            time.sleep(sleep)
            counter += 1
            pass
        tf.print(f"Samples counted: {counter}")
        e = time.time()
        tf.print(f'{e - s}s elapsed.')
    tf.print("Total execution time:", time.perf_counter() - start_time)


def dataflow_len(dataflow):
    """
    Compute the length of a dataflow.
    """
    tot = 0
    for data in dataflow:
        tot += data[0].shape[0]
        print(tot)
    return tot

def create_direct_dataflow(tf_dataset,
                           batch_size,
                           augmentations=(),
                           gpu_augmentations=(),
                           label_augmentations=(),
                           num_proc=cpu_count(),
                           test_flow=True,
                           ):

    # Create a dataflow
    dataflow = D.DataFromGenerator(tf_dataset)
    # Map the tensors to numpy arrays
    dataflow = D.MapData(dataflow, func=lambda x: (x[0].numpy(), x[1].numpy()))
    # Batch the data
    dataflow = D.BatchData(dataflow, batch_size=batch_size)
    # Repeat the data only once, we use a custom loop over epochs
    dataflow = D.RepeatedData(dataflow, 1)
    # Create a function for data augmentations
    daug = lambda x: compose_augmentations((compose_augmentations(x[0], augmentations), x[1]), label_augmentations)
    # Map the function onto the data
    dataflow = D.MapData(dataflow, func=daug)
    # Create a function for gpu data augmentations
    gpu_daug = lambda x: (compose_augmentations(x, gpu_augmentations))
    # Map the function onto the data
    dataflow = D.MapDataComponent(dataflow, func=gpu_daug, index=0)

    if test_flow:
        # A quick runthrough of all the data
        D.TestDataSpeed(dataflow, size=128).start()
    else:
        # Reset state manually
        dataflow.reset_state()

    return dataflow


def create_paired_direct_dataflow(tf_dataset_1,
                                  tf_dataset_2,
                                  batch_size,
                                  augmentations,
                                  x_only=False,
                                  num_proc=cpu_count(),
                                  test_flow=True,
                                  cache_dir1='',
                                  cache_dir2='',
                                  shuffle=True,
                                  shuffle_buffer=1000):
    # Cache the dataset first
    tf_dataset_1 = tf_dataset_1.cache(cache_dir1).prefetch(tf.data.experimental.AUTOTUNE)
    tf_dataset_2 = tf_dataset_2.cache(cache_dir2).prefetch(tf.data.experimental.AUTOTUNE)

    try:
        # Unbatch them
        tf_dataset_1 = tf_dataset_1.unbatch()
        tf_dataset_2 = tf_dataset_2.unbatch()
    except ValueError:
        pass

    if shuffle:
        # Shuffle the data
        tf_dataset_1 = tf_dataset_1.shuffle(shuffle_buffer, seed=1)
        tf_dataset_2 = tf_dataset_2.shuffle(shuffle_buffer, seed=2)

    # Run through to cache the datasets: this is necessary to do, otherwise it won't work
    for _ in tf_dataset_1.batch(batch_size):
        print('.', end='')
        pass

    for _ in tf_dataset_2.batch(batch_size):
        print('.', end='')
        pass

    # Create a dataflow
    dataflow_1 = D.DataFromGenerator(tf_dataset_1)
    dataflow_2 = D.DataFromGenerator(tf_dataset_2)
    # Map the tensors to numpy arrays
    dataflow_1 = D.MapData(dataflow_1, func=lambda x: (x[0].numpy(), x[1].numpy()))
    dataflow_2 = D.MapData(dataflow_2, func=lambda x: (x[0].numpy(), x[1].numpy()))
    # Select some indices in the data
    if x_only:
        dataflow_1 = D.SelectComponent(dataflow_1, [0])
        dataflow_2 = D.SelectComponent(dataflow_2, [0])
    # Zip them
    dataflow = D.JoinData([dataflow_1, dataflow_2])
    # Batch data
    dataflow = D.BatchData(dataflow, batch_size=batch_size, remainder=True)
    # Repeat data only once, we use a custom loop over epochs
    dataflow = D.RepeatedData(dataflow, 1)
    # Create a function for data augmentations
    if not x_only:
        daug = lambda x: (compose_augmentations(x[0], augmentations), x[1],
                          compose_augmentations(x[2], augmentations), x[3])
    else:
        daug = lambda x: (compose_augmentations(x[0], augmentations),
                          compose_augmentations(x[1], augmentations))
    # Map the function onto the data
    dataflow = D.MapData(dataflow, func=daug)
    if test_flow:
        # A quick runthrough of all the data
        D.TestDataSpeed(dataflow).start()
    else:
        # Reset state manually
        dataflow.reset_state()
    return dataflow


def create_parallel_dataflow_via_numpy(tf_dataset,
                                       batch_size,
                                       augmentations=(),
                                       gpu_augmentations=(),
                                       x_only=False,
                                       num_proc=cpu_count(),
                                       test_flow=True):
    X, y = [], []
    # Materialize the dataset as a numpy array: this is memory intensive for large datasets!
    for data in tf_dataset:
        X.append(data[0].numpy())
        y.append(data[1].numpy())
    numpy_dataset = list(zip(np.array(X), np.array(y)))
    # Create a dataflow
    dataflow = D.DataFromList(numpy_dataset)
    # Select some indices in the data
    if x_only:
        dataflow = D.SelectComponent(dataflow, [0])
    # Batch data
    dataflow = D.BatchData(dataflow, batch_size=batch_size)
    # Repeat data only once, we use a custom loop over epochs
    dataflow = D.RepeatedData(dataflow, 1)
    # Create a function for data augmentations
    if not x_only:
        daug = lambda x: (compose_augmentations(x[0], augmentations), x[1])
    else:
        daug = lambda x: (compose_augmentations(x[0], augmentations))
    # Map the function onto the data with parallelism
    dataflow = D.MultiProcessMapData(dataflow, num_proc=num_proc, map_func=daug, strict=True)
    # Create a function for gpu data augmentations
    gpu_daug = lambda x: (compose_augmentations(x, gpu_augmentations))
    # Map the function onto the data
    dataflow = D.MapDataComponent(dataflow, func=gpu_daug, index=0)
    if test_flow:
        # A quick runthrough of all the data
        D.TestDataSpeed(dataflow).start()
    return dataflow


def create_paired_parallel_dataflow_via_numpy(tf_dataset_1,
                                              tf_dataset_2,
                                              batch_size,
                                              augmentations,
                                              x_only=False,
                                              num_proc=cpu_count(),
                                              test_flow=True):
    X_1, y_1 = [], []
    X_2, y_2 = [], []
    # Materialize the dataset as a numpy array: this is memory intensive for large datasets!
    for data in tf_dataset_1:
        X_1.append(data[0].numpy())
        y_1.append(data[1].numpy())

    for data in tf_dataset_2:
        X_2.append(data[0].numpy())
        y_2.append(data[1].numpy())

    numpy_dataset_1 = list(zip(np.array(X_1), np.array(y_1)))
    numpy_dataset_2 = list(zip(np.array(X_2), np.array(y_2)))
    # Create a dataflow
    dataflow_1 = D.DataFromList(numpy_dataset_1)
    dataflow_2 = D.DataFromList(numpy_dataset_2)
    # Select some indices in the data
    if x_only:
        dataflow_1 = D.SelectComponent(dataflow_1, [0])
        dataflow_2 = D.SelectComponent(dataflow_2, [0])
    # Zip them
    dataflow = D.JoinData([dataflow_1, dataflow_2])
    # Batch data
    dataflow = D.BatchData(dataflow, batch_size=batch_size)
    # Repeat data only once, we use a custom loop over epochs
    dataflow = D.RepeatedData(dataflow, 1)
    # Create a function for data augmentations
    if not x_only:
        daug = lambda x: (compose_augmentations(x[0], augmentations), x[1],
                          compose_augmentations(x[2], augmentations), x[3])
    else:
        daug = lambda x: (compose_augmentations(x[0], augmentations),
                          compose_augmentations(x[1], augmentations))
    # Map the function onto the data with parallelism
    dataflow = D.MultiProcessMapData(dataflow, num_proc=num_proc, map_func=daug, strict=True)
    if test_flow:
        # A quick runthrough of all the data
        D.TestDataSpeed(dataflow).start()
    return dataflow


def build_basic_data_pipeline(datasets, n_examples, batch_size, map_fn, map_fn_args):
    """
    Builds a basic data pipeline for multiple datasets by,
    - Restricting the dataset to a few examples with take (use n_examples = -1 as default to fetch whole dataset)
    - Batching data with batch
    - Prefetching batches of data, keeping CPU busy for speedup
    - Applying multiple augmentations to the data in sequence using a map_fn and map_fn_args. map_fn_args
    is a list of list of arguments. Each list of arguments corresponds to an augmentation we would like to map on
    the dataset.

    Example:
    map_fn = data_map
    map_fn_args = [[BasicImagePreprocessingPipeline], [NoAugmentationPipeline]]
    applies the two augmentations to the dataset in sequence.

    :param datasets: list of datasets
    :param n_examples: number of examples to take
    :param batch_size: batch size
    :param map_fn: function to map over the data for augmentation
    :param map_fn_args: list of lists. Each inner list contains arguments to pass to map_fn.
    :return: the list of augmented datasets
    """

    augmented_datasets = []
    for dataset in datasets:
        # Take some examples, batch the dataset and enable prefetching
        dataset = dataset.take(n_examples).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        # Map some transformation over the dataset
        for args in map_fn_args:
            dataset = dataset.map(lambda image, label: map_fn(image, label, *args))
        # Append the augmented dataset
        augmented_datasets.append(dataset)

    return augmented_datasets
