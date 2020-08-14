from types import SimpleNamespace
import tensorflow as tf
import augmentation.datasets.utils

WATERBIRDS_CLASSES = ['landbird', 'waterbird']
WATERBIRDS_DOMAINS = ['land', 'water']

# Group Sizes
# ------------------------------------
# [y, z] = [[0, 0], [0, 1], [1, 0], [1, 1]]
#
# Training Set (split = 0)
#  [3498,  184,   56, 1057]
#
# Validation Set (split = 1)
#  [467, 466, 133, 133]
#
# Test Set (split = 2)
#  [2255, 2255,  642,  642]
# ------------------------------------------
train_group_sizes = {(0, 0): 3498, (0, 1): 184, (1, 0): 56, (1, 1): 1057}
val_group_sizes = {(0, 0): 467, (0, 1): 466, (1, 0): 133, (1, 1): 133}
test_group_sizes = {(0, 0): 2255, (0, 1): 2255, (1, 0): 642, (1, 1): 642}


def read_waterbirds_tfrecord(example, batched=True, parallelism=8):
    features = {"image": tf.io.FixedLenFeature([], tf.string),
                "img_id": tf.io.FixedLenFeature([], tf.int64),
                "img_filename": tf.io.FixedLenFeature([], tf.string),
                "place_filename": tf.io.FixedLenFeature([], tf.string),
                "y": tf.io.FixedLenFeature([], tf.int64),
                "split": tf.io.FixedLenFeature([], tf.int64),
                "place": tf.io.FixedLenFeature([], tf.int64)}

    if batched:
        # Parse the TFRecord
        example = tf.io.parse_example(example, features)
        # Decode the image
        image = tf.map_fn(augmentation.datasets.utils.decode_raw_image, example['image'],
                          dtype=tf.uint8, back_prop=False, parallel_iterations=parallelism)
    else:
        # Parse the TFRecord
        example = tf.io.parse_single_example(example, features)
        # Decode the image
        image = augmentation.datasets.utils.decode_raw_image(example['image'])

    # Get all the other tags
    img_id = example['img_id']
    img_filename = example["img_filename"]
    place_filename = example['place_filename']
    y = example['y']
    split = example['split']
    place = example['place']

    return image, img_id, img_filename, place_filename, y, split, place


def get_label_selection_function(label_type):
    if label_type == 'y':
        # Keep only the y labels
        return lambda image, img_id, img_filename, place_filename, y, split, place: \
            (image, y)
    elif label_type == 'z':
        # Keep only the z labels
        return lambda image, img_id, img_filename, place_filename, y, split, place: \
            (image, place)
    elif label_type == 'full':
        # Keep everything
        return lambda image, img_id, img_filename, place_filename, y, split, place: \
            (image, img_id, img_filename, place_filename, y, split, place)
    else:
        raise NotImplementedError


def load_base_variant(data_dir, y_label, z_label, label_type, proc_batch=128):
    # Load up the list of .tfrec files for the train/val/test sets
    waterbirds_dataset = tf.data.Dataset.list_files(f'{data_dir}/*.tfrec', shuffle=False)

    # Construct the TF Dataset from the list of .tfrec files
    waterbirds_dataset = augmentation.datasets.utils. \
        get_dataset_from_list_files_dataset(waterbirds_dataset, proc_batch=proc_batch,
                                            tfrecord_example_reader=read_waterbirds_tfrecord).unbatch()

    # Split the data into train, validation and test
    waterbirds_train = waterbirds_dataset.filter(lambda image, img_id, img_filename, place_filename, y, split, place:
                                                 (split == 0))
    waterbirds_val = waterbirds_dataset.filter(lambda image, img_id, img_filename, place_filename, y, split, place:
                                               (split == 1))
    waterbirds_test = waterbirds_dataset.filter(lambda image, img_id, img_filename, place_filename, y, split, place:
                                                (split == 2))

    if y_label == 0 or y_label == 1:
        # Keep only one of the y_labels
        waterbirds_train = waterbirds_train.filter(lambda image, img_id, img_filename, place_filename, y, split, place:
                                                   (y == y_label))
        waterbirds_val = waterbirds_val.filter(lambda image, img_id, img_filename, place_filename, y, split, place:
                                               (y == y_label))
        waterbirds_test = waterbirds_test.filter(lambda image, img_id, img_filename, place_filename, y, split, place:
                                                 (y == y_label))

    if z_label == 0 or z_label == 1:
        # Keep only one of the z_labels
        waterbirds_train = waterbirds_train.filter(lambda image, img_id, img_filename, place_filename, y, split, place:
                                                   (place == z_label))
        waterbirds_val = waterbirds_val.filter(lambda image, img_id, img_filename, place_filename, y, split, place:
                                               (place == z_label))
        waterbirds_test = waterbirds_test.filter(lambda image, img_id, img_filename, place_filename, y, split, place:
                                                 (place == z_label))

    # Get the label selection function
    label_selection_fn = get_label_selection_function(label_type)

    # Apply the label selection function and cache the dataset into memory since it's quite small
    # \approx 11000 * (224 * 224 * 3)/(1024 * 1024) < 2 GiB
    waterbirds_train = waterbirds_train.map(label_selection_fn).cache()
    waterbirds_val = waterbirds_val.map(label_selection_fn).cache()
    waterbirds_test = waterbirds_test.map(label_selection_fn).cache()

    return waterbirds_train, waterbirds_val, waterbirds_test


def get_waterbirds_dataset_len(y_label, z_label):
    if y_label == -1:
        if z_label == -1:
            entries_to_sum = [(0, 0), (0, 1), (1, 0), (1, 1)]
        else:
            entries_to_sum = [(0, z_label), (1, z_label)]
    else:
        if z_label == -1:
            entries_to_sum = [(y_label, 0), (y_label, 1)]
        else:
            entries_to_sum = [(y_label, z_label)]

    return sum([train_group_sizes[k] for k in entries_to_sum]), \
           sum([val_group_sizes[k] for k in entries_to_sum]), \
           sum([test_group_sizes[k] for k in entries_to_sum])


def load_waterbirds(dataset_name, dataset_version, data_dir):
    assert dataset_name.startswith(
        'waterbirds'), f'Dataset name is {dataset_name}, should be waterbirds/<which_y>/<which_z>/<y or z>'

    # Grab the name of the variant and label type
    y_label, z_label, label_type = dataset_name.split("/")[1:]
    y_label, z_label = int(y_label), int(z_label)
    assert y_label in [-1, 0, 1], f'y_label should be in {-1, 0, 1}, not {y_label}.'
    assert z_label in [-1, 0, 1], f'z_label should be in {-1, 0, 1}, not {z_label}.'
    assert label_type in ['y', 'z', 'full'], 'Label types must be in {y, z, full}.'

    if dataset_version == '1.*.*':
        # Load up the basic dataset
        waterbirds_train, waterbirds_val, waterbirds_test = load_base_variant(data_dir, y_label, z_label, label_type)

        # Compute the lengths of the dataset
        train_dataset_len, val_dataset_len, test_dataset_len = get_waterbirds_dataset_len(y_label, z_label)

        # Make a dataset info namespace to ensure downstream compatibility
        num_classes = 2
        classes = WATERBIRDS_DOMAINS if label_type == 'z' else WATERBIRDS_CLASSES
        dataset_info = SimpleNamespace(features={'label': SimpleNamespace(num_classes=num_classes),
                                                 'image': SimpleNamespace(shape=(224, 224, 3))},
                                       splits={'train': SimpleNamespace(num_examples=train_dataset_len),
                                               'val': SimpleNamespace(num_examples=val_dataset_len),
                                               'test': SimpleNamespace(num_examples=test_dataset_len)},
                                       classes=classes)

        # Return the data sets
        return SimpleNamespace(dataset_info=dataset_info,
                               train_dataset=waterbirds_train,
                               val_dataset=waterbirds_val,
                               test_dataset=waterbirds_test)

    else:
        raise NotImplementedError
