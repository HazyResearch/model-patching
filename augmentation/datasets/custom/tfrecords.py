import tensorflow as tf
import os
import augmentation.datasets.utils


# Basic feature construction, taken from the tutorial on TFRecords
def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def _int_feature(list_of_ints):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def _float_feature(list_of_floats):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def image_label_to_tfrecord(img_bytes, label):
    # Construct a TFRecord Example using an (image, label) pair
    feature = {"image": _bytestring_feature([img_bytes]),
               "label": _int_feature([int(label)])}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def read_image_label_tfrecord(example, batched=True, parallelism=8):
    # Read a TFRecord Example that contains an (image, label) pair
    features = {"image": tf.io.FixedLenFeature([], tf.string),
                "label": tf.io.FixedLenFeature([], tf.int64)}
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
    label = example['label']

    return image, label
