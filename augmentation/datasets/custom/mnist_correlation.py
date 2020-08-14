from types import SimpleNamespace

import augmentation.datasets.utils
from augmentation.datasets.custom.mnist import MNIST_CORRUPTED_VARIANTS
import tensorflow as tf


# TODO multihead should be specified as an option to the dataset instead of a separate one
def load_mnist_correlation_yz_multihead(dataset_name, dataset_version, data_dir, validation_frac):
    """
    Dataset of the form 'mnist_correlation_yz_multihead/zigzag/{a}/{b}/{size}/{test_size}/[z]'
    This loads a training set with Y=a and Z=b, of total size {size},
    where Y is the existence of the spurious feature and Z is the digit parity
    If the last option "z" is included, dataset is labeled by z instead of y
    """
    params = dataset_name.split("/")
    assert params[0] == 'mnist_correlation_yz_multihead', f'Dataset name is {dataset_name}, ' \
                                                          f'should be mnist_correlation_yz_multihead/<variant>/<y_class>/<z_class>/<size>/<test_size>/<label>.'
    variant = params[1]
    y = int(params[2])
    z = int(params[3])
    size = int(params[4])
    test_size = int(params[5])
    label_var = params[6] if len(params) > 6 else 'y'
    assert variant in MNIST_CORRUPTED_VARIANTS, f'Dataset variant {variant} is not available.'
    assert y in [0, 1] and z in [0, 1], f'Classes Y={y} and Z={z} must be in {0, 1}.'
    assert label_var == 'y' or label_var == 'z'

    if z == 0:
        # Load up the standard MNIST dataset
        mnists_dataset_payload = augmentation.datasets.utils.load_dataset(dataset_name='mnist',
                                                                          dataset_version='3.*.*',
                                                                          data_dir=data_dir,
                                                                          validation_frac=validation_frac)
    else:
        # Load up the corrupted MNIST dataset
        mnists_dataset_payload = augmentation.datasets.utils.load_dataset(dataset_name=f'mnist_corrupted/{variant}',
                                                                          dataset_version='1.*.*',
                                                                          data_dir=data_dir,
                                                                          validation_frac=validation_frac)

    mnists_train = mnists_dataset_payload.train_dataset. \
        filter(lambda image, label: label % 2 == y)
    mnists_val = mnists_dataset_payload.val_dataset. \
        filter(lambda image, label: label % 2 == y)
    mnists_test = mnists_dataset_payload.test_dataset. \
        filter(lambda image, label: label % 2 == y)
    train_sz_ = augmentation.datasets.utils.dataset_len(mnists_train)
    val_sz_ = augmentation.datasets.utils.dataset_len(mnists_val)
    test_sz_ = augmentation.datasets.utils.dataset_len(mnists_test)

    size = train_sz_ if size == -1 else size
    test_size = test_sz_ if test_size == -1 else min(test_size, test_sz_)
    assert size <= train_sz_ + val_sz_, f'Dataset size {size} for {dataset_name} should be at most {train_sz_ + val_sz_}.'
    val_size = int(size * validation_frac)

    if z == 0:
        mnists_train = mnists_train.take(size - val_size)
        mnists_val = mnists_val.take(val_size)
        mnists_test = mnists_test.take(test_size)
    else:
        mnists_train = mnists_train.skip(train_sz_ - (size - val_size))
        mnists_val = mnists_val.skip(val_sz_ - val_size)
        mnists_test = mnists_test.skip(test_sz_ - test_size)

    # relabel labels to 0/1
    if label_var == 'y':
        mnists_train = mnists_train.map(lambda image, label: (image, y))
        mnists_val = mnists_val.map(lambda image, label: (image, y))
        mnists_test = mnists_test.map(lambda image, label: (image, y))
    if label_var == 'z':
        mnists_train = mnists_train.map(lambda image, label: (image, z))
        mnists_val = mnists_val.map(lambda image, label: (image, z))
        mnists_test = mnists_test.map(lambda image, label: (image, z))

    print(
        f'{dataset_name} splits: {augmentation.datasets.utils.dataset_len(mnists_train)}, '
        f'{augmentation.datasets.utils.dataset_len(mnists_val)}, {augmentation.datasets.utils.dataset_len(mnists_test)}')

    # Make a dataset info namespace to ensure downstream compatibility
    # num_classes = mnists_dataset_payload.dataset_info.features['label'].num_classes
    num_classes = 2
    shape = mnists_dataset_payload.dataset_info.features['image'].shape
    num_examples = size

    # Change to multihead binary classification
    num_classes = 1
    mnists_train = mnists_train.map(lambda x, y: (x, tf.convert_to_tensor(y)[..., None]))
    mnists_val = mnists_val.map(lambda x, y: (x, tf.convert_to_tensor(y)[..., None]))
    mnists_test = mnists_test.map(lambda x, y: (x, tf.convert_to_tensor(y)[..., None]))

    dataset_info = SimpleNamespace(features={'label': SimpleNamespace(num_classes=num_classes),
                                             'image': SimpleNamespace(shape=shape)},
                                   splits={'train': SimpleNamespace(num_examples=num_examples)})

    if label_var == 'y':
        dataset_info.classes = ['parity']
    else:
        dataset_info.classes = ['corruption']

    return SimpleNamespace(dataset_info=dataset_info,
                           train_dataset=mnists_train,
                           val_dataset=mnists_val,
                           test_dataset=mnists_test)


def load_mnist_correlation_yz(dataset_name, dataset_version, data_dir, validation_frac):
    """
    Dataset of the form 'mnist_correlation_yz/zigzag/{a}/{b}/{size}/{test_size}/[z]'
    This loads a training set with Y=a and Z=b, of total size {size},
    where Y is the existence of the spurious feature and Z is the digit parity
    If the last option "z" is included, dataset is labeled by z instead of y
    """
    params = dataset_name.split("/")
    assert params[0] == 'mnist_correlation_yz', f'Dataset name is {dataset_name}, ' \
                                                f'should be mnist_correlation_yz/<variant>/<y_class>/<z_class>/<size>/<test_size>/<label>.'
    variant = params[1]
    y = int(params[2])
    z = int(params[3])
    size = int(params[4])
    test_size = int(params[5])
    label_var = params[6] if len(params) > 6 else 'y'
    assert variant in MNIST_CORRUPTED_VARIANTS, f'Dataset variant {variant} is not available.'
    assert y in [0, 1] and z in [0, 1], f'Classes Y={y} and Z={z} must be in {0, 1}.'
    assert label_var == 'y' or label_var == 'z'

    if z == 0:
        # Load up the standard MNIST dataset
        mnists_dataset_payload = augmentation.datasets.utils.load_dataset(dataset_name='mnist',
                                                                          dataset_version='3.*.*',
                                                                          data_dir=data_dir,
                                                                          validation_frac=validation_frac)
    else:
        # Load up the corrupted MNIST dataset
        mnists_dataset_payload = augmentation.datasets.utils.load_dataset(dataset_name=f'mnist_corrupted/{variant}',
                                                                          dataset_version='1.*.*',
                                                                          data_dir=data_dir,
                                                                          validation_frac=validation_frac)

    mnists_train = mnists_dataset_payload.train_dataset. \
        filter(lambda image, label: label % 2 == y). \
        cache()
    mnists_val = mnists_dataset_payload.val_dataset. \
        filter(lambda image, label: label % 2 == y). \
        cache()
    mnists_test = mnists_dataset_payload.test_dataset. \
        filter(lambda image, label: label % 2 == y). \
        cache()
    train_sz_ = augmentation.datasets.utils.dataset_len(mnists_train)
    val_sz_ = augmentation.datasets.utils.dataset_len(mnists_val)
    test_sz_ = augmentation.datasets.utils.dataset_len(mnists_test)

    size = train_sz_ if size == -1 else size
    test_size = test_sz_ if test_size == -1 else min(test_size, test_sz_)
    assert size <= train_sz_ + val_sz_, f'Dataset size {size} for {dataset_name} should be at most {train_sz_ + val_sz_}.'
    val_size = int(size * validation_frac)

    if z == 0:
        mnists_train = mnists_train.take(size - val_size)
        mnists_val = mnists_val.take(val_size)
        mnists_test = mnists_test.take(test_size)
    else:
        mnists_train = mnists_train.skip(train_sz_ - (size - val_size))
        mnists_val = mnists_val.skip(val_sz_ - val_size)
        mnists_test = mnists_test.skip(test_sz_ - test_size)

    # relabel labels to 0/1
    if label_var == 'y':
        mnists_train = mnists_train.map(lambda image, label: (image, y))
        mnists_val = mnists_val.map(lambda image, label: (image, y))
        mnists_test = mnists_test.map(lambda image, label: (image, y))
    if label_var == 'z':
        mnists_train = mnists_train.map(lambda image, label: (image, z))
        mnists_val = mnists_val.map(lambda image, label: (image, z))
        mnists_test = mnists_test.map(lambda image, label: (image, z))

    print(
        f'{dataset_name} splits: {augmentation.datasets.utils.dataset_len(mnists_train)}, '
        f'{augmentation.datasets.utils.dataset_len(mnists_val)}, {augmentation.datasets.utils.dataset_len(mnists_test)}')

    # Make a dataset info namespace to ensure downstream compatibility
    num_classes = 2
    shape = mnists_dataset_payload.dataset_info.features['image'].shape
    num_examples = size

    dataset_info = SimpleNamespace(features={'label': SimpleNamespace(num_classes=num_classes),
                                             'image': SimpleNamespace(shape=shape)},
                                   splits={'train': SimpleNamespace(num_examples=num_examples)})

    return SimpleNamespace(dataset_info=dataset_info,
                           train_dataset=mnists_train,
                           val_dataset=mnists_val,
                           test_dataset=mnists_test)


def load_mnist_correlation_y(dataset_name, dataset_version, data_dir, validation_frac):
    """
    Dataset of the form 'mnist_correlation_y/zigzag/{a}/{p}/{size}/[z]'
    This loads a training set with Y=a and p(Z=a) = p, of total size {size},
    where Y is the existence of the spurious feature and Z is the digit parity
    """
    params = dataset_name.split("/")
    assert params[0] == 'mnist_correlation_y', f'Dataset name is {dataset_name}, ' \
                                               f'should be mnist_correlation_y/<variant>/<y_class>/<z_prob>/<size>/<label>.'
    variant = params[1]
    y = int(params[2])
    p_z = float(params[3])
    size = int(params[4])
    label_var = params[5] if len(params) > 5 else 'y'
    if size == -1: size = 30000  # TODO FIX THIS - WHY ISN'T MNIST CLASS BALANCED
    assert variant in MNIST_CORRUPTED_VARIANTS, f'Dataset variant {variant} is not available.'
    assert y in [0, 1], f'Class Y={y} must be in {0, 1}.'
    assert 0. <= p_z <= 1., f'Probability p(Z=y)={p_z} should be in [0.0, 1.0].'
    assert size <= 30000, f'Dataset size {size} should be at most 30000.'
    assert label_var == 'y' or label_var == 'z'

    size_z = int(size * p_z)
    test_size_z = int(5000 * p_z)
    dataset_z = load_mnist_correlation_yz(
        f'mnist_correlation_yz/{variant}/{y}/{y}/{size_z}/{test_size_z}/{label_var}',
        dataset_version, data_dir, validation_frac)
    dataset_z_ = load_mnist_correlation_yz(
        f'mnist_correlation_yz/{variant}/{y}/{1 - y}/{size - size_z}/{5000 - test_size_z}/{label_var}',
        dataset_version, data_dir, validation_frac)
    dataset_z.train_dataset = dataset_z.train_dataset.concatenate(dataset_z_.train_dataset)
    dataset_z.val_dataset = dataset_z.val_dataset.concatenate(dataset_z_.val_dataset)
    dataset_z.test_dataset = dataset_z.test_dataset.concatenate(dataset_z_.test_dataset)

    dataset_z.dataset_info.splits['train'].num_examples += dataset_z_.dataset_info.splits['train'].num_examples
    return dataset_z


def load_mnist_correlation_partial(dataset_name, dataset_version, data_dir, validation_frac):
    """
    Dataset of the form 'mnist_correlation_partial/{variant}/{z}/{size}/[label_var]'.
    Creates a balanced dataset with Pr(Y = 1) = Pr(Y = 0) = 1/2 and Pr(Z = z) = 1.

    Use this for training CycleGANs:
    E.g. for 'mnist_correlation/zigzag/p/size/y' as your main dataset
    you can create source and target datasets using
    'mnist_correlation_partial/zigzag/0/some_size/y' and
    'mnist_correlation_partial/zigzag/1/some_size/y'
    """
    params = dataset_name.split("/")
    assert params[0] == 'mnist_correlation_partial', \
        f'Dataset name is {dataset_name}, should be mnist_correlation_partial/<variant>/<z_class>/<size>/<label>.'

    variant = params[1]
    z = int(params[2])
    size = int(params[3])
    label_var = params[4] if len(params) > 4 else 'y'
    size = 30000 if size == -1 else size

    assert variant in MNIST_CORRUPTED_VARIANTS, f'Dataset variant {variant} is not available.'
    assert z == 1 or z == 0, f'Clean should be 0 or 1 not {z}.'
    assert size <= 30000, f'Dataset size {size} should be at most 30000.'
    assert size % 2 == 0, f"C'mon why would you use an odd dataset size..."
    assert label_var == 'y' or label_var == 'z'

    dataset_evens = load_mnist_correlation_yz(
        f'mnist_correlation_yz/{variant}/{0}/{z}/{size // 2}/{5000}/{label_var}',
        dataset_version, data_dir, validation_frac)
    dataset_odds = load_mnist_correlation_yz(
        f'mnist_correlation_yz/{variant}/{1}/{z}/{size // 2}/{5000}/{label_var}',
        dataset_version, data_dir, validation_frac)

    dataset_evens.train_dataset = dataset_evens.train_dataset.concatenate(dataset_odds.train_dataset)
    dataset_evens.val_dataset = dataset_evens.val_dataset.concatenate(dataset_odds.val_dataset)
    dataset_evens.test_dataset = dataset_evens.test_dataset.concatenate(dataset_odds.test_dataset)
    dataset_evens.dataset_info.splits['train'].num_examples += dataset_odds.dataset_info.splits['train'].num_examples
    return dataset_evens


def load_mnist_correlation(dataset_name, dataset_version, data_dir, validation_frac):
    """
    Dataset of the form 'mnist_correlation/{variant}/{p}/{size}/[label_var]'
    This loads a training+val set of total size ~{size}, where the spurious feature {variant} and digit parity are correlated
    More precisely:
      - Y=parity and Z=variant are marginally balanced [p(Y=0) = p(Y=1) = P(Z=0) = P(Z=1) = 1/2]
      - P(Y=a | Z=a) = P(Z=a | Y=a) = p
      - Alternatively, Y and Z are correlated with strength (2p-1)
    """
    params = dataset_name.split("/")
    assert params[0] == 'mnist_correlation', \
        f'Dataset name is {dataset_name}, should be mnist_correlation/<variant>/<prob>/<size>/<label>.'

    variant = params[1]
    p = float(params[2])
    size = int(params[3])
    label_var = params[4] if len(params) > 4 else 'y'
    size = 60000 if size == -1 else size

    assert variant in MNIST_CORRUPTED_VARIANTS, f'Dataset variant {variant} is not available.'
    assert 0. <= p <= 1., f'Probability p(Z=y)={p} should be in [0.0, 1.0].'
    assert size <= 60000, f'Dataset size {size} should be at most 30000.'
    assert size % 2 == 0, f"C'mon why would you use an odd dataset size..."
    assert label_var == 'y' or label_var == 'z'

    dataset_evens = load_mnist_correlation_y(
        f'mnist_correlation_y/{variant}/{0}/{p}/{size // 2}/{label_var}',
        dataset_version, data_dir, validation_frac)
    dataset_odds = load_mnist_correlation_y(
        f'mnist_correlation_y/{variant}/{1}/{p}/{size // 2}/{label_var}',
        dataset_version, data_dir, validation_frac)
    dataset_evens.train_dataset = dataset_evens.train_dataset.concatenate(dataset_odds.train_dataset)
    dataset_evens.val_dataset = dataset_evens.val_dataset.concatenate(dataset_odds.val_dataset)
    dataset_evens.test_dataset = dataset_evens.test_dataset.concatenate(dataset_odds.test_dataset)
    dataset_evens.dataset_info.splits['train'].num_examples += dataset_odds.dataset_info.splits['train'].num_examples
    return dataset_evens


def load_mnist_correlation_multihead(dataset_name, dataset_version, data_dir, validation_frac):
    """
    Dataset of the form 'mnist_correlation_multihead/{variant}/{p}/{size}/[label_var]'
    This loads a training+val set of total size ~{size}, where the spurious feature {variant} and digit parity are correlated
    More precisely:
      - Y=parity and Z=variant are marginally balanced [p(Y=0) = p(Y=1) = P(Z=0) = P(Z=1) = 1/2]
      - P(Y=a | Z=a) = P(Z=a | Y=a) = p
      - Alternatively, Y and Z are correlated with strength (2p-1)
    """
    params = dataset_name.split("/")
    assert params[0] == 'mnist_correlation_multihead', \
        f'Dataset name is {dataset_name}, should be mnist_correlation_multihead/<variant>/<prob>/<size>/<label>.'

    variant = params[1]
    p = float(params[2])
    size = int(params[3])
    label_var = params[4] if len(params) > 4 else 'y'
    size = 60000 if size == -1 else size

    assert variant in MNIST_CORRUPTED_VARIANTS, f'Dataset variant {variant} is not available.'
    assert 0. <= p <= 1., f'Probability p(Z=y)={p} should be in [0.0, 1.0].'
    assert size <= 60000, f'Dataset size {size} should be at most 30000.'
    assert size % 2 == 0, f"C'mon why would you use an odd dataset size..."
    assert label_var == 'y' or label_var == 'z'

    dataset_evens = load_mnist_correlation_y(
        f'mnist_correlation_y/{variant}/{0}/{p}/{size // 2}/{label_var}',
        dataset_version, data_dir, validation_frac)
    dataset_odds = load_mnist_correlation_y(
        f'mnist_correlation_y/{variant}/{1}/{p}/{size // 2}/{label_var}',
        dataset_version, data_dir, validation_frac)
    dataset_evens.train_dataset = dataset_evens.train_dataset.concatenate(dataset_odds.train_dataset)
    dataset_evens.val_dataset = dataset_evens.val_dataset.concatenate(dataset_odds.val_dataset)
    dataset_evens.test_dataset = dataset_evens.test_dataset.concatenate(dataset_odds.test_dataset)
    dataset_evens.dataset_info.splits['train'].num_examples += dataset_odds.dataset_info.splits['train'].num_examples

    # To turn this into a "multihead" method: pretend there is 1 binary head
    dataset_evens.dataset_info.num_classes = 1
    if label_var == 'y':
        dataset_evens.dataset_info.classes = ['parity']
    else:
        dataset_evens.dataset_info.classes = ['corruption']
    dataset_evens.train_dataset = dataset_evens.train_dataset.map(lambda x, y: (x, tf.convert_to_tensor(y)[..., None]))
    dataset_evens.val_dataset = dataset_evens.val_dataset.map(lambda x, y: (x, tf.convert_to_tensor(y)[..., None]))
    dataset_evens.test_dataset = dataset_evens.test_dataset.map(lambda x, y: (x, tf.convert_to_tensor(y)[..., None]))
    return dataset_evens


