from types import SimpleNamespace

import augmentation.datasets.utils

MNIST_CORRUPTED_VARIANTS = ['identity',
                            'shot_noise',
                            'impulse_noise',
                            'glass_blur',
                            'motion_blur',
                            'shear',
                            'scale',
                            'rotate',
                            'brightness',
                            'translate',
                            'stripe',
                            'fog',
                            'spatter',
                            'dotted_line',
                            'zigzag',
                            'canny_edges']


def load_mnist_spurious_variants(dataset_variant, data_dir, modified_class=7, validation_frac=0.):
    assert dataset_variant in MNIST_CORRUPTED_VARIANTS, f'The requested variant _{dataset_variant}_ does not exist.'

    # Load up the standard MNIST dataset
    mnist_dataset_payload = augmentation.datasets.utils.load_dataset('mnist', '3.*.*', data_dir,
                                                                     validation_frac=validation_frac)
    # Load up the corrupted MNIST dataset
    mnistc_dataset_payload = augmentation.datasets.utils.load_dataset(f'mnist_corrupted/{dataset_variant}', '1.*.*',
                                                                      data_dir,
                                                                      validation_frac=validation_frac)

    # Grab the training and test sets
    mnist_train = mnist_dataset_payload.train_dataset
    mnistc_train = mnistc_dataset_payload.train_dataset

    mnist_test = mnist_dataset_payload.test_dataset
    mnistc_test = mnistc_dataset_payload.test_dataset

    # Construct a dataset -- MNIST spurious,
    # where a class in MNIST is replaced with a corrupted variant of it from MNIST-C
    mnists_train = mnist_train.filter(lambda image, label: label != modified_class).concatenate(
        mnistc_train.filter(lambda image, label: label == modified_class))
    mnists_test = mnist_test.filter(lambda image, label: label != modified_class).concatenate(
        mnistc_test.filter(lambda image, label: label == modified_class))

    # Construct a dataset -- MNIST combined,
    # where each class has digits from both MNIST and MNIST-C (for the chosen corruption)
    mnistcom_train = mnist_train.concatenate(mnistc_train)
    mnistcom_test = mnist_test.concatenate(mnistc_test)

    return mnist_train, mnist_test, mnistc_train, mnistc_test, mnists_train, mnists_test, mnistcom_train, mnistcom_test


def load_mnist_spurious(dataset_name, dataset_version, data_dir, validation_frac):
    assert dataset_name.startswith('mnist_spurious'), f'Dataset name is {dataset_name}, ' \
                                                      f'should be mnist_spurious/<variant>/<modified_class>.'
    # Grab the name of the variant requested
    variant = dataset_name.split("/")[1]
    modified_class = int(dataset_name.split("/")[2])
    assert variant in MNIST_CORRUPTED_VARIANTS, f'Dataset variant {variant} is not available.'
    assert modified_class in range(10), f'Cannot modify class {modified_class}. Pick a class between 0-9.'

    if dataset_version == '1.*.*':
        # Load up the standard MNIST dataset
        mnist_dataset_payload = augmentation.datasets.utils.load_dataset(dataset_name='mnist',
                                                                         dataset_version='3.*.*',
                                                                         data_dir=data_dir,
                                                                         validation_frac=validation_frac)
        # Load up the corrupted MNIST dataset
        mnistc_dataset_payload = augmentation.datasets.utils.load_dataset(dataset_name=f'mnist_corrupted/{variant}',
                                                                          dataset_version='1.*.*',
                                                                          data_dir=data_dir,
                                                                          validation_frac=validation_frac)

        # Construct the dataset by replacing a class in MNIST with a corrupted variant of it from MNIST-C
        mnists_train = mnist_dataset_payload.train_dataset.filter(lambda image, label: label != modified_class). \
            concatenate(mnistc_dataset_payload.train_dataset.filter(lambda image, label: label == modified_class))
        mnists_val = mnist_dataset_payload.val_dataset.filter(lambda image, label: label != modified_class). \
            concatenate(mnistc_dataset_payload.val_dataset.filter(lambda image, label: label == modified_class))
        mnists_test = mnist_dataset_payload.test_dataset.filter(lambda image, label: label != modified_class). \
            concatenate(mnistc_dataset_payload.test_dataset.filter(lambda image, label: label == modified_class))

        # Make a dataset info namespace to ensure downstream compatibility
        num_classes = mnist_dataset_payload.dataset_info.features['label'].num_classes
        shape = mnist_dataset_payload.dataset_info.features['image'].shape
        num_examples = mnist_dataset_payload.dataset_info.splits['train'].num_examples

        dataset_info = SimpleNamespace(features={'label': SimpleNamespace(num_classes=num_classes),
                                                 'image': SimpleNamespace(shape=shape)},
                                       splits={'train': SimpleNamespace(num_examples=num_examples)})

        return SimpleNamespace(dataset_info=dataset_info,
                               train_dataset=mnists_train,
                               val_dataset=mnists_val,
                               test_dataset=mnists_test)
    else:
        raise NotImplementedError


def load_mnist_combined(dataset_name, dataset_version, data_dir, validation_frac):
    assert dataset_name.startswith('mnist_combined'), f'Dataset name is {dataset_name}, ' \
                                                      f'should be mnist_combined/<variant>.'
    # Grab the name of the variant requested
    variant = dataset_name.split("/")[1]
    assert variant in MNIST_CORRUPTED_VARIANTS, f'Dataset variant {variant} is not available.'

    if dataset_version == '1.*.*':
        # Load up the standard MNIST dataset
        mnist_dataset_payload = augmentation.datasets.utils.load_dataset(dataset_name='mnist',
                                                                         dataset_version='3.*.*',
                                                                         data_dir=data_dir,
                                                                         validation_frac=validation_frac)
        # Load up the corrupted MNIST dataset
        mnistc_dataset_payload = augmentation.datasets.utils.load_dataset(dataset_name=f'mnist_corrupted/{variant}',
                                                                          dataset_version='1.*.*',
                                                                          data_dir=data_dir,
                                                                          validation_frac=validation_frac)

        # Construct the dataset by combining MNIST and MNIST-Corrupted (for the chosen corruption)
        mnistcom_train = mnist_dataset_payload.train_dataset.concatenate(mnistc_dataset_payload.train_dataset)
        mnistcom_val = mnist_dataset_payload.val_dataset.concatenate(mnistc_dataset_payload.val_dataset)
        mnistcom_test = mnist_dataset_payload.test_dataset.concatenate(mnistc_dataset_payload.test_dataset)

        # Make a dataset info namespace to ensure downstream compatibility
        num_classes = mnist_dataset_payload.dataset_info.features['label'].num_classes
        shape = mnist_dataset_payload.dataset_info.features['image'].shape
        num_examples = mnist_dataset_payload.dataset_info.splits['train'].num_examples + \
                       mnistc_dataset_payload.dataset_info.splits['train'].num_examples

        dataset_info = SimpleNamespace(features={'label': SimpleNamespace(num_classes=num_classes),
                                                 'image': SimpleNamespace(shape=shape)},
                                       splits={'train': SimpleNamespace(num_examples=num_examples)})

        return SimpleNamespace(dataset_info=dataset_info,
                               train_dataset=mnistcom_train,
                               val_dataset=mnistcom_val,
                               test_dataset=mnistcom_test)
    else:
        raise NotImplementedError
