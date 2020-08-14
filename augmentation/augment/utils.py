import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmenters import *
from augmentation.methods.cyclegan.models import *
from augmentation.autoaugment import augmentation_transforms
from augmentation.autoaugment.augmentation_transforms import MEANS, STDS
from augmentation.autoaugment.policies import good_policies
from augmentation.utilities.wandb import *
from scipy import ndimage


def compose_augmentations(x, augmentations):
    for f in augmentations:
        x = f(x)
    return x


def create_augmentation_pipeline(daug_pipeline, daug_pipeline_args, broadcast_to=1):
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


def create_augmentation_pipelines(train_daug_pipeline, train_daug_pipeline_args,
                                  val_daug_pipeline, val_daug_pipeline_args,
                                  test_daug_pipeline, test_daug_pipeline_args):
    # Setup the augmentation pipeline we'll be using
    train_augmentations = create_augmentation_pipeline(train_daug_pipeline, train_daug_pipeline_args)
    val_augmentations = create_augmentation_pipeline(val_daug_pipeline, val_daug_pipeline_args)
    test_augmentations = create_augmentation_pipeline(test_daug_pipeline, test_daug_pipeline_args)

    return train_augmentations, val_augmentations, test_augmentations


def create_multiple_train_eval_augmentation_pipelines(train_augmentation_pipelines,
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
        (create_augmentation_pipeline(*z))
        for z in zip(train_augmentation_pipelines * broadcast_train_to,
                     train_augmentation_pipelines_args * broadcast_train_to)
    ]

    eval_augmentations = [
        (create_augmentation_pipeline(*z))
        for z in zip(eval_augmentation_pipelines * broadcast_eval_to,
                     eval_augmentation_pipelines_args * broadcast_eval_to)
    ]
    return train_augmentations, eval_augmentations


def create_multiple_augmentation_pipelines(train_daug_pipelines, train_daug_pipelines_args,
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
        create_augmentation_pipelines(*z)
        for z in zip(train_daug_pipelines * broadcast_train_to,
                     train_daug_pipelines_args * broadcast_train_to,
                     val_daug_pipelines * broadcast_val_to,
                     val_daug_pipelines_args * broadcast_val_to,
                     test_daug_pipelines * broadcast_test_to,
                     test_daug_pipelines_args * broadcast_test_to)
    ]
    return tuple(zip(*augmentations))


class AugmentationPipeline:
    """
    Base class for performing augmentations.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, data, *args, **kwargs):
        if len(data.shape) == 4:
            return np.array([self.transform(e) for e in data])
        elif len(data.shape) == 3:
            return np.array(self.transform(data))
        else:
            raise NotImplementedError

    def improve(self, *args, **kwargs):
        pass

    def transform(self, data, *args, **kwargs):
        pass


class NoAugmentationPipeline(AugmentationPipeline):
    """
    An empty augmentation pipeline that returns the data as-is.
    """

    def __init__(self, *args, **kwargs):
        super(NoAugmentationPipeline, self).__init__(*args, **kwargs)

    def transform(self, data, *args, **kwargs):
        return data


class ResizeImage(AugmentationPipeline):

    def __init__(self, size, *args, **kwargs):
        super(ResizeImage, self).__init__(*args, **kwargs)

        self.resizer = iaa.Sequential([iaa.Resize(size=size)])

    def transform(self, data, *args, **kwargs):
        return self.resizer(images=data)

    def __call__(self, data, *args, **kwargs):
        if len(data.shape) == 4 or len(data.shape) == 3:
            return np.array(self.transform(data))
        else:
            raise NotImplementedError


class ImgAugAugmentationPipeline(AugmentationPipeline):

    def __init__(self, pipeline, *args, **kwargs):
        super(ImgAugAugmentationPipeline, self).__init__(*args, **kwargs)

        self.iaa_pipeline = iaa.Sequential([])
        if pipeline == 'fliplr:crop':
            self.iaa_pipeline.append(iaa.Fliplr(0.5))
            self.iaa_pipeline.append(iaa.Crop(percent=(0, 0.10), keep_size=True, sample_independently=True))
        elif pipeline == 'heavy':
            self.iaa_pipeline.append(self.create_heavy_augmentation_pipeline())
        else:
            raise NotImplementedError

    def create_heavy_augmentation_pipeline(self):
        # Adapting most of what AugMix/AutoAugment/RandAugment uses
        # -----------------
        # Shear (-30, 30): this is simplified from the shear_x and shear_y ops used
        shear = iaa.Affine(shear=(-30, 30))

        # Translation (-150 pixels, 150 pixels): this is simplified from the translate_x and translate_y ops used
        # We translate 20% of the image independently in either direction
        translate = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})

        # Rotation (-30 degrees, 30 degrees)
        rotate = iaa.Affine(rotate=(-30, 30))

        # Auto Contrast: can't find this in imgaug
        # auto_contrast = iaa.Identity()

        # Invert
        invert = iaa.Invert()

        # Equalize
        equalize = iaa.HistogramEqualization()

        # Solarize (0, 255)
        solarize = iaa.Invert(threshold=(0, 255))

        # Posterize (4, 8) bits
        posterize = iaa.Posterize(nb_bits=(4, 8))

        # Contrast
        contrast = iaa.GammaContrast(gamma=(0.1, 1.9))

        # Color
        color = iaa.MultiplyHue()

        # Brightness
        brightness = iaa.Multiply((0.1, 1.9))

        # Sharpness
        sharpness = iaa.Sharpen(alpha=(0.1, 1.0), lightness=1.0)

        # Cutout: approximates Cutout
        cutout = iaa.CoarseDropout(p=0.1, size_percent=0.02)

        # Sample Pairing: linearly mixes images (by convex combination)
        mixup = iaa.Lambda(self.linear_mixup)

        # Flip
        flip = iaa.Fliplr(0.5)

        # Sample between 1 and 3 of these augmentations and chain them
        return iaa.SomeOf((1, 3), [shear,
                                   translate,
                                   rotate,
                                   invert,
                                   equalize,
                                   solarize,
                                   posterize,
                                   contrast,
                                   color,
                                   brightness,
                                   sharpness,
                                   cutout,
                                   mixup,
                                   flip], random_order=True)

    def linear_mixup(self, images, random_state, parents, hooks):
        randomized_images = images[random_state.permutation(images.shape[0])]
        scale = random_state.uniform(0.5, 1.0, size=images.shape[0]).reshape(images.shape[0], 1, 1, 1)
        return (scale * images + (1 - scale) * randomized_images).astype(np.uint8)

    def transform(self, data, *args, **kwargs):
        return self.iaa_pipeline(images=data)

    def __call__(self, data, *args, **kwargs):
        if len(data.shape) == 4 or len(data.shape) == 3:
            return np.array(self.transform(data))
        else:
            raise NotImplementedError


class BasicImagePreprocessingPipeline(AugmentationPipeline):
    """
    A basic image preprocessing pipeline that
    (1) casts an image to tf.float32,
    (2) normalizes pixel values to lie in [0, 1] or [-1, 1].
    """

    def __init__(self, type='zero-one', *args, **kwargs):
        super(BasicImagePreprocessingPipeline, self).__init__(*args, **kwargs)
        if type == 'zero-one':
            self.transform = self.zero_one_normalization
            self.zero_one_conversion = lambda x: x
        elif type == 'minusone-one':
            self.transform = self.minusone_one_normalization
            self.zero_one_conversion = self.minuseone_one_to_zero_one_normalization
        elif type == 'minusone-one-to-zero-one':
            self.transform = self.minuseone_one_to_zero_one_normalization
        elif type == 'grayscale':
            self.transform = self.grayscale
        elif type == 'none':
            self.transform = lambda x: x
        else:
            raise NotImplementedError

    def zero_one_normalization(self, image):
        return image.astype(np.float32) / 255.

    def inverse_zero_one_normalization(self, image):
        return (image * 255.).astype(np.uint8)

    def minusone_one_normalization(self, image):
        return (image.astype(np.float32) / 127.5) - 1.

    def minuseone_one_to_zero_one_normalization(self, image):
        return image * 0.5 + 0.5

    def grayscale(self, image):
        # See https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])


class CIFAR10PreprocessingPipeline(BasicImagePreprocessingPipeline):
    """
    A basic image preprocessing pipeline for the CIFAR10 dataset. It first calls the BasicImagePreprocessingPipeline,
    followed by standardizing the images using a precomputed mean and standard deviation.

    The mean and std values are taken from the AutoAugment repository.
    """

    def __init__(self, *args, **kwargs):
        super(CIFAR10PreprocessingPipeline, self).__init__(*args, **kwargs)

    def transform(self, image, *args, **kwargs):
        # First do basic preprocessing
        image = BasicImagePreprocessingPipeline.transform(self, image, *args, **kwargs)
        # Then subtract off the mean and std
        return (image - MEANS) / STDS


class OnlyImageNetPreprocessingPipeline(AugmentationPipeline):
    """
    A basic image preprocessing pipeline for ImageNet.
    """

    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    def __init__(self, *args, **kwargs):
        super(OnlyImageNetPreprocessingPipeline, self).__init__(*args, **kwargs)

    def transform(self, image, *args, **kwargs):
        # Subtract off the mean and std
        return (image - self.MEANS) / self.STDS

    def __call__(self, data, *args, **kwargs):
        if len(data.shape) == 4 or len(data.shape) == 3:
            return np.array(self.transform(data))
        else:
            raise NotImplementedError


class ImageNetPreprocessingPipeline(AugmentationPipeline):
    """
    A basic image preprocessing pipeline for the CIFAR10 dataset. It first calls the BasicImagePreprocessingPipeline,
    followed by standardizing the images using a precomputed mean and standard deviation.

    The mean and std values are taken from the AutoAugment repository.
    """

    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    def __init__(self, *args, **kwargs):
        super(ImageNetPreprocessingPipeline, self).__init__(*args, **kwargs)
        self.basic_preprocessor = BasicImagePreprocessingPipeline()

    def transform(self, image, *args, **kwargs):
        # First do basic preprocessing
        image = self.basic_preprocessor(image)
        # Then subtract off the mean and std
        return (image - self.MEANS) / self.STDS

    def __call__(self, data, *args, **kwargs):
        if len(data.shape) == 4 or len(data.shape) == 3:
            return np.array(self.transform(data))
        else:
            raise NotImplementedError


class HeuristicImageAugmentationPipeline(AugmentationPipeline):
    """
    A variety of heuristic pipelines for data augmentation.
    """

    def __init__(self, heuristic, *args, **kwargs):
        super(HeuristicImageAugmentationPipeline, self).__init__(*args, **kwargs)
        if heuristic == 'pad:crop:flip':
            self.transform = self.pad_crop_flip
        elif heuristic == 'pad:crop':
            self.transform = self.pad_crop
        elif heuristic == 'cutout':
            self.transform = self.cutout
        elif heuristic == 'pad:crop:flip:cutout':
            self.transform = self.pad_crop_flip_cutout
        elif heuristic == 'pad16:crop:flip:cutout':
            self.transform = lambda x: self.pad_crop_flip_cutout(x, padding=16)
        elif heuristic == 'rotr':
            self.transform = self.rotate_random
        else:
            raise NotImplementedError

    def pad_crop_flip(self, image, padding=4):
        return augmentation_transforms.random_flip(augmentation_transforms.zero_pad_and_crop(image, padding))

    def pad_crop(self, image, padding=4):
        return augmentation_transforms.zero_pad_and_crop(image, padding)

    def cutout(self, image, size=16):
        return augmentation_transforms.cutout_numpy(image, size)

    def pad_crop_flip_cutout(self, image, padding=4, cutout_size=16):
        image = self.pad_crop_flip(image, padding)
        return self.cutout(image, cutout_size)

    def rotate_random(self, image, max_angle=45):
        return ndimage.rotate(image, np.random.uniform(-max_angle, max_angle), reshape=False)


class AutoAugmentPipeline(AugmentationPipeline):
    """
    Implements the augmentation pipeline learned by AutoAugment.

    Code for AutoAugment is taken from
    https://github.com/tensorflow/models/tree/048f5a9541c1400c0345bab4e3d9b5c9eb234989/research/autoaugment
    """

    def __init__(self, dataset, *args, **kwargs):
        super(AutoAugmentPipeline, self).__init__(*args, **kwargs)
        if dataset == 'cifar10':
            self.policy = self._cifar_policy()
        elif dataset == 'imagenet':
            self.policy = self._imagenet_policy()
        elif dataset == 'svhn':
            self.policy = self._svhn_policy()
        else:
            raise NotImplementedError('AutoAugment only supports (\'cifar10\', \'imagenet\', \'svhn\') policies.')

    def transform(self, image, *args, **kwargs):
        # Implementation is borrowed from
        # lines 152-162 in
        # https://github.com/tensorflow/models/blob/048f5a9541c1400c0345bab4e3d9b5c9eb234989/research/autoaugment/data_utils.py

        # Convert tensor to a numpy array
        image = np.array(image)
        # Randomly sample one of the AutoAugment policies
        epoch_policy = self.policy[np.random.choice(len(self.policy))]
        # Apply the policy transformation to the image
        image = augmentation_transforms.apply_policy(epoch_policy, image)
        # Zero-pad, crop and flip the image randomly
        image = augmentation_transforms.random_flip(augmentation_transforms.zero_pad_and_crop(image, 4))
        # Apply cutout to the image
        image = augmentation_transforms.cutout_numpy(image)
        return image

    def _cifar_policy(self):
        return good_policies()

    def _imagenet_policy(self):
        raise NotImplementedError

    def _svhn_policy(self):
        raise NotImplementedError


class AutoAugmentCIFAR10Pipeline(CIFAR10PreprocessingPipeline, AutoAugmentPipeline):

    def __init__(self, *args, **kwargs):
        super(AutoAugmentCIFAR10Pipeline, self).__init__(dataset='cifar10', *args, **kwargs)

    def transform(self, image, *args, **kwargs):
        image = CIFAR10PreprocessingPipeline.transform(self, image, *args, **kwargs)
        image = AutoAugmentPipeline.transform(self, image, *args, **kwargs)
        return image


class RandomPolicyImageAugmentationPipeline(AugmentationPipeline):

    def __init__(self, policy, *args, **kwargs):
        super(RandomPolicyImageAugmentationPipeline, self).__init__()

        if policy == 'basic':
            pass
        else:
            raise NotImplementedError

    def _basic_policy(self):
        pass

    def transform(self, image, *args, **kwargs):
        return image


class TandaPipeline(AugmentationPipeline):

    def __init__(self, *args, **kwargs):
        super(TandaPipeline, self).__init__(*args, **kwargs)
        pass

    def improve(self):
        pass


class WandbModelPseudoLabelingPipeline(AugmentationPipeline):
    LABELING_METHODS = ['argmax', 'sigmoid_argmax', 'sigmoid_threshold']

    def __init__(self,
                 wandb_entity,
                 wandb_project,
                 wandb_run_id,
                 input_shape,
                 n_classes,
                 checkpoint_path='checkpoints/',
                 labeling_method='argmax',
                 placeholder_labels=(),
                 *args, **kwargs):

        super(WandbModelPseudoLabelingPipeline, self).__init__(*args, **kwargs)

        # Load up the Weights and Biases run, get information about the model source and architecture and
        # create the model.
        wandb_run = load_wandb_run(wandb_run_id, wandb_project, wandb_entity)
        self.keras_model = \
            load_pretrained_keras_classification_model(source=wandb_run.cfg['model_source']['value'],
                                                       architecture=wandb_run.cfg['architecture']['value'],
                                                       input_shape=input_shape,
                                                       n_classes=n_classes,
                                                       imagenet_pretrained=False,
                                                       pretraining_source='wandb',
                                                       pretraining_info=f'{wandb_run_id}:{wandb_project}:{wandb_entity}',
                                                       checkpoint_path=checkpoint_path)

        # Assume we only need to normalize to [0, 1] to run the Keras model
        self.basic_preprocessor = BasicImagePreprocessingPipeline()

        # What labels need to be pseudo-labeled? These are the placeholder labels we're replacing
        # If empty, pseudolabel all the data
        # TODO: add support for nonempty placeholder labels (only those labels are pseudolabeled)
        assert len(placeholder_labels) == 0
        self.placeholder_labels = np.array(placeholder_labels)

        assert labeling_method in self.LABELING_METHODS, f'Labeling method {labeling_method} is invalid.'
        self.labeling_method = labeling_method

    def pseudolabel(self, outputs):
        if self.labeling_method == 'argmax':
            return np.argmax(outputs, axis=-1)
        else:
            raise NotImplementedError

    def __call__(self, data, *args, **kwargs):
        return self.transform(data)

    def transform(self, data, *args, **kwargs):
        # The data consists of inputs and labels
        inputs, labels = data

        # Transform the inputs using the model
        outputs = self.keras_model(self.basic_preprocessor(inputs))

        # Create pseudolabels
        pseudolabels = self.pseudolabel(outputs)

        # Return the data along with the pseudolabels
        return inputs, pseudolabels


class BinaryMNISTWandbModelPseudoLabelingPipeline(WandbModelPseudoLabelingPipeline):

    def __init__(self, wandb_entity, wandb_project, wandb_run_id, *args, **kwargs):
        # Initialize the pseudolabeler: we just use the argmax labeling method since this is MNIST and
        # pseudolabel everything
        super(BinaryMNISTWandbModelPseudoLabelingPipeline, self).__init__(wandb_entity=wandb_entity,
                                                                          wandb_project=wandb_project,
                                                                          wandb_run_id=wandb_run_id,
                                                                          input_shape=(28, 28, 1),
                                                                          n_classes=2, *args, **kwargs)


def shuffle_and_split_data(data, proportion):
    """
    Shuffle the data, split the data and return the shuffled data splits along with the permutation applied to the data.
    """
    perm = np.random.permutation(len(data))
    shuffled = data[perm]
    return shuffled[:int(proportion * len(data))], shuffled[int(proportion * len(data)):], perm


def unshuffle_data(data, permutation):
    """
    Unshuffle data that was shuffled using a permutation.
    """
    return data[np.argsort(permutation)]


class PretrainedGenerativeModelAugmentationPipeline(AugmentationPipeline):

    def __init__(self,
                 wandb_entity,
                 wandb_project,
                 wandb_run_id,
                 model_name,
                 keras_model_creation_fn,
                 keras_model_creation_fn_args,
                 basic_preprocessing='minusone-one',
                 step_extractor=None,
                 aug_proportion=0.5,
                 run_in_eval_mode=True,
                 *args, **kwargs):
        super(PretrainedGenerativeModelAugmentationPipeline, self).__init__(*args, **kwargs)

        self.keras_model, _ = load_pretrained_keras_model_from_wandb(wandb_run_id=wandb_run_id,
                                                                     wandb_project=wandb_project,
                                                                     wandb_entity=wandb_entity,
                                                                     keras_model_creation_fn=keras_model_creation_fn,
                                                                     keras_model_creation_fn_args=
                                                                     keras_model_creation_fn_args,
                                                                     model_name=model_name,
                                                                     step_extractor=step_extractor)

        self.basic_preprocessor = BasicImagePreprocessingPipeline(type=basic_preprocessing)
        self.aug_proportion = aug_proportion
        self.training = not run_in_eval_mode

    def __call__(self, data, *args, **kwargs):
        if len(data.shape) == 4:
            return np.array(self.transform(data))
        else:
            raise NotImplementedError

    def transform(self, data, *args, **kwargs):
        # Rescale the data
        data = self.basic_preprocessor(data)
        # Get splits of the data
        split_1, split_2, permutation = shuffle_and_split_data(data, self.aug_proportion)
        # Pass it through the generator
        split_1 = self.keras_model(split_1, training=self.training)
        # Combine the data
        data = np.concatenate([split_1, split_2], axis=0)
        # Unshuffle the data
        data = unshuffle_data(data, permutation)
        # Rescale output to [0, 1]
        return self.basic_preprocessor.zero_one_conversion(data)


class PretrainedMNISTCycleGANAugmentationPipeline(PretrainedGenerativeModelAugmentationPipeline):

    def __init__(self,
                 wandb_entity,
                 wandb_project,
                 wandb_run_id,
                 model_name,
                 aug_proportion=1.0,
                 run_in_eval_mode=False,
                 norm_type='batchnorm',
                 checkpoint_step=-1,
                 *args, **kwargs):
        assert model_name in ['generator_g', 'generator_f'], 'model_name must be {generator_g, generator_f}.'

        super(PretrainedMNISTCycleGANAugmentationPipeline,
              self).__init__(wandb_entity=wandb_entity,
                             wandb_project=wandb_project,
                             wandb_run_id=wandb_run_id,
                             model_name=model_name,
                             keras_model_creation_fn='mnist_unet_generator',
                             keras_model_creation_fn_args={'norm_type': norm_type},
                             step_extractor=particular_checkpoint_step_extractor(checkpoint_step),
                             aug_proportion=aug_proportion,
                             run_in_eval_mode=run_in_eval_mode,
                             *args, **kwargs)


class PretrainedCycleGANAugmentationPipeline(AugmentationPipeline):

    def __init__(self,
                 wandb_entity,
                 wandb_project,
                 wandb_run_id,
                 model_name,
                 keras_model_creation_fn,
                 keras_model_creation_fn_args,
                 step_extractor=None,
                 aug_proportion=0.5,
                 *args, **kwargs):
        super(PretrainedCycleGANAugmentationPipeline, self).__init__(*args, **kwargs)
        raise DeprecationWarning("Please use PretrainedGenerativeModelAugmentationPipeline "
                                 "instead as a drop-in replacement (with an optional argument for the preprocessor).")
        # Load the run
        wandb_run = load_wandb_run(wandb_run_id, wandb_project, wandb_entity)
        # Create the model architecture
        self.keras_model = globals()[keras_model_creation_fn](**keras_model_creation_fn_args)
        # Load up the model weights
        if step_extractor is None:
            load_most_recent_keras_model_weights(self.keras_model, wandb_run, model_name=model_name)
        else:
            load_most_recent_keras_model_weights(self.keras_model, wandb_run,
                                                 model_name=model_name,
                                                 step_extractor=step_extractor)

        self.basic_preprocessor = BasicImagePreprocessingPipeline(type='minusone-one')
        self.aug_proportion = aug_proportion

    def __call__(self, data, *args, **kwargs):
        if len(data.shape) == 4:
            return np.array(self.transform(data))
        else:
            raise NotImplementedError

    def transform(self, data, *args, **kwargs):
        # Rescale the data to [-1, 1]
        data = self.basic_preprocessor(data)
        # Get splits of the data
        split_1, split_2, permutation = shuffle_and_split_data(data, self.aug_proportion)
        # Pass it through the generator
        split_1 = self.keras_model(split_1, training=False)
        # Combine the data
        data = np.concatenate([split_1, split_2], axis=0)
        # Unshuffle the data
        data = unshuffle_data(data, permutation)
        # Rescale output to [0, 1]
        return data * 0.5 + 0.5


class PretrainedCycleGANBatchBalancingAugmentationPipeline(AugmentationPipeline):

    def __init__(self,
                 wandb_entity,
                 wandb_project,
                 wandb_run_id,
                 generator_1_name,
                 generator_2_name,
                 keras_model_creation_fn,
                 keras_model_creation_fn_args,
                 step_extractor=None,
                 basic_preprocessing='minusone-one',
                 aug_proportion=1.0,
                 generator_1_balance=0.5,
                 run_in_eval_mode=True,
                 *args, **kwargs):
        super(PretrainedCycleGANBatchBalancingAugmentationPipeline, self).__init__(*args, **kwargs)

        # Load up the generators for both domains
        self.generator_1, _ = load_pretrained_keras_model_from_wandb(wandb_run_id=wandb_run_id,
                                                                     wandb_project=wandb_project,
                                                                     wandb_entity=wandb_entity,
                                                                     keras_model_creation_fn=keras_model_creation_fn,
                                                                     keras_model_creation_fn_args=
                                                                     keras_model_creation_fn_args,
                                                                     model_name=generator_1_name,
                                                                     step_extractor=step_extractor)

        self.generator_2, _ = load_pretrained_keras_model_from_wandb(wandb_run_id=wandb_run_id,
                                                                     wandb_project=wandb_project,
                                                                     wandb_entity=wandb_entity,
                                                                     keras_model_creation_fn=keras_model_creation_fn,
                                                                     keras_model_creation_fn_args=
                                                                     keras_model_creation_fn_args,
                                                                     model_name=generator_2_name,
                                                                     step_extractor=step_extractor)

        # Set up the preprocessing
        self.basic_preprocessor = BasicImagePreprocessingPipeline(type=basic_preprocessing)
        # The proportion of examples that are augmented in a data batch
        self.aug_proportion = aug_proportion
        # The proportion of augmented examples that are augmented by generator_1
        self.generator_1_balance = generator_1_balance
        # The mode to run the Keras model in
        self.training = not run_in_eval_mode

    def __call__(self, data, *args, **kwargs):
        if len(data.shape) == 4:
            return np.array(self.transform(data))
        else:
            raise NotImplementedError

    def transform(self, data, *args, **kwargs):
        # Rescale the data to [-1, 1]
        data = self.basic_preprocessor(data)
        # Get splits of the data
        aug_split, unchanged_split, permutation = shuffle_and_split_data(data, self.aug_proportion)
        # Get splits of the data to be augmented
        gen1_split, gen2_split, aug_permutation = shuffle_and_split_data(aug_split, self.generator_1_balance)
        # Pass the splits through the generators
        gen1_split = self.generator_1(gen1_split, training=self.training)
        gen2_split = self.generator_2(gen2_split, training=self.training)
        # Combine to recover the augmented data split
        aug_split = np.concatenate([gen1_split, gen2_split], axis=0)
        # Unshuffle the augmented data split
        aug_split = unshuffle_data(aug_split, aug_permutation)
        # Combine to recover the data
        data = np.concatenate([aug_split, unchanged_split], axis=0)
        # Unshuffle to recover the original data
        data = unshuffle_data(data, permutation)
        # Rescale output to [0, 1]
        return self.basic_preprocessor.zero_one_conversion(data)


class GenerativeAugmentationPipeline(AugmentationPipeline):

    def __init__(self, *args, **kwargs):
        super(GenerativeAugmentationPipeline, self).__init__()
        pass

    def improve(self):
        pass
