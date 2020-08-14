import augmentation.datasets.utils
from augmentation.augment.utils import WandbModelPseudoLabelingPipeline, BinaryMNISTWandbModelPseudoLabelingPipeline


def configure_pseudolabeler(pseudolabel: bool, pseudolabeler_builder, pseudolabeler_builder_args):
    """Pass in a class that can build a pseudolabeler (implementing __call__) or a builder function
    that returns a pseudolabeling function.
    """
    if pseudolabel:
        return globals()[pseudolabeler_builder](*pseudolabeler_builder_args)
    return None


def apply_pseudolabeler(pseudolabel: bool,
                        pseudolabeler_builder,
                        pseudolabeler_builder_args,
                        tf_datasets,
                        dataset_aliases,
                        dataset_lens,
                        labeler_batch_size,
                        keep_datasets=False):
    assert len(tf_datasets) == len(dataset_aliases), 'Must specify one alias per dataset.'

    if pseudolabel:
        # If pseudolabeling, create the pseudolabeler and apply it
        print("Pseudolabeling the dataset.")
        pseudolabeler = configure_pseudolabeler(pseudolabel, pseudolabeler_builder, pseudolabeler_builder_args)

        updated_datasets, updated_aliases, \
        updated_dataset_lens, variants_per_dataset = \
            augmentation.datasets.utils.split_datasets_by_pseudolabels(tf_datasets=tf_datasets,
                                                                       dataset_aliases=dataset_aliases,
                                                                       pseudolabeler=pseudolabeler,
                                                                       batch_size=labeler_batch_size)

        if keep_datasets:
            # Append the datasets that pseudolabeling generated
            updated_datasets = list(tf_datasets) + list(updated_datasets)
            updated_aliases = dataset_aliases + updated_aliases
            updated_dataset_lens = dataset_lens + updated_dataset_lens if dataset_lens is not None else None

        #TODO: stable for the moment, but figure out how to handle variants_per_dataset more elegantly
        # (e.g. it's used to replicate the augmentations in robust/train.py)
        return updated_datasets, updated_aliases, updated_dataset_lens, variants_per_dataset

    # Just return everything as is, if not pseudolabeling
    return tf_datasets, dataset_aliases, dataset_lens, [1] * len(dataset_aliases)
