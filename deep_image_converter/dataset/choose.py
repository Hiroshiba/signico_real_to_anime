import glob
import numpy

from deep_image_converter import config

from .image_dataset import *


def _process_augmentation(dataset, test, name, *args):
    process_dataset_class = {
        'crop': RandomCropImageDataset,
        'flip': RandomFlipImageDataset,
        'scale': RandomScaleImageDataset,
    }[name]
    return process_dataset_class(dataset, test, *args)


def _get_output_class(name):
    return {
        'luminance': LuminanceImageArrayDataset,
        'linedrawing': LinedrawingImageArrayDataset,
        'rawline': RawLineImageArrayDataset,
        'onehot': OneHotImageArrayDataset,
    }[name]


def _process_output(dataset, test, name, *args):
    process_dataset_class = _get_output_class(name)
    return process_dataset_class(dataset, test, *args)


def _mix_dataset(paths, augmentation_recipe_list, output_recipe, test):
    _dataset = PILImageDataset(paths)
    for recipe in augmentation_recipe_list:
        _dataset = _process_augmentation(_dataset, test, recipe[0], *recipe[1:])

    _dataset = ImageArrayDataset(_dataset, test=test)
    if output_recipe is not None:
        _dataset = _process_output(_dataset, test, output_recipe[0], *output_recipe[1:])
    return _dataset


def _choose_one(random_state, paths, num_test, augmentation, output):
    paths = random_state.permutation(paths)

    train_paths = paths[num_test:]
    test_paths = paths[:num_test]
    train_for_evaluate_paths = train_paths[:num_test]

    return {
        'train': _mix_dataset(train_paths, augmentation, output, test=False),
        'test': _mix_dataset(test_paths, augmentation, output, test=True),
        'train_eval': _mix_dataset(train_for_evaluate_paths, augmentation, output, test=True),
    }


def choose_output_class(dataset_config: config.DatasetConfig):
    if dataset_config.output is None:
        return ImageArrayDataset
    else:
        return _get_output_class(dataset_config.output[0])


def choose(dataset_config: config.DatasetConfig):
    random_state = numpy.random.RandomState(seed=dataset_config.seed_evaluation)

    if dataset_config.b_only:
        paths = glob.glob(dataset_config.b_domain_images_path + '/*')

        one_dataset = _choose_one(
            random_state=random_state,
            paths=paths,
            num_test=dataset_config.num_test,
            augmentation=dataset_config.augmentation,
            output=dataset_config.output,
        )

        return one_dataset

    else:
        paths_list = [
            glob.glob(dataset_config.a_domain_images_path + '/*'),
            glob.glob(dataset_config.b_domain_images_path + '/*'),
        ]

        dataset_a_b = []

        for paths in paths_list:
            one_dataset = _choose_one(
                random_state=random_state,
                paths=paths,
                num_test=dataset_config.num_test,
                augmentation=dataset_config.augmentation,
                output=dataset_config.output,
            )
            dataset_a_b.append(one_dataset)

        return {
            'train_a': dataset_a_b[0]['train'],
            'train_b': dataset_a_b[1]['train'],
            'test': PairImageDataset(dataset_a_b[0]['test'], dataset_a_b[1]['test']),
            'train_eval': PairImageDataset(dataset_a_b[0]['train_eval'], dataset_a_b[1]['train_eval']),
        }
