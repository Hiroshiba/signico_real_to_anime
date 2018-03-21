import numpy

from deep_image_converter import utility

from .image_dataset import *


def array_to_image(dataset_instance_or_class, images_array: numpy.ndarray):
    d = dataset_instance_or_class

    def decider(v, cla):
        return isinstance(v, cla) or v == cla

    if decider(d, ImageArrayDataset):
        return utility.image.array_to_image(images_array)
    elif decider(d, LuminanceImageArrayDataset):
        return utility.image.array_to_image(images_array, mode='L')
    elif decider(d, RawLineImageArrayDataset):
        return utility.image.array_to_image(images_array, mode='L')
    elif decider(d, OneHotImageArrayDataset):
        return utility.image.array_to_image(images_array)
    else:
        raise NotImplementedError(str(d))
