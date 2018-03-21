import math
import numpy
import os
import subprocess
import typing
from PIL import Image


def array_to_image(images_array: numpy.ndarray, minmax=None, mode='RGB') -> typing.List[Image.Image]:
    images_array = images_array.transpose(0, 2, 3, 1)

    if mode == 'L':
        images_array = images_array.repeat(3, axis=3)
        mode = 'RGB'

    # to uint8
    if minmax is None:
        minmax = (-1, 1)

    def clip_image(x):
        x = (x - minmax[0]) / (minmax[1] - minmax[0]) * 255  # normalize to 0~255
        return numpy.float32(0 if x < 0 else (255 if x > 255 else x))

    images_array = numpy.vectorize(clip_image)(images_array)
    images_array = images_array.astype(numpy.uint8)
    return [Image.fromarray(image_array, mode=mode) for image_array in images_array]


def label_array_to_image(images_array: numpy.ndarray, num_label: int) -> typing.List[Image.Image]:
    return array_to_image(images_array, (0, num_label - 1))


def save_images(images: typing.List[Image.Image], path_directory, prefix_filename):
    """
    save image as [prefix_filename][index of image].png
    """
    if not os.path.exists(path_directory):
        os.mkdir(path_directory)

    filepath_list = []
    for i, image in enumerate(images):
        filename = prefix_filename + str(i) + '.png'
        filepath = os.path.join(path_directory, filename)
        image.save(filepath)
        filepath_list += [filepath]

    return filepath_list


def save_tiled_image(paths_input: typing.List[str], path_output=None, col=None, row=None, border=5):
    num_image = len(paths_input)

    if path_output is None:
        commonpath = os.path.commonprefix(paths_input)
        path_output = commonpath + 'tiled.png'

    if col is None:
        col = math.ceil(math.sqrt(num_image))
    else:
        assert isinstance(col, int)

    if row is None:
        row = math.ceil(num_image / col)
    else:
        assert isinstance(row, int)

    assert isinstance(border, int)

    command = \
        '''
        montage \
        -tile {col}x{row} \
        -geometry +0 \
        -border {border}x{border} \
        {paths_input} \
        {path_output}
        '''.format(
            col=col,
            row=row,
            border=border,
            paths_input=' '.join(paths_input),
            path_output=path_output,
        )
    subprocess.check_output(command, shell=True)
