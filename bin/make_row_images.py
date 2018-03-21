import argparse
import glob
import numpy
import os
import sys

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT_PATH)

from deep_image_converter.config import Config
from deep_image_converter import dataset
from deep_image_converter.utility.image import save_images, save_tiled_image

parser = argparse.ArgumentParser()
parser.add_argument('--path_result_directory')
parser.add_argument('--config_json_path')
parser.add_argument('--path_save_directory')
parser.add_argument('--num_image', type=int, default=100)
args = parser.parse_args()

path_result_directory = args.path_result_directory
config_json_path = args.config_json_path
path_save_directory = args.path_save_directory
num_image = args.num_image

if config_json_path is None:
    config_json_path = Config.get_config_path(path_result_directory)
config = Config(config_json_path)

os.path.exists(path_save_directory) or os.mkdir(path_save_directory)

path_save = os.path.join(path_save_directory, 'test')
os.path.exists(path_save) or os.mkdir(path_save)

dataset_test = dataset.choose(config.dataset_config)['test']

# save images
try:
    images_a = []
    images_b = []
    for i, data in enumerate(dataset_test[:num_image]):
        image_a = data['image_a']['input'][numpy.newaxis]
        image_b = data['image_b']['input'][numpy.newaxis]
        image_a = dataset.array_to_image(dataset_test.dataset_a, image_a)
        image_b = dataset.array_to_image(dataset_test.dataset_b, image_b)
        images_a += image_a
        images_b += image_b

    save_images(images_a, path_save, 'a_')
    save_images(images_b, path_save, 'b_')

    # save tiled images
    for num_tile_image in range(10, 110, 10):
        paths_input = [os.path.join(path_save, 'a_{}.png'.format(i)) for i in range(num_tile_image)]
        save_tiled_image(paths_input, os.path.join(path_save, 'tile{}_a.png'.format(num_tile_image)), col=num_tile_image)
        paths_input = [os.path.join(path_save, 'b_{}.png'.format(i)) for i in range(num_tile_image)]
        save_tiled_image(paths_input, os.path.join(path_save, 'tile{}_b.png'.format(num_tile_image)), col=num_tile_image)
except:
    pass

try:
    images = []
    for i, data in enumerate(dataset_test[:num_image]):
        image = data['input'][numpy.newaxis]
        image = dataset.array_to_image(dataset_test, image)
        images += image

    save_images(images, path_save, '')

    # save tiled images
    for num_tile_image in range(10, 110, 10):
        paths_input = [os.path.join(path_save, '{}.png'.format(i)) for i in range(num_tile_image)]
        save_tiled_image(paths_input, os.path.join(path_save, 'tile{}.png'.format(num_tile_image)), col=num_tile_image)
except:
    pass
