from abc import ABCMeta, abstractmethod
import chainer
import cv2
import numpy
import os
from PIL import Image
from skimage.color import rgb2lab


class BaseProcessDataset(chainer.dataset.DatasetMixin):
    def __init__(self, base_dataset, test: bool):
        self._base_dataset = base_dataset
        self._test = test

    def __len__(self):
        return len(self._base_dataset)


class PILImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, paths):
        self._paths = paths

    def __len__(self):
        return len(self._paths)

    def get_example(self, i) -> Image:
        path = self._paths[i]
        return Image.open(path)


class RandomScaleImageDataset(BaseProcessDataset):
    def __init__(self, base_image_dataset, test, min_scale=1.0, max_scale=1.3):
        super().__init__(base_image_dataset, test)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def get_example(self, i):
        image = self._base_dataset[i]
        base_size = image.size

        rand = numpy.random.rand(1) if not self._test else 0.5

        scale = rand * (self.max_scale - self.min_scale) + self.min_scale
        size_resize = (int(image.size[0] * scale), int(image.size[1] * scale))

        if base_size != size_resize:
            image = image.resize(size_resize, resample=Image.BICUBIC)

        return image


class RandomCropImageDataset(BaseProcessDataset):
    def __init__(self, base_image_dataset, test, crop_width, crop_height):
        super().__init__(base_image_dataset, test)
        self._crop_width = crop_width
        self._crop_height = crop_height

    def get_example(self, i):
        image = self._base_dataset[i]
        width, height = image.size
        assert width >= self._crop_width and height >= self._crop_height

        if not self._test:
            top = numpy.random.randint(height - self._crop_height + 1)
            left = numpy.random.randint(width - self._crop_width + 1)
        else:
            top = (height - self._crop_height) // 2
            left = (width - self._crop_width) // 2

        bottom = top + self._crop_height
        right = left + self._crop_width

        image = image.crop((left, top, right, bottom))
        return image


class RandomFlipImageDataset(BaseProcessDataset):
    def __init__(self, base_image_dataset, test, p_flip_horizontal=0.5, p_flip_vertical=0.0):
        super().__init__(base_image_dataset, test)
        self.p_flip_horizontal = p_flip_horizontal
        self.p_flip_vertical = p_flip_vertical

    def get_example(self, i):
        image = self._base_dataset[i]

        if not self._test:
            if numpy.random.rand(1) < self.p_flip_horizontal:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            if numpy.random.rand(1) < self.p_flip_vertical:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)

        return image


class ImageArrayDataset(BaseProcessDataset):
    def __init__(self, pil_image_dataset: PILImageDataset, test, normalize=True, dtype=numpy.float32):
        super().__init__(pil_image_dataset, test)
        self._base_image_dataset = pil_image_dataset
        self._normalize = normalize
        self._dtype = dtype

    def get_example(self, i):
        image = self._base_image_dataset[i]
        image = numpy.asarray(image, dtype=self._dtype).transpose(2, 0, 1)[:3, :, :]

        if self._normalize:
            image = image / 255 * 2 - 1

        return {
            'input': image,
            'target': image,
        }


class LuminanceImageArrayDataset(BaseProcessDataset):
    def __init__(self, base_image_dataset, test, normalize=True, keep_dimension=True):
        super().__init__(base_image_dataset, test)
        self._normalize = normalize
        self._keep_dimension = keep_dimension

    def get_example(self, i):
        image = self._base_dataset[i]['input']
        dtype = image.dtype

        image = (image.transpose(1, 2, 0) + 1) / 2
        image = rgb2lab(image)[:, :, 0]

        if self._normalize:
            image = image / 100 * 2 - 1

        if self._keep_dimension:
            image = image[numpy.newaxis]

        image = image.astype(dtype)
        return {
            'input': image,
            'target': image,
        }


class LinedrawingImageArrayDataset(BaseProcessDataset):
    def __init__(self, base_image_dataset, test, binarization=False):
        super().__init__(base_image_dataset, test)
        self.binarization = binarization

    def get_example(self, i):
        image = self._base_dataset[i]['input']
        dtype = image.dtype

        image = rgb = (image.transpose(1, 2, 0) + 1) / 2
        image = lab = rgb2lab(image) / 100

        def dilate_diff(image):
            dil = cv2.dilate(image, numpy.ones((3, 3), numpy.float32), iterations=1)
            image = cv2.absdiff(image, dil)
            return image

        image = numpy.dstack([dilate_diff(one_dim) for one_dim in numpy.dsplit(image, 3)])
        image = image.mean(axis=2, keepdims=True).transpose(2, 0, 1)

        image = image / image.max()
        image = 1 - image

        if not self.binarization:
            image = image ** 2

        else:
            def blur_diff(image, range):
                dil = cv2.blur(image, (range, range))
                image = cv2.absdiff(image, dil)
                return image

            d = image

            # 減色して領域を作成し、その領域の境界を取得
            image = rgb
            image = numpy.round(image * (4 - 1))  # 減色
            image = numpy.dstack([blur_diff(rgb, range=2) for rgb in numpy.dsplit(image, 3)])  # ブラー差分
            image = (image.sum(axis=2) > 0).astype(numpy.float32)
            image = cv2.erode(image, numpy.ones(2))
            c = 1 - image

            image = numpy.squeeze(d)
            dc = image[c != 1]
            image = image > numpy.median(dc)
            image = image.astype(numpy.int32)
            image = image[numpy.newaxis]

        image = image.astype(dtype)
        return {
            'input': image,
            'target': image,
        }


class RawLineImageArrayDataset(BaseProcessDataset):
    def __init__(self, base_image_dataset, test):
        super().__init__(base_image_dataset, test)

    def get_example(self, i):
        from scipy import stats

        image = self._base_dataset[i]['input']
        dtype = image.dtype

        def dilate_diff(image, range, iterations=1):
            dil = cv2.dilate(image, numpy.ones((range, range), numpy.float32), iterations=iterations)
            image = cv2.absdiff(image, dil)
            return image

        rgb = (image.transpose(1, 2, 0) + 1) / 2
        lab = rgb2lab(rgb) / 100

        image = lab[:, :, 0]
        image = dilate_diff(image, 3).astype(numpy.float32)

        rand = numpy.random.randn(1) / 20 if not self._test else 0
        rand = 0.000001 if rand <= 0 else rand
        image = cv2.GaussianBlur(image, (5, 5), 0.2 + rand)

        rand = numpy.random.randn(1) * 2.5 if not self._test else 0
        a = stats.scoreatpercentile(image, 60 + rand)
        rand = numpy.random.randn(1) / 2 if not self._test else 0
        b = stats.scoreatpercentile(image, 90 + rand)

        image = numpy.clip((image - a) / (b - a), 0, 1)

        rand = numpy.random.randn(1) / 20 if not self._test else 0
        image = cv2.GaussianBlur(image, (5, 5), 0.4 + rand)

        rand = numpy.random.randn(1) / 40 if not self._test else 0
        image = numpy.power(image, 0.8 + rand)

        image = image.astype(dtype)[numpy.newaxis] * -2 + 1
        return {
            'input': image,
            'target': image,
        }


class OneHotImageArrayDataset(BaseProcessDataset):
    def __init__(self, base_image_dataset, test, num_label, input_range=(-1, 1), dtype=numpy.int32):
        super().__init__(base_image_dataset, test)
        self._base_image_dataset = base_image_dataset
        self._num_label = num_label
        self._input_range = input_range
        self._dtype = dtype

    def get_example(self, i):
        image = self._base_image_dataset[i]['input']
        image = (image - self._input_range[0]) / (self._input_range[1] - self._input_range[0])
        image *= (self._num_label - 1)
        image_onehot = numpy.around(image).astype(self._dtype)

        return {
            'input': image,
            'target': image_onehot,
        }


class PairImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dataset_a, dataset_b):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        assert len(dataset_a) == len(dataset_b)

    def __len__(self):
        return len(self.dataset_a)

    def get_example(self, i):
        return {
            'image_a': self.dataset_a[i],
            'image_b': self.dataset_b[i],
        }
