import chainer
import os

from deep_image_converter.config import Config
from deep_image_converter import dataset
from deep_image_converter.model import *
from deep_image_converter import utility


class Drawer(object):
    def __init__(self, path_result_directory, gpu):
        config_path = Config.get_config_path(path_result_directory)
        config = Config(config_path)

        self.model = None

        self.path_result_directory = path_result_directory
        self.dataset_config = config.dataset_config
        self.model_config = config.model_config
        self.gpu = gpu
        self.target_iteration = None

    def _get_path_model(self, iteration):
        return os.path.join(self.path_result_directory, '{}.model'.format(iteration))

    def exist_save_model(self, iteration):
        path_model = self._get_path_model(iteration)
        return os.path.exists(path_model)

    def load_model(self, iteration):
        if not self.exist_save_model(iteration):
            print("warning! iteration {iteration} model is not found.".format(iteration=iteration))
            return False

        self.model = prepare_model(self.model_config)
        path_model = self._get_path_model(iteration)

        print("load {} ...".format(path_model))
        chainer.serializers.load_npz(path_model, self.model)
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()
            self.model.to_gpu(self.gpu)

        self.target_iteration = iteration
        return True

    def _forward(
            self,
            input_images_array,
            forwarder,
    ):
        if self.gpu >= 0:
            input_images_array = utility.chainer.to_variable_recursive(input_images_array, device=self.gpu)

        output_images_array = forwarder(input_images_array)[0]

        if self.gpu >= 0:
            output_images_array.to_cpu()

        return output_images_array.data

    def _make_image(
            self,
            images_array,
    ):
        dataset_class = dataset.choose_output_class(self.dataset_config)
        return dataset.array_to_image(dataset_class, images_array)

    def random_draw(
            self,
            batchsize: int,
    ):
        model = self.model
        assert isinstance(model, BaseAutoEncoderModel)

        latent_array = model.get_random_feature(batchsize=batchsize)
        return self.latent_draw(latent_array)

    def latent_draw(
            self,
            latent_array,
    ):
        model = self.model
        assert isinstance(model, BaseAutoEncoderModel)

        return self._make_image(self._forward(latent_array, lambda x: model.decode(
            x,
            test=True,
        )))

    def auto_encode(
            self,
            input_images_array,
    ):
        model = self.model  # type: BaseAutoEncoderModel
        assert isinstance(model, BaseAutoEncoderModel)

        return self._make_image(self._forward(input_images_array, lambda x: model(
            x,
            test=True,
        )))

    def convert(
            self,
            input_images_array,
    ):
        model = self.model  # type: BaseConvertModel
        assert isinstance(model, BaseConvertModel)

        return self._make_image(self._forward(input_images_array, lambda x: model(
            x,
            test=True,
        )))

    def convert_from_to(
            self,
            input_images_array,
            from_a_b,
            to_a_b,
    ):
        model = self.model  # type: BaseEachConvertModel
        assert isinstance(model, BaseEachConvertModel)

        return self._make_image(self._forward(input_images_array, lambda x: model(
            x,
            from_a_b=from_a_b,
            to_a_b=to_a_b,
            test=True,
        )))
