import json
import os
import shutil


class Config(object):
    @staticmethod
    def get_config_path(project_path):
        return os.path.join(project_path, 'config.json')

    def __init__(self, path_json):
        self.path_json = path_json
        self.config = json.load(open(path_json, encoding='utf-8'))

        self.dataset_config = DatasetConfig(self.config.get('dataset'))
        self.loss_config = LossConfig(self.config.get('loss'))
        self.model_config = ModelConfig(self.config.get('model'))
        self.train_config = TrainConfig(self.config.get('train'))

        self.validation()

        project_path = self.train_config.get_project_path()
        os.path.exists(project_path) or os.mkdir(project_path)

    def validation(self):
        if self.loss_config.name == 'facebook':
            assert self.model_config.name == 'poor_dilate_convert_model'

    def copy_config_json(self):
        project_path = self.train_config.get_project_path()
        config_path = self.get_config_path(project_path)
        shutil.copy(self.path_json, config_path)


class DatasetConfig(object):
    def __init__(self, config):
        self.a_domain_images_path = config.get('a_domain_images_path')
        self.b_domain_images_path = config.get('b_domain_images_path')
        self.b_only = config.get('b_only')
        self.augmentation = config.get('augmentation')
        self.output = config.get('output')
        self.seed_evaluation = config.get('seed_evaluation')
        self.num_test = config.get('num_test')


class LossConfig(object):
    def __init__(self, config):
        self.name = config.get('name')
        self.blend = config.get('blend')
        self.blend_discriminator = config.get('blend_discriminator')
        self.blend_latent_discriminator = config.get('blend_latent_discriminator')
        self.other = config.get('other')


class ModelConfig(object):
    def __init__(self, config):
        self.name = config.get('name')
        self.num_encoder_block = config.get('num_encoder_block')
        self.num_decoder_block = config.get('num_decoder_block')
        self.num_z_base_encoder = config.get('num_z_base_encoder')
        self.num_z_base_decoder = config.get('num_z_base_decoder')
        self.num_z_feature = config.get('num_z_feature')
        self.num_z_base_discriminator = config.get('num_z_base_discriminator')
        self.image_width = config.get('image_width')
        self.method_output = config.get('method_output')
        self.initialW = config.get('initialW')
        self.other = config.get('other')
        self.discriminator = config.get('discriminator')
        self.latent_discriminator = config.get('latent_discriminator')
        self.pre_trained_path = config.get('pre_trained_path')


class TrainConfig(object):
    def __init__(self, config):
        self.batchsize = config.get('batchsize')
        self.gpu = config.get('gpu')
        self.optimizer = config.get('optimizer')
        self.log_iteration = config.get('log_iteration')
        self.save_result_iteration = config.get('save_result_iteration')
        self.project_name = config.get('project_name')
        self.result_path = config.get('result_path')
        self.tags = config.get('tags')
        self.comment = config.get('comment')

    def get_project_path(self):
        return os.path.join(self.result_path, self.project_name)
