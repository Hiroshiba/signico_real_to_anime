import chainer
from chainer.training import extensions
import typing

from deep_image_converter.config import TrainConfig
from deep_image_converter import utility


class TrainManager(object):
    def __init__(self, config: TrainConfig):
        self.config = config

    def make_optimizer(self, model, model_name: str):
        config = self.config.optimizer[model_name]
        if config == 'same':
            config = self.config.optimizer['main']

        optimizer = None
        if config['name'] == 'adam':
            alpha = 0.001 if 'alpha' not in config else config['alpha']
            beta1 = 0.9 if 'beta1' not in config else config['beta1']
            beta2 = 0.999 if 'beta2' not in config else config['beta2']
            optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        elif config['name'] == 'RMSprop':
            lr = 0.01 if 'lr' not in config else config['lr']
            alpha = 0.99 if 'alpha' not in config else config['alpha']
            optimizer = chainer.optimizers.RMSprop(lr=lr, alpha=alpha)
        else:
            assert "{name} is not defined.".format(name=config['name'])

        optimizer.setup(model)

        if 'weight_decay' in config and config['weight_decay'] is not None:
            optimizer.add_hook(chainer.optimizer.WeightDecay(config['weight_decay']))

        if 'gradient_clipping' in config and config['gradient_clipping'] is not None:
            optimizer.add_hook(chainer.optimizer.GradientClipping(config['gradient_clipping']))

        return optimizer

    def make_trainer(
            self,
            updater,
            model: typing.Dict,
            eval_func,
            iterator_test,
            iterator_train_eval,
            loss_names,
            converter=chainer.dataset.convert.concat_examples,
    ):
        trainer = chainer.training.Trainer(updater, out=self.config.get_project_path())

        log_trigger = (self.config.log_iteration, 'iteration')
        save_trigger = (self.config.save_result_iteration, 'iteration')

        eval_test_name = 'validation/test'
        eval_train_name = 'validation/train'

        snapshot = extensions.snapshot_object(model['main'], '{.updater.iteration}.model')
        trainer.extend(snapshot, trigger=save_trigger)

        trainer.extend(extensions.dump_graph('main/' + loss_names[0], out_name='main_graph.dot'))

        def make_evaluator(iterator):
            return extensions.Evaluator(
                iterator,
                target=model,
                converter=converter,
                eval_func=eval_func,
                device=self.config.gpu,
            )

        trainer.extend(make_evaluator(iterator_test), name=eval_test_name, trigger=log_trigger)
        trainer.extend(make_evaluator(iterator_train_eval), name=eval_train_name, trigger=log_trigger)

        report_target = []
        for evaluator_name in ['', eval_test_name + '/', eval_train_name + '/']:
            for model_name in ['main/', 'dis/', 'dis_a/', 'dis_b/']:
                for loss_name in loss_names:
                    report_target.append(evaluator_name + model_name + loss_name)

        trainer.extend(extensions.LogReport(trigger=log_trigger, log_name="log.txt"))
        trainer.extend(extensions.PrintReport(report_target))

        return trainer
