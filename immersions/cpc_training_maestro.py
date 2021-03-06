from test_tube import HyperOptArgumentParser, Experiment
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.arg_parse import add_default_args
import os.path
import sys

from immersions.cpc_system_maestro import ContrastivePredictiveSystemMaestro
from immersions.classication_task import MaestroClassificationTaskModel

def main(hparams, cluster=None, results_dict=None):
    """
    Main training routine specific for this project
    :param hparams:
    :return:
    """
    # init experiment
    log_dir = os.path.dirname(os.path.realpath(__file__))
    exp = Experiment(
        name='test_tube_exp',
        debug=True,
        save_dir=log_dir,
        version=0,
        autosave=False,
        description='maestro dataset experiment'
    )

    #hparams.training_set_path = '/Volumes/Elements/Datasets/maestro-v2.0.0'
    #hparams.validation_set_path = '/Volumes/Elements/Datasets/maestro-v2.0.0'
    #hparams.test_task_set_path = '/Volumes/Elements/Datasets/maestro-v2.0.0'
    hparams.training_set_path = 'C:/Users/HEV7RNG/Documents/data/maestro-v2.0.0'
    hparams.validation_set_path = 'C:/Users/HEV7RNG/Documents/data/maestro-v2.0.0'
    hparams.test_task_set_path = 'C:/Users/HEV7RNG/Documents/data/maestro-v2.0.0'
    hparams.audio_noise = 3e-3
    hparams.ar_kernel_sizes = (5, 4, 1, 3, 3, 1, 3, 1, 6)
    hparams.ar_self_attention = (False, False, False, False, False, False, False, False, False)
    hparams.batch_size = 4
    hparams.learning_rate = 2e-4
    hparams.warmup_steps = 1000
    hparams.annealing_steps = 100000
    hparams.score_over_all_timesteps = False
    hparams.visible_steps = 62

    # set the hparams for the experiment
    exp.argparse(hparams)
    exp.save()

    # build model
    model = ContrastivePredictiveSystemMaestro(hparams)
    task_model = MaestroClassificationTaskModel(model, task_dataset_path=hparams.validation_set_path)
    model.test_task_model = task_model

    # callbacks
    early_stop = EarlyStopping(
        monitor=hparams.early_stop_metric,
        patience=hparams.early_stop_patience,
        verbose=True,
        mode=hparams.early_stop_mode
    )

    model_save_path = '{}/{}/{}'.format(hparams.model_save_path, exp.name, exp.version)
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        verbose=True,
        monitor=hparams.model_save_monitor_value,
        mode=hparams.model_save_monitor_mode
    )

    # configure trainer
    trainer = Trainer(
        experiment=exp,
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stop,
        # distributed_backend='dp',
        #gpus=[0],
        nb_sanity_val_steps=2
    )

    # train model
    trainer.fit(model)

if __name__ == '__main__':

    # use default args given by lightning
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=False)
    add_default_args(parent_parser, root_dir)

    # allow model to overwrite or extend args
    parser = ContrastivePredictiveSystemMaestro.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # train model
    main(hyperparams)