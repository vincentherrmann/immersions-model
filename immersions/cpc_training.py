from test_tube import HyperOptArgumentParser, Experiment
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.arg_parse import add_default_args
import os.path
import sys

from immersions.cpc_system import ContrastivePredictiveSystem


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
        description='test demo'
    )

    hparams.training_set_path = '/Volumes/Elements/Datasets/Immersions/house_data_mp3/training'
    hparams.validation_set_path = '/Volumes/Elements/Datasets/Immersions/house_data_mp3/validation'
    hparams.test_task_set_path = '/Volumes/Elements/Datasets/Immersions/house_data_mp3/test_task'
    hparams.batch_size = 4

    # set the hparams for the experiment
    exp.argparse(hparams)
    exp.save()

    # build model
    model = ContrastivePredictiveSystem(hparams)

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
    parser = ContrastivePredictiveSystem.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # train model
    main(hyperparams)