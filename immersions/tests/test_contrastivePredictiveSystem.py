from test_tube import HyperOptArgumentParser, Experiment
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.arg_parse import add_default_args
import os.path
import sys

from immersions.cpc_system import ContrastivePredictiveSystem
from unittest import TestCase


class TestContrastivePredictiveSystem(TestCase):
    def test_training(self):
        # use default args given by lightning
        root_dir = '/Volumes/Elements/Projekte/Immersions'
        parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=False)
        add_default_args(parent_parser, root_dir)

        # allow model to overwrite or extend args
        parser = ContrastivePredictiveSystem.add_model_specific_args(parent_parser, root_dir)
        hparams = parser.parse_args()

        name = 'immersions_scalogram_resnet_test'
        version = 0
        hparams.log_dir = '/Volumes/Elements/Projekte/Immersions/logs'
        hparams.checkpoint_dir = '/Volumes/Elements/Projekte/Immersions/checkpoints'
        hparams.training_set_path = '/Volumes/Elements/Datasets/Immersions/house_data_mp3/training'
        hparams.validation_set_path = '/Volumes/Elements/Datasets/Immersions/house_data_mp3/validation'
        hparams.dummy_datasets = False
        hparams.batch_size = 64
        hparams.learning_rate = 2e-4
        hparams.warmup_steps = 1000
        hparams.annealing_steps = 100000

        # init experiment
        exp = Experiment(
            name=name,
            debug=False,
            save_dir=hparams.log_dir,
            version=version,
            autosave=False,
            description='test demo'
        )

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

        checkpoint = ModelCheckpoint(
            filepath=hparams.checkpoint_dir,
            save_best_only=False,
            verbose=True,
            monitor=hparams.model_save_monitor_value,
            mode=hparams.model_save_monitor_mode
        )

        # configure trainer
        trainer = Trainer(
            experiment=exp,
            checkpoint_callback=checkpoint,
            # early_stop_callback=early_stop,
            # distributed_backend='dp',
            gpus=[0],
            nb_sanity_val_steps=5,
            val_check_interval=0.2,
            train_percent_check=0.01,
            max_nb_epochs=1
        )

        # train model
        trainer.fit(model)