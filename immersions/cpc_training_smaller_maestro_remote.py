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
    name = 'immersions_maestro_new_4_score_over_all_steps'
    version = 0
    hparams.log_dir = '/home/idivinci3005/experiments/logs'
    hparams.checkpoint_dir = '/home/idivinci3005/experiments/checkpoints/' + name + '/' + str(version)
    hparams.training_set_path = '/home/idivinci3005/data/maestro-v2.0.0'
    hparams.validation_set_path = '/home/idivinci3005/data/maestro-v2.0.0'
    hparams.test_task_set_path = '/home/idivinci3005/data/maestro-v2.0.0'
    hparams.dummy_datasets = False
    hparams.audio_noise = 3e-3

    hparams.cqt_fmin = 40.
    hparams.cqt_bins_per_octave = 24
    hparams.cqt_n_bins = 216
    hparams.cqt_hop_length = 512
    hparams.cqt_filter_scale = 0.43

    hparams.enc_channels = (1, 8, 16, 32, 64, 128, 256, 512, 512)
    hparams.enc_kernel_1_w = (3, 1, 3, 1, 3, 1, 3, 1)
    hparams.enc_kernel_1_h = (3, 3, 3, 3, 3, 3, 3, 3)
    hparams.enc_kernel_2_w = (1, 1, 1, 1, 1, 1, 1, 1)
    hparams.enc_kernel_2_h = (25, 3, 15, 3, 15, 3, 5, 3)
    hparams.enc_padding_1 = (0, 0, 0, 0, 0, 0, 0, 0)
    hparams.enc_padding_2 = (0, 0, 0, 0, 0, 0, 0, 0)
    hparams.enc_stride_1 = (1, 1, 1, 1, 1, 1, 1, 1)
    hparams.enc_stride_2 = (1, 1, 1, 1, 1, 1, 1, 1)
    hparams.enc_pooling_1 = (2, 1, 1, 1, 2, 1, 1, 1)

    hparams.ar_kernel_sizes = (4, 4, 1, 4, 4, 1, 4, 1, 4)
    hparams.ar_pooling = (1, 1, 2, 1, 1, 2, 1, 1, 1)
    hparams.ar_self_attention = (False, False, False, False, False, False, False, False, False)
    hparams.batch_size = 32
    hparams.learning_rate = 3e-4
    hparams.warmup_steps = 1000
    hparams.annealing_steps = 100000
    hparams.score_over_all_timesteps = True
    hparams.visible_steps = 64
    hparams.prediction_steps = 16
    hparams.prediction_gap = 4

    # init experiment
    exp = Experiment(
        name=name,
        debug=False,
        save_dir=hparams.log_dir,
        version=version,
        autosave=False,
        description='maestro dataset experiment'
    )

    # set the hparams for the experiment
    exp.argparse(hparams)
    exp.save()

    # build model
    model = ContrastivePredictiveSystemMaestro(hparams)
    task_model = MaestroClassificationTaskModel(model, task_dataset_path=hparams.test_task_set_path)
    model.test_task_model = task_model

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
        #early_stop_callback=early_stop,
        #distributed_backend='dp',
        gpus=[0],
        nb_sanity_val_steps=5,
        val_check_interval=0.1,
        val_percent_check=0.25
        #gradient_clip=0.5,
        #track_grad_norm=2
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