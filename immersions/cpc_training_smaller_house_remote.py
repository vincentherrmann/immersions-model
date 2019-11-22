from test_tube import HyperOptArgumentParser, Experiment
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.arg_parse import add_default_args
import os.path
import sys

from immersions.cpc_system import ContrastivePredictiveSystem
from immersions.classication_task import ClassificationTaskModel


def main(hparams, cluster=None, results_dict=None):
    """
    Main training routine specific for this project
    :param hparams:
    :return:
    """
    name = 'immersions_scalogram_resnet_house_smaller'
    version = 1
    hparams.log_dir = '/home/idivinci3005/experiments/logs'
    hparams.checkpoint_dir = '/home/idivinci3005/experiments/checkpoints/' + name + '/' + str(version)
    hparams.training_set_path = '/home/idivinci3005/data/immersions/training'
    hparams.validation_set_path = '/home/idivinci3005/data/immersions/validation'
    hparams.test_task_set_path = '/home/idivinci3005/data/immersions/test_task'
    hparams.dummy_datasets = False
    hparams.audio_noise = 3e-3

    hparams.cqt_fmin = 40.
    hparams.cqt_bins_per_octave = 24
    hparams.cqt_n_bins = 216
    hparams.cqt_hop_length = 512
    hparams.cqt_filter_scale = 0.43

    hparams.enc_channels = (1, 8, 16, 32, 64,
                            128, 256, 512, 512)
    hparams.enc_kernel_1_w = (3, 3, 3, 3, 3, 3, 3, 3)
    hparams.enc_kernel_1_h = (3, 3, 3, 3, 3, 3, 3, 3)
    hparams.enc_kernel_2_w = (1, 3, 1, 3, 1, 3, 1, 3)
    hparams.enc_kernel_2_h = (25, 3, 25, 3, 25, 3, 4, 3)
    hparams.enc_padding_1 = (1, 1, 1, 1, 1, 1, 1, 1)
    hparams.enc_padding_2 = (0, 1, 0, 1, 0, 1, 0, 0)
    hparams.enc_stride_1 = (1, 1, 1, 1, 1, 1, 1, 1)
    hparams.enc_stride_2 = (1, 1, 1, 1, 1, 1, 1, 1)
    hparams.enc_pooling_1 = (2, 1, 1, 1, 2, 1, 1, 1)

    hparams.ar_kernel_sizes = (5, 4, 1, 3, 3, 1, 3, 1, 6)
    hparams.ar_self_attention = (False, False, False, False, False, False, False, False, False)
    hparams.batch_size = 4
    hparams.learning_rate = 3e-4
    hparams.warmup_steps = 1000
    hparams.annealing_steps = 100000
    hparams.score_over_all_timesteps = False
    hparams.visible_steps = 60

    hparams.batch_size = 32
    hparams.learning_rate = 3e-4
    hparams.warmup_steps = 1000
    hparams.annealing_steps = 100000
    hparams.score_over_all_timesteps = False
    hparams.visible_steps = 60

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
    task_model = ClassificationTaskModel(model, task_dataset_path=hparams.test_task_set_path)

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
        val_check_interval=0.2,
        gradient_clip=0.5,
        track_grad_norm=2
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