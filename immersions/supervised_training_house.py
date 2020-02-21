from test_tube import HyperOptArgumentParser, Experiment
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.arg_parse import add_default_args
import os.path
import sys

from immersions.supervised_system import SupervisedTaskSystem
from immersions.cpc_system import ContrastivePredictiveSystem
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
        name='house_supervised_0',
        debug=True,
        save_dir=log_dir,
        version=0,
        autosave=False,
        description='house supervised training'
    )

    hparams.training_set_path = '/home/idivinci3005/data/immersions/training'
    hparams.validation_set_path = '/home/idivinci3005/data/immersions/validation'
    hparams.test_task_set_path = '/home/idivinci3005/data/immersions/test_task'
    #hparams.training_set_path = 'C:/Users/HEV7RNG/Documents/data/maestro-v2.0.0'
    #hparams.validation_set_path = 'C:/Users/HEV7RNG/Documents/data/maestro-v2.0.0'
    #hparams.test_task_set_path = 'C:/Users/HEV7RNG/Documents/data/maestro-v2.0.0'
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
    hparams.batch_size = 4
    hparams.learning_rate = 3e-4
    hparams.warmup_steps = 1000
    hparams.annealing_steps = 100000
    hparams.score_over_all_timesteps = False
    hparams.visible_steps = 64

    # debug testing
    hparams.enc_batch_norm = False
    hparams.ar_batch_norm = False

    # set the hparams for the experiment
    exp.argparse(hparams)
    exp.save()

    # build model
    model = SupervisedTaskSystem(hparams, dataset_path=hparams.test_task_set_path)

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
        #early_stop_callback=early_stop,
        # distributed_backend='dp',
        gpus=[0],
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