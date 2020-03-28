from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
import os.path
import sys
import argparse

from immersions.cpc_system import convert_hparams_to_string
from immersions.cpc_system_maestro import ContrastivePredictiveSystem
from immersions.classication_task import ClassificationTaskModel


def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data-path', metavar='DIR', type=str,
                               help='path to dataset')
    parent_parser.add_argument('--save-path', metavar='DIR', default=".", type=str,
                               help='path to save output')
    parent_parser.add_argument('--gpus', type=int, default=1,
                               help='how many gpus')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--use-16bit', dest='use_16bit', action='store_true',
                               help='if true uses 16 bit precision')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')

    parser = ContrastivePredictiveSystem.add_model_specific_args(parent_parser)
    return parser.parse_args()

def main(hparams, cluster=None, results_dict=None):
    """
    Main training routine specific for this project
    :param hparams:
    :return:
    """
    # init experiment


    name = "immersions_house_7_wasserstein_0,01"
    logs_dir = "/home/vincent/Projects/Immersions/logs"
    checkpoint_dir = "/home/vincent/Projects/Immersions/checkpoints/" + name
    hparams.training_set_path = '/home/vincent/data/house_data_mp3/training'
    hparams.validation_set_path = '/home/vincent/data/house_data_mp3/validation'
    hparams.test_task_set_path = '/home/vincent/data/house_data_mp3/test_task'
    hparams.data_path = hparams.training_set_path
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
    hparams.batch_size = 32
    hparams.learning_rate = 3e-4
    hparams.warmup_steps = 1000
    hparams.annealing_steps = 100000
    hparams.score_over_all_timesteps = False
    hparams.wasserstein_penalty = 0.01
    hparams.visible_steps = 64
    hparams.prediction_steps = 16
    hparams.prediction_gap = 4

    # build model
    model = ContrastivePredictiveSystem(hparams)
    model.activation_register.active = False

    def create_task_model():
        task_model = ClassificationTaskModel(cpc_system=model,
                                             task_dataset_path=hparams.validation_set_path,
                                             feature_size=model.ar_model.ar_size,
                                             hidden_layers=0)
        return task_model

    model.get_test_task_model = create_task_model

    logger = TensorBoardLogger(save_dir=logs_dir, name=name)
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_dir, save_top_k=-1)
    # configure trainer
    trainer = Trainer(gpus=1,
                      train_percent_check=1.,
                      val_percent_check=1.,
                      val_check_interval=0.5,
                      logger=logger,
                      checkpoint_callback=checkpoint_callback,
                      fast_dev_run=False,
                      early_stop_callback=False)

    # train model
    convert_hparams_to_string(model)
    trainer.fit(model)

if __name__ == '__main__':
    hyperparams = get_args()

    # train model
    main(hyperparams)