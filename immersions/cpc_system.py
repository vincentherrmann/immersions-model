import os
from collections import OrderedDict
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from test_tube import HyperOptArgumentParser
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.root_module.root_module import LightningModule

from immersions.model.preprocessing import PreprocessingModule
from immersions.model.scalogram_encoder import ScalogramResidualEncoder
from immersions.model.autoregressive import ConvolutionalArModel
from immersions.model.predictive_coding import PredictiveCodingModel
from immersions.model.utilities import ActivationRegister
try:
    from immersions.audio_dataset import AudioDataset, FileBatchSampler
except:
    print("cannot load audio dataset")
from immersions.lr_schedules import *
from immersions.input_optimization.activation_utilities import ActivationStatistics

import ast


class ContrastivePredictiveSystem(LightningModule):
    def __init__(self, hparams, load_datasets=True, test_task_model=None):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(ContrastivePredictiveSystem, self).__init__()
        self.hparams = hparams
        if type(self.hparams.enc_channels) is str:
            self.hparams = decode_hparams_strings(self.hparams)

        self.batch_size = hparams.batch_size
        self.prediction_steps = self.hparams.prediction_steps
        self.test_task_model = test_task_model

        # build model
        self.__build_model()
        if load_datasets:
            self.setup_datasets()

        # if you specify an example input, the summary will show input/output for each layer
        #self.example_input_array = torch.rand(5, 1, 350000)

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        :return:
        """

        multi_gpu = torch.cuda.device_count() > 1 and self.hparams.use_all_GPUs
        if multi_gpu:
            self.activation_register = ActivationRegister(devices=[i for i in range(torch.cuda.device_count())])
        else:
            self.activation_register = ActivationRegister()
        self.activation_register.active = False

        with torch.no_grad():
            self.preprocessing = PreprocessingModule(self.hparams)
        self.encoder = ScalogramResidualEncoder(self.hparams, preprocessing_module=self.preprocessing,
                                                activation_register=self.activation_register)
        self.ar_model = ConvolutionalArModel(self.hparams, activation_register=self.activation_register)
        self.model = PredictiveCodingModel(encoder=self.encoder,
                                           autoregressive_model=self.ar_model,
                                           hparams=self.hparams,
                                           activation_register=self.activation_register)
        self.item_length = self.model.item_length
        self.preprocessing_downsampling = self.preprocessing.downsampling_factor
        self.preprocessing_receptive_field = self.preprocessing.receptive_field

        if True: #multi_gpu:
            self.preprocessing = torch.nn.DataParallel(self.preprocessing)
            self.model = torch.nn.DataParallel(self.model)

        self.validation_examples_per_batch = self.batch_size
        if self.hparams.score_over_all_timesteps:
            self.validation_examples_per_batch *= self.hparams.prediction_steps

        prediction_template = torch.arange(0, self.validation_examples_per_batch, dtype=torch.long)
        if self.hparams.score_over_all_timesteps:
            prediction_template = prediction_template.view(self.batch_size, self.hparams.prediction_steps)
        else:
            prediction_template = prediction_template.unsqueeze(1).repeat(1, self.hparams.prediction_steps)
        self.register_buffer('prediction_template', prediction_template)
        #self.example_input_array = torch.zeros(1, 1, self.model.item_length)

    def setup_datasets(self):
        if not os.path.exists(self.hparams.training_set_path):
            print("dataset paths are not valid, loading the model without datasets")
            return
        self.training_set = AudioDataset(self.hparams.training_set_path,
                                         item_length=self.item_length,
                                         unique_length=self.encoder.downsampling_factor * self.hparams.unique_steps,
                                         sampling_rate=self.hparams.sampling_rate,
                                         dummy=self.hparams.dummy_datasets)
        print("training set length:", len(self.training_set))
        self.validation_set = AudioDataset(self.hparams.validation_set_path,
                                           item_length=self.item_length,
                                           unique_length=self.encoder.downsampling_factor * self.hparams.unique_steps,
                                           sampling_rate=self.hparams.sampling_rate,
                                           dummy=self.hparams.dummy_datasets)
        print("validation set length:", len(self.validation_set))
        self.train_sampler = FileBatchSampler(index_count_per_file=self.training_set.get_example_count_per_file(),
                                              batch_size=self.batch_size,
                                              file_batch_size=self.hparams.file_batch_size,
                                              drop_last=True)

        self.validation_sampler = FileBatchSampler(index_count_per_file=self.validation_set.get_example_count_per_file(),
                                                   batch_size=self.batch_size,
                                                   file_batch_size=self.hparams.file_batch_size,
                                                   drop_last=True,
                                                   seed=123)


    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """

        scal = self.preprocessing(x)
        predicted_z, targets, z, c = self.model(scal)

        return predicted_z, targets, z, c

    def loss(self, scores):
        batch_size = scores.shape[0]

        # scores: data_batch, data_step, target_batch, target_step
        if self.hparams.score_over_all_timesteps:
            n_scores = scores.view(-1, batch_size, self.prediction_steps)  # data_batch*data_step, target_batch. target_step
            noise_scoring = torch.logsumexp(n_scores, dim=0)  # target_batch, target_step
            valid_scores = torch.diagonal(scores, dim1=0, dim2=2)  # data_step, target_step, batch
            valid_scores = torch.diagonal(valid_scores, dim1=0, dim2=1)  # batch, step
        else:
            scores = torch.diagonal(scores, dim1=1, dim2=3)  # data_batch, target_batch, step
            noise_scoring = torch.logsumexp(scores, dim=0)  # target_batch, target_step
            valid_scores = torch.diagonal(scores, dim1=0, dim2=1).permute([1, 0])  # batch, step

        prediction_losses = -torch.mean(valid_scores - noise_scoring, dim=1)
        loss = torch.mean(prediction_losses)

        return loss

    def training_step(self, batch, batch_i):
        """
        Lightning calls this inside the training loop
        :param data_batch:
        :return:
        """

        batch = batch[0]

        self.eval()

        # forward pass
        predicted_z, targets, _, _ = self.forward(batch)
        scores = torch.tensordot(predicted_z, targets, dims=([2], [1]))  # data_batch, data_step, target_batch, target_step
        loss = self.loss(scores)

        if self.hparams.wasserstein_penalty != 0.:
            score_sum = torch.sum(scores)
            batch_grad = torch.autograd.grad(outputs=score_sum,
                                             inputs=batch,
                                             create_graph=True,
                                             retain_graph=True,
                                             only_inputs=True)
            gradient_penalty = ((batch_grad[0].norm(2, dim=1) - 1) ** 2).mean()
            loss += self.hparams.wasserstein_penalty * gradient_penalty

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss = loss.unsqueeze(0)

        output = OrderedDict({
            'loss': loss,
            'prog': {'tng_loss': loss, 'lr': self.current_learning_rate}
        })

        return output

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i):
        lr_scale = self.lr_scheduler.lr_lambda(self.global_step)
        for pg in optimizer.param_groups:
            self.current_learning_rate = lr_scale * self.hparams.learning_rate
            pg['lr'] = self.current_learning_rate

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def validation_step(self, batch, batch_i):
        """
        Lightning calls this inside the validation loop
        :param data_batch:
        :return:
        """

        batch = batch[0]

        # debug testing
        #batch[:, :, -1] = float('inf')

        # forward pass
        predicted_z, targets, _, _ = self.forward(batch)
        # predicted_z: batch, step, features
        # targets: batch, features, step
        scores = torch.tensordot(predicted_z, targets, dims=([2], [1]))  # data_batch, data_step, target_batch, target_step
        loss = self.loss(scores)  # valid scores: batch, time_step

        if self.hparams.wasserstein_penalty != 0.:
            score_sum = torch.sum(scores)
            batch_grad = torch.autograd.grad(outputs=score_sum,
                                             inputs=batch,
                                             create_graph=True,
                                             retain_graph=True,
                                             only_inputs=True)
            gradient_penalty = ((batch_grad[0].norm(2, dim=1) - 1) ** 2).mean()
            loss += self.hparams.wasserstein_penalty * gradient_penalty

        # calculate prediction accuracy as the proportion of scores that are highest for the correct target
        prediction_template = torch.arange(0, scores.shape[0], dtype=torch.long)
        if self.hparams.score_over_all_timesteps:
            prediction_template = torch.arange(0, scores.shape[0]*scores.shape[1], dtype=torch.long, device=scores.device)
            prediction_template = prediction_template.view(scores.shape[0], scores.shape[1])
            max_score = torch.argmax(scores.view(scores.shape[0], scores.shape[1], -1), dim=2)  # batch, step
        else:
            scores = torch.diagonal(scores, dim1=1, dim2=3)  # data_batch, target_batch, step
            prediction_template = torch.arange(0, scores.shape[0], dtype=torch.long, device=scores.device)
            prediction_template = prediction_template[:, None].repeat(1, self.hparams.prediction_steps)
            max_score = torch.argmax(scores, dim=1) # batch, step

        correctly_predicted = torch.eq(prediction_template, max_score)
        prediction_accuracy = torch.sum(correctly_predicted).float() / prediction_template.numel() # per prediction step  # self.validation_examples_per_batch

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss = loss.unsqueeze(0)
            prediction_accuracy = prediction_accuracy.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss,
            'val_acc': torch.mean(prediction_accuracy)
        })

        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)

        test_task_dict = {}
        if self.test_task_model is not None:
            self.test_task_model.calculate_data()

            if not self.experiment.debug:
                self.experiment.add_embedding(self.test_task_model.task_data,
                                              metadata=self.test_task_model.task_labels,
                                              global_step=self.global_step)
            if torch.cuda.is_available():
                trainer = pl.Trainer(max_nb_epochs=20,
                                     gpus=[0])
            else:
                trainer = pl.Trainer(max_nb_epochs=20)
            trainer.fit(self.test_task_model)

            test_task_dict['val_task_acc'] = trainer.tng_tqdm_dic['avg_val_accuracy']
            test_task_dict['val_task_loss'] = trainer.tng_tqdm_dic['avg_val_loss']

        tqdm_dic = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        return {**tqdm_dic, **test_task_dict}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.lr_scheduler = WarmupCosineSchedule(optimizer=optimizer,
                                                 warmup_steps=self.hparams.warmup_steps,
                                                 t_total=self.hparams.annealing_steps)
        self.current_learning_rate = 0.
        return optimizer

    def calc_silence_statistics(self, activation_ranges, num_batches=10, device='cpu'):
        register_was_active = self.activation_register.active
        self.activation_register.active = True

        activation_statistics = ActivationStatistics()
        for i in range(num_batches):
            silence_input = torch.randn(self.batch_size, 1, self.item_length) * 1e-6
            _ = self.forward(silence_input.to(device))
            activations = self.activation_register.get_activations()
            for key in activations.keys():
                activations[key] = activations[key].cpu()
                if key in activation_ranges.keys():
                    activation_range = activation_ranges[key]
                    if len(activations[key].shape) == 4:
                        activations[key] = activations[key][:, :, :, activation_range[0]:activation_range[1]]
                    else:
                        activations[key] = activations[key][:, :, activation_range[0]:activation_range[1]]
            activation_statistics.add_activation_batch(activations)

        t_mean, t_std, e_mean, e_std = activation_statistics.condense_statistics()

        result_dict = {'total_mean': t_mean,
                       'total_std': t_std,
                       'element_mean': e_mean,
                       'element_std': e_std}

        return result_dict

    def calc_data_statistics(self, activation_ranges, num_batches=10, device='cpu'):
        register_was_active = self.activation_register.active
        self.activation_register.active = True

        activation_statistics = ActivationStatistics()
        for i, batch in enumerate(self.val_dataloader):
            if i >= num_batches:
                break
            _ = self.forward(batch[0].to(device))
            activations = self.activation_register.get_activations()
            for key in activations.keys():
                activations[key] = activations[key].cpu()
                if key in activation_ranges.keys():
                    activation_range = activation_ranges[key]
                    if len(activations[key].shape) == 4:
                        activations[key] = activations[key][:, :, :, activation_range[0]:activation_range[1]]
                    else:
                        activations[key] = activations[key][:, :, activation_range[0]:activation_range[1]]
            activation_statistics.add_activation_batch(activations)

        t_mean, t_std, e_mean, e_std = activation_statistics.condense_statistics()

        result_dict = {'total_mean': t_mean,
                       'total_std': t_std,
                       'element_mean': e_mean,
                       'element_std': e_std}

        return result_dict

    @property
    def tng_dataloader(self):
        print('tng data loader called')
        # loader = DataLoader(
        #     dataset=self.training_set,
        #     batch_sampler=self.train_sampler,
        #     num_workers=1
        # )
        if torch.cuda.is_available():
            torch.cuda.seed()
        else:
            torch.seed()
        loader = DataLoader(
            dataset=self.training_set,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        return loader

    @pl.data_loader
    def val_dataloader(self):
        print('val data loader called')
        # loader = DataLoader(
        #     dataset=self.validation_set,
        #     batch_sampler=self.validation_sampler,
        #     num_workers=1
        # )
        if torch.cuda.is_available():
            torch.cuda.manual_seed(12345)
        else:
            torch.manual_seed(12345)
        loader = DataLoader(
            dataset=self.validation_set,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        return loader

    @pl.data_loader
    def test_dataloader(self):
        print('test data loader called')
        return None

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])

        parser.add_argument('--log_dir', default='logs', type=str)
        parser.add_argument('--checkpoint_dir', default='checkpoints', type=str)

        # data
        parser.add_argument('--training_set_path', default='data/training', type=str)
        parser.add_argument('--validation_set_path', default='data/validation', type=str)
        parser.add_argument('--test_task_set_path', default='data/test_task', type=str)
        parser.add_argument('--unique_steps', default=8., type=float)
        parser.add_argument('--sampling_rate', default=44100, type=float)
        parser.add_argument('--dummy_datasets', default=False, type=bool)

        # preprocessing
        parser.add_argument('--cqt_fmin', default=30., type=float)
        parser.add_argument('--cqt_n_bins', default=292, type=int)
        parser.add_argument('--cqt_bins_per_octave', default=32, type=int)
        parser.add_argument('--cqt_hop_length', default=256, type=int)
        parser.add_argument('--cqt_filter_scale', default=0.5, type=float)
        parser.add_argument('--scalogram_output_power', default=1., type=float)
        parser.add_argument('--scalogram_pooling', nargs='+', default=(1, 2), type=int)
        parser.add_argument('--phase', default=True, type=bool)
        parser.add_argument('--audio_noise', default=0., type=float)

        # training
        parser.opt_list('--optimizer_name', default='adam', type=str, options=['adam'], tunable=False)
        parser.opt_list('--learning_rate', default=0.0001, type=float, options=[0.0001, 0.0005, 0.001], tunable=True)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--score_over_all_timesteps', default=True, type=bool)
        parser.add_argument('--visible_steps', default=60, type=int)
        parser.add_argument('--prediction_steps', default=16, type=int)
        parser.add_argument('--prediction_gap', default=8, type=int)
        parser.add_argument('--wasserstein_penalty', default=0., type=float)
        parser.add_argument('--detach_targets', default=False, type=bool)
        parser.add_argument('--file_batch_size', default=1, type=int)
        parser.add_argument('--use_all_GPUs', default=True, type=bool)
        parser.add_argument('--warmup_steps', default=0, type=int)
        parser.add_argument('--annealing_steps', default=1e6, type=int)

        # encoder
        parser.add_argument('--enc_channels', nargs='+', default=(1, 32, 32, 64, 64,
                                                                  128, 128, 256, 512), type=int)
        parser.add_argument('--enc_kernel_1_w', nargs='+', default=(1, 3, 1, 3, 1, 3, 1, 3), type=int)
        parser.add_argument('--enc_kernel_1_h', nargs='+', default=(65, 3, 33, 3, 16, 3, 9, 3), type=int)
        parser.add_argument('--enc_kernel_2_w', nargs='+', default=(3, 3, 3, 3, 3, 3, 3, 3), type=int)
        parser.add_argument('--enc_kernel_2_h', nargs='+', default=(3, 3, 3, 3, 3, 3, 3, 3), type=int)
        parser.add_argument('--enc_padding_1', nargs='+', default=(0, 1, 0, 1, 0, 1, 0, 0), type=int)
        parser.add_argument('--enc_padding_2', nargs='+', default=(1, 1, 1, 1, 1, 1, 1, 0), type=int)
        parser.add_argument('--enc_pooling_1', nargs='+', default=(1, 1, 1, 1, 1, 1, 1, 1), type=int)
        parser.add_argument('--enc_pooling_2', nargs='+', default=(1, 1, 1, 1, 1, 1, 1, 1), type=int)
        parser.add_argument('--enc_stride_1', nargs='+', default=(1, 1, 1, 1, 1, 1, 1, 1), type=int)
        parser.add_argument('--enc_stride_2', nargs='+', default=(2, 1, 2, 1, 2, 1, 1, 1), type=int)
        parser.add_argument('--enc_batch_norm', default=True, type=bool)
        parser.add_argument('--enc_batch_norm_affine', default=False, type=bool)
        parser.add_argument('--enc_dropout', default=0.2, type=float)
        parser.add_argument('--enc_residual', default=True, type=bool)
        parser.add_argument('--enc_bias', default=True, type=bool)

        # ar model
        parser.add_argument('--ar_channels', nargs='+', default=(512, 512, 512, 512, 512,
                                                                 256, 256, 256, 256, 256), type=int)
        parser.add_argument('--ar_kernel_sizes', nargs='+', default=(5, 4, 1, 3, 3,
                                                                     1, 3, 1, 5), type=int)
        parser.add_argument('--ar_pooling', nargs='+', default=(1, 1, 2, 1, 1,
                                                                2, 1, 1, 1), type=int)
        parser.add_argument('--ar_stride', nargs='+', default=(1, 1, 1, 1, 1,
                                                               1, 1, 1, 1), type=int)
        parser.add_argument('--ar_self_attention', nargs='+', default=(False, True, False, False, True,
                                                                       False, True, False, False), type=bool)
        parser.add_argument('--ar_batch_norm', default=True, type=bool)
        parser.add_argument('--ar_batch_norm_affine', default=False, type=bool)
        parser.add_argument('--ar_dropout', default=0.2, type=float)
        parser.add_argument('--ar_residual', default=True, type=bool)
        parser.add_argument('--ar_bias', default=True, type=bool)

        return parser


def decode_hparams_strings(hparams):
    for key, value in vars(hparams).items():
        if type(value) is str:
            try:
                new_value = ast.literal_eval(value)
            except:
                continue
            setattr(hparams, key, new_value)
    return hparams