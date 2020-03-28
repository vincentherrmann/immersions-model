import os
import torch
import random
from torch.nn import functional as F
from torch.utils.data import DataLoader
from _collections import OrderedDict

import pytorch_lightning as pl
from immersions.audio_dataset import AudioTestingDataset, MaestroTestingDataset
from immersions.model.preprocessing import PreprocessingModule
from immersions.model.scalogram_encoder import ScalogramResidualEncoder
from immersions.model.autoregressive import ConvolutionalArModel


class SupervisedTaskModel(torch.nn.Module):
    def __init__(self, encoder, autoregressive_model, hparams, num_classes):
        super().__init__()
        self.encoder = encoder
        self.ar_model = autoregressive_model
        self.classifier_model = torch.nn.Sequential(
            torch.nn.Linear(in_features=hparams.ar_channels[-1], out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        z = self.encoder(x)
        c = self.ar_model(z)
        result = self.classifier_model(c[:, :, 0])
        return result


class SupervisedTaskSystem(pl.LightningModule):
    def __init__(self, hparams, dataset_path, evaluation_ratio=0.2):
        super(SupervisedTaskSystem, self).__init__()
        # not the best model...
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.task_dataset_path = dataset_path
        self._load_dataset()
        self.__build_model()

        self.num_items = len(self.dataset)
        self.classifier_model = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.hparams.ar_channels[-1], out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=128, out_features=self.dataset.num_classes)
        )

        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        index_list = list(range(self.num_items))
        random.seed(123)
        random.shuffle(index_list)
        self.validation_indices = index_list[:int(self.num_items*evaluation_ratio)]
        self.training_indices = index_list[int(self.num_items*evaluation_ratio):]

    def __build_model(self):
        """
        Layout model
        :return:
        """

        multi_gpu = torch.cuda.device_count() > 1 and self.hparams.use_all_GPUs

        with torch.no_grad():
            self.preprocessing = PreprocessingModule(self.hparams)
        self.encoder = ScalogramResidualEncoder(self.hparams, preprocessing_module=self.preprocessing)
        self.ar_model = ConvolutionalArModel(self.hparams)
        self.model = SupervisedTaskModel(self.encoder, self.ar_model, hparams=self.hparams,
                                         num_classes=self.dataset.num_classes)
        self.item_length = 8 * 44100
        self.preprocessing_downsampling = self.preprocessing.downsampling_factor
        self.preprocessing_receptive_field = self.preprocessing.receptive_field

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
        # self.example_input_array = torch.zeros(1, 1, self.model.item_length)

    def _load_dataset(self):
        if "maestro" in self.task_dataset_path:
            self.dataset = MaestroTestingDataset(location=self.task_dataset_path,
                                                 item_length=44100 * 8,
                                                 sampling_rate=self.hparams.sampling_rate,
                                                 unique_length=44100 * 4)
        else:
            self.dataset = AudioTestingDataset(location=self.task_dataset_path,
                                               item_length=44100*8,
                                               sampling_rate=self.hparams.sampling_rate,
                                               unique_length=44100*4)

    def forward(self, x):
        scal = self.preprocessing(x[:, None, :])
        r = self.model(scal)
        return r

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        output = OrderedDict({
            'loss': loss,
            'progress_bar': {'train_loss': loss},
            'log': {'tng_loss': loss}
        })

        return output

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        accuracy = torch.mean((torch.argmax(y_hat, dim=1) == y).float())
        return {'val_loss': F.cross_entropy(y_hat, y), 'val_accuracy': accuracy}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()

        result = {
            "log": {"val_loss": avg_loss,
                    "val_acc": avg_acc},
            "val_loss": avg_loss
        }
        return result

        return output

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        print("call task training data loader")
        dataset = torch.utils.data.Subset(self.dataset, self.training_indices)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        print("call task validation data loader")
        dataset = torch.utils.data.Subset(self.dataset, self.validation_indices)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)