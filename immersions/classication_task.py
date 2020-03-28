import os
import torch
import random
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from immersions.audio_dataset import AudioTestingDataset, MaestroTestingDataset


class ClassificationTaskModel(pl.LightningModule):
    def __init__(self, cpc_system, task_dataset_path, feature_size, hidden_layers=1, evaluation_ratio=0.2):
        super(ClassificationTaskModel, self).__init__()
        # not the best model...
        self.__cpc_system = cpc_system,
        self.task_dataset_path = task_dataset_path
        self.audio_dataset = self._load_dataset()
        self.feature_size = feature_size

        self.num_items = len(self.audio_dataset)

        if hidden_layers <= 0:
            self.classifier_model = torch.nn.Linear(in_features=self.feature_size,
                                                    out_features=self.audio_dataset.num_classes)
        else:
            modules = [torch.nn.Linear(in_features=self.feature_size, out_features=128)]
            for _ in range(1, hidden_layers):
                modules.append(torch.nn.ReLU())
                modules.append(torch.nn.Dropout(0.5))
                modules.append(torch.nn.Linear(in_features=128, out_features=128))

            modules.append(torch.nn.ReLU())
            modules.append(torch.nn.Dropout(0.5))
            modules.append(torch.nn.Linear(in_features=128,
                                           out_features=self.audio_dataset.num_classes))
            self.classifier_model = torch.nn.Sequential(*modules)

        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        index_list = list(range(self.num_items))
        random.seed(123)
        random.shuffle(index_list)
        self.batch_size = 64
        self.validation_indices = index_list[:int(self.num_items*evaluation_ratio)]
        self.training_indices = index_list[int(self.num_items*evaluation_ratio):]

    def _load_dataset(self):
        return AudioTestingDataset(location=self.task_dataset_path,
                                   item_length=self.cpc_system.item_length,
                                   sampling_rate=self.cpc_system.hparams.sampling_rate,
                                   unique_length=44100*4)

    def calculate_data(self):
        # calculate data for test task
        batch_size = self.cpc_system.batch_size
        task_data = torch.FloatTensor(self.num_items, self.feature_size)
        task_labels = torch.LongTensor(self.num_items)
        task_data.needs_grad = False
        task_labels.needs_grad = False
        t_dataloader = torch.utils.data.DataLoader(self.audio_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=4,
                                                   pin_memory=False,
                                                   shuffle=False)
        print("calculate test task data")
        for step, (batch, labels) in enumerate(iter(t_dataloader)):
            #print("step", step)
            batch = batch.to(device=self.device).unsqueeze(1)
            # if self.preprocessing is not None:
            #     batch = self.preprocessing(batch)
            predictions, targets, z, c, scal = self.cpc_system(batch)
            c = c.view(batch.shape[0], -1, c.shape[1])
            c = c[:, 0, :]
            # z = self.model.encoder(batch.unsqueeze(1))
            # c = self.model.autoregressive_model(z)
            task_data[step * batch_size:step*batch_size + c.shape[0], :] = c.detach().cpu()
            task_labels[step * batch_size:step*batch_size + labels.shape[0]] = labels.detach().cpu()

        del t_dataloader

        self.task_data = task_data.detach()
        self.task_labels = task_labels.detach()
        self.encoding_dataset = torch.utils.data.TensorDataset(self.task_data, self.task_labels)

    def forward(self, x):
        return self.classifier_model(x)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return {'loss': loss,
                'progress_bar': {'train_loss': loss}}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        y_hat = self.forward(x)
        accuracy = torch.mean((torch.argmax(y_hat, dim=1) == y).float())
        loss = F.cross_entropy(y_hat, y)
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        return {'val_loss': avg_loss, 'val_accuracy': avg_acc}

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        print("call task training data loader")
        dataset = torch.utils.data.Subset(self.encoding_dataset, self.training_indices)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        print("call task validation data loader")
        dataset = torch.utils.data.Subset(self.encoding_dataset, self.validation_indices)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

    # def train(self, mode=True):
    #     self.classifier_model.train(mode)
    #
    @property
    def cpc_system(self):
        return self.__cpc_system[0]


class MaestroClassificationTaskModel(ClassificationTaskModel):
    def _load_dataset(self):
        return MaestroTestingDataset(location=self.task_dataset_path,
                                     item_length=self.cpc_system.item_length,
                                     sampling_rate=self.cpc_system.hparams.sampling_rate,
                                     unique_length=44100*4,
                                     mode='validation',
                                     max_file_count=None, # 10,
                                     shuffle_with_seed=123)