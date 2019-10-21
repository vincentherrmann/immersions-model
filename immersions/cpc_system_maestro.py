import os
from immersions.cpc_system import ContrastivePredictiveSystem
from immersions.audio_dataset import MaestroDataset, FileBatchSampler


class ContrastivePredictiveSystemMaestro(ContrastivePredictiveSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_datasets(self):
        if not os.path.exists(self.hparams.training_set_path):
            print("dataset paths are not valid, loading the model without datasets")
            return
        self.training_set = MaestroDataset(self.hparams.training_set_path,
                                           item_length=self.item_length,
                                           unique_length=self.encoder.downsampling_factor * self.hparams.unique_steps,
                                           sampling_rate=self.hparams.sampling_rate,
                                           dummy=self.hparams.dummy_datasets,
                                           mode='train')
        print("training set length:", len(self.training_set))
        self.validation_set = MaestroDataset(self.hparams.validation_set_path,
                                             item_length=self.item_length,
                                             unique_length=self.encoder.downsampling_factor * self.hparams.unique_steps,
                                             sampling_rate=self.hparams.sampling_rate,
                                             dummy=self.hparams.dummy_datasets,
                                             mode='validation')
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