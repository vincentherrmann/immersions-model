from unittest import TestCase
from immersions.model.preprocessing import PreprocessingModule
from immersions.cpc_system import ContrastivePredictiveSystem
from immersions.audio_dataset import AudioDataset, MaestroDataset
from test_tube import HyperOptArgumentParser
from pytorch_lightning.utilities.arg_parse import add_default_args
from argparse import ArgumentParser



class TestPreprocessingModule(TestCase):

    def test_preprocessing_noise(self):
        dataset = MaestroDataset('/Volumes/Elements/Datasets/maestro-v2.0.0', item_length=176400, sampling_rate=44100,
                                 mode='validation', max_file_count=20, shuffle_with_seed=123)
        parser = HyperOptArgumentParser(strategy='random_search', add_help=False)
        #parser = ArgumentParser()
        parser = ContrastivePredictiveSystem.add_model_specific_args(parser,
                                                                     root_dir='../')
        hparams = parser.parse_args()
        preprocessing = PreprocessingModule(hparams)
        pass
