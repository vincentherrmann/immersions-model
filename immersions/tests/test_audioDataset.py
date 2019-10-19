from unittest import TestCase
from immersions.audio_dataset import AudioDataset, MaestroDataset
from matplotlib import pyplot as plt

class TestAudioDataset(TestCase):
    def test_dataset_mp3(self):
        dataset = AudioDataset('/Volumes/Elements/Datasets/Immersions/house_data_mp3/validation',
                               item_length=176400, unique_length=88200, sampling_rate=44100)
        item_0 = dataset[0]
        pass

    def test_maestro_dataset(self):
        dataset = MaestroDataset('/Volumes/Elements/Datasets/maestro-v2.0.0', item_length=176400, sampling_rate=44100,
                                 mode='validation', max_file_count=20, shuffle_with_seed=123)
        pass

