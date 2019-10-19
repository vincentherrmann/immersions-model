from unittest import TestCase
from immersions.model.preprocessing import PreprocessingModule
from immersions.cpc_system import ContrastivePredictiveSystem
from immersions.audio_dataset import AudioDataset, MaestroDataset
from test_tube import HyperOptArgumentParser
from pytorch_lightning.utilities.arg_parse import add_default_args
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import sounddevice as sd
import time
import torch
import torch.nn.functional as F
dataset = MaestroDataset('/Volumes/Elements/Datasets/maestro-v2.0.0', item_length=44100*8, sampling_rate=44100,
                         mode='validation', max_file_count=20, shuffle_with_seed=123)
parser = HyperOptArgumentParser(strategy='random_search', add_help=False)
#parser = ArgumentParser()
parser = ContrastivePredictiveSystem.add_model_specific_args(parser,
                                                             root_dir='../')
hparams = parser.parse_args()
hparams.audio_noise = 3e-3
preprocessing = PreprocessingModule(hparams)
preprocessing.train()

item = dataset[1000][0][None, :]

# filter = 1 - (torch.cos(torch.linspace(0, 2*3.14159, steps=17)[1:-1]) * 0.5 + 0.5)
# filter = filter[None, None, :] / filter.sum()
#
# noise = torch.randn_like(item)
# noise = F.conv1d(noise, filter, padding=7)
#
# item += noise * 3e-3

scal = preprocessing(item)

plt.imshow(scal[0, 0], origin='lower')
plt.show()
plt.imshow(scal[0, 1], origin='lower')
plt.show()

sd.play(item[0, 0], 44100.)
time.sleep(8.)
sd.stop()
pass