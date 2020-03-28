import torch
import torch.nn.functional as F
#import torchaudio
import pprint
import os.path
import pickle
import time
import glob
import numpy as np
from collections import OrderedDict
import librosa as lr
import gc

from matplotlib import pyplot as plt
from immersions.model.utilities import ActivationRegister
from immersions.cpc_system import ContrastivePredictiveSystem
from immersions.cpc_system_maestro import ContrastivePredictiveSystemMaestro
from immersions.input_optimization.activation_utilities import ModelActivations, activation_selection_dict, ActivationNormalization
#from immersions_control_app.streaming import SocketDataExchangeClient
import immersions.input_optimization.optimization_utilities as util

default_control_dict = {
            'lr': 1e-4,
            'selected_clip': 'silence',
            'mix_original': 0.,
            'batch_size': 8,
            'time_jitter': 0.,
            'time_masking': 0.,
            'pitch_masking': 0.,
            'activation_loss': 0.,
            'noise_loss': 0.,
            'high_freq_loss': 0.,
            'eq_bands': None,
            'activation_selection': None
        }


class Optimization:
    def __init__(self, weights_path, tags_path, model_shapes_path, ranges_path, noise_statistics_path,
                 data_statistics_path, soundclips_path, communicator=None,
                 dev='cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.activation_mask = None

        self.dev = dev

        self.communicator = communicator
        self.losses = [0.] * 100
        self.step_durations = []
        self.step_tic = time.time()
        self.load_soundclips(soundclips_path)
        self.load_statistics(noise_statistics_path, data_statistics_path)
        self.load_model(weights_path, tags_path, model_shapes_path, ranges_path)
        self.setup_optimization()
        self.control_dict = default_control_dict
        self.tic = time.time()
        self.active = True

    def setup_optimization(self):
        self.optimizer = torch.optim.SGD([self.audio_input], lr=1e-3)
        #self.optimizer = torch.optim.Adam([self.audio_input], lr=1e-3)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda s: 1.05 ** s)
        total_receptive_field = self.model.encoder.receptive_field * self.model.ar_model.downsampling_factor
        #zero_point = self.activation_ranges['scalogram'][0] * self.model.preprocessing.downsampling_factor + self.model.preprocessing.receptive_field
        #zero_point -= 6000
        zero_point = 8000

        item_length = 550000  # self.model.item_length  #self.model.preprocessing_receptive_field + 100 * 4096
        self.jitter_loop_module = util.JitterLoop(output_length=item_length, jitter_batches=8,
                                                  jitter_size=64000,
                                                  zero_position=-int(zero_point),
                                                  first_batch_offset=-int(zero_point))
        #self.input_extender = df.JitterLoop(output_length=self.input_length, dim=2, jitter_size=0)
        self.time_masking = util.Masking2D(size=0, axis='width', value=torch.FloatTensor([0., 0.]),
                                           exclude_first_batch=True)
        self.pitch_masking = util.Masking2D(size=0, axis='height', value=torch.FloatTensor([0., 0.]), exclude_first_batch=True)

    def load_statistics(self, mean_statistics_path, variance_statistics_path):
        with open(mean_statistics_path, 'rb') as handle:
            f = pickle.load(handle)
            #self.mean_statistics = f['element_mean']
            self.mean_statistics = f['total_mean']

        with open(variance_statistics_path, 'rb') as handle:
            f = pickle.load(handle)
            #self.std_statistics = f['element_std']
            self.std_statistics = f['total_std']

    def load_soundclips(self, clip_directory):
        self.sr = 44100
        clip_paths = sorted(glob.glob(clip_directory + '/*.wav'))
        self.soundclip_dict = OrderedDict()
        for path in clip_paths:
            #clip = torchaudio.load(path)[0]
            clip, _ = lr.load(path, sr=self.sr)
            clip = torch.from_numpy(clip)
            if len(clip.shape) == 1:
                clip = clip[None]
            if len(clip.shape) == 2:
                clip = clip[0:1]
            if clip.shape[1] != self.sr * 4:
                clip = F.pad(clip, (0, self.sr*4 - clip.shape[1]))
            clip = clip.unsqueeze(0).to(self.dev) * 0.5
            self.soundclip_dict[os.path.basename(path)] = clip

        audio_input = self.soundclip_dict['silence.wav'].clone()
        self.original_input = audio_input.clone()

        audio_input.requires_grad = True
        self.audio_input = audio_input

    def load_model(self, weights_path, tags_path, model_shapes_path, ranges_path):
        if "maestro" in weights_path:
            system = ContrastivePredictiveSystemMaestro
        else:
            system = ContrastivePredictiveSystem
        if tags_path is None:
            model = system.load_from_checkpoint(weights_path)
        else:
            model = system.load_from_metrics(weights_path, tags_csv=tags_path)

        if self.dev != 'cpu' and torch.cuda.is_available():
            model.cuda()
            model.on_gpu = True
        model.freeze()
        model.eval()
        model.activation_register.active = True
        model.model.module.viz_mode = True
        self.model = model

        # remove DataParallel
        self.model.model = self.model.model.module
        self.model.model.activation_register.devices = None
        self.model.model.activation_register.activations = OrderedDict()

        self.register = self.model.activation_register
        self.register.active = True

        pprint.pprint(self.model)

        self.activation_normalization = ActivationNormalization(self.mean_statistics, self.std_statistics,
                                                                var_epsilon=1e-1, norm_epsilon=1e-2)

        self.model_activation = ModelActivations(shapes_path=model_shapes_path,
                                                 ignore_time_dimension=False,
                                                 remove_results=True)
        with open(ranges_path, 'rb') as handle:
            self.activation_ranges = pickle.load(handle)

        self.high_freq_filter = torch.linspace(-1., 1., steps=self.model_activation.shapes['scalogram'][1])
        self.high_freq_filter = torch.clamp(self.high_freq_filter, 0., 1.)[None, :, None]

        if self.dev != 'cpu' and torch.cuda.is_available():
            self.activation_normalization.cuda()
            self.high_freq_filter = self.high_freq_filter.cuda()
        else:
            self.activation_normalization.cpu()
            self.high_freq_filter.cpu()

    def input_preprocessing(self):
        self.audio_input.data += torch.rand_like(self.audio_input.data) * 1e-11
        normalized_audio_input = self.audio_input[0]

        jittered_input = self.jitter_loop_module(normalized_audio_input)[:, None]
        scal = self.model.preprocessing(jittered_input)
        scal.retain_grad()

        self.time_masking.size = int(self.control_dict['time_masking'] * scal.shape[3])
        self.pitch_masking.size = int(self.control_dict['pitch_masking'] * scal.shape[2])
        masked_scal = self.time_masking(scal)
        masked_scal = self.pitch_masking(masked_scal)

        masked_scal.retain_grad()
        return masked_scal

    def get_activation_shapes(self):
        scal = self.input_preprocessing()
        predicted_z, targets, z, c = self.model(scal)
        registered_activations = self.register.get_activations()

        activation_shapes = OrderedDict()
        for key, value in registered_activations.items():
            activation_shapes[key] = value.shape
        pass

    def step(self):
        self.receive_data()

        toc = time.time()
        #print("optimization step communication:", toc - self.tic)
        self.tic = toc
        self.step_durations.append(toc - self.step_tic)
        self.step_durations = self.step_durations[-10:]
        self.step_tic = toc

        scal = self.input_preprocessing()

        if not self.active:
            self.scal = scal[0, 0]
            self.scal_grad = scal[0, 0] * 0.
            dummy_activations = torch.ones(1, self.model_activation.num_activations)
            self.send_data(dummy_activations)


        toc = time.time()
        print("optimization step preprocessing:", toc - self.tic)
        self.tic = toc

        predicted_z, targets, z, c = self.model.model(scal)

        toc = time.time()
        print("optimization step forward:", toc - self.tic)
        self.tic = toc

        registered_activations = self.register.get_activations()

        del registered_activations['c_code']
        del registered_activations['z_code']
        del registered_activations['prediction']

        for key, range in self.activation_ranges.items():
            registered_activations[key] = registered_activations[key][..., -range:]

        normalized_activations = self.activation_normalization(registered_activations)
        #normalized_activations = registered_activations

        flat_activations = self.flatten_activations(normalized_activations, exclude_first_dimension=True)
        # set nans to zero
        #flat_activations[torch.bitwise_not(torch.isfinite(flat_activations))] = 0.
        flat_activations[flat_activations != flat_activations] = 0.
        selected_activations = flat_activations * self.activation_mask.unsqueeze(0)

        loss = -torch.mean(torch.mean(selected_activations, dim=0)**2)  # TODO which exponent?

        activation_energy_loss = torch.mean(torch.abs(flat_activations))
        activation_energy_loss *= self.control_dict['activation_loss']

        noise_loss = torch.mean(torch.abs(scal[:, 0]))
        noise_loss *= self.control_dict['noise_loss']

        high_freq_loss = torch.mean(torch.abs(scal[:, 0] * self.high_freq_filter))
        high_freq_loss *= self.control_dict['high_freq_loss']

        loss += activation_energy_loss + noise_loss + high_freq_loss

        if self.audio_input.grad is not None:
            self.audio_input.grad *= 0

        self.losses.pop(0)
        self.losses.append(loss.item())

        nan_input = torch.isnan(self.audio_input).sum() > 0
        if nan_input:
            #if loss != loss: # if loss is nan
            print("nan in audio input!")
            self.audio_input.data = self.original_input.clone()
        print("loss:", loss.item())

        toc = time.time()
        print("optimization step loss calc:", toc - self.tic)
        self.tic = toc

        loss.backward()
        # normalize gradient
        self.audio_input.grad /= torch.std(self.audio_input.grad).squeeze() + 1e-7
        if not nan_input:
            self.optimizer.step()

        toc = time.time()
        print("optimization step backward:", toc - self.tic)
        self.tic = toc

        self.scal = scal[0, 0]
        self.scal_grad = scal.grad[0, 0]

        mix_o = self.control_dict['mix_original']
        self.audio_input.data *= (1 - mix_o)
        self.audio_input.data += mix_o * self.original_input

        amplitude = torch.abs(self.audio_input.max()).item()
        if amplitude > 1.:
            self.audio_input.data /= amplitude

        self.send_data(flat_activations)

    def send_data(self, activations):
        print("optimization send data")
        scal_range = self.activation_ranges['scalogram']
        #input_scal = self.scal #self.preprocessing_module(self.input_extender(self.audio_input))
        #input_grad_scal = self.scal.grad #self.preprocessing_module(self.input_extender(self.audio_input.grad))
        viz_activations = activations[0].detach().cpu().type(torch.float16)

        data_dict = {'activations': viz_activations,
                     'scalogram': self.scal[:, -scal_range:].detach().cpu().numpy(),
                     'scalogram_grad': self.scal_grad[:, -scal_range:].detach().cpu().numpy(),
                     'losses': self.losses,
                     'step_durations': self.step_durations}

        signal_data = self.audio_input.clone().detach().squeeze().cpu().numpy()
        # if amplitude > 1.:
        #     signal_data /= amplitude
        signal_data *= 32000.
        data_dict['audio'] = signal_data.astype(np.int16)

        data = pickle.dumps(data_dict)
        if self.communicator is not None:
            #print("optimization set new data")
            print("send data with size ")
            self.communicator.set_new_data(data)

    def receive_data(self):
        #print("optimization receive data")
        if self.communicator is not None and self.communicator.new_data_available:
            self.control_dict = pickle.loads(self.communicator.get_received_data())
            self.eq_bands = self.control_dict['eq_bands']
            if self.eq_bands is not None:
                self.eq_bands = torch.from_numpy(self.eq_bands).to(self.dev)
            #print("select soundclip", self.control_dict['selected_clip'])
            self.original_input = self.soundclip_dict[self.control_dict['selected_clip']]
        else:
            if self.activation_mask is not None:
                return
        if self.control_dict is None:
            time.sleep(0.01)
            return

        lr = 10**self.control_dict['lr']
        if self.control_dict['lr'] < -4.9:
            lr = 0.
        for g in self.optimizer.param_groups:
            g['lr'] = lr

        batch_size = int(self.control_dict['batch_size'])
        self.jitter_loop_module.jitter_size = int(self.sr * self.control_dict['time_jitter'])
        self.jitter_loop_module.jitter_batches = batch_size

        self.model_activation.select_activations(self.control_dict['activation_selection'])
        self.activation_mask = torch.from_numpy(self.model_activation.focus).type(torch.float32).to(self.dev)

    def run(self):
        while True:
            self.step()

    @staticmethod
    def flatten_activations(activation_dict, exclude_first_dimension=False):
        activations = []
        if exclude_first_dimension:
            for key, value in activation_dict.items():
                activations.append(value.contiguous().view(value.shape[0], -1))
            activations = torch.cat(activations, dim=1)
        else:
            for key, value in activation_dict.items():
                activations.append(value.contiguous().view(-1))
            activations = torch.cat(activations)
        return activations

    @staticmethod
    def convert_activation_dict_type(activation_dict, dtype=torch.float, select_batch=None):
        converted_activations = {}
        if select_batch is not None:
            for key, value in activation_dict.items():
                converted_activations[key] = value[select_batch].detach().cpu().contiguous().clone().type(dtype)
        else:
            for key, value in activation_dict.items():
                converted_activations[key] = value.detach().cpu().contiguous().clone().type(dtype)
        return converted_activations

    @staticmethod
    def memReport():
        num_elements = 0
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                num_elements += obj.numel()
                print(type(obj), print(obj.dtype), obj.size())
        print("total elements in memory:", num_elements)

    @staticmethod
    def mean_activations_by_time(activations, plot=True):
        mean_activations = {}
        for key, value in activations.items():
            length = value.shape[-1]
            a = value.detach().cpu().view(-1, length).mean(dim=0)
            mean_activations[key] = a
            plt.plot(a)
            plt.show()


if __name__ == "__main__":
    optimization = Optimization(experiment='e32',
                                model_path='/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13/snapshots_model_2019-08-13_run_0_90000',
                                model_shapes_path='/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13/activation_shapes.pickle',
                                noise_statistics_path='/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13/noise_statistics_snapshots_model_2019-08-13_run_0_90000.pickle',
                                data_statistics_path='/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13/data_statistics_snapshots_model_2019-08-13_run_0_90000.pickle',
                                soundclips_path='/Users/vincentherrmann/Documents/Projekte/Immersions/soundclips_44khz')
    optimization.control_dict = default_control_dict
    optimization.run()