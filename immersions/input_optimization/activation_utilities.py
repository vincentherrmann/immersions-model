import pickle
import torch
import numpy as np
from collections import OrderedDict

activation_selection_dict = {
    'layer': 'scalogram',
    'channel': None,
    'channel_region': None,
    'pitch': None,
    'pitch_region': None,
    'time': None,
    'time_region': None,
    'keep_selection': False
}


class ModelActivations:
    def __init__(self, shapes_path, ignore_time_dimension=False, remove_results=False):
        with open(shapes_path, 'rb') as handle:
            self.shapes = pickle.load(handle)

        if remove_results:
            del self.shapes['z_code']
            del self.shapes['c_code']
            del self.shapes['prediction']

        self.ignore_time_dimensions = ignore_time_dimension
        self.num_activations = 0
        self.layer_starts = OrderedDict()
        for k, v in self.shapes.items():
            if self.ignore_time_dimensions and len(v) > 1:
                v = v[:-1]
                self.shapes[k] = v
            self.layer_starts[k] = self.num_activations
            self.num_activations += np.prod(v)

        self.focus = np.zeros(self.num_activations, dtype=np.bool)
        self.floored_time_step_lookup = None

    def select_activations(self, sel=activation_selection_dict):
        if sel is None:
            return

        if not sel['keep_selection']:
            self.focus = np.zeros(self.num_activations, dtype=np.bool)

        if sel['layer'] not in self.shapes.keys():
            return

        shape = self.shapes[sel['layer']]
        focus = np.zeros(shape, np.bool)

        channel_dim, pitch_dim, time_dim = None, None, None
        if len(shape) == 3:
            channel_dim = 0
            pitch_dim = 1
            time_dim = 2
        elif self.ignore_time_dimensions:
            if len(shape) == 2:
                channel_dim = 0
                pitch_dim = 1
            else:
                channel_dim = 0
        else:
            if len(shape) == 2:
                channel_dim = 0
                time_dim = 1
            else:
                time_dim = 0

        if pitch_dim is not None:
            pitch = sel['pitch']
            pitch_region = sel['pitch_region']
            if pitch is None:
                pitch = 0.
            if pitch_region is None:
                pitch_region = 1.
            num_pitch = shape[pitch_dim]
            pitch_start = int(pitch * (1 - pitch_region) * num_pitch)
            pitch_end = int(pitch_start + pitch_region * num_pitch + 1)

        if channel_dim is not None:
            channel = sel['channel']
            channel_region = sel['channel_region']
            if channel is None:
                channel=0.
            if channel_region is None:
                channel_region = 1.
            num_channels = shape[channel_dim]
            channel_start = int(channel * (1 - channel_region) * num_channels)
            channel_end = int(channel_start + channel_region * num_channels + 1)

        if time_dim is not None:
            time = sel['time']
            time_region = sel['time_region']
            if time is None:
                time=0.
            if time_region is None:
                time_region = 1.
            num_time = shape[time_dim]
            time_start = int(time * (1 - time_region) * num_time)
            time_end = int(time_start + time_region * num_time + 1)

        if len(shape) == 3:
            focus[channel_start:channel_end, pitch_start:pitch_end, time_start:time_end] = True
        elif self.ignore_time_dimensions:
            if len(shape) == 2:
                focus[channel_start:channel_end, pitch_start:pitch_end] = True
            else:
                focus[channel_start:channel_end] = True
        else:
            if len(shape) == 2:
                focus[channel_start:channel_end, time_start:time_end] = True
            else:
                focus[time_start:time_end] = True

        l = np.prod(shape)
        o = self.layer_starts[sel['layer']]
        self.focus[o:o+l] += focus.flatten()

    def construct_time_step_lookup(self, num_time_steps):
        if self.ignore_time_dimensions:
            raise Exception("time step lookup not possible when ignoring time dimension")

        time_step_size = 0
        shapes_without_time = OrderedDict()
        layer_starts_without_time = OrderedDict()
        layer_time_size = OrderedDict()
        for k, v in self.shapes.items():
            shapes_without_time[k] = v[:-1]
            layer_starts_without_time[k] = time_step_size
            layer_time_size[k] = v[-1]
            time_step_size += np.prod(v[:-1])

        floored_lookup = np.zeros([num_time_steps, time_step_size], dtype=np.long)
        ceiled_lookup = np.zeros([num_time_steps, time_step_size], dtype=np.long)
        interpolation_factors = np.zeros([num_time_steps, time_step_size], dtype=np.float32)

        for layer, shape in shapes_without_time.items():
            shape_with_time = self.shapes[layer]
            layer_start = self.layer_starts[layer]
            layer_end = layer_start + np.prod(shape_with_time)
            timeless_layer_start = layer_starts_without_time[layer]
            timeless_layer_end = timeless_layer_start + np.prod(shape)

            time_size = shape_with_time[-1]
            time_indices = np.linspace(0., time_size, num_time_steps, endpoint=False)
            floored_time_indices = np.floor(time_indices).astype(np.long)
            ceiled_time_indices = np.ceil(time_indices).astype(np.long)
            ceiled_time_indices[ceiled_time_indices == time_size] = 0
            interp = np.mod(time_indices, 1.)
            interp = np.repeat(interp[:, np.newaxis], timeless_layer_end-timeless_layer_start, 1)

            indices = np.arange(layer_start, layer_end, dtype=np.long).reshape(shape_with_time)
            if len(shape_with_time) == 3:
                floored_time_step_indices = indices[:, :, floored_time_indices]
                ceiled_time_step_indices = indices[:, :, ceiled_time_indices]
            elif len(shape_with_time) == 2:
                floored_time_step_indices = indices[:, floored_time_indices]
                ceiled_time_step_indices = indices[:, ceiled_time_indices]
            floored_time_step_indices = floored_time_step_indices.reshape(-1, num_time_steps).transpose()
            ceiled_time_step_indices = ceiled_time_step_indices.reshape(-1, num_time_steps).transpose()
            floored_lookup[:, timeless_layer_start:timeless_layer_end] = floored_time_step_indices
            ceiled_lookup[:, timeless_layer_start:timeless_layer_end] = ceiled_time_step_indices
            interpolation_factors[:, timeless_layer_start:timeless_layer_end] = interp

        self.floored_time_step_lookup = floored_lookup
        self.ceiled_time_step_lookup = ceiled_lookup
        self.interpolation_lookup = interpolation_factors

    def convert_activations_to_timestep_weights(self, activations):
        weights = (1-self.interpolation_lookup) * activations[self.floored_time_step_lookup] \
                  + self.interpolation_lookup * activations[self.ceiled_time_step_lookup]
        return weights

    def focus_with_time_steps(self):
        if self.ignore_time_dimensions:
            return self.focus[np.newaxis]
        if self.floored_time_step_lookup is None:
            raise Exception("no time step lookup calculated")
        return self.focus[self.floored_time_step_lookup]


class ActivationStatistics:
    def __init__(self):
        self.total_mean = {}
        self.total_std = {}
        self.element_mean = {}
        self.element_std = {}

    def add_activation_batch(self, activations):
        for key, value in activations.items():
            if key not in self.total_mean.keys():
                self.total_mean[key] = []
                self.total_std[key] = []
                self.element_mean[key] = []
                self.element_std[key] = []
            self.total_mean[key].append(torch.mean(value))
            self.total_std[key].append(torch.std(value))
            self.element_mean[key].append(torch.mean(value, dim=0))
            self.element_std[key].append(torch.std(value, dim=0))

    def condense_statistics(self):
        total_mean = {}
        total_std = {}
        element_mean = {}
        element_std = {}

        for key in self.total_std.keys():
            total_mean[key] = torch.mean(torch.stack(self.total_mean[key], dim=0), dim=0)
            total_std[key] = torch.mean(torch.stack(self.total_std[key], dim=0), dim=0)
            element_mean[key] = torch.mean(torch.stack(self.element_mean[key], dim=0), dim=0)
            element_std[key] = torch.mean(torch.stack(self.element_std[key], dim=0), dim=0)

        return total_mean, total_std, element_mean, element_std


class ActivationNormalization(torch.nn.Module):
    def __init__(self, means, variances, var_epsilon=1e-3, norm_epsilon=0.1):
        super().__init__()
        self.means = torch.nn.ParameterDict({key: torch.nn.Parameter(value.unsqueeze(0), requires_grad=False)
                                             for (key, value) in  means.items()})
        self.stds = torch.nn.ParameterDict({key: torch.nn.Parameter(torch.sqrt(value).unsqueeze(0), requires_grad=False)
                                            for (key, value) in variances.items()})
        self.std_epsilon = var_epsilon
        self.norm_epsilon = norm_epsilon

    def forward(self, activations):
        for key in activations.keys():
            mean = self.means[key]
            std = self.stds[key]
            old_activations = activations[key]
            statistically_normalized_activations = (old_activations - mean) / (std + self.std_epsilon)
            intra_normalized_activations = statistically_normalized_activations #/ (torch.std(statistically_normalized_activations) + self.norm_epsilon)
            activations[key] = intra_normalized_activations
        return activations


def activation_downsampling(activation_dict, target_length):
    activation_dict = activation_dict.copy()
    for key, value in activation_dict.items():
        if key == 'c_code' or key == 'prediction':
            continue
        dim = len(value.shape) - 1
        downsampling_factor = int(value.shape[dim] / target_length)
        if downsampling_factor > 1:
            unsqueeze = len(value.shape) == 2
            if unsqueeze:
                value = value.unsqueeze(1)
            activation_dict[key] = F.avg_pool1d(value, kernel_size=downsampling_factor)
            if unsqueeze:
                value.squeeze(1)

    return activation_dict


def convert_activation_dict_type(activation_dict, dtype=torch.float, select_batch=None):
    converted_activations = {}
    if select_batch is not None:
        for key, value in activation_dict.items():
            converted_activations[key] = value[select_batch].detach().cpu().contiguous().clone().type(dtype)
    else:
        for key, value in activation_dict.items():
            converted_activations[key] = value.detach().cpu().contiguous().clone().type(dtype)
    return converted_activations


def select_activation_slice(activations, channel=0., channel_region=1.,
                            pitch=0., pitch_region=1.,
                            time=0., time_region=1.):

    # batch, channel, pitch, time  or
    # batch, channel, time

    num_channels = activations.shape[1]
    channel_region = max(1, int(channel_region * num_channels))
    channel_pos = int(channel * (num_channels - channel_region + 1))
    channel_start = max(0, channel_pos - channel_region + 1)
    channel_end = min(num_channels, channel_pos + channel_region)

    num_time = activations.shape[-1]
    time_region = max(1, int(time_region * num_time))
    time_pos = int(time * (num_time - time_region + 1))
    time_start = max(0, time_pos - time_region + 1)
    time_end = min(num_time, time_pos + time_region)

    if len(activations.shape) == 3:
        slice = activations[:, channel_start:channel_end, time_start:time_end]
    else:
        num_pitch = activations.shape[2]
        pitch_region = max(1, int(pitch_region * num_pitch))
        pitch_pos = int(pitch * (num_pitch - pitch_region + 1))
        pitch_start = max(0, pitch_pos - pitch_region + 1)
        pitch_end = min(num_pitch, pitch_pos + pitch_region)
        slice = activations[:, channel_start:channel_end, pitch_start:pitch_end, time_start:time_end]

    return slice


def flatten_activations(activation_dict, exclude_first_dimension=False):
    activations = []
    if exclude_first_dimension:
        for key, value in activation_dict.items():
            activations.append(value.view(value.shape[0], -1))
        activations = torch.cat(activations, dim=1)
    else:
        for key, value in activation_dict.items():
            activations.append(value.view(-1))
        activations = torch.cat(activations)
    return activations


if __name__ == '__main__':
    path = '/Users/vincentherrmann/Documents/Projekte/Immersions/models/e32-2019-08-13/activation_shapes.pickle'
    activations = ModelActivations(path, ignore_time_dimension=False, remove_results=True)
    activations.construct_time_step_lookup(240)
    time_step_focus = activations.focus_with_time_steps()
    selection = activation_selection_dict.copy()
    selection['pitch_region'] = 0.5
    activations.select_activations(selection)
    pass