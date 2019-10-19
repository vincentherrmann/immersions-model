import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa as lr

pi = np.pi


def complex_multiply(a, b, complex_dim_a=None, complex_dim_b=None):
    # if a.shape != b.shape:
    #    print('a and b must have the same shape')
    #    print('shape a:', a.shape, 'shape b:', b.shape)

    r = torch.LongTensor([0]).to(a.device)

    if complex_dim_a is None:
        complex_dim_a = len(a.shape) - 1

    if complex_dim_b is None:
        complex_dim_b = len(b.shape) - 1

    real_a = torch.index_select(a, complex_dim_a, r).squeeze(complex_dim_a)
    imag_a = torch.index_select(a, complex_dim_a, r+1).squeeze(complex_dim_a)
    real_b = torch.index_select(b, complex_dim_b, r).squeeze(complex_dim_b)
    imag_b = torch.index_select(b, complex_dim_b, r+1).squeeze(complex_dim_b)

    product_real = real_a * real_b - imag_a * imag_b
    product_imag = real_a * imag_b + imag_a * real_b

    stack_dim = max(complex_dim_a, complex_dim_b)
    return torch.stack([product_real, product_imag], dim=stack_dim)


def amplitude(z, complex_dim=None):
    r = torch.LongTensor([0]).to(z.device)

    if complex_dim is None:
        complex_dim = len(z.shape) - 1
    real = torch.index_select(z, complex_dim, r).squeeze(dim=complex_dim)
    imag = torch.index_select(z, complex_dim, r+1).squeeze(dim=complex_dim)
    return torch.sqrt(real ** 2 + imag ** 2)


def angle(z, complex_dim=None):
    r = torch.LongTensor([0]).to(z.device)
    if complex_dim is None:
        complex_dim = len(z.shape) - 1
    real = torch.index_select(z, complex_dim, r).squeeze(dim=complex_dim)
    imag = torch.index_select(z, complex_dim, r+1).squeeze(dim=complex_dim)
    return torch.atan2(imag, real)


def polar_to_complex(abs, angle, complex_dim=None):
    real = abs * torch.cos(angle)
    imag = abs * torch.sin(angle)
    if complex_dim is None:
        complex_dim = len(abs.shape)
    return torch.stack([real, imag], dim=complex_dim)


def to_complex(real, imag, complex_dim=None):
    if complex_dim is None:
        complex_dim = len(real.shape)
    return torch.stack([real, imag], dim=complex_dim)


def unwrap(x):
    y = x.clone()
    y[torch.gt(x, pi)] = x[torch.gt(x, pi)] - 2*pi
    y[torch.lt(x, -pi)] = x[torch.lt(x, -pi)] + 2*pi
    return y


class CQT(nn.Module):
    def __init__(self, sr=16000, fmin=30, n_bins=256, bins_per_octave=32, filter_scale=1., hop_length=128, trainable=False):
        super().__init__()

        self.hop_length = hop_length

        self.sr = sr
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.filter_scale = filter_scale
        self.hop_length = hop_length

        # load filters
        cqt_filters, cqt_filter_lenghts = lr.filters.constant_q(sr,
                                                                fmin=fmin,
                                                                n_bins=n_bins,
                                                                bins_per_octave=bins_per_octave,
                                                                filter_scale=filter_scale)
        self.cqt_filter_lengths = cqt_filter_lenghts

        # one convolution operation per octave
        self.conv_kernel_sizes = []  # the kernel sizes of the octaves
        self.conv_index_ranges = []  # the indices belonging to each convolution operation
        current_kernel_size = None
        last_change_index = 0
        for i, l in enumerate(cqt_filter_lenghts):
            kernel_size = 2 ** math.ceil(np.log2(l))
            if current_kernel_size is not None and kernel_size >= current_kernel_size:
                # continue if this is in the same octave
                continue
            self.conv_kernel_sizes.append(kernel_size)
            current_kernel_size = kernel_size
            if i != 0:
                self.conv_index_ranges.append(range(last_change_index, i))
            last_change_index = i
        self.conv_index_ranges.append(range(last_change_index, len(self.cqt_filter_lengths)))

        filter_length = cqt_filters.shape[-1]
        self.conv_modules = nn.ModuleList()
        for i, size in enumerate(self.conv_kernel_sizes):
            this_range = self.conv_index_ranges[i]
            offset = (filter_length - size) // 2
            if offset > 0:
                this_filter = cqt_filters[this_range, offset:-offset]
            else:
                this_filter = cqt_filters[this_range, :]
            this_filter = torch.cat([torch.from_numpy(np.real(this_filter)),
                                     torch.from_numpy(np.imag(this_filter))], dim=0).type(torch.FloatTensor)
            this_conv = nn.Conv1d(in_channels=1, out_channels=this_filter.shape[0], kernel_size=size, bias=False,
                                  stride=hop_length)  # , padding=size // 2)
            this_conv.weight = torch.nn.Parameter(this_filter.unsqueeze(1), requires_grad=False) # should be False
            self.conv_modules.append(this_conv)

        self._trainable = False
        self.trainable = trainable

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        for p in self.parameters():
            p.requires_grad = value
        self._trainable = value

    def forward(self, x):
        real = []
        imag = []
        for i, conv in enumerate(self.conv_modules):
            offset = (self.conv_kernel_sizes[0] - self.conv_kernel_sizes[i]) // 2
            conv_result = conv(x[:, :, offset:-(offset+1)])
            r, i = torch.chunk(conv_result, 2, dim=1)
            real.append(r)
            imag.append(i)
        real = torch.cat(real, dim=1)
        imag = torch.cat(imag, dim=1)
        return torch.stack([real, imag], dim=3)


class PhaseDifference(nn.Module):
    def __init__(self, sr=16000, fmin=30, n_bins=256, bins_per_octave=32, hop_length=128):
        super().__init__()

        freqs = lr.time_frequency.cqt_frequencies(fmin=fmin,
                                                  bins_per_octave=bins_per_octave,
                                                  n_bins=n_bins)
        self.fixed_phase_diff = torch.from_numpy((((1.0 * freqs * hop_length / sr) + 0.5) % 1 - 0.5) * 2 * np.pi)
        self.fixed_phase_diff = self.fixed_phase_diff.type(torch.FloatTensor).view(1, -1, 1)
        self.fixed_phase_diff = torch.nn.Parameter(self.fixed_phase_diff, requires_grad=False)
        self.scaling = torch.from_numpy(1 / np.log(freqs))
        self.scaling = self.scaling.type(torch.FloatTensor).view(1, -1, 1)
        self.scaling = torch.nn.Parameter(self.scaling, requires_grad=False)

    def forward(self, x):
        phase_diff = x[:, :, 1:] - x[:, :, :-1]
        pd = phase_diff + self.fixed_phase_diff
        pd = unwrap(pd) * self.scaling
        return pd


class PhaseAccumulation(nn.Module):
    def __init__(self, sr=16000, fmin=30, n_bins=256, bins_per_octave=32, hop_length=128):
        super().__init__()

        freqs = lr.time_frequency.cqt_frequencies(fmin=fmin,
                                                  bins_per_octave=bins_per_octave,
                                                  n_bins=n_bins)
        self.fixed_phase_diff = torch.from_numpy((((1.0 * freqs * hop_length / sr) + 0.5) % 1 - 0.5) * 2 * np.pi)
        self.fixed_phase_diff = self.fixed_phase_diff.type(torch.FloatTensor).view(1, -1, 1)
        self.scaling = torch.from_numpy(1 / np.log(freqs))
        self.scaling = self.scaling.type(torch.FloatTensor).view(1, -1, 1)
        self.start_phase = torch.zeros_like(self.scaling)
        self.scaling = torch.nn.Parameter(self.scaling, requires_grad=False)
        self.start_phase = torch.nn.Parameter(self.start_phase, requires_grad=False)

    def forward(self, x):
        x = (x / self.scaling) - self.fixed_phase_diff
        x = torch.cat([self.start_phase, x], dim=2)
        x = torch.cumsum(x, dim=2)
        return x % (2 * np.pi) - np.pi