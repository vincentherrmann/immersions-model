import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import pi as pi

from immersions.model.constant_q_transform import CQT, PhaseDifference, angle, amplitude


class PreprocessingModule(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.downsampling_factor = 1
        self.receptive_field = 1

        filter = 1 - (torch.cos(torch.linspace(0, 2 * pi, steps=17)[1:-1]) * 0.5 + 0.5)
        filter = filter[None, None, :] / filter.sum()
        self.register_buffer('filter', filter)

        self.cqt = None
        self.cqt = CQT(sr=hparams.sampling_rate,
                       fmin=hparams.cqt_fmin,
                       n_bins=hparams.cqt_n_bins,
                       bins_per_octave=hparams.cqt_bins_per_octave,
                       filter_scale=hparams.cqt_filter_scale,
                       hop_length=hparams.cqt_hop_length,
                       trainable=False)
        self.downsampling_factor = hparams.cqt_hop_length
        self.receptive_field = self.cqt.conv_kernel_sizes[0]

        self.phase_diff = None
        if hparams.phase:
            self.phase_diff = PhaseDifference(sr=hparams.sampling_rate,
                                              fmin=hparams.cqt_fmin,
                                              n_bins=hparams.cqt_n_bins,
                                              bins_per_octave=hparams.cqt_bins_per_octave,
                                              hop_length=hparams.cqt_hop_length)

        self.output_power = hparams.scalogram_output_power

        self.offset = 1e-9
        self.log_offset = -math.log(self.offset)
        self.normalization_factor = 4. / self.log_offset
        self.pooling = hparams.scalogram_pooling
        self.downsampling_factor *= self.pooling[1]
        self.audio_noise = hparams.audio_noise
        self.output = None

    def forward(self, x):
        # x shape:  batch, channels, samples

        if self.audio_noise > 0. and self.training:
            noise = torch.randn_like(x)
            noise = F.conv1d(noise, self.filter, padding=7)
            x = x + noise*self.audio_noise

        if self.cqt is None:
            return x
        else:
            x = self.cqt(x)

        if self.phase_diff is not None:
            amp = torch.pow(amplitude(x[:, :, 1:]), 2)
            amp = torch.log(amp + self.offset) + self.log_offset
            phi = self.phase_diff(angle(x))
            x = torch.stack([amp, phi], dim=1)
        else:
            x = torch.pow(amplitude(x), 2)
            x = torch.log(x + self.offset).unsqueeze(1)
            x += self.log_offset

        if self.pooling is not None:
            x = F.max_pool2d(x, self.pooling)

        x *= self.normalization_factor
        x = x**self.output_power

        #if self.output_requires_grad:
        #    x.requires_grad = True
        #    x.retain_grad()
        self.output = x
        return x