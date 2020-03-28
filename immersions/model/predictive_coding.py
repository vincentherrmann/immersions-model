import torch
import torch.nn as nn
import torch.nn.functional as F

from immersions.model.utilities import ActivationWriter


class PredictiveCodingModel(nn.Module):
    def __init__(self, encoder, autoregressive_model, hparams,
                 activation_register=None):  # TODO: test if targets should be detached
        super().__init__()
        self.enc_size = autoregressive_model.encoding_size
        self.ar_size = autoregressive_model.ar_size
        self.visible_steps = hparams.visible_steps
        self.prediction_steps = hparams.prediction_steps
        self.prediction_gap = hparams.prediction_gap

        self.encoder = encoder
        self.autoregressive_model = autoregressive_model
        self.prediction_model = nn.Linear(in_features=self.ar_size, out_features=self.enc_size*self.prediction_steps,
                                          bias=False)
        self.activation_register = activation_register

        self.input_activation_writer = ActivationWriter(register=self.activation_register,
                                                        name='scalogram')
        self.z_activation_writer = ActivationWriter(register=self.activation_register,
                                                    name='z_code')
        self.c_activation_writer = ActivationWriter(register=self.activation_register,
                                                    name='c_code')
        self.prediction_activation_writer = ActivationWriter(register=self.activation_register,
                                                             name='prediction')
        self.detach_targets = hparams.detach_targets
        self.viz_mode = False
        #self.group_norm = nn.GroupNorm(num_groups=prediction_steps, num_channels=enc_size*prediction_steps, affine=False)

    @property
    def item_length(self):
        item_length = self.encoder.receptive_field
        item_length += (self.visible_steps + self.prediction_gap + self.prediction_steps) * self.encoder.downsampling_factor
        return item_length

    def forward(self, x):
        x = self.input_activation_writer(x)

        z = self.encoder(x)  # batch x enc_dim x steps
        n, e, total_steps = z.shape

        #   |                   |---|---|--targets--|
        #   |----------z----------| gap |   steps   |
        # z | | | | | | | | | | | | | | | | | | | | |
        # c |         | c | c | c |

        z_end = total_steps - (self.prediction_gap + self.prediction_steps)
        if self.viz_mode:
            z_end = total_steps

        c = self.autoregressive_model(z[:, :, :z_end])
        num_predictions = c.shape[2]

        if not self.viz_mode:
            additional_prediction_offset = ((num_predictions-1) * self.autoregressive_model.downsampling_factor)
            target_start = z_end + self.prediction_gap - additional_prediction_offset
            target_end = target_start + self.prediction_steps + additional_prediction_offset

            targets = z[:, :, target_start:target_end]  # batch, enc_size, step
            if self.detach_targets:
                targets = targets.detach()

            prediction_indices = torch.arange(self.prediction_steps).to(x.device)
            prediction_offsets = torch.arange(num_predictions).to(x.device) * self.autoregressive_model.downsampling_factor
            prediction_indices = prediction_offsets[:, None] + prediction_indices[None, :]
            targets = targets[:, :, prediction_indices]  # batch, enc_size, c_step, step
            targets = targets.permute(0, 2, 1, 3).contiguous().view(n*num_predictions, e, self.prediction_steps)  # batchxc_step, enc_size, step

        z = z[:, :, :z_end]

        self.z_activation_writer(z[:, :, None, :])
        self.c_activation_writer(c[:, :, None, :])

        c = c.permute(0, 2, 1).contiguous().view(n*num_predictions, -1)  # batch, c_step, c_enc_size

        predicted_z = self.prediction_model(c)  # batch, step*enc_size
        predicted_z = predicted_z.view(-1, self.prediction_steps, self.enc_size)  # batch, step, enc_size

        self.prediction_activation_writer(predicted_z[:, :, None, :])

        if self.viz_mode:
            targets = torch.zeros_like(predicted_z)

        return predicted_z, targets, z, c