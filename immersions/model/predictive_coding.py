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
        #self.group_norm = nn.GroupNorm(num_groups=prediction_steps, num_channels=enc_size*prediction_steps, affine=False)

    @property
    def item_length(self):
        item_length = self.encoder.receptive_field
        item_length += (self.visible_steps + self.prediction_steps) * self.encoder.downsampling_factor
        return item_length

    def forward(self, x):
        x = self.input_activation_writer(x)

        z = self.encoder(x)  # batch x enc_dim x steps
        n, e, _ = z.shape
        if self.detach_targets:
            targets = z[:, :, self.autoregressive_model.receptive_field:].detach()  # batch, enc_size, step
        else:
            targets = z[:, :, self.autoregressive_model.receptive_field:]  # batch, enc_size, step

        num_predictions = (targets.shape[2] - self.prediction_steps) // self.autoregressive_model.downsampling_factor + 1
        prediction_indices = torch.arange(self.prediction_steps).to(x.device)
        prediction_offsets = torch.arange(num_predictions).to(x.device) * self.autoregressive_model.downsampling_factor
        prediction_indices = prediction_offsets[:, None] + prediction_indices[None, :]
        targets = targets[:, :, prediction_indices]  # batch, enc_size, c_step, step
        targets = targets.permute(0, 2, 1, 3).contiguous().view(n*num_predictions, e, self.prediction_steps)  # batchxc_step, enc_size, step



        z = z[:, :, :-self.prediction_steps]

        z = self.z_activation_writer(z)

        c = self.autoregressive_model(z)  # batch, enc_size, c_step
        c = c[:, :, :num_predictions]
        # if len(c.shape) == 3:
        #     c = c[:, :, 0]

        c = self.c_activation_writer(c)

        c = c.permute(0, 2, 1).contiguous().view(n*num_predictions, -1)  # batch, c_step, c_enc_size


        predicted_z = self.prediction_model(c)  # batch, step*enc_size
        predicted_z = predicted_z.view(-1, self.prediction_steps, self.enc_size)  # batch, step, enc_size

        predicted_z = self.prediction_activation_writer(predicted_z)

        return predicted_z, targets, z, c