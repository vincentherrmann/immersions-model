import torch
import torch.nn as nn
import torch.nn.functional as F

from immersions.model.utilities import ActivationWriter


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x):
        # x: batch, channel, L
        x = x.permute(2, 0, 1)
        output, _ = self.attn(x, x, x) # L, batch, channel
        return output.permute(1, 2, 0)


class ConvolutionalArBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 pooling=1,
                 stride=1,
                 bias=True,
                 residual=False,
                 batch_norm=False,
                 batch_norm_affine=True,
                 name='ar_block',
                 activation_register=None,
                 self_attention=False,
                 dropout=0.):
        super().__init__()

        self.name = name
        self.downsampling_factor = stride*pooling

        self.main_modules = nn.ModuleList()
        if pooling > 1:
            self.main_modules.append(nn.MaxPool1d(pooling, ceil_mode=True))
        self.main_modules.append(nn.Conv1d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           bias=bias))
        if batch_norm:
            self.main_modules.append(nn.BatchNorm1d(out_channels, affine=batch_norm_affine))

        if self_attention:
            self.main_modules.append(SelfAttention(embed_dim=out_channels, num_heads=8))

        if dropout > 0.:
            self.main_modules.append(nn.Dropout(dropout))

        self.residual_modules = None
        self.residual = residual
        if self.residual:
            self.residual_modules = nn.ModuleList()
            if self.downsampling_factor > 1:
                self.residual_modules.append(nn.MaxPool1d(pooling*stride, ceil_mode=True))
            if in_channels != out_channels:
                self.residual_modules.append(nn.Conv1d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=1))

            if dropout > 0.:
                self.main_modules.append(nn.Dropout(dropout))

        self.output_activation_writer = ActivationWriter(register=activation_register,
                                                         name=self.name)

    def forward(self, x):
        l = x.shape[2]
        b = -l % self.downsampling_factor
        if b != 0:
            padding = torch.zeros([x.shape[0], x.shape[1], b]).to(x.device)
            x = torch.cat([padding, x], dim=2)
            #print("start padding:", b)

        original_x = x
        for m in self.main_modules:
            x = m(x)
        main_x = x
        if self.residual:
            x = original_x
            for m in self.residual_modules:
                x = m(x)
            main_x += x[:, :, -main_x.shape[2]:]

        main_x = F.relu(main_x)

        self.output_activation_writer(main_x[:, :, None, :])
        return main_x


class ConvolutionalArModel(nn.Module):
    def __init__(self, hparams, activation_register=None):
        super().__init__()
        self.module_list = nn.ModuleList()
        self.receptive_field = 1
        self.downsampling_factor = 1
        for l in range(len(hparams.ar_kernel_sizes)):
            self.module_list.append(ConvolutionalArBlock(in_channels=hparams.ar_channels[l],
                                                         out_channels=hparams.ar_channels[l + 1],
                                                         kernel_size=hparams.ar_kernel_sizes[l],
                                                         stride=hparams.ar_stride[l],
                                                         pooling=hparams.ar_pooling[l],
                                                         bias=hparams.ar_bias,
                                                         batch_norm=hparams.ar_batch_norm,
                                                         batch_norm_affine=hparams.ar_batch_norm_affine,
                                                         residual=hparams.ar_residual,
                                                         name='ar_block_' + str(l),
                                                         activation_register=activation_register,
                                                         self_attention=hparams.ar_self_attention[l],
                                                         dropout=hparams.ar_dropout))
            self.receptive_field += (hparams.ar_kernel_sizes[l] - 1) * self.downsampling_factor
            self.downsampling_factor *= hparams.ar_pooling[l] * hparams.ar_stride[l]

        self.encoding_size = hparams.ar_channels[0]
        self.ar_size = hparams.ar_channels[-1]

    def forward(self, x):
        for m in self.module_list:
            x = m(x)
        return x  # [:, :, -1]
