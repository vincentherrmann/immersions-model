import torch
import torch.nn as nn
import torch.nn.functional as F

from immersions.model.utilities import ActivationWriter, Conv2dSeparable


default_encoder_block_dict = {'in_channels': 64,
                              'hidden_channels': None,
                              'out_channels': 64,
                              'kernel_size_1': (3, 3),
                              'kernel_size_2': (3, 3),
                              'top_padding_1': None,
                              'top_padding_2': None,
                              'padding_1': 0,
                              'padding_2': 0,
                              'stride_1': 1,
                              'stride_2': 1,
                              'pooling_1': 1,
                              'pooling_2': 1,
                              'bias': True,
                              'separable': False,
                              'residual': True,
                              'batch_norm': False,
                              'dropout': 0.}


class ScalogramEncoderBlock(nn.Module):
    def __init__(self, args_dict=default_encoder_block_dict,
                 name='scalogram_block', activation_register=None):
        super().__init__()

        self.name = name

        #   +--------------- pooling -- conv_1x1 ----------------+
        #   |                                                    |
        # --+-- conv_a -- pooling -- ReLU -- padding -- conv_b --+--

        if args_dict['hidden_channels'] is None:
            args_dict['hidden_channels'] = args_dict['out_channels']
        conv_module = Conv2dSeparable if args_dict['separable'] else nn.Conv2d

        self.main_modules = nn.ModuleList()

        if args_dict['top_padding_1'] is not None:
            self.main_modules.append(nn.ZeroPad2d((0, 0, args_dict['top_padding_1'], 0)))

        self.main_modules.append(conv_module(in_channels=args_dict['in_channels'],
                                             out_channels=args_dict['hidden_channels'],
                                             kernel_size=args_dict['kernel_size_1'],
                                             bias=args_dict['bias'],
                                             padding=args_dict['padding_1'],
                                             stride=args_dict['stride_1']))

        if args_dict['batch_norm']:
            self.main_modules.append(nn.BatchNorm2d(args_dict['hidden_channels'], affine=args_dict['batch_norm_affine']))

        if args_dict['pooling_1'] > 1:
            self.main_modules.append(nn.MaxPool2d(kernel_size=args_dict['pooling_1'],
                                                  ceil_mode=args_dict['ceil_pooling']))

        self.main_modules.append(nn.ReLU())

        self.main_modules.append(ActivationWriter(register=activation_register,
                                                  name=self.name + '_main_conv_1'))

        if args_dict['dropout'] > 0.:
            self.main_modules.append(nn.Dropout(args_dict['dropout']))

        if args_dict['top_padding_2'] is not None:
            self.main_modules.append(nn.ZeroPad2d((0, 0, args_dict['top_padding_2'], 0)))

        self.main_modules.append(conv_module(in_channels=args_dict['hidden_channels'],
                                             out_channels=args_dict['out_channels'],
                                             kernel_size=args_dict['kernel_size_2'],
                                             bias=args_dict['bias'],
                                             padding=args_dict['padding_2'],
                                             stride=args_dict['stride_2']))

        if args_dict['batch_norm']:
            self.main_modules.append(nn.BatchNorm2d(args_dict['out_channels'], affine=args_dict['batch_norm_affine']))

        if args_dict['pooling_2'] > 1:
            self.main_modules.append(nn.MaxPool2d(kernel_size=args_dict['pooling_2'],
                                                  ceil_mode=args_dict['ceil_pooling']))

        self.main_modules.append(nn.ReLU())

        if args_dict['dropout'] > 0.:
            self.main_modules.append(nn.Dropout(args_dict['dropout']))

        self.residual = args_dict['residual']
        if self.residual:
            self.residual_modules = nn.ModuleList()

            stride_pool = args_dict['stride_1'] * args_dict['stride_2'] * args_dict['pooling_1'] * args_dict['pooling_2']
            if stride_pool > 1:
                self.residual_modules.append(nn.MaxPool2d(kernel_size=stride_pool, ceil_mode=True))

            if args_dict['in_channels'] != args_dict['out_channels']:
                self.residual_modules.append(nn.Conv2d(in_channels=args_dict['in_channels'],
                                                       out_channels=args_dict['out_channels'],
                                                       kernel_size=1,
                                                       padding=args_dict['padding_1']+args_dict['padding_2'],
                                                       bias=False))

            if args_dict['dropout'] > 0.:
                self.residual_modules.append(nn.Dropout(args_dict['dropout']))

        self.output_activation_writer = ActivationWriter(register=activation_register,
                                                         name=self.name + '_main_conv_2')

        self.downsampling_factor = args_dict["stride_1"] * args_dict["pooling_1"] * args_dict["stride_2"] * args_dict["pooling_2"]

    def forward(self, x):
        l = x.shape[3]
        b = -l % self.downsampling_factor
        if b != 0:
            padding = torch.zeros([x.shape[0], x.shape[1], x.shape[2], b]).to(x.device)
            x = torch.cat([padding, x], dim=3)
            #print("start padding:", b)

        original_input = x
        for m in self.main_modules:
            x = m(x)

        main = x
        if self.residual:
            x = original_input
            for m in self.residual_modules:
                x = m(x)
            res = x[:, :, :, -main.shape[3]:]
            r_h = res.shape[2]
            m_h = main.shape[2]
            o_h = (r_h - m_h + 1) / 2
            if int(o_h) > 0:
                res = res[:, :, -int(o_h + m_h):-int(o_h), :]
            main = main + res
            if main.shape[2] == 0:
                print("faulty shape:")
                print(main.shape)

        main = F.relu(main)

        self.output_activation_writer(main)

        return main


scalogram_encoder_resnet_dict = {'phase': True,
                                 'blocks': [default_encoder_block_dict,
                                            default_encoder_block_dict,
                                            default_encoder_block_dict]}


class ScalogramResidualEncoder(nn.Module):
    def __init__(self, hparams, preprocessing_module=None, activation_register=None, verbose=0):
        super().__init__()

        self.verbose = verbose

        self.phase = hparams.phase
        if self.phase:
            hparams.enc_channels = (2,) + hparams.enc_channels[1:]

        if preprocessing_module is None:
            self.receptive_field = 1
            self.downsampling_factor = 1
        else:
            self.receptive_field = preprocessing_module.receptive_field
            self.downsampling_factor = preprocessing_module.downsampling_factor

        self.blocks = nn.ModuleList()

        for i in range(len(hparams.enc_kernel_1_h)):
            block_dict = {'in_channels': hparams.enc_channels[i],
                          'hidden_channels': None,
                          'out_channels': hparams.enc_channels[i+1],
                          'kernel_size_1': (hparams.enc_kernel_1_h[i], hparams.enc_kernel_1_w[i]),
                          'kernel_size_2': (hparams.enc_kernel_2_h[i], hparams.enc_kernel_2_w[i]),
                          'top_padding_1': None,
                          'top_padding_2': None,
                          'padding_1': hparams.enc_padding_1[i],
                          'padding_2': hparams.enc_padding_2[i],
                          'stride_1': hparams.enc_stride_1[i],
                          'stride_2': hparams.enc_stride_2[i],
                          'pooling_1': hparams.enc_pooling_1[i],
                          'pooling_2': hparams.enc_pooling_2[i],
                          'bias': hparams.enc_bias,
                          'separable': False,
                          'residual': hparams.enc_residual,
                          'batch_norm': hparams.enc_batch_norm,
                          'batch_norm_affine': hparams.enc_batch_norm_affine,
                          'dropout': hparams.enc_dropout,
                          'ceil_pooling': True}
            self.blocks.append(ScalogramEncoderBlock(block_dict,
                                                     name='scalogram_block_' + str(i),
                                                     activation_register=activation_register))

            self.receptive_field += (block_dict['kernel_size_1'][1] - 1) * self.downsampling_factor
            self.downsampling_factor *= block_dict['pooling_1'] * block_dict['stride_1']
            self.receptive_field += (block_dict['kernel_size_2'][1] - 1) * self.downsampling_factor
            self.downsampling_factor *= block_dict['pooling_2'] * block_dict['stride_2']

            if self.verbose > 0:
                print("receptive field after block", i, ":", self.receptive_field)
        self.receptive_field = self.calc_receptive_field(hparams)

    def calc_receptive_field(self, hparams):
        n = len(hparams.enc_stride_1)
        flatten = lambda l: [item for sublist in l for item in sublist]
        kernel_sizes = flatten([[hparams.enc_kernel_1_w[i], hparams.enc_kernel_2_w[i]] for i in range(n)])
        strides = flatten([[hparams.enc_stride_1[i] * hparams.enc_pooling_1[i],
                            hparams.enc_stride_2[i] * hparams.enc_pooling_2[i]] for i in range(n)])
        r = 1
        s = 1
        for l in range(n*2):
            if l > 0:
                s *= strides[l-1]
            r += (kernel_sizes[l] - 1) * s
        return r

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(2)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.verbose > 1:
                print("activation shape after block", i, ":", x.shape)
        return x[:, :, 0, :]
