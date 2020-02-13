from unittest import TestCase
from immersions.model.scalogram_encoder import ScalogramResidualEncoder
import torch
from types import SimpleNamespace
from matplotlib import pyplot as plt


class TestScalogramResidualEncoder(TestCase):
    def test_encoder(self):
        args = SimpleNamespace(
            phase=True,
            enc_channels=(1, 8, 16, 32, 64, 128, 256, 512, 512),
            enc_kernel_1_w=(3, 1, 3, 1, 3, 1, 3, 1),
            enc_kernel_1_h=(3, 3, 3, 3, 3, 3, 3, 3),
            enc_kernel_2_w=(1, 1, 1, 1, 1, 1, 1, 1),
            enc_kernel_2_h=(25, 3, 15, 3, 15, 3, 5, 3),
            enc_padding_1=(0, 0, 0, 0, 0, 0, 0, 0),
            enc_padding_2=(0, 0, 0, 0, 0, 0, 0, 0),
            enc_pooling_1=(2, 1, 1, 1, 2, 1, 1, 1),
            enc_pooling_2=(1, 1, 1, 1, 1, 1, 1, 1),
            enc_stride_1=(1, 1, 1, 1, 1, 1, 1, 1),
            enc_stride_2=(1, 1, 1, 1, 1, 1, 1, 1),
            enc_batch_norm=False,
            enc_batch_norm_affine=False,
            enc_dropout=0.2,
            enc_residual=True,
            enc_bias=True
        )

        encoder = ScalogramResidualEncoder(args, verbose=0)

        l = 200
        encoder_outputs = []

        for i in range(190, l):
            inf_pos = i
            dummy_input = torch.zeros([1, 2, 216, l])
            dummy_input[:, :, :, inf_pos] = float('inf')
            r = encoder(dummy_input)
            encoder_outputs.append(r[0, 0])
            print("inf_pos:", inf_pos, r[0, 0, :])

        encoder_outputs = torch.stack(encoder_outputs)
        plt.ion()
        plt.imshow(encoder_outputs.detach())
        #plt.show()
        pass

