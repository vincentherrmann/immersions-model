import torch
import torch.nn as nn
from collections import OrderedDict


class ActivationRegister:
    def __init__(self, writing_condition=None, clone_activations=False, batch_filter=None, move_to_cpu=False, devices=None):
        self.devices = devices
        if devices is not None:
            self.activations = {}
            for dev in devices:
                self.activations[dev] = OrderedDict()
        else:
            self.activations = OrderedDict()
        self.active = True
        self.writing_condition = writing_condition
        self.clone_activations = clone_activations
        self.batch_filter = batch_filter
        self.move_to_cpu = move_to_cpu

    def write_activation(self, name, value):
        if not self.active:
            return

        if self.writing_condition is not None:
            if not self.writing_condition(value):
                return

        if self.batch_filter is not None:
            value = value[self.batch_filter]

        if self.devices is not None:
            dev = value.device.index

        if self.move_to_cpu:
            value = value.cpu()

        if self.clone_activations:
            value = value.clone()

        if self.devices is not None:
            self.activations[dev][name] = value
        else:
            self.activations[name] = value

    def get_activations(self):
        if self.devices is None:
            return self.activations
        else:
            act = {
                key: torch.cuda.comm.gather([self.activations[dev][key] for dev in self.devices],
                                            dim=0, destination=self.devices[0])
                for key in self.activations[self.devices[0]].keys()
            }
            return act


class ActivationWriter(nn.Module):
    def __init__(self, register, name):
        super().__init__()
        self.register = register
        self.name = name

    def forward(self, x):
        if self.register is not None:
            self.register.write_activation(self.name, x)
        return x


class Conv2dSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False, groups=in_channels)
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)

    @property
    def weight(self):
        return self.conv.weight

    def forward(self, x):
        return self.conv_1x1(self.conv(x))
