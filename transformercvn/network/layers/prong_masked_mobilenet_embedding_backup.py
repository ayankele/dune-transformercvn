from typing import Optional, List, Union, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence

from transformercvn.network.layers.masked_sequential import MaskedSequential
from transformercvn.network.layers.masked_batchnorm_2d import MaskedBatchNorm2D
from transformercvn.network.layers.packed_data import masked_pack_3d, masked_pad_3d, masked_pad_1d

def make_divisible_channel_count(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MaskedConv2D(nn.Module):
    """
    A quick override of nn.Conv2d which allows mask as a
    second input to make sequential layers simpler.
    """

    def __init__(self,
                 in_channel: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 0,
                 padding_mode: str = 'zeros',
                 dilation: Union[int, Tuple[int]] = 1,
                 groups: int = 1,
                 bias: bool = True):
        super(MaskedConv2D, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              padding_mode=padding_mode,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)

    def forward(self, images: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.conv(images)


class MaskedConvolutionBlock(nn.Module):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            dropout: float = 0.1
    ):
        super(MaskedConvolutionBlock, self).__init__()

        padding = (kernel_size - 1) // 2 * dilation
        self.out_channels = out_planes

        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size,
                              stride,
                              padding,
                              dilation=dilation,
                              groups=groups,
                              bias=False)

        self.norm = MaskedBatchNorm2D(out_planes)
        self.activation = nn.PReLU(out_planes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, C, H, W = images.shape

        features = self.conv(images)
        features = self.norm(features, mask)
        features = self.activation(features)
        features = self.dropout(features)

        if mask is not None:
            features = features * mask.view(B, 1, 1, 1)

        return features


class MaskedSqueezeAndExcitation(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """
    __constants__ = ["reduction_ratio"]

    def __init__(self,
                 num_channels: int,
                 reduction_ratio: int = 2):
        super(MaskedSqueezeAndExcitation, self).__init__()

        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.prelu = nn.PReLU(num_channels_reduced)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, C, H, W = images.size()

        # Average along each channel
        squeeze_tensor = images.view(B, C, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.prelu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(images, fc_out_2.view(a, b, 1, 1))

        if mask is not None:
            output_tensor = output_tensor * mask.view(B, 1, 1, 1)

        return output_tensor


class MaskedInvertedResidual(nn.Module):
    __constants__ = ["use_res_connect", "out_channels", "_is_cn"]

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            stride: int,
            expand_ratio: int,
            dropout: float = 0.1):
        super(MaskedInvertedResidual, self).__init__()

        self.use_res_connect = stride == 1 and input_dim == output_dim
        self.out_channels = output_dim
        self._is_cn = stride > 1
        self.stride = stride

        assert stride in [1, 2]

        hidden_dim = int(round(input_dim * expand_ratio))

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(MaskedConvolutionBlock(input_dim, hidden_dim, kernel_size=1, dropout=dropout))

        layers.extend([
            MaskedConvolutionBlock(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, dropout=dropout),
            MaskedConv2D(hidden_dim, output_dim, 1, 1, 0, bias=False),
            MaskedSqueezeAndExcitation(output_dim),
            MaskedBatchNorm2D(output_dim),
        ])

        self.convolution_blocks = MaskedSequential(*layers)

    def forward(self, images: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, C, H, W = images.size()

        hidden = self.convolution_blocks(images, mask)

        if mask is not None:
            hidden = hidden * mask.view(B, 1, 1, 1)

        if self.use_res_connect:
            return images + hidden
        else:
            return hidden


class MaskedProngMobileNetEmbedding(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            width_multiplier: float = 1.0,
            dropout: float = 0.1,
            initial_dimension: int = 32,
            inverted_residual_setting: Optional[List[List[int]]] = None,
            round_nearest: int = 8):
        """
        MobileNet V2 main class
        Args:
            width_multiplier (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MaskedProngMobileNetEmbedding, self).__init__()

        input_channel = initial_dimension
        last_channel = hidden_dim

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 8, 1, 1],
                [6, 16, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 128, 3, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        self.last_channel = make_divisible_channel_count(last_channel * max(1.0, width_multiplier), round_nearest)
        input_channel = make_divisible_channel_count(input_channel * width_multiplier, round_nearest)
        layers: List[nn.Module] = [MaskedConvolutionBlock(input_dim, input_channel, stride=2, dropout=dropout)]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible_channel_count(c * width_multiplier, round_nearest)

            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(MaskedInvertedResidual(input_channel,
                                                     output_channel,
                                                     stride,
                                                     expand_ratio=t,
                                                     dropout=dropout))
                input_channel = output_channel

        # building last several layers
        layers.append(MaskedConvolutionBlock(input_channel, self.last_channel, kernel_size=1, dropout=dropout))

        # make it nn.Sequential
        self.layers = nn.ModuleList(layers)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (MaskedBatchNorm2D, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, images: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, C, H, W = images.shape

        # Masked sequential layer
        features = images
        for layer in self.layers:
            features = layer(features, mask)

        # Cannot use "squeeze" as batch-size can be 1
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)

        return features

    def forward(self, pixels: Tensor, mask: Tensor) -> Tensor:
        batch_size, max_particles, channels, height, width = pixels.shape

        hidden = masked_pack_3d(pixels, mask)
        hidden = self._forward_impl(hidden)
        return masked_pad_1d(hidden, mask)

        # lengths = mask.sum(dim=1).cpu()
        # packed_pixels = pack_padded_sequence(pixels, lengths, batch_first=True, enforce_sorted=False)
        # pixels = packed_pixels.data

        # Apply the CNN to every timestep independently
        # pixels = pixels.view(batch_size * max_particles, channels, height, width)
        # pixel_mask = mask.view(batch_size * max_particles)

        # Apply forward model
        # hidden = self._forward_impl(pixels, pixel_mask)


        # hidden = PackedSequence(hidden,
        #                         packed_pixels.batch_sizes,
        #                         packed_pixels.sorted_indices,
        #                         packed_pixels.unsorted_indices)
        #
        # hidden = pad_packed_sequence(hidden, batch_first=True, total_length=max_particles)[0]
        # return hidden

        # hidden = hidden.reshape(batch_size, max_particles, self.last_channel)
        # return hidden * mask.view(batch_size, max_particles, 1)
