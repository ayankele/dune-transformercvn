from typing import Optional

import torch
from torch import nn, Tensor, jit
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence

from transformercvn.options import Options
from transformercvn.network.layers.masked_sequential import MaskedSequential
from transformercvn.network.layers.masked_batchnorm_0d import MaskedBatchNorm0D
from transformercvn.network.layers.packed_data import masked_pack_1d, masked_pad_1d


class MaskedLinearBlock(nn.Module):
    def __init__(self, options: Options, input_dim: int, output_dim: int):
        super(MaskedLinearBlock, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

        if options.linear_batch_norm:
            self.norm = MaskedBatchNorm0D(output_dim)
        else:
            self.norm = nn.Identity()

        if options.linear_prelu_activation:
            self.activation = nn.PReLU(output_dim)
        else:
            self.activation = nn.ReLU()

        self.dropout = nn.Dropout(options.dropout)

    def forward(self, data: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, C = data.shape

        features = self.linear(data)
        features = self.norm(features, mask)
        features = self.activation(features)
        features = self.dropout(features)

        if mask is not None:
            features = features * mask.view(B, 1)

        return features


class ProngFeatureEmbedding(nn.Module):

    def __init__(self, options: Options, sequence_dim: int, extra_dim: int, output_dim: int):
        super(ProngFeatureEmbedding, self).__init__()

        self.options = options
        self.extra_dim = extra_dim
        self.output_dim = output_dim
        self.sequence_dim = sequence_dim

        self.embedding = jit.script(self.create_embedding_layers(options, sequence_dim + extra_dim, output_dim))
        self.embedding_dim = self.options.hidden_dim

    @staticmethod
    def create_embedding_layers(options, input_dim, output_dim):
        embedding_layers = [MaskedLinearBlock(options, input_dim, options.initial_feature_dim)]
        current_embedding_dim = options.initial_feature_dim

        for i in range(options.num_embedding_layers):
            next_embedding_dim = 2 * current_embedding_dim
            if next_embedding_dim >= output_dim:
                break

            embedding_layers.append(MaskedLinearBlock(options, current_embedding_dim, next_embedding_dim))
            current_embedding_dim = next_embedding_dim

        embedding_layers.append(MaskedLinearBlock(options, current_embedding_dim, output_dim))

        return MaskedSequential(*embedding_layers)

    def forward(self, data: Tensor, extra: Tensor, mask: Tensor) -> Tensor:
        batch_size, max_particles, input_dim = data.shape

        # Create some extra views of our mask
        embedding_mask = mask.view(batch_size * max_particles)

        # Add the extra variables as constant event-level features to each prong
        extra = extra.reshape(batch_size, 1, self.extra_dim)
        extra = extra.expand(batch_size, max_particles, self.extra_dim)

        # Create the combined feature vector and pass through embedding
        hidden = torch.cat([data, extra], dim=2)

        # Pack the sequence to get rid of unused terms
        hidden = masked_pack_1d(hidden, mask)
        hidden = self.embedding(hidden)
        return masked_pad_1d(hidden, mask)

        # packed = pack_padded_sequence(hidden, lengths, batch_first=True, enforce_sorted=False)
        # hidden = packed.data

        # Perform main embedding
        # hidden = hidden.view(batch_size * max_particles, self.extra_dim + self.sequence_dim)

        # hidden = hidden.view(batch_size, max_particles, self.output_dim)

        # Transform packed sequence back into a padded form for transfomer.
        # hidden = PackedSequence(hidden, packed.batch_sizes, packed.sorted_indices, packed.unsorted_indices)
        # hidden = pad_packed_sequence(hidden, batch_first=True, total_length=max_particles)[0]
        # return hidden


        # return hidden * mask.view(batch_size, max_particles, 1)
