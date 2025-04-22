from transformercvn.options import Options
from transformercvn.network.layers.sparse_dense_net import SparseDenseNet
from transformercvn.network.layers.sparse_convnext import SparseConvNeXt

from transformercvn.network.networks.neutrino_full_base_network import BaseProngEmbedding, NeutrinoBaseNetwork


class SparseProngEmbedding(BaseProngEmbedding):
    def create_pixel_embedding(self, options: Options, pixel_dim: int, output_dim: int):
        return SparseDenseNet(
            input_features=pixel_dim * 256 if self.one_hot_pixels else pixel_dim,
            output_features=output_dim,
            initial_latent_features=options.initial_pixel_dim,
            growth_rate=options.densenet_growth_rate,
            batch_norm_size=options.densenet_batch_norm_size,
            block_config=tuple(options.densenet_structure),
            dropout=options.dropout
        )

        # return SparseConvNeXt(
        #     input_features=pixel_dim * 256 if self.one_hot_pixels else pixel_dim,
        #     output_features=output_dim,
        #     kernel_size=5,
        #     hidden_features=(32, 64, 128, 256),
        #     hidden_depths=(1, 1, 1, 1),
        #     drop_path_rate=options.dropout,
        #     layer_scale_init_value=1e-6
        # )


class NeutrinoSparseNetwork(NeutrinoBaseNetwork):
    def create_prong_embedding(self, options: Options, features_dim: int, extra_dim: int, pixel_dim: int):
        return SparseProngEmbedding(options, features_dim, extra_dim, pixel_dim)
