from transformercvn.options import Options
from transformercvn.network.layers.dense_net import DenseNet
from transformercvn.network.networks.neutrino_full_base_network import BaseProngEmbedding, NeutrinoBaseNetwork


class DenseProngEmbedding(BaseProngEmbedding):
    def create_pixel_embedding(self, options: Options, pixel_dim: int, output_dim: int):
        return DenseNet(
            input_features=pixel_dim * 256 if self.one_hot_pixels else pixel_dim,
            output_features=output_dim,
            initial_latent_features=options.initial_pixel_dim,
            growth_rate=options.densenet_growth_rate,
            batch_norm_size=options.densenet_batch_norm_size,
            block_config=tuple(options.densenet_structure),
            dropout=options.dropout
        )


class NeutrinoDenseNetwork(NeutrinoBaseNetwork):
    def create_prong_embedding(self, options: Options, features_dim: int, extra_dim: int, pixel_dim: int):
        return DenseProngEmbedding(options, features_dim, extra_dim, pixel_dim)
