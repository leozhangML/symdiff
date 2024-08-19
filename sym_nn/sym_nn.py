import torch
import torch.nn as nn
import numpy as np

from equivariant_diffusion.utils import remove_mean_with_mask
from sym_nn.utils import qr, orthogonal_haar
from sym_nn.perceiver import SymDiffPerceiverConfig, SymDiffPerceiver, SymDiffPerceiverDecoder, TensorPreprocessor, t_emb_dim


"""
# kv_dim is given by dim of input_preprocessor
# first cross-att is q_dim=d_latents, kv_dim=input_preprocessor.num_channels, and projecting to d_latents
# then q_dim=d_latents, kv_dim=d_latents
config = SymDiffPerceiverConfig(  
    num_latents=4,
    d_latents=64,
    num_blocks=1,  # number of times to apply the same block
    num_self_attends_per_block=2,  # num of self-att to encode latents in block
    num_self_attention_heads=1,
    num_cross_attention_heads=1,
    qk_channels=None,  # default is q_dim for self-att
    v_channels=None,  # default is qk_channel for self-att
    cross_attention_shape_for_attention="kv",  # what to take for qk_channels - i.e. kv_dim or q_dim
    self_attention_widening_factor=1,
    cross_attention_widening_factorint=1,
    hidden_act="gelu",
    attention_probs_dropout_prob=0.,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    use_query_residual=True,
    num_bands=4,
    max_resolution=100
)

decoder = SymDiffPerceiverDecoder(
    config,
    pos_num_channels=3,
    output_num_channels=3,
    index_dims=-1,
    qk_channels=None,
    v_channels=None,
    num_heads=1,
    widening_factor=1,
    use_query_residual=False,
    final_project=False,
    decoder_self_attention=False
    )
"""


class SymDiff_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        gamma_num_latents: int, 
        gamma_d_latents: int, 
        gamma_num_blocks: int, 
        gamma_num_self_attends_per_block: int, 
        gamma_num_self_attention_heads: int, 
        gamma_num_cross_attention_heads: int,
        gamma_pos_num_channels: int,
        gamma_num_heads: int,

        k_num_latents: int,
        k_d_latents: int,
        k_num_blocks: int,
        k_num_self_attends_per_block: int,
        k_num_self_attention_heads: int,
        k_num_cross_attention_heads: int,
        k_pos_num_channels: int,
        k_num_heads: int,
        k_decoder_self_attention: bool,

        num_bands: int, 
        max_resolution: float,
        device: str = "cpu"
    ) -> None:
        
        self.args = args

        self.gamma_config = SymDiffPerceiverConfig(
            num_latents=gamma_num_latents, d_latents=gamma_d_latents, num_blocks=gamma_num_blocks, 
            num_self_attends_per_block=gamma_num_self_attends_per_block, 
            num_self_attention_heads=gamma_num_self_attention_heads,
            num_cross_attention_heads=gamma_num_cross_attention_heads,
            attention_probs_dropout_prob=0.,
            num_bands=num_bands, max_resolution=max_resolution
        )
        self.k_config = SymDiffPerceiverConfig(
            num_latents=k_num_latents, d_latents=k_d_latents, num_blocks=k_num_blocks, 
            num_self_attends_per_block=k_num_self_attends_per_block, 
            num_cross_attention_heads=k_num_cross_attention_heads,
            num_self_attention_heads=k_num_self_attention_heads,
            attention_probs_dropout_prob=0.,
            num_bands=num_bands, max_resolution=max_resolution
        )

        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        self.t_dim = t_emb_dim(self.gamma_config)
        self.node_dim = 3 + in_node_nf

        self.gamma_input_preprocessor = TensorPreprocessor(3, self.gamma_config)  # pos+t
        self.k_input_preprocessor = TensorPreprocessor(self.node_dim+context_node_nf, self.k_config)  # pos+h_int+h_cat+context+t - i.e concat context to features

        # query shape is preserved by default - hence final_project to output_num_channels
        self.gamma_decoder = SymDiffPerceiverDecoder(
            self.gamma_config, pos_num_channels=gamma_pos_num_channels,
            output_num_channels=3, index_dims=3, num_heads=gamma_num_heads,
            final_project=True
        )
        # we use the same trainable query params for each node position
        self.k_decoder = SymDiffPerceiverDecoder(
            self.k_config, pos_num_channels=k_pos_num_channels, 
            output_num_channels=self.node_dim, index_dims=-1, num_heads=k_num_heads,
            final_project=True, decoder_self_attention=k_decoder_self_attention
        )

        self.gamma = SymDiffPerceiver(self.gamma_config, decoder=self.gamma_decoder, 
                                      input_preprocessor=self.gamma_input_preprocessor)
        self.k = SymDiffPerceiver(self.k_config, decoder=self.k_decoder,
                                  input_preprocessor=self.k_input_preprocessor)

        self.device = device

    def forward(self):
        raise NotImplementedError

    def _forward(self, t, xh, node_mask, edge_mask, context):  # computation for predicting noise - might need to change for 
        # xh: [bs, n_nodes, dims]
        # t: [bs, 1]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]
        
        x = xh[:, :, :3]
        h = xh[:, :, 3:]
        g = orthogonal_haar(dim=3, target_tensor=x)  # [bs, 3, 3]

        x = remove_mean_with_mask(x, node_mask)
        g_inv_x = torch.bmm(x, g)  # as x is represented row-wise

        # [bs, 3, 3]
        gamma = torch.bmm(
            self.gamma(g_inv_x, t, enc_attention_mask=node_mask, 
                       dec_attention_mask=torch.ones(len(x), 3, device=self.device)),
            g.transpose(2, 1)
            )

        gamma_inv_x = torch.bmm(x, gamma)
        xh = torch.cat([gamma_inv_x, h], dim=2)

        # dec_attention_mask=node_mask is used as the query has the same sequence dim as n_node
        # [bs, n_nodes, dims]
        k_output = torch.bmm(
            self.k(xh, t, enc_attention_mask=node_mask,
                   dec_attention_mask=None),
            gamma.transpose(2, 1)
        )

        return k_output