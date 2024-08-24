import torch
import torch.nn as nn
import numpy as np

from equivariant_diffusion.utils import remove_mean_with_mask, assert_mean_zero_with_mask, assert_correctly_masked
from sym_nn.utils import qr, orthogonal_haar
from sym_nn.perceiver import SymDiffPerceiverConfig, SymDiffPerceiver, SymDiffPerceiverDecoder, TensorPreprocessor, \
                      t_emb_dim, concat_t_emb, positional_encoding, IdentityPreprocessor
from sym_nn.dit import DiT

from qm9.models import EGNN_dynamics_QM9
from timm.models.vision_transformer import Mlp

"""
def remove_mean_with_mask(x, node_mask, return_mean=False):  # [bs, n_nodes, 3], [bs, n_nodes, 1]
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'  # checks for mistakes with masked positions - why?
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N  # [bs, 1, 3]
    x = x - mean * node_mask
    if return_mean:
        return x, mean
    else:
        return x
"""
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


class SymDiffPerceiver_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        gamma_num_latents: int, 
        gamma_d_latents: int,
        gamma_n_pad: int,
        gamma_num_blocks: int, 
        gamma_num_self_attends_per_block: int, 
        gamma_num_self_attention_heads: int, 
        gamma_num_cross_attention_heads: int,
        gamma_attention_probs_dropout_prob: float,
        gamma_pos_num_channels: int,
        gamma_num_heads: int,

        k_num_latents: int,
        k_d_latents: int,
        k_n_pad: int,
        k_num_blocks: int,
        k_num_self_attends_per_block: int,
        k_num_self_attention_heads: int,
        k_num_cross_attention_heads: int,
        k_attention_probs_dropout_prob: float,
        k_pos_num_channels: int,
        k_num_heads: int,
        k_decoder_self_attention: bool,
        k_num_self_heads: int,

        num_bands: int,
        max_resolution: float,
        concat_t: bool = False,
        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        
        super().__init__()

        self.args = args

        self.gamma_config = SymDiffPerceiverConfig(
            num_latents=gamma_num_latents, d_latents=gamma_d_latents, n_pad=gamma_n_pad, 
            num_blocks=gamma_num_blocks, num_self_attends_per_block=gamma_num_self_attends_per_block, 
            num_self_attention_heads=gamma_num_self_attention_heads,
            num_cross_attention_heads=gamma_num_cross_attention_heads,
            attention_probs_dropout_prob=gamma_attention_probs_dropout_prob,
            num_bands=num_bands, max_resolution=max_resolution, concat_t=concat_t
        )
        self.k_config = SymDiffPerceiverConfig(
            num_latents=k_num_latents, d_latents=k_d_latents, n_pad=k_n_pad, 
            num_blocks=k_num_blocks, num_self_attends_per_block=k_num_self_attends_per_block, 
            num_self_attention_heads=k_num_self_attention_heads,
            num_cross_attention_heads=k_num_cross_attention_heads,
            attention_probs_dropout_prob=k_attention_probs_dropout_prob,
            num_bands=num_bands, max_resolution=max_resolution, concat_t=concat_t
        )

        self.n_dims = n_dims
        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        self.t_dim = t_emb_dim(self.gamma_config)
        self.node_dim = n_dims + in_node_nf

        # pos+pad+t - want these to be even
        self.gamma_input_preprocessor = TensorPreprocessor(n_dims, self.gamma_config).to(device)
        # pos+h_int+h_cat+context+pad+t - i.e concat context to features
        self.k_input_preprocessor = TensorPreprocessor(self.node_dim+context_node_nf, self.k_config).to(device)

        # query shape is preserved by default - hence final_project to output_num_channels
        self.gamma_decoder = SymDiffPerceiverDecoder(
            self.gamma_config, pos_num_channels=gamma_pos_num_channels,
            output_num_channels=n_dims, index_dims=n_dims, num_heads=gamma_num_heads,
            final_project=True
        )
        # we use the same trainable query params for each node position (index_dims=-1)
        self.k_decoder = SymDiffPerceiverDecoder(
            self.k_config, pos_num_channels=k_pos_num_channels, 
            output_num_channels=self.node_dim, index_dims=-1, num_heads=k_num_heads,
            final_project=True, decoder_self_attention=k_decoder_self_attention, num_self_heads=k_num_self_heads
        )

        self.gamma = SymDiffPerceiver(self.gamma_config, decoder=self.gamma_decoder, 
                                      input_preprocessor=self.gamma_input_preprocessor)
        self.k = SymDiffPerceiver(self.k_config, decoder=self.k_decoder,
                                  input_preprocessor=self.k_input_preprocessor)

        self.device = device

    def forward(self):
        raise NotImplementedError

    def _forward(self, t, xh, node_mask, edge_mask, context):  # computation for predicting noise - might need to change for MiDi
        # xh: [bs, n_nodes, dims]
        # t: [bs, 1]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]
        g = orthogonal_haar(dim=self.n_dims, target_tensor=x)  # [bs, 3, 3]

        x = remove_mean_with_mask(x, node_mask)
        g_inv_x = torch.bmm(x, g)  # as x is represented row-wise

        # [bs, 3, 3] - no noise in gamma
        gamma = self.gamma(g_inv_x, t, enc_attention_mask=node_mask.squeeze(-1), 
                           dec_attention_mask=torch.ones(len(x), self.n_dims, device=self.device))[0]
        gamma = torch.bmm(qr(gamma)[0], g.transpose(2, 1))

        gamma_inv_x = torch.bmm(x, gamma)
        if context is None:
            xh = torch.cat([gamma_inv_x, h], dim=-1)
        else:
            xh = torch.cat([gamma_inv_x, h, context], dim=-1)

        # dec_attention_mask=node_mask is used as the query has the same sequence dim as n_node
        # [bs, n_nodes, dims]
        xh = self.k(xh, t, enc_attention_mask=node_mask.squeeze(-1), dec_attention_mask=None)[0]
        xh *= node_mask

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        gamma_x = torch.bmm(x, gamma.transpose(2, 1))
        xh = torch.cat([gamma_x, h], dim=-1)

        return xh


class SymDiffPerceiverFourier_dynamics(nn.Module):  # using fourier positional embeddings for all input dims

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
        gamma_attention_probs_dropout_prob: float,
        gamma_pos_num_channels: int,
        gamma_num_heads: int,

        k_num_latents: int,
        k_d_latents: int,
        k_num_blocks: int,
        k_num_self_attends_per_block: int,
        k_num_self_attention_heads: int,
        k_num_cross_attention_heads: int,
        k_attention_probs_dropout_prob: float,
        k_pos_num_channels: int,
        k_num_heads: int,
        k_decoder_self_attention: bool,
        k_num_self_heads: int,

        sigma: float,
        m: int,
        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        
        super().__init__()

        self.args = args

        self.gamma_config = SymDiffPerceiverConfig(
            num_latents=gamma_num_latents, d_latents=gamma_d_latents, n_pad=0, 
            num_blocks=gamma_num_blocks, num_self_attends_per_block=gamma_num_self_attends_per_block, 
            num_self_attention_heads=gamma_num_self_attention_heads,
            num_cross_attention_heads=gamma_num_cross_attention_heads,
            attention_probs_dropout_prob=gamma_attention_probs_dropout_prob,
            num_bands=m, max_resolution=sigma, concat_t=False, use_pos_embed=True  # for consistency
        )
        self.k_config = SymDiffPerceiverConfig(
            num_latents=k_num_latents, d_latents=k_d_latents, n_pad=0, 
            num_blocks=k_num_blocks, num_self_attends_per_block=k_num_self_attends_per_block, 
            num_self_attention_heads=k_num_self_attention_heads,
            num_cross_attention_heads=k_num_cross_attention_heads,
            attention_probs_dropout_prob=k_attention_probs_dropout_prob,
            num_bands=m, max_resolution=sigma, concat_t=False, use_pos_embed=True
        )

        self.n_dims = n_dims
        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        self.node_dim = n_dims + in_node_nf

        self.gamma_dim = 2 * (n_dims + 1) * m
        self.k_dim = 2 * (self.node_dim+context_node_nf+1) * m

        self.sigma = sigma
        self.m = m

        self.gamma_input_preprocessor = IdentityPreprocessor(self.gamma_dim).to(device)
        self.k_input_preprocessor = IdentityPreprocessor(self.k_dim).to(device)

        # query shape is preserved by default - hence final_project to output_num_channels
        self.gamma_decoder = SymDiffPerceiverDecoder(
            self.gamma_config, pos_num_channels=gamma_pos_num_channels,
            output_num_channels=n_dims, index_dims=n_dims, num_heads=gamma_num_heads,
            final_project=True
        )
        # we use the same trainable query params for each node position (index_dims=-1)
        self.k_decoder = SymDiffPerceiverDecoder(
            self.k_config, pos_num_channels=k_pos_num_channels, 
            output_num_channels=self.node_dim, index_dims=-1, num_heads=k_num_heads,
            final_project=True, decoder_self_attention=k_decoder_self_attention, num_self_heads=k_num_self_heads
        )

        self.gamma = SymDiffPerceiver(self.gamma_config, decoder=self.gamma_decoder, 
                                      input_preprocessor=self.gamma_input_preprocessor)
        self.k = SymDiffPerceiver(self.k_config, decoder=self.k_decoder,
                                  input_preprocessor=self.k_input_preprocessor)

        self.device = device

    def forward(self):
        raise NotImplementedError

    def _forward(self, t, xh, node_mask, edge_mask, context):  # computation for predicting noise - might need to change for MiDi
        # xh: [bs, n_nodes, dims]
        # t: [bs, 1]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        bs, n_nodes, _ = xh.shape

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]
        x = remove_mean_with_mask(x, node_mask)

        g = orthogonal_haar(dim=self.n_dims, target_tensor=x)  # [bs, 3, 3]
        g_inv_x = torch.bmm(x, g)  # as x is represented row-wise
        g_inv_x = positional_encoding(
            torch.cat([g_inv_x, t.unsqueeze(1).expand(bs, n_nodes, 1)], dim=-1), 
            self.sigma,
            self.m
            )

        # [bs, 3, 3] - no noise in gamma
        gamma = self.gamma(g_inv_x, t, enc_attention_mask=node_mask.squeeze(-1), 
                           dec_attention_mask=torch.ones(len(x), self.n_dims, device=self.device))[0]
        gamma = torch.bmm(qr(gamma)[0], g.transpose(2, 1))

        gamma_inv_x = torch.bmm(x, gamma)
        if context is None:
            xh = torch.cat([gamma_inv_x, h, t.unsqueeze(1).expand(bs, n_nodes, 1)], dim=-1)
        else:
            xh = torch.cat([gamma_inv_x, h, context, t.unsqueeze(1).expand(bs, n_nodes, 1)], dim=-1)
        xh = positional_encoding(xh, self.sigma, self.m)

        # dec_attention_mask=node_mask is used as the query has the same sequence dim as n_node
        # [bs, n_nodes, dims]
        xh = self.k(xh, t, enc_attention_mask=node_mask.squeeze(-1), dec_attention_mask=None)[0]
        xh *= node_mask

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        gamma_x = torch.bmm(x, gamma.transpose(2, 1))
        #assert_mean_zero_with_mask(gamma_x, node_mask)  # ???
        xh = torch.cat([gamma_x, h], dim=-1)
        #assert_correctly_masked(xh, node_mask)  
        
        return xh


class SymDiffTransformer_dynamics(nn.Module):

    """Use torch.nn.transformer encoder here"""

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int, 

        gamma_num_enc_layers: int,
        gamma_num_dec_layers: int,
        gamma_d_model: int, 
        gamma_nhead: int,
        gamma_dim_feedforward: int, 
        gamma_dropout: float,

        k_num_layers: int,
        k_d_model: int, 
        k_nhead: int,
        k_dim_feedforward: int, 
        k_dropout: float,

        num_bands: int,
        max_resolution: float,
        t_fourier: bool = True,
        concat_t: bool = False,

        activation: str = "gelu", 
        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:

        super().__init__()

        self.args = args

        # whether to add t via fourier embeddings or concat
        self.t_fourier = t_fourier
        self.t_config = SymDiffPerceiverConfig(
            num_bands=num_bands, max_resolution=max_resolution, 
            concat_t=concat_t
        )
        self.t_dim = t_emb_dim(self.t_config) if t_fourier else 1

        # dimensions of inputs
        self.n_dims = n_dims
        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        self.node_dim = n_dims + in_node_nf

        # define encoder-decoder and decoder-only transformer for gamma and k
        # project input dims to d_model 
        self.gamma_linear_in = nn.Linear(n_dims+self.t_dim, gamma_d_model, device=device)
        self.gamma_linear_out = nn.Linear(gamma_d_model, n_dims, device=device)
        self.gamma_query = nn.Parameter(torch.randn(n_dims, gamma_d_model))

        self.k_linear_in = nn.Linear(self.node_dim+self.t_dim+context_node_nf, k_d_model, device=device)
        self.k_linear_out = nn.Linear(k_d_model, self.node_dim, device=device)
        self.k_encoder_layer = nn.TransformerEncoderLayer(
            k_d_model, k_nhead, k_dim_feedforward, k_dropout, 
            activation=activation, batch_first=True, norm_first=True, 
            device=device
        )

        # need encoder-decoder for gamma as we need the shape [bs, 3, 3]
        self.gamma = nn.Transformer(
            d_model=gamma_d_model, nhead=gamma_nhead, num_encoder_layers=gamma_num_enc_layers, 
            num_decoder_layers=gamma_num_dec_layers, dim_feedforward=gamma_dim_feedforward, 
            dropout=gamma_dropout, activation="gelu", batch_first=True, 
            norm_first=True, device=device
        )
        self.k = nn.TransformerEncoder(self.k_encoder_layer, k_num_layers)

        self.device = device

    def forward(self):
        raise NotImplementedError

    def _forward(self, t, xh, node_mask, edge_mask, context):
        # t: [bs, 1]
        # xh: [bs, n_nodes, dims]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        bs, n_nodes, _ = xh.shape

        print("norm of original xh: ", torch.mean(torch.norm(xh, dim=(1, 2))))

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]
        x = remove_mean_with_mask(x, node_mask)

        g = orthogonal_haar(dim=self.n_dims, target_tensor=x)  # [bs, 3, 3]
        g_inv_x = torch.bmm(x, g)  # as x is represented row-wise

        if self.t_fourier:
            g_inv_x = concat_t_emb(self.t_config, g_inv_x, t)
        else:
            g_inv_x = torch.cat([g_inv_x, t.expand(bs, n_nodes, -1)], dim=-1)  # [bs, n_nodes, n_dims+1]

        g_inv_x = self.gamma_linear_in(g_inv_x)  # [bs, n_nodes, gamma_d_model]

        print("norm of g_inv_x after gamma_lin_in: ", torch.mean(torch.norm(g_inv_x, dim=(1, 2))))

        att_mask = ~node_mask.bool()

        # [bs, 3, 3] - no noise in gamma
        gamma = self.gamma(
            g_inv_x, self.gamma_query.expand(bs, self.n_dims, -1), 
            src_key_padding_mask=att_mask.squeeze(-1), 
            memory_key_padding_mask=att_mask.squeeze(-1)
        )
        print("norm of gamma before qr: ", torch.mean(torch.norm(gamma, dim=-1)) )
        gamma = qr(self.gamma_linear_out(gamma))[0]
        gamma = torch.bmm(gamma, g.transpose(2, 1))

        print("norm of gamma: ", torch.mean(torch.norm(gamma, dim=-1)))

        # compute k
        gamma_inv_x = torch.bmm(x, gamma)
        if context is None:
            xh = torch.cat([gamma_inv_x, h], dim=-1)
        else:
            xh = torch.cat([gamma_inv_x, h, context], dim=-1)

        print("norm of gamma to xh: ", torch.mean(torch.norm(xh, dim=(1, 2))))

        if self.t_fourier:
            xh = concat_t_emb(self.t_config, xh, t)
        else:
            xh = torch.cat([xh, t.expand(bs, n_nodes, -1)], dim=-1)  # [bs, n_nodes, n_dims+in_node_nf+context_node_nf+1]

        print("norm of xh with t: ", torch.mean(torch.norm(xh, dim=(1, 2))))

        xh = self.k_linear_in(xh)
        print("norm of xh after k_lin_in: ", torch.mean(torch.norm(xh, dim=(1, 2))))
        xh = self.k(xh, src_key_padding_mask=att_mask.squeeze(-1))
        print("norm of xh after k: ", torch.mean(torch.norm(xh, dim=(1, 2))))
        xh = self.k_linear_out(xh)  # NOTE: bias of nn.Linear
        print("norm of xh after k_lin_out: ", torch.mean(torch.norm(xh, dim=(1, 2))))
        print("norm of xh masked: ", torch.mean(torch.norm((1-node_mask)*xh, dim=(1, 2))))
        
        if torch.isnan(xh).any():
            print("NANs")
            print(torch.sum(torch.isnan(xh)))
            print(torch.isnan(xh).nonzero())
            print(xh[torch.isnan(xh).nonzero()[0]], t[torch.isnan(xh).nonzero()[0]])
            print(xh, t)
        xh *= node_mask

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U
        gamma_x = torch.bmm(x, gamma.transpose(2, 1))
        xh = torch.cat([gamma_x, h], dim=-1)
        print("norm of xh after com_free/end: ", torch.mean(torch.norm(xh, dim=(1, 2))), "\n")
        return xh


class Transformer_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        trans_num_layers: int,
        trans_d_model: int, 
        trans_nhead: int,
        trans_dim_feedforward: int, 
        trans_dropout: float,

        num_bands: int,
        max_resolution: float,
        t_fourier: bool = True,
        concat_t: bool = False,

        activation: str = "gelu", 
        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.args = args

        self.n_dims = n_dims
        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        self.node_dim = n_dims + in_node_nf

        # whether to add t via fourier embeddings or concat
        self.t_fourier = t_fourier
        self.t_config = SymDiffPerceiverConfig(
            num_bands=num_bands, max_resolution=max_resolution, 
            concat_t=concat_t
        )
        self.t_dim = t_emb_dim(self.t_config) if t_fourier else 1

        self.linear_in = nn.Linear(self.node_dim+self.t_dim+context_node_nf, trans_d_model, device=device)
        self.linear_out = nn.Linear(trans_d_model, self.node_dim, device=device)

        self.encoder_layer = nn.TransformerEncoderLayer(
            trans_d_model, trans_nhead, trans_dim_feedforward, trans_dropout, 
            activation=activation, batch_first=True, norm_first=True,
            device=device
        )
        self.model = nn.TransformerEncoder(self.encoder_layer, trans_num_layers)

    def forward(self):
        raise NotImplementedError

    def _forward(self, t, xh, node_mask, edge_mask, context):
        # t: [bs, 1]
        # xh: [bs, n_nodes, dims]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        bs, n_nodes, _ = xh.shape

        print("initial xh: ", torch.mean(torch.norm(xh, dim=(1, 2))))

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]
        x = remove_mean_with_mask(x, node_mask)

        x_0 = x * 1.

        print("initial xh with mean removed: ", torch.mean(torch.norm(xh, dim=(1, 2))))

        if context is None:
            xh = torch.cat([x, h], dim=-1)
        else:
            xh = torch.cat([x, h, context], dim=-1)

        if self.t_fourier:
            xh = concat_t_emb(self.t_config, xh, t)
        else:
            xh = torch.cat([xh, t.expand(bs, n_nodes, -1)], dim=-1)  # [bs, n_nodes, n_dims+1]
        print("initial xh with t: ", torch.mean(torch.norm(xh, dim=(1, 2))))

        xh = self.linear_in(xh)
        print("xh after lin_in: ", torch.mean(torch.norm(xh, dim=(1, 2))))
 
        att_mask = ~node_mask.bool()
        xh = self.model(xh, src_key_padding_mask=att_mask.squeeze(-1))
        print("xh after transformer: ", torch.mean(torch.norm(xh, dim=(1, 2))))

        xh = self.linear_out(xh) * node_mask 
        print("xh after lin_out: ", torch.mean(torch.norm(xh, dim=(1, 2))))

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U
            x -= x_0

        xh = torch.cat([x, h], dim=-1)
        
        print("xh with com_free: ", torch.mean(torch.norm(xh, dim=(1, 2))))

        return xh


class DiT_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        x_scale: float,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        use_fused_attn: bool,
        subtract_x_0: bool,
        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        assert use_fused_attn

        self.args = args
        self.n_dims = n_dims

        self.subtract_x_0 = subtract_x_0

        self.model = DiT(
            out_channels=n_dims+in_node_nf+context_node_nf, x_scale=x_scale, 
            hidden_size=hidden_size, depth=depth, 
            num_heads=num_heads, mlp_ratio=mlp_ratio, 
            use_fused_attn=use_fused_attn
            )

        self.device = device

    def forward(self):
        raise NotImplementedError

    def _forward(self, t, xh, node_mask, edge_mask, context):
        # t: [bs, 1]
        # xh: [bs, n_nodes, dims]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        bs, n_nodes, _ = xh.shape
        print("initial xh: ", torch.mean(torch.norm(xh, dim=(1, 2))))

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]
        x = remove_mean_with_mask(x, node_mask)
        x_0 = x * 1.
        print("initial xh with mean removed: ", torch.mean(torch.norm(xh, dim=(1, 2))))

        assert context is None
        xh = torch.cat([x, h], dim=-1)

        xh = self.model(xh, t.squeeze(-1), node_mask.squeeze(-1)) * node_mask
        print("xh after transformer: ", torch.mean(torch.norm(xh, dim=(1, 2))))

        x = xh[:, :, :self.n_dims] 
        h = xh[:, :, self.n_dims:]
        if self.subtract_x_0:
            x -= x_0

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        xh = torch.cat([x, h], dim=-1)
        print("xh with com_free: ", torch.mean(torch.norm(xh, dim=(1, 2))))
        #assert_correctly_masked(xh, node_mask)

        return xh


class DiT_GNN_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        enc_out_channels: int,
        enc_x_scale: float,
        enc_hidden_size: int,
        enc_depth: int,
        enc_num_heads: int,
        enc_mlp_ratio: float,
        use_fused_attn: bool,

        dec_hidden_features: int,

        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.args = args

        self.in_node_nf = in_node_nf  # dynamics_in_node_nf (includes time)
        self.context_node_nf = context_node_nf
        self.n_dims = n_dims

        self.gamma_enc = DiT(
            out_channels=enc_out_channels, x_scale=enc_x_scale, 
            hidden_size=enc_hidden_size, depth=enc_depth, 
            num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio, 
            use_fused_attn=use_fused_attn
        ).to(device)

        self.gamma_dec = Mlp(
            in_features=enc_hidden_size, hidden_features=dec_hidden_features,
            out_features=n_dims**2
        ).to(device)

        # use GNN for k
        self.gnn_dynamics = EGNN_dynamics_QM9(
            in_node_nf=in_node_nf, context_node_nf=args.context_node_nf,
            n_dims=n_dims, device=device, hidden_nf=args.nf,
            act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
            attention=args.attention, tanh=args.tanh, mode="gnn_dynamics", norm_constant=args.norm_constant,
            inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
            normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method
        )

        self.device = device

    def forward(self):
        raise NotImplementedError

    def _forward(self, t, xh, node_mask, edge_mask, context):
        # t: [bs, 1]
        # xh: [bs, n_nodes, dims]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        assert context is None

        bs, n_nodes, _ = xh.shape

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]
        x = remove_mean_with_mask(x, node_mask)

        g = orthogonal_haar(dim=self.n_dims, target_tensor=x)  # [bs, 3, 3]
        g_inv_x = torch.bmm(x, g)  # as x is represented row-wise

        # [bs, n_nodes, hidden_size]
        gamma = node_mask * self.gamma_enc(
            g_inv_x, t.squeeze(-1), node_mask.squeeze(-1), 
            use_final_layer=False
            )

        # decoded summed representation into gamma - this is S_n-invariant
        N = node_mask.sum(1)  # [bs, 1]
        gamma = torch.sum(gamma, dim=1) / N  # [bs, hidden_size] 
        # [bs, 3, 3]
        gamma = qr(
            self.gamma_dec(gamma).reshape(-1, self.n_dims, self.n_dims)
            )[0]
        gamma = torch.bmm(gamma, g.transpose(2, 1))

        gamma_inv_x = torch.bmm(x, gamma)
        xh = torch.cat([gamma_inv_x, h], dim=-1)
        xh = self.gnn_dynamics._forward(t, xh, node_mask, edge_mask, context)  # [bs, n_nodes, dims] - com_free

        x = xh[:, :, :self.n_dims] 
        h = xh[:, :, self.n_dims:]

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        x = torch.bmm(x, gamma.transpose(2, 1))
        xh = torch.cat([x, h], dim=-1)

        assert_correctly_masked(xh, node_mask)

        return xh


class DiT_DiT_dynamics(nn.Module):

    def __init__(
        self,
        args,
        in_node_nf: int,
        context_node_nf: int,

        enc_out_channels: int,
        enc_x_scale: float,
        enc_hidden_size: int,
        enc_depth: int,
        enc_num_heads: int,
        enc_mlp_ratio: float,
        use_fused_attn: bool,

        dec_hidden_features: int,

        x_scale: float,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,

        n_dims: int = 3,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.args = args

        self.in_node_nf = in_node_nf  # standard in_node_nf
        self.context_node_nf = context_node_nf
        self.n_dims = n_dims

        self.gamma_enc = DiT(
            out_channels=enc_out_channels, x_scale=enc_x_scale, 
            hidden_size=enc_hidden_size, depth=enc_depth, 
            num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio, 
            use_fused_attn=use_fused_attn
        ).to(device)

        self.gamma_dec = Mlp(
            in_features=enc_hidden_size, hidden_features=dec_hidden_features,
            out_features=n_dims**2
        ).to(device)

        self.k = DiT(
            out_channels=n_dims+in_node_nf+context_node_nf, x_scale=x_scale, 
            hidden_size=hidden_size, depth=depth, 
            num_heads=num_heads, mlp_ratio=mlp_ratio, 
            use_fused_attn=use_fused_attn
            )

        self.device = device

    def forward(self):
        raise NotImplementedError

    def _forward(self, t, xh, node_mask, edge_mask, context):
        # t: [bs, 1]
        # xh: [bs, n_nodes, dims]
        # node_mask: [bs, n_nodes, 1]
        # context: [bs, n_nodes, context_node_nf]
        # return [bs, n_nodes, dims]

        assert context is None

        bs, n_nodes, _ = xh.shape

        x = xh[:, :, :self.n_dims]
        h = xh[:, :, self.n_dims:]
        x = remove_mean_with_mask(x, node_mask)

        g = orthogonal_haar(dim=self.n_dims, target_tensor=x)  # [bs, 3, 3]
        g_inv_x = torch.bmm(x, g)  # as x is represented row-wise

        # [bs, n_nodes, hidden_size]
        gamma = node_mask * self.gamma_enc(
            g_inv_x, t.squeeze(-1), node_mask.squeeze(-1), 
            use_final_layer=False
            )

        # decoded summed representation into gamma - this is S_n-invariant
        N = node_mask.sum(1)  # [bs, 1]
        gamma = torch.sum(gamma, dim=1) / N  # [bs, hidden_size] 
        # [bs, 3, 3]
        gamma = qr(
            self.gamma_dec(gamma).reshape(-1, self.n_dims, self.n_dims)
            )[0]
        gamma = torch.bmm(gamma, g.transpose(2, 1))

        gamma_inv_x = torch.bmm(x, gamma)
        xh = torch.cat([gamma_inv_x, h], dim=-1)
        xh = self.k(xh, t.squeeze(-1), node_mask.squeeze(-1)) * node_mask  # use DiT

        x = xh[:, :, :self.n_dims] 
        h = xh[:, :, self.n_dims:]

        if self.args.com_free:
            x = remove_mean_with_mask(x, node_mask)  # k: U -> U

        x = torch.bmm(x, gamma.transpose(2, 1))
        xh = torch.cat([x, h], dim=-1)

        assert_correctly_masked(xh, node_mask)

        return xh


if __name__ == "__main__":

    # print parameter number

    class Args:
        def __init__(self) -> None:
            self.com_free = True

            # GNN
            self.nf = 256
            self.n_layers = 9

    args = Args()

    model = SymDiffPerceiver_dynamics(
        args,
        in_node_nf=6,
        context_node_nf=0,

        gamma_num_latents=64, 
        gamma_d_latents=128,
        gamma_n_pad=61,
        gamma_num_blocks=2, 
        gamma_num_self_attends_per_block=3, 
        gamma_num_self_attention_heads=4, 
        gamma_num_cross_attention_heads=4,
        gamma_attention_probs_dropout_prob=0,
        gamma_pos_num_channels=64,
        gamma_num_heads=4,

        k_num_latents=128,
        k_d_latents=256,
        k_n_pad=55,
        k_num_blocks=1,
        k_num_self_attends_per_block=10,
        k_num_self_attention_heads=4,
        k_num_cross_attention_heads=4,
        k_attention_probs_dropout_prob=0,
        k_pos_num_channels=64,
        k_num_heads=4,
        k_decoder_self_attention=True,
        k_num_self_heads=4,
        num_bands=32,
        max_resolution=100,
        concat_t=False,
        device="cpu"
    )

    """
    gamma params:  507264
    k params:  4554752
    total params:  5062016
    """
    print("symdiff_perceiver:")
    print("gamma params: ", sum(p.numel() for p in model.gamma.parameters() if p.requires_grad))
    print("k params: ", sum(p.numel() for p in model.k.parameters() if p.requires_grad))
    print("total params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(model._forward(torch.rand(2, 1), torch.randn(2, 5, 9), torch.ones(2, 5, 1), None, None).shape, "\n")

    model = SymDiffTransformer_dynamics(
        args,
        in_node_nf=6,
        context_node_nf=0, 

        gamma_num_enc_layers=2,
        gamma_num_dec_layers=2,
        gamma_d_model=128, 
        gamma_nhead=4,
        gamma_dim_feedforward=128, 
        gamma_dropout=0,

        k_num_layers=4,
        k_d_model=256, 
        k_nhead=8,
        k_dim_feedforward=256, 
        k_dropout=0,

        num_bands=64,
        max_resolution=100,
        t_fourier=True,
        concat_t=False,
    )

    print("symdiff_transformer:")
    print("gamma params: ", sum(p.numel() for p in model.gamma.parameters() if p.requires_grad))
    print("k params: ", sum(p.numel() for p in model.k.parameters() if p.requires_grad))
    print("total params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(model._forward(torch.rand(2, 1), torch.randn(2, 5, 9), torch.ones(2, 5, 1), None, None).shape, "\n")

    model = SymDiffPerceiverFourier_dynamics(
        args,
        in_node_nf=6,
        context_node_nf=0,

        gamma_num_latents=64, 
        gamma_d_latents=128,
        gamma_num_blocks=2, 
        gamma_num_self_attends_per_block=3, 
        gamma_num_self_attention_heads=4, 
        gamma_num_cross_attention_heads=4,
        gamma_attention_probs_dropout_prob=0,
        gamma_pos_num_channels=64,
        gamma_num_heads=4,

        k_num_latents=128,
        k_d_latents=256,
        k_num_blocks=1,
        k_num_self_attends_per_block=10,
        k_num_self_attention_heads=4,
        k_num_cross_attention_heads=4,
        k_attention_probs_dropout_prob=0,
        k_pos_num_channels=64,
        k_num_heads=4,
        k_decoder_self_attention=True,
        k_num_self_heads=4,

        sigma=100,
        m=20,
        device="cpu"
    )

    print("symdiff_perceiver_fourier")
    print("gamma params: ", sum(p.numel() for p in model.gamma.parameters() if p.requires_grad))
    print("k params: ", sum(p.numel() for p in model.k.parameters() if p.requires_grad))
    print("total params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(model._forward(torch.rand(2, 1), torch.randn(2, 5, 9), torch.ones(2, 5, 1), None, None).shape, "\n")

    model = DiT_dynamics(
        args,
        in_node_nf=6,
        context_node_nf=0,

        x_scale=25.0,
        hidden_size=32,
        depth=4,
        num_heads=2,
        mlp_ratio=2.0,
        use_fused_attn=True,
        subtract_x_0=False,
    )
    
    print("DiT")
    print("model params: ", sum(p.numel() for p in model.model.parameters() if p.requires_grad))