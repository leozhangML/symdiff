import torch
import numpy as np

from perceiver import EDMPerceiverConfig, EDMPerceiverDecoder


# kv_dim is given by dim of input_preprocessor
# first cross-att is q_dim=d_latents, kv_dim=input_preprocessor.num_channels, and projecting to d_latents
# then q_dim=d_latents, kv_dim=d_latents
config = EDMPerceiverConfig(  
    num_latents=4,
    d_latents=64,
    num_blocks=1,
    num_self_attends_per_block=2,
    num_self_attention_heads=1,
    num_cross_attention_heads=1,
    qk_channels=None,  # default is q_dim for self-att
    v_channels=None,  # default is qk_channel for self-att
    cross_attention_shape_for_attention="q",  # what to take for qk_channels - i.e. kv_dim or q_dim
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

decoder = EDMPerceiverDecoder(
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