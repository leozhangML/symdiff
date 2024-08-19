from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import nn
import numpy as np

from transformers.configuration_utils import PretrainedConfig
from transformers.models.perceiver.modeling_perceiver import AbstractPreprocessor, PerceiverPreTrainedModel, PerceiverAttention, \
                                                             PerceiverEmbeddings, PerceiverEncoder, PerceiverModelOutput, PerceiverDecoderOutput, \
                                                             PerceiverLayer, PerceiverAbstractDecoder, build_position_encoding


"""NOTE: adapted from https://huggingface.co/docs/transformers/v4.44.0/en/model_doc/perceiver"""


ModalitySizeType = Mapping[str, int]
PreprocessorOutputType = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
PreprocessorType = Callable[..., PreprocessorOutputType]
PostprocessorType = Callable[..., Any]


def restructure(modality_sizes: ModalitySizeType, inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
    """
    Partitions a [B, N, C] tensor into tensors for each modality.

    Args:
        modality_sizes
            dict specifying the size of the modality
        inputs:
            input tensor

    Returns:
        dict mapping name of modality to its associated tensor.
    """
    outputs = {}
    index = 0
    # Apply a predictable ordering to the modalities
    for modality in sorted(modality_sizes.keys()):
        size = modality_sizes[modality]
        inp = inputs[:, index : index + size]
        index += size
        outputs[modality] = inp
    return outputs


def generate_fourier_features(
    t: torch.FloatTensor, 
    num_bands: int, 
    max_resolution: float = 100, 
    concat_t: bool = True, 
    sine_only: bool = False, 
    **kwargs
    ) -> torch.FloatTensor:

    """
    For time embedding
    t: [bs, 1]
    out: [bs, *]
    """

    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = torch.linspace(start=min_freq, end=max_resolution / 2, steps=num_bands)  # [num_bands]

    # Get frequency bands for each spatial dimension.
    per_t_features = t * freq_bands[None, :]
    if sine_only:
        per_t_features = torch.sin(np.pi * per_t_features)  # [bs, num_bands]
    else:
        per_t_features = torch.cat(
            [torch.sin(np.pi * per_t_features), torch.cos(np.pi * per_t_features)], dim=-1  # [bs, 2*num_bands]
        )
    # Concatenate the raw input positions.
    if concat_t:
        per_t_features = torch.cat([t, per_t_features], dim=-1)
    return per_t_features


def invert_attention_mask(attention_mask: torch.Tensor, mode="enc") -> torch.Tensor:
        """
        We extend the mask to be compatible with [bs, num_heads, query_len, key_value_len]
        attention_mask: [bs, n_nodes] of 0, 1

        Note that it doesn't matter that the attention probs for the padding positions
        will be uniform as these values are not used in loss
        """
        assert attention_mask.ndim == 2
        if mode == "enc":
            extended_attention_mask = attention_mask[:, None, None, :]
        elif mode == "dec":
            extended_attention_mask = attention_mask[:, None, :, None]
        elif mode == "self_pad":
            extended_attention_mask = torch.bmm(
                attention_mask.unsqueeze(-1), 
                attention_mask.unsqueeze(-2)
                ).unsqueeze(1)  # for decoding_self_attention
        else:
            ValueError
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(attention_mask.dtype).min

        return extended_attention_mask


class SymDiffPerceiverConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PerceiverModel`]. It is used to instantiate an
    Perceiver model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Perceiver
    [deepmind/language-perceiver](https://huggingface.co/deepmind/language-perceiver) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_latents (`int`, *optional*, defaults to 256):
            The number of latents.
        d_latents (`int`, *optional*, defaults to 1280):
            Dimension of the latent embeddings.
        d_model (`int`, *optional*, defaults to 768):
            Dimension of the inputs. Should only be provided in case [*PerceiverTextPreprocessor*] is used or no
            preprocessor is provided.
        num_blocks (`int`, *optional*, defaults to 1):
            Number of blocks in the Transformer encoder.
        num_self_attends_per_block (`int`, *optional*, defaults to 26):
            The number of self-attention layers per block.
        num_self_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each self-attention layer in the Transformer encoder.
        num_cross_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each cross-attention layer in the Transformer encoder.
        qk_channels (`int`, *optional*):
            Dimension to project the queries + keys before applying attention in the cross-attention and self-attention
            layers of the encoder. Will default to preserving the dimension of the queries if not specified.
        v_channels (`int`, *optional*):
            Dimension to project the values before applying attention in the cross-attention and self-attention layers
            of the encoder. Will default to preserving the dimension of the queries if not specified.
        cross_attention_shape_for_attention (`str`, *optional*, defaults to `'kv'`):
            Dimension to use when downsampling the queries and keys in the cross-attention layer of the encoder.
        self_attention_widening_factor (`int`, *optional*, defaults to 1):
            Dimension of the feed-forward layer in the cross-attention layer of the Transformer encoder.
        cross_attention_widening_factor (`int`, *optional*, defaults to 1):
            Dimension of the feed-forward layer in the self-attention layers of the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_query_residual (`float`, *optional*, defaults to `True`):
            Whether to add a query residual in the cross-attention layer of the encoder.
        vocab_size (`int`, *optional*, defaults to 262):
            Vocabulary size for the masked language modeling model.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that the masked language modeling model might ever be used with. Typically set
            this to something large just in case (e.g., 512 or 1024 or 2048).
        image_size (`int`, *optional*, defaults to 56):
            Size of the images after preprocessing, for [`PerceiverForImageClassificationLearned`].
        train_size (`List[int]`, *optional*, defaults to [368, 496]):
            Training size of the images for the optical flow model.
        num_frames (`int`, *optional*, defaults to 16):
            Number of video frames used for the multimodal autoencoding model.
        audio_samples_per_frame (`int`, *optional*, defaults to 1920):
            Number of audio samples per frame for the multimodal autoencoding model.
        samples_per_patch (`int`, *optional*, defaults to 16):
            Number of audio samples per patch when preprocessing the audio for the multimodal autoencoding model.
        output_num_channels (`int`, *optional*, defaults to 512):
            Number of output channels for each modalitiy decoder.
        output_shape (`List[int]`, *optional*, defaults to `[1, 16, 224, 224]`):
            Shape of the output (batch_size, num_frames, height, width) for the video decoder queries of the multimodal
            autoencoding model. This excludes the channel dimension.
    ```"""
    model_type = "symdiff_perceiver"

    def __init__(
        self,
        num_latents=256,
        d_latents=1280,
        d_model=768,
        n_pad=0,
        num_blocks=1,
        num_self_attends_per_block=26,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
        num_bands=4,
        concat_t=False,
        max_resolution=100,
        sine_only=False,
        vocab_size=262,
        max_position_embeddings=2048,
        image_size=56,
        train_size=[368, 496],
        num_frames=16,
        audio_samples_per_frame=1920,
        samples_per_patch=16,
        output_shape=[1, 16, 224, 224],
        output_num_channels=512,
        _label_trainable_num_channels=1024,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_latents = num_latents
        self.d_latents = d_latents
        self.d_model = d_model
        self.n_pad = n_pad  # when input dims is odd
        self.num_blocks = num_blocks
        self.num_self_attends_per_block = num_self_attends_per_block
        self.num_self_attention_heads = num_self_attention_heads
        self.num_cross_attention_heads = num_cross_attention_heads
        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.cross_attention_shape_for_attention = cross_attention_shape_for_attention
        self.self_attention_widening_factor = self_attention_widening_factor
        self.cross_attention_widening_factor = cross_attention_widening_factor
        self.hidden_act = hidden_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_query_residual = use_query_residual

        # for time embeddings for context and query

        self.num_bands = num_bands
        self.concat_t = concat_t
        self.max_resolution = max_resolution
        self.sine_only = sine_only

        # masked language modeling attributes
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        # image classification attributes
        self.image_size = image_size
        # flow attributes
        self.train_size = train_size
        # multimodal autoencoding attributes
        self.num_frames = num_frames
        self.audio_samples_per_frame = audio_samples_per_frame
        self.samples_per_patch = samples_per_patch
        self.output_shape = output_shape
        self.output_num_channels = output_num_channels
        self._label_trainable_num_channels = _label_trainable_num_channels


def concat_t_emb(config: SymDiffPerceiverConfig, inputs: torch.FloatTensor, t: torch.FloatTensor):
    bs, seq_len, _ = inputs.shape
    t_emb = generate_fourier_features(t, **config.__dict__)  # [bs, t_emb_dim]
    t_emb = torch.broadcast_to(
        t_emb.unsqueeze(1), 
        [bs, seq_len, t_emb.shape[-1]]
        )
    inputs = torch.cat([inputs, t_emb], dim=-1)
    return inputs


def t_emb_dim(config):
    sine_only = 1 if config.sine_only else 2
    concat_t = int(config.concat_t)
    num_bands = config.num_bands
    return sine_only * num_bands + concat_t


class TensorPreprocessor(AbstractPreprocessor):  # NOTE: preprocess context before this

    def __init__(self, dim: int, config: SymDiffPerceiverConfig) -> None:
        super().__init__()
        self.dim = dim  # dim of input tensor - i.e. xh: [bs, n_nodes, dim]
        self.config = config
        if config.n_pad > 0:
            self.pad = nn.Parameter(torch.randn(1, config.n_pad))

    @property
    def num_channels(self) -> int:
        t_dim = t_emb_dim(self.config)
        return self.dim + t_dim + self.config.n_pad

    def forward(self, inputs, t, attention_mask):  # we append t to inputs features here via ff and config - NOTE: expand should be fine
        assert(inputs.shape[-1] == self.dim)
        if self.config.n_pad > 0:
            bs, seq_len, _ = inputs.shape
            inputs = torch.cat([inputs, self.pad.expand(bs, seq_len, -1)], dim=-1)
        inputs = concat_t_emb(self.config, inputs, t)
        return inputs, None, attention_mask  # inputs, modality_sizes, enc_attention_mask


class ConditionalSymDiffPerceiverPreprocessor(AbstractPreprocessor):  # NOTE: might not be useful for now
    """
    Multimodal preprocessing for Perceiver Encoder for molecular conditioning

    Inputs for each modality are preprocessed, then padded with trainable position embeddings to have the same number
    of channels.

    Args:
        modalities (`Mapping[str, PreprocessorType]`):
            Dict mapping modality name to preprocessor.
        mask_probs (`Dict[str, float]`):
            Dict mapping modality name to masking probability of that modality.
        min_padding_size (`int`, *optional*, defaults to 2):
            The minimum padding size for all modalities. The final output will have num_channels equal to the maximum
            channels across all modalities plus min_padding_size.
    """

    def __init__(
        self,
        modalities: Mapping[str, PreprocessorType],  # use TensorPreprocessing
        mask_probs: Optional[Mapping[str, float]] = None,  # might be useful for conditioning
        min_padding_size: int = 2,
    ):
        super().__init__()
        self.modalities = nn.ModuleDict(modalities)
        self.min_padding_size = min_padding_size
        self.mask_probs = mask_probs if mask_probs is not None else {}
        self.padding = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.randn(1, self.num_channels - preprocessor.num_channels))
                for modality, preprocessor in modalities.items()
            }
        )
        self.mask = nn.ParameterDict(
            {modality: nn.Parameter(torch.randn(1, self.num_channels)) for modality, _ in self.mask_probs.items()}
        )

    @property
    def num_channels(self) -> int:
        max_channel_size = max(processor.num_channels for _, processor in self.modalities.items())
        common_channel_size = max_channel_size + self.min_padding_size
        return common_channel_size

    def forward(
        self, inputs: Mapping[str, torch.Tensor], t: torch.FloatTensor, attention_masks: Mapping[str, torch.FloatTensor]
    ) -> PreprocessorOutputType:
        padded = {}
        modality_sizes = {}
        enc_attention_masks = {}
        for modality, preprocessor in self.modalities.items():
            # preprocess each modality using the respective preprocessor.
            output, _, enc_attention_mask = preprocessor(
                inputs[modality], t, attention_masks[modality]
            )

            # pad to the same common_channel_size.
            batch_size, num_samples, num_channels = output.shape
            pos_enc = self.padding[modality].expand(batch_size, -1, -1)

            padding = torch.broadcast_to(
                pos_enc,
                [batch_size, num_samples, self.num_channels - num_channels],
            )
            output_padded = torch.cat([output, padding], dim=2)

            # mask if required
            if modality in self.mask_probs:
                mask_token = self.mask[modality].expand(batch_size, -1, -1)
                mask_prob = self.mask_probs[modality]
                mask = torch.bernoulli(torch.full([batch_size, num_samples], mask_prob))
                mask = torch.unsqueeze(mask, dim=2).to(mask_token.device)
                output_padded = (1 - mask) * output_padded + mask * mask_token

            padded[modality] = output_padded
            modality_sizes[modality] = output_padded.shape[1]
            enc_attention_masks[modality] = enc_attention_mask

        # Apply a predictable ordering to the modalities and concat along seq dim
        final_inputs = torch.cat(
            [padded[k] for k in sorted(padded.keys())],
            dim=1
        )
        enc_attention_masks = torch.cat(
            [enc_attention_masks[k] for k in sorted(padded.keys())],
            dim=1
        )

        return final_inputs, modality_sizes, enc_attention_masks


class SymDiffPerceiver(PerceiverPreTrainedModel):  # could use multi-modal decoder later
    def __init__(
        self,
        config,
        decoder=None,
        input_preprocessor: PreprocessorType = None,
        output_postprocessor: PostprocessorType = None,
    ):
        super().__init__(config)
        self.config = config

        self.input_preprocessor = input_preprocessor
        self.output_postprocessor = output_postprocessor
        self.embeddings = PerceiverEmbeddings(config)
        self.encoder = PerceiverEncoder(
            config, kv_dim=input_preprocessor.num_channels if input_preprocessor is not None else config.d_model  # q_dim of cross-att is d_latents
        )
        self.decoder = decoder

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.latents  # nn.Parameter

    def set_input_embeddings(self, value):
        self.embeddings.latents = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        inputs: torch.FloatTensor,
        t: torch.FloatTensor,
        enc_attention_mask: torch.FloatTensor,
        dec_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PerceiverModelOutput]:

        """Probably don't need subsampling or head mask"""

        assert enc_attention_mask.ndim == 2
        assert enc_attention_mask.shape == inputs.shape[:-1]  # assume shape [bs, n_nodes]
        assert self.input_preprocessor is not None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs, modality_sizes, enc_attention_mask = self.input_preprocessor(
            inputs, t, enc_attention_mask
            )

        batch_size = inputs.size()[0]

        # Make the attention mask broadcastable to [batch_size, num_heads, query_len, key_value_len]
        extended_enc_attention_mask = invert_attention_mask(enc_attention_mask, mode="enc")
        if dec_attention_mask is None:
            dec_attention_mask = enc_attention_mask

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_blocks x num_heads]
        # and head_mask is converted to shape [num_blocks x batch x num_heads x N x N]
        head_mask = self.get_head_mask(head_mask, self.config.num_blocks * self.config.num_self_attends_per_block)

        embedding_output = self.embeddings(batch_size=batch_size)  # expand to [bs, **latent_shape]

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=None,
            head_mask=head_mask,
            inputs=inputs,
            inputs_mask=extended_enc_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        logits = None
        if self.decoder:
            output_modality_sizes = modality_sizes
            decoder_query = self.decoder.decoder_query(
                inputs, t, modality_sizes=modality_sizes
            )
            decoder_outputs = self.decoder(
                decoder_query,
                z=sequence_output,
                query_mask=dec_attention_mask,  # input_mask and att_mask are the same for cross-att - NOTE: this is not extended
                output_attentions=output_attentions,
            )
            logits = decoder_outputs.logits

            # add cross-attentions of decoder
            if output_attentions and decoder_outputs.cross_attentions is not None:
                if return_dict:
                    encoder_outputs.cross_attentions = (
                        encoder_outputs.cross_attentions + decoder_outputs.cross_attentions
                    )
                else:
                    encoder_outputs = encoder_outputs + decoder_outputs.cross_attentions

            if self.output_postprocessor:
                logits = self.output_postprocessor(logits, modality_sizes=output_modality_sizes)

        if not return_dict:
            if logits is not None:
                return (logits, sequence_output) + encoder_outputs[1:]
            else:
                return (sequence_output,) + encoder_outputs[1:]

        return PerceiverModelOutput(
            logits=logits,
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class SymDiffPerceiverDecoder(PerceiverAbstractDecoder):
    """
    Cross-attention-based decoder. This class can be used to decode the final hidden states of the latents using a
    cross-attention operation, in which the latents produce keys and values.

    The shape of the output of this class depends on how one defines the output queries (also called decoder queries).

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        output_num_channels (`int`, *optional*):
            The number of channels in the output. Will only be used in case *final_project* is set to `True`.
        position_encoding_type (`str`, *optional*, defaults to "trainable"):
            The type of position encoding to use. Can be either "trainable", "fourier", or "none".
        output_index_dims (`int`, *optional*):
            The number of dimensions of the output queries. Ignored if 'position_encoding_type' == 'none'.
        num_channels (`int`, *optional*, defaults to 128):
            The number of channels of the decoder queries. Ignored if 'position_encoding_type' == 'none'.
        qk_channels (`int`, *optional*):
            The number of channels of the queries and keys in the cross-attention layer.
        v_channels (`int`, *optional*):
            The number of channels of the values in the cross-attention layer.
        num_heads (`int`, *optional*, defaults to 1):
            The number of attention heads in the cross-attention layer.
        widening_factor (`int`, *optional*, defaults to 1):
            The widening factor of the cross-attention layer.
        use_query_residual (`bool`, *optional*, defaults to `False`):
            Whether to use a residual connection between the query and the output of the cross-attention layer.
        concat_preprocessed_input (`bool`, *optional*, defaults to `False`):
            Whether to concatenate the preprocessed input to the query.
        final_project (`bool`, *optional*, defaults to `True`):
            Whether to project the output of the cross-attention layer to a target dimension.
        position_encoding_only (`bool`, *optional*, defaults to `False`):
            Whether to only use this class to define output queries.
    """

    def __init__(
        self,
        config: SymDiffPerceiverConfig,
        pos_num_channels: int,
        output_num_channels: int,  # used if final_project
        index_dims: int = -1,
        qk_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        num_heads: Optional[int] = 1,
        widening_factor: Optional[int] = 1,
        use_query_residual: Optional[bool] = False,
        final_project: Optional[bool] = True,
        decoder_self_attention: Optional[bool] = False
    ) -> None:
        super().__init__()

        self.config = config
        self.t_dim = t_emb_dim(self.config)
        self.pos_num_channels = pos_num_channels  # number of query channels before adding t_emb
        self.output_num_channels = output_num_channels  # final channel output of decoder if using final_layer
        self.index_dims = index_dims  # to output matrix or node positions

        if index_dims > 0:
            self.position_embeddings = nn.Parameter(torch.randn(index_dims, pos_num_channels))
        else:
            self.position_embeddings = nn.Parameter(torch.randn(1, pos_num_channels))

        self.final_project = final_project
        self.decoder_self_attention = decoder_self_attention

        self.decoding_cross_attention = PerceiverLayer(
            config,
            is_cross_attention=True,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=self.num_query_channels,
            kv_dim=config.d_latents,
            widening_factor=widening_factor,
            use_query_residual=use_query_residual,
            )
        self.decoding_self_attention = PerceiverLayer(
            config,
            is_cross_attention=False,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=self.num_query_channels,
            kv_dim=self.num_query_channels,
            widening_factor=widening_factor,
            use_query_residual=use_query_residual,
        ) if decoder_self_attention else nn.Identity()
        self.final_layer = nn.Linear(self.num_query_channels, output_num_channels) if final_project else nn.Identity()

    @property
    def num_query_channels(self) -> int:  # channel of final output
        #if self.final_project:
        #    return self.output_num_channels
        return self.pos_num_channels + self.t_dim

    def decoder_query(self, inputs, t, modality_sizes=None):
        bs, seq_len, _ = inputs.shape
        query_index_dims = self.index_dims if self.index_dims > 0 else seq_len

        # Construct the position encoding.
        pos_emb = self.position_embeddings.expand(bs, query_index_dims, -1)  # [bs, seq_len or 3, pos_num_channels]
        pos_emb = concat_t_emb(self.config, pos_emb, t)  # add t_emb to channels

        # Optionally project them to a target dimension. Should be id
        #pos_emb = self.positions_projection(pos_emb)
        return pos_emb

    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: Optional[torch.FloatTensor] = None,  # NOTE: this is not extended
        output_attentions: Optional[bool] = False,
    ) -> PerceiverDecoderOutput:
        # Cross-attention decoding.
        # key, value: B x N x K; query: B x M x K
        # Attention maps -> B x N x M
        # Output -> B x M x K
        cross_attentions = () if output_attentions else None

        extended_query_mask = invert_attention_mask(query_mask, mode="dec")

        layer_outputs = self.decoding_cross_attention(
            query,
            attention_mask=extended_query_mask,
            head_mask=None,
            inputs=z,
            inputs_mask=None,  # equal to att-mask due to cross-att
            output_attentions=output_attentions,
        )
        output = layer_outputs[0]

        if output_attentions:
            cross_attentions = cross_attentions + (layer_outputs[1],)

        if self.decoder_self_attention:  # if using a second decoder block
            extended_self_query_mask = invert_attention_mask(query_mask, mode="self_pad")
            layer_outputs = self.decoding_self_attention(
                query,
                attention_mask=extended_self_query_mask,
                head_mask=None,
                output_attentions=output_attentions,
            )
            output = layer_outputs[0]

            if output_attentions:
                cross_attentions = cross_attentions + (layer_outputs[1],)

        logits = self.final_layer(output)

        return PerceiverDecoderOutput(logits=logits, cross_attentions=cross_attentions)
