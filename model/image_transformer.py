# encoding: utf-8
from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
    EncoderOut,
    base_architecture as transformer_base_architecture
)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
logger = logging.getLogger(__name__)


@register_model("img-transformer")
class ImageTransformerModel(TransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument('--img-dim', type=int, metavar='N', default=1000,
                            help='image feature dimension')
        parser.add_argument('--use-img', default=False, action='store_true',
                            help='if set, use image features')
        parser.add_argument('--strict', default=False, action='store_true',
                        help='if set, use image features')  
        
    def forward(self, src_tokens, src_imgs, src_lengths, prev_output_tokens, **kwargs):

        encoder_out = self.encoder(src_tokens, src_imgs=src_imgs, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ImageTransformerEncoder(args, src_dict, embed_tokens)


class ImageTransformerEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        embed_dim = embed_tokens.embedding_dim
        self.img_dim = args.img_dim
        self.fuse_img_token = nn.Linear(embed_dim + self.img_dim, embed_dim) if args.use_img else None

        self.use_img = args.use_img

    def forward_embedding(
        self, src_tokens, src_imgs=None, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding #x torch.Size([96, 104, 512])
        # concat imgs features and reduce dimension
        if self.use_img:
            assert src_imgs is not None
            # [B, T, C'] self.dictionary.eos_index = 2
            token_img_idxs = torch.cumsum((src_tokens == self.dictionary.eos_index).long(), dim=1).unsqueeze(-1).expand([-1, -1, self.img_dim])
            # token_img_idxs torch.Size([96, 104, 1000])
            # [B, T, C']  f[b][t][c] = src_imgs[b][token_img_idxs[b][t][c]][c]
            token_img_features = torch.gather(src_imgs, 1, token_img_idxs)
            # token_img_features torch.Size([96, 104, 1000])
            # [B, T, C]
            x = self.fuse_img_token(torch.cat([x, token_img_features], dim=-1))
            # x torch.Size([96, 104, 512])

        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_imgs,
        src_lengths,
        cls_input=None,
        return_all_hiddens=False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_imgs (FloatTensor): images features in the source sentences
                `(batch, img_num, dim)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

        x, encoder_embedding = self.forward_embedding(src_tokens, src_imgs, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        encoder_padding_mask = src_tokens.eq(self.padding_idx) #encoder_padding_mask (ByteTensor): the positions of padding elements of shape (batch, src_len)
        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:

            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


@register_model_architecture('img-transformer', 'baseline-img-transformer')
def base_architecture(args):
    transformer_base_architecture(args)
