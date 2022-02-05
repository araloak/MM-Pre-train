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
CLS_TOKEN = 2
logger = logging.getLogger(__name__)


@register_model("diy-img-transformer")
class DIYImageTransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument('--use-img', default=False, action='store_true',
                            help='if set, use image features')
        parser.add_argument('--img-dim', type=int, metavar='N', default=1000,
                            help='image feature dimension')
        parser.add_argument('--strict', default=False, action='store_true',
                        help='if set, use image features')  
    def forward(self, src_tokens, src_imgs, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_imgs (FloatTensor): images features in the source sentences
                `(batch, img_num, dim)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_imgs=src_imgs, src_lengths=src_lengths, **kwargs)
        decoder_out,extra = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        
        
        
        
        

        return (decoder_out,extra)
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
        self.token_type_embedding = nn.Embedding(2, args.encoder_embed_dim) 
        self.special_token_embedding = nn.Embedding(2,args.encoder_embed_dim)
        self.resnet2transformer = nn.Linear(args.img_dim,args.encoder_embed_dim)


    def forward_embedding(
        self, src_tokens, src_imgs=None, token_embedding: Optional[torch.Tensor] = None
    ):
        
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding 
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)

        x += self.token_type_embedding(torch.ones_like(src_tokens))  
        




        img_emb = self.resnet2transformer(src_imgs)
        
        img_emb= torch.cat([self.special_token_embedding(torch.zeros([img_emb.size(0),1]).long().cuda()),img_emb],dim=1) 
        img_type_emb = self.token_type_embedding(torch.zeros_like(img_emb[:, :, 0]).long())  
        
        

        y = img_emb + img_type_emb
        

        if self.embed_positions is not None:
            img_pos_emb = self.embed_positions(torch.ones_like(img_emb[:, :, 0].long()))  
            y = y + img_pos_emb

        x = torch.cat([y,x], dim=1)
        
        

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, None

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

        
        x = x.transpose(0, 1)

        encoder_token_padding_mask = src_tokens.eq(self.padding_idx) 
        encoder_emotion_padding_mask = src_imgs[:,:,0].eq(0)
        encoder_cls_padding_mask = torch.ones([src_imgs.size(0),1]).eq(0).cuda()
        
        
        encoder_padding_mask = torch.cat([encoder_cls_padding_mask,encoder_emotion_padding_mask,encoder_token_padding_mask],dim=-1)
        encoder_states = [] if return_all_hiddens else None

        
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


@register_model_architecture('diy-img-transformer', 'baseline-diy-img-transformer')
def base_architecture(args):
    transformer_base_architecture(args)
