# encoding: utf-8
"""
@author: Yuxian Meng, Zhenghao Wu
@contact: yuxian_meng@shannonai.com; zwubq@connect.ust.hk

@version: 1.0
@file: transformer_encoder
@time: 2020/11/18 11:35
@desc: Transformer encoder with src-tokens and img-features as inputs

"""

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
from fairseq import utils
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.bart.hub_interface import BARTHubInterface

# DEFAULT_MAX_SOURCE_POSITIONS = 1024
# DEFAULT_MAX_TARGET_POSITIONS = 1024

logger = logging.getLogger(__name__)

CLS_TOKEN = 2
@register_model("img-seq-cls-bart")
class ImageBARTSeqClsModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.classification_layer = IMGClassificationHead(768,args.num_classes)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()
        if hasattr(self.encoder, "dictionary"):
            self.eos: int = self.encoder.dictionary.eos()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        # super(ImageBARTModel, ImageBARTModel).add_args(parser)  # TODO
        #parser.add_argument('--img-dim', type=int, metavar='N', default=1000,help='image feature dimension')
        parser.add_argument('--use-img', default=False, action='store_true',
                            help='if set, use image features')
        parser.add_argument('--strict', default=False, action='store_true',
                            help='if set, use image features')
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            help="Apply spectral normalization on the classification head",
        )

    @property
    def supported_targets(self):
        return {"self"}

    def forward(self,
                src_tokens,
                src_imgs,
                src_lengths,
                prev_output_tokens,
                features_only: bool = False,
                classification_head_name: Optional[str] = None,
                token_embeddings: Optional[torch.Tensor] = None,
                return_all_hiddens: bool = True,
                alignment_layer: Optional[int] = None,
                alignment_heads: Optional[int] = None,
                **kwargs):
        if classification_head_name is not None:
            features_only = True

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
        encoder_out = self.encoder(src_tokens,
                                   src_imgs=src_imgs,
                                   src_lengths=src_lengths,
                                   token_embeddings=token_embeddings,
                                   return_all_hiddens=return_all_hiddens,
                                   **kwargs)
        img_representation = encoder_out.encoder_out # todo : extract CLS vevtor from encoder_out
        #print(img_representation.shape)
        # img_representation.shape == [seq_len(2),batch_size,dim(512)]
        x = self.classification_layer(
            img_representation[1:,:,:] # extract CLS token for classification
        )

        return x

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ImageBARTSeqClsEncoder(args, src_dict, embed_tokens)

    @classmethod
    def from_pretrained(
            cls,
            model_name_or_path,
            checkpoint_file="model.pt",
            data_name_or_path=".",
            bpe="gpt2",
            sample_break_mode="eos",
            **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            sample_break_mode=sample_break_mode,
            **kwargs,
        )
        return BARTHubInterface(x["args"], x["task"], x["models"][0])

    def register_classification_head(
            self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = BARTClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            do_spectral_norm=getattr(
                self.args, "spectral_norm_classification_head", False
            ),
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads."):].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
                ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
                ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                        num_classes
                        != self.classification_heads[head_name].out_proj.out_features
                        or inner_dim
                        != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
        if (
                loaded_dict_size == len(self.encoder.dictionary) + 1
                and "<mask>" not in self.encoder.dictionary
        ):
            truncate_emb("encoder.embed_tokens.weight")
            truncate_emb("decoder.embed_tokens.weight")
            truncate_emb("encoder.output_projection.weight")
            truncate_emb("decoder.output_projection.weight")

        # When continued pretraining on new set of languages for mbart,
        # add extra lang embeddings at the end of embed_tokens.
        # Note: newly added languages are assumed to have been added at the end.
        if self.args.task == "multilingual_denoising" and loaded_dict_size < len(
                self.encoder.dictionary
        ):
            logger.info(
                "Adding extra language embeddings not found in pretrained model for "
                "continued pretraining of MBART on new set of languages."
            )
            loaded_mask_token_embedding = state_dict["encoder.embed_tokens.weight"][
                                          -1, :
                                          ]

            num_langids_to_add = len(self.encoder.dictionary) - loaded_dict_size
            embed_dim = state_dict["encoder.embed_tokens.weight"].size(1)

            new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
            nn.init.normal_(new_lang_embed_to_add, mean=0, std=embed_dim ** -0.5)
            new_lang_embed_to_add = new_lang_embed_to_add.to(
                dtype=state_dict["encoder.embed_tokens.weight"].dtype,
            )

            state_dict["encoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["encoder.embed_tokens.weight"][
                    : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )
            state_dict["decoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["decoder.embed_tokens.weight"][
                    : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


class ImageBARTSeqClsEncoder(TransformerEncoder):
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

        self.img_dim = args.img_dim
        self.token_type_embedding = nn.Embedding(2, args.encoder_embed_dim) # 0 stands for image, 1 stands for text, 2 stands for [CLS] token
        self.special_token_embedding = nn.Embedding(2,args.encoder_embed_dim)
        self.resnet2transformer = nn.Linear(args.img_dim,args.encoder_embed_dim)

    def forward_embedding(
        self, src_tokens, src_imgs=None, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        # if token_embedding is None:
        #     token_embedding = self.embed_tokens(src_tokens)
        # x = embed = self.embed_scale * token_embedding #x torch.Size([96, 104, 512])
        # if self.embed_positions is not None:
        #     x = embed + self.embed_positions(src_tokens)
        #
        # x += self.token_type_embedding(torch.ones_like(src_tokens))  #segment embedding of token
        #print(" before cat x", x.shape)

        img_emb = self.resnet2transformer(src_imgs)
        cls_embedding = self.special_token_embedding(torch.zeros([img_emb.size(0),1]).long().cuda())

        img_emb= torch.cat([cls_embedding,img_emb],dim=1) # add [CLS] token at the begining of img squences

        img_type_emb = self.token_type_embedding(torch.zeros_like(img_emb[:, :, 0]).long())  # segment embedding of obj
        #print("img_type_emb",img_type_emb.shape)
        #print("img_emb",img_emb.shape)

        y = img_emb + img_type_emb
        #print("y",y.shape)

        if self.embed_positions is not None:# todo:rethink about this position embedding
            img_pos_emb = self.embed_positions(torch.ones_like(img_emb[:, :, 0].long()))  
            y = y + img_pos_emb

        #x = torch.cat([ y,x], dim=1)
        #print(" after cat x",x.shape)
        # encoder_padding_mask for images

        if self.layernorm_embedding is not None:
            y = self.layernorm_embedding(y)
        y = self.dropout_module(y)
        if self.quant_noise is not None:
            y = self.quant_noise(y)
        return y, None

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

        encoder_emotion_padding_mask = src_imgs[:,:,0].eq(0) 
        encoder_cls_padding_mask = torch.ones([src_imgs.size(0),1]).eq(0).cuda()
        encoder_padding_mask = torch.cat([encoder_cls_padding_mask,encoder_emotion_padding_mask],dim=-1)
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
class IMGClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(
        self,
        input_dim,
        num_classes,
        do_spectral_norm=False,
    ):
        super().__init__()

        self.out_proj = nn.Linear(input_dim, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        if do_spectral_norm:
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, x, **kwargs):
        x = self.out_proj(x)
        x = self.logsoftmax(x)
        return x


class BARTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

        if do_spectral_norm:
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture('img-seq-cls-bart', 'baseline-img-seq-cls-bart')
def base_architecture(args):
    # transformer_base_architecture(args)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)  # 512
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)  # 2048
    args.encoder_layers = getattr(args, "encoder_layers", 12)  # 6
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)  # 8
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)  # 512
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )  # 2048
    args.decoder_layers = getattr(args, "decoder_layers", 12)  # 6
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)  # 8
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )  # False
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)  # False

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)


@register_model_architecture("img-seq-cls-bart", "baseline-img-seq-cls-bart-base")
def bart_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    base_architecture(args)


@register_model_architecture("img-seq-cls-bart", "baseline-img-seq-cls-bart-trimed")
def bart_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    base_architecture(args)

