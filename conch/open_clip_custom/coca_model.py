from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass
import logging

from .transformer import MultimodalTransformer

from .vision_tower import VisualModel
from timm.models.vision_transformer import VisionTransformer
from .transformer import TextTransformer

try:
    from transformers import (
        LogitsProcessorList,
        TopPLogitsWarper,
        TopKLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        MinLengthLogitsProcessor,
        MaxLengthCriteria,
        StoppingCriteriaList
    )

    GENERATION_TYPES = {
        "top_k": TopKLogitsWarper,
        "top_p": TopPLogitsWarper
    }
    _has_transformers = True
except ImportError as e:
    GENERATION_TYPES = {
        "top_k": None,
        "top_p": None
    }
    _has_transformers = False

@dataclass
class CoCaVisionCfg:
    layers: int = 12
    width: int = 768
    num_heads: int = 12
    mlp_ratio: int = 4
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    attentional_pool_contrast: bool = False # perceiver resampler for contrastive loss
    attentional_pool_caption: bool = False # perceiver resampler for captioning
    n_queries_contrast: int = 1 # n_queries for contrastive loss
    n_queries_caption: int = 256 # n_queries for captioning
    attn_pooler_heads: int = 8 # n heads for attentional_pooling
    output_tokens: bool = False

@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False

def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CoCaVisionCfg,
        embed_dim_caption: Optional[int] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CoCaVisionCfg(**vision_cfg)

    trunk = VisionTransformer(embed_dim=vision_cfg.width, 
                              depth=vision_cfg.layers, 
                              num_heads=vision_cfg.num_heads, 
                              mlp_ratio=vision_cfg.mlp_ratio,
                              img_size=vision_cfg.image_size, 
                              patch_size=vision_cfg.patch_size,
                              num_classes=0,
                              dynamic_img_size=True)

    trunk_kwargs = {}
    trunk.forward = trunk.forward_features

    visual = VisualModel(
        trunk=trunk,
        trunk_kwargs=trunk_kwargs,
        use_attentional_pool_contrast=vision_cfg.attentional_pool_contrast,
        use_attentional_pool_caption=vision_cfg.attentional_pool_caption,
        n_queries_contrast=vision_cfg.n_queries_contrast,
        n_queries_caption=vision_cfg.n_queries_caption,
        output_tokens=vision_cfg.output_tokens,
        embed_dim_contrast=embed_dim,
        embed_dim_caption=embed_dim_caption,
        image_size=vision_cfg.image_size,
    )
    return visual

def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)
    act_layer = nn.GELU
    norm_layer = nn.LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        output_tokens=text_cfg.output_tokens,
        pad_id=text_cfg.pad_id,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return text


def _build_text_decoder_tower(
        embed_dim,
        multimodal_cfg
):
    multimodal_cfg = CLIPTextCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
    act_layer = nn.GELU
    norm_layer = nn.LayerNorm

    decoder = MultimodalTransformer(
        context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return decoder


class CoCa(nn.Module):
    def __init__(
            self,
            embed_dim,
            embed_dim_caption,
            multimodal_cfg: CLIPTextCfg,
            text_cfg: CLIPTextCfg,
            vision_cfg: CoCaVisionCfg,
            pad_id: int = 0,
    ):
        super().__init__()
        multimodal_cfg = CLIPTextCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CoCaVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        
        self.text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg
        )

        vocab_size = text_cfg.vocab_size

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            embed_dim_caption=embed_dim_caption,
            vision_cfg=vision_cfg
        )
        
        if multimodal_cfg.layers > 0:
            self.text_decoder = _build_text_decoder_tower(
                vocab_size,
                multimodal_cfg=multimodal_cfg
            )
        else:
            # no decoder
            self.text_decoder = None
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pad_id = pad_id
        self.context_length = text_cfg.context_length

        self.embed_dim = embed_dim
        self.embed_dim_caption = embed_dim_caption
    
    def lock_temperature(self):
        self.logit_scale.requires_grad = False

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    def _encode_image(self, images=None, normalize=True):
        image_latent, tokens_embs = self.visual(images)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return image_latent, tokens_embs

    def _encode_text(self, text, normalize=True, embed_cls=True):
        text = text[:, :-1] if embed_cls else text # make space for CLS token
        text_latent, token_emb = self.text(text)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, images, normalize=True, proj_contrast=True):
        if proj_contrast:
            image_latent, _ = self._encode_image(images, normalize=normalize)
        else:
            image_latent = self.visual.forward_no_head(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, normalize=True, embed_cls=True):
        text_latent, _ = self._encode_text(text, normalize=normalize, embed_cls=embed_cls)
        return text_latent

    def forward(self, image, text, embed_cls=True, image_latent=None, image_embs=None):
        text_latent, token_embs = self._encode_text(text, embed_cls=embed_cls)
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self._encode_image(image)

        labels = text[:, -token_embs.shape[1]:] 
        if self.text_decoder is not None:
            logits = self.text_decoder(image_embs, token_embs)
        else:
            logits = torch.empty(text.shape[0], 1, device=text.device)
        return {
            "image_features": image_latent,
            "text_features": text_latent,
            "logits": logits,
            "labels": labels,
            "logit_scale": self.logit_scale.exp()
        }
    
    def generate(
        self,
        image,
        text=None,
        seq_len=30,
        max_seq_len=77,
        temperature=1.,
        generation_type="beam_search",
        top_p=0.1,  # keep tokens in the 1 - top_p quantile
        top_k=1,  # keeps the top_k most probable tokens
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False # if True output.shape == (batch_size, seq_len)
    ):
        # taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert _has_transformers, "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"

        with torch.no_grad():
            sot_token_id = 1 if sot_token_id is None else sot_token_id
            eos_token_id = 2 if eos_token_id is None else eos_token_id
            pad_token_id = self.pad_id if pad_token_id is None else pad_token_id
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),
                ]
            )

            if stopping_criteria is None:
                stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]

            stopping_criteria = StoppingCriteriaList(
                stopping_criteria
            )

            device = image.device
            if generation_type == "top_p":
                logit_warper = GENERATION_TYPES[generation_type](top_p)
            elif generation_type == "top_k":
                logit_warper = GENERATION_TYPES[generation_type](top_k)
            else:
                raise ValueError(
                    f"generation_type has to be one of "
                    f"{'| ' + ' | '.join(list(GENERATION_TYPES.keys())) + ' |'}."
                )

            image_latent, image_embs = self._encode_image(image)

            if text is None:
                text = torch.ones((image.shape[0], 1), device=device, dtype=torch.long) * sot_token_id

            was_training = self.training
            num_dims = len(text.shape)

            if num_dims == 1:
                text = text[None, :]

            cur_len = text.shape[1]
            self.eval()
            out = text

            while True:
                x = out[:, -max_seq_len:]
                cur_len = x.shape[1]
                logits = self(image, x, image_latent=image_latent, image_embs=image_embs, embed_cls=False)["logits"][:, -1]
                mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)
                sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id

                if mask.all():
                    if not fixed_output_length:
                        break
                else:
                    logits = logits[~mask, :]
                    filtered_logits = logit_processor(x[~mask, :], logits)
                    filtered_logits = logit_warper(x[~mask, :], filtered_logits)
                    probs = F.softmax(filtered_logits / temperature, dim=-1)

                    if (cur_len + 1 == seq_len):
                        sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                    else:
                        sample[~mask, :] = torch.multinomial(probs, 1)

                out = torch.cat((out, sample), dim=-1)

                cur_len += 1

                if stopping_criteria(out, None):
                    break

            if num_dims == 1:
                out = out.squeeze(0)

            self.train(was_training)
            return out

def resize_pos_embed(state_dict, model):
    resized = False
    pos_embed_w = state_dict['visual.trunk.pos_embed']
    if pos_embed_w.shape != model.visual.trunk.pos_embed.shape:
        # see https://github.com/rwightman/pytorch-image-models/blob/624266148d8fa5ddb22a6f5e523a53aaf0e8a9eb/timm/models/vision_transformer.py#L509
        interpolation = 'bilinear'
        antialias = False

        from timm.layers import resample_abs_pos_embed
        num_prefix_tokens = 0 if getattr(model.visual.trunk, 'no_embed_class', False) else getattr(model.visual.trunk, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
                    pos_embed_w,
                    new_size=model.visual.trunk.patch_embed.grid_size,
                    num_prefix_tokens=num_prefix_tokens,
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        
        resized = True
    if not resized:
        logging.info('pos embedding not resized.')
    state_dict['visual.trunk.pos_embed'] = pos_embed_w


