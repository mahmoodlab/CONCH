import logging
import pdb
from collections import OrderedDict

import torch
import torch.nn as nn

from .transformer import AttentionalPooler
from timm.models.layers import Mlp, to_2tuple

from .utils import freeze_batch_norm_2d

class VisualModel(nn.Module):
    def __init__(
            self,
            embed_dim_contrast,
            embed_dim_caption,
            trunk,
            image_size=224,
            proj='',
            proj_bias=False,
            drop=0.,
            global_average_pool=False,
            use_attentional_pool_contrast=False,
            use_attentional_pool_caption=False,
            n_queries_contrast=1,
            n_queries_caption=256,
            attn_pooler_heads=8,
            norm_layer=nn.LayerNorm,
            output_tokens=False,
            trunk_kwargs={}
    ):
        super().__init__()

        self.trunk = trunk
        self.trunk_kwargs = trunk_kwargs
        self.image_size = to_2tuple(image_size)
        prev_chs = self.trunk.num_features
        head_layers = OrderedDict()
        
        # whether to use attentional pooling
        self.use_attentional_pool_contrast = use_attentional_pool_contrast
        self.use_attentional_pool_caption = use_attentional_pool_caption
        self.global_average_pool = global_average_pool
        self.output_tokens = output_tokens
        if use_attentional_pool_contrast:
            scale = prev_chs ** -0.5
            self.attn_pool_contrast = AttentionalPooler(d_model=embed_dim_contrast, context_dim=prev_chs, n_head=attn_pooler_heads, n_queries=n_queries_contrast)
            self.ln_contrast = norm_layer(embed_dim_contrast)
            self.proj_contrast = nn.Parameter(scale * torch.randn(embed_dim_contrast, embed_dim_contrast))
        else:
            assert proj, 'projection layer needed if not using attentional pooling.'
            # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
            if proj == 'linear':
                head_layers['drop'] = nn.Dropout(drop)
                head_layers['proj'] = nn.Linear(prev_chs, embed_dim_contrast, bias=proj_bias)
            elif proj == 'mlp':
                head_layers['mlp'] = Mlp(prev_chs, 2 * embed_dim_contrast, embed_dim_contrast, drop=(drop, 0), bias=(True, proj_bias))

        self.head = nn.Sequential(head_layers)

        if use_attentional_pool_caption:
            self.attn_pool_caption = AttentionalPooler(d_model=embed_dim_caption, context_dim=prev_chs, n_head=attn_pooler_heads, n_queries=n_queries_caption)
            self.ln_caption = norm_layer(embed_dim_caption)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """ lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            # lock full model
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            from timm.models.helpers import group_parameters, group_modules
            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        try:
            self.trunk.set_grad_checkpointing(enable)
        except Exception as e:
            logging.warning('grad checkpointing not supported for this timm image tower, continuing without...')

    def _global_pool(self, x):
        if self.global_average_pool:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]

    def forward_project(self, x):
        if self.use_attentional_pool_contrast:
            x = x @ self.proj_contrast
            return x
        else:
            x = self.head(x)
            return x
        
    def forward_attn_pool_caption(self, tokens, attn_mask=None):
        if self.use_attentional_pool_caption:
            tokens = self.attn_pool_caption(tokens, attn_mask=attn_mask)
            tokens = self.ln_caption(tokens)
            return tokens
        else:
            raise NotImplementedError
        
    def forward_no_head(self, x, normalize=False):
        x = self.trunk(x, **self.trunk_kwargs)
        if self.use_attentional_pool_contrast:
            pooled = self.attn_pool_contrast(x)[:, 0]
            pooled = self.ln_contrast(pooled)
        else:
            pooled, _ = self._global_pool(x)
        if normalize:
            pooled = nn.functional.normalize(pooled, dim=-1)
        return pooled

    def forward(self, x):
        x = self.trunk(x, **self.trunk_kwargs)
        tokens = None
        if self.use_attentional_pool_contrast:
            pooled = self.attn_pool_contrast(x)[:, 0] # single query
            pooled = self.ln_contrast(pooled)
            pooled = pooled @ self.proj_contrast
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.head(x)

        if self.use_attentional_pool_caption:
            tokens = self.attn_pool_caption(x)
            tokens = self.ln_caption(tokens)
        else:
            tokens = None
        
        if self.output_tokens:
            return pooled, tokens
        else:
            return pooled

        
