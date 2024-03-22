import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .coca_model import CoCa, resize_pos_embed
from .transform import image_transform
from functools import partial
from huggingface_hub import hf_hub_download


CFG_DIR = Path(__file__).parent / 'model_configs'

def read_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict

def load_checkpoint(model, checkpoint_path):
    state_dict = read_state_dict(checkpoint_path)
    resize_pos_embed(state_dict, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

def create_model(
    model_cfg: str,
    checkpoint_path: Optional[str] = None,
    device: Union[str, torch.device] = 'cpu',
    jit: bool = False,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    cache_dir: Optional[str] = None,
    hf_auth_token: Optional[str] = None
):
    if not isinstance(model_cfg, dict):
        model_cfg_path = CFG_DIR / f'{model_cfg}.json'
        with open(model_cfg_path, 'r') as f:
            model_cfg = json.load(f)

    if isinstance(device, str):
        device = torch.device(device)

    if force_image_size is not None:
        # override model config's image size
        model_cfg["vision_cfg"]["image_size"] = force_image_size

    _ = model_cfg.pop('custom_text', None)
        
    model = CoCa(**model_cfg)

    if checkpoint_path.startswith("hf_hub:"): 
        _ = hf_hub_download(checkpoint_path[len("hf_hub:"):], 
                                          cache_dir=cache_dir, filename="meta.yaml",
                                          token=hf_auth_token)
        checkpoint_path = hf_hub_download(checkpoint_path[len("hf_hub:"):], 
                                          cache_dir=cache_dir, filename="pytorch_model.bin",
                                          token=hf_auth_token)
        
    load_checkpoint(model, checkpoint_path)

    model.to(device=device)

    # set image / mean metadata
    
    model.visual.image_mean = OPENAI_DATASET_MEAN
    model.visual.image_std = OPENAI_DATASET_STD

    if jit:
        model = torch.jit.script(model)

    return model

def create_model_from_pretrained(
        model_cfg: Union[str, Dict],
        checkpoint_path: Optional[str] = None,
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        return_transform: bool = True,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        cache_dir: Optional[str] = None,
        hf_auth_token: Optional[str] = None
):
    model = create_model(
        model_cfg,
        checkpoint_path=checkpoint_path,
        device=device,
        jit=jit,
        force_image_size=force_image_size,
        cache_dir=cache_dir,
        hf_auth_token=hf_auth_token
    )

    if not return_transform:
        return model

    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)

    preprocess = image_transform(
        model.visual.image_size,
        mean=image_mean,
        std=image_std,
    )

    return model, preprocess
