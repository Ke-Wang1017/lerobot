"""S2 image and prompt preprocessing.

Handles camera images → SigLIP format and task text → tokenized prompt.
Shared between S2 inference and S2 training.
"""

import logging

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import Tensor

logger = logging.getLogger(__name__)

# Default camera keys matching SO107 bimanual setup
DEFAULT_IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
    "base_1_rgb",
)

IMAGE_RESOLUTION = (224, 224)


def preprocess_images(
    images: dict[str, np.ndarray | Tensor],
    image_keys: tuple[str, ...] = DEFAULT_IMAGE_KEYS,
    resolution: tuple[int, int] = IMAGE_RESOLUTION,
    device: torch.device | str = "cpu",
) -> tuple[list[Tensor], list[Tensor]]:
    """Preprocess camera images for S2 VLM.

    Accepts numpy uint8 HWC or torch tensors. Resizes to resolution,
    normalizes to [-1, 1], returns list of [1, C, H, W] tensors + masks.

    Args:
        images: Dict mapping camera key → image (HWC uint8 numpy or CHW tensor).
        image_keys: Which cameras to use and in what order.
        resolution: Target (H, W) for SigLIP.
        device: Target device for output tensors.

    Returns:
        (image_tensors, image_masks) — lists aligned with image_keys.
    """
    out_images = []
    out_masks = []

    for key in image_keys:
        if key not in images:
            # Missing camera → zero tensor with mask=False
            h, w = resolution
            out_images.append(torch.zeros(1, 3, h, w, dtype=torch.float32, device=device))
            out_masks.append(torch.zeros(1, dtype=torch.bool, device=device))
            continue

        img = images[key]

        # Convert numpy to tensor if needed
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(np.ascontiguousarray(img))
            if img.ndim == 3 and img.shape[2] == 3:
                img = img.permute(2, 0, 1)  # HWC → CHW

        # Ensure CHW format
        if img.ndim == 3 and img.shape[0] != 3 and img.shape[2] == 3:
            img = img.permute(2, 0, 1)

        # Resize on CPU (small tensor → fast GPU transfer)
        if img.shape[1:] != resolution:
            img = TF.resize(
                img, list(resolution), interpolation=TF.InterpolationMode.BILINEAR, antialias=True
            )

        # Normalize: uint8 [0,255] → float [-1, 1]
        if img.dtype == torch.uint8:
            img = img.float().div_(127.5).sub_(1.0)
        elif img.dtype in (torch.float32, torch.float16, torch.bfloat16):
            # Already float — assume [0, 1] range, convert to [-1, 1]
            if img.max() <= 1.0:
                img = img * 2.0 - 1.0

        out_images.append(img.unsqueeze(0).to(device))
        out_masks.append(torch.ones(1, dtype=torch.bool, device=device))

    return out_images, out_masks


def prepare_prompt_tokens(
    task: str,
    tokenizer,
    max_length: int = 256,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Tokenize task prompt for PaliGemma.

    Returns: (token_ids [1, seq_len], attention_mask [1, seq_len])
    """
    encoded = tokenizer(
        task,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return (
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device).bool(),
    )
