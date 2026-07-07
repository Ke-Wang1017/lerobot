"""S2 process: loads VLM, runs extraction loop, writes latent to shared memory.

Spawned by launch.py via 'spawn' context. Has its own CUDA context.
"""

import json
import logging
import math
import time

import numpy as np
import torch

from lerobot.policies.hvla.ipc import SharedImageBuffer, SharedLatentCache
from lerobot.policies.hvla.s2.config import S2VLMConfig
from lerobot.policies.hvla.s2.model import S2VLMModel
from lerobot.policies.hvla.s2.preprocessing import preprocess_images
from lerobot.policies.hvla.s2.tokenizer import PaligemmaTokenizer

logger = logging.getLogger(__name__)


def run_s2(
    checkpoint_path: str,
    shared_cache: SharedLatentCache,
    shared_images: SharedImageBuffer,
    task: str,
    config: S2VLMConfig | None = None,
    device: str = "cuda",
    image_keys: tuple[str, ...] = (),
    decode_subtask: bool = False,
    norm_stats_path: str | None = None,
    stop_event=None,
    throttle_ms: int = 100,
    inject_active=None,  # threading.Event — when set, skip inference (injection overrides)
):
    """S2 extraction loop entry point. Runs in a spawned process."""
    # Spawned process needs its own logging config
    from lerobot.policies.hvla.logging_utils import setup_process_logging

    setup_process_logging()

    if config is None:
        config = S2VLMConfig()
    if not image_keys:
        image_keys = config.image_keys if hasattr(config, "image_keys") else shared_images.camera_keys

    try:
        _run_s2_inner(
            checkpoint_path,
            shared_cache,
            shared_images,
            task,
            config,
            device,
            image_keys,
            decode_subtask,
            norm_stats_path,
            stop_event,
            throttle_ms=throttle_ms,
            inject_active=inject_active,
        )
    except Exception:
        logger.exception("S2 process crashed")


def _run_s2_inner(
    checkpoint_path,
    shared_cache,
    shared_images,
    task,
    config,
    device,
    image_keys,
    decode_subtask,
    norm_stats_path,
    stop_event,
    throttle_ms=100,
    inject_active=None,
):
    # Load model
    logger.info("S2: Loading VLM from %s...", checkpoint_path)
    model = S2VLMModel.from_pretrained(checkpoint_path, config)
    model.to(device)
    model.eval()
    logger.info("S2: VLM loaded on %s", device)

    # Tokenizer
    tokenizer = PaligemmaTokenizer(max_len=config.max_token_len)

    # State normalization
    norm_q01, norm_q99 = None, None
    if norm_stats_path:
        with open(norm_stats_path) as f:
            stats = json.load(f)
        if "state" in stats:
            norm_q01 = np.array(stats["state"]["q01"], dtype=np.float32)
            norm_q99 = np.array(stats["state"]["q99"], dtype=np.float32)
            logger.info("S2: Loaded norm stats from %s", norm_stats_path)

    def normalize_state(state: np.ndarray) -> np.ndarray:
        if norm_q01 is not None:
            denom = np.where(np.abs(norm_q99 - norm_q01) < 1e-6, 1.0, norm_q99 - norm_q01)
            return (state - norm_q01[: len(state)]) / denom[: len(state)] * 2.0 - 1.0
        return state

    logger.info("S2: Ready, entering extraction loop")

    query_count = 0
    last_log_time = time.time()
    prev_latent_norm = 0.0

    while stop_event is None or not stop_event.is_set():
        # Skip inference when injection is active (keyboard override)
        if inject_active is not None and inject_active.is_set():
            time.sleep(0.1)
            continue

        images = shared_images.read_images()
        if images is None:
            time.sleep(0.05)
            continue

        t0 = time.perf_counter()

        # State normalization
        state_raw = images.pop("_state", None)
        state_norm = None
        if state_raw is not None:
            state_norm = normalize_state(np.asarray(state_raw, dtype=np.float32))
            if len(state_norm) < 32:
                state_norm = np.pad(state_norm, (0, 32 - len(state_norm)))

        # Tokenize — when decoding subtask, use truncated prompt (stops at "Subtask: ")
        # so the model AR-decodes subtask text instead of FAST action tokens.
        # The latent from the truncated prompt is slightly different but functional.
        token_ids, token_mask = tokenizer.tokenize_prompt(
            task,
            low_prompt="",
            state=state_norm,
            subtask_only=decode_subtask,
        )
        lang_tokens = torch.from_numpy(token_ids).unsqueeze(0).long().to(device)
        lang_masks = torch.from_numpy(token_mask).unsqueeze(0).bool().to(device)

        # Preprocess images
        image_tensors, image_masks = preprocess_images(
            images,
            image_keys=image_keys,
            resolution=config.image_resolution,
            device=device,
        )

        if query_count == 0:
            logger.info(
                "S2: First query — %d images, running extract_prefix_latent (may take 30-60s)...",
                len(image_tensors),
            )

        # Extract latent
        confidence = 0.0
        if decode_subtask:
            latent, subtask_tokens, _, token_log_probs = model.extract_prefix_latent_and_subtask(
                image_tensors,
                image_masks,
                lang_tokens,
                lang_masks,
                temperature=config.subtask_temperature,
            )
            subtask_text = ""
            if subtask_tokens:
                try:
                    subtask_text = tokenizer.detokenize(np.array(subtask_tokens))
                except Exception:
                    subtask_text = f"<{len(subtask_tokens)} tokens>"
            # Compute confidence as geometric mean of per-token probs, only
            # over the subtask portion (before ";"). Action tokens after ";"
            # are irrelevant and would drag confidence to ~0.
            if subtask_text and token_log_probs:
                semi_pos = subtask_text.find(";")
                if semi_pos >= 0:
                    # Count tokens up to ";" by detokenizing incrementally
                    n_subtask_tokens = 0
                    partial = ""
                    for i, _ in enumerate(subtask_tokens):
                        partial = tokenizer.detokenize(np.array(subtask_tokens[: i + 1]))
                        n_subtask_tokens = i + 1
                        if ";" in partial:
                            break
                    # +1 for the first token (before loop)
                    n_conf = min(n_subtask_tokens + 1, len(token_log_probs))
                else:
                    n_conf = len(token_log_probs)
                lp = sum(token_log_probs[:n_conf])
                confidence = math.exp(lp / n_conf)
        else:
            latent = model.extract_prefix_latent(image_tensors, image_masks, lang_tokens, lang_masks)
            subtask_text = None

        # Write to shared memory (latent + subtask text + confidence if decoding)
        subtask_label_for_shm = None
        if decode_subtask and subtask_text:
            subtask_label_for_shm = subtask_text.split(";")[0].strip()
        shared_cache.write(latent[0], subtask=subtask_label_for_shm, confidence=confidence)
        query_count += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if query_count == 1:
            logger.info(
                "S2: First latent extracted (norm=%.1f, %.0fms)", latent.float().norm().item(), elapsed_ms
            )

        # Track unique subtask labels and their latest latents (for injection)
        latent_norm = latent[0].float().norm().item()
        if decode_subtask and subtask_text:
            subtask_label = subtask_text.split(";")[0].strip()
            prev_label = getattr(run_s2, "_prev_subtask_label", None)

            # Update the subtask latent bank (overwrite same label)
            if not hasattr(run_s2, "subtask_bank"):
                run_s2.subtask_bank = {}  # {label: latent_tensor}
                run_s2.subtask_order = []  # ordered list of labels
            if subtask_label not in run_s2.subtask_bank:
                run_s2.subtask_order.append(subtask_label)
                idx = len(run_s2.subtask_order)
                logger.info('S2: New subtask [%d] "%s" (norm=%.1f)', idx, subtask_label, latent_norm)
                # Show all available subtasks
                logger.info(
                    "S2: Available subtasks: %s",
                    " | ".join(f"[{i + 1}] {l}" for i, l in enumerate(run_s2.subtask_order)),
                )
            run_s2.subtask_bank[subtask_label] = latent[0].cpu().float().clone()

            if subtask_label != prev_label:
                idx = run_s2.subtask_order.index(subtask_label) + 1
                logger.info(
                    'S2 #%d: SUBTASK → [%d] "%s" (norm=%.1f, %.0fms)',
                    query_count,
                    idx,
                    subtask_label,
                    latent_norm,
                    elapsed_ms,
                )
            run_s2._prev_subtask_label = subtask_label

        # Periodic logging (every 10s)
        if time.time() - last_log_time >= 10.0:
            delta = abs(latent_norm - prev_latent_norm)
            subtask_str = f' | subtask="{subtask_text}"' if subtask_text else ""
            logger.info(
                "S2 #%d: %.0fms | norm=%.1f | Δnorm=%.2f%s",
                query_count,
                elapsed_ms,
                latent_norm,
                delta,
                subtask_str,
            )
            prev_latent_norm = latent_norm
            last_log_time = time.time()

        # Yield GPU time to S1 — sleep between queries to reduce contention.
        # S2 at 130ms/query + 100ms sleep = ~4Hz (sufficient for scene understanding).
        # Without this, S1 inference doubles from 29ms to 80ms due to GPU contention.
        if throttle_ms > 0:
            time.sleep(throttle_ms / 1000.0)

    logger.info("S2: Shutdown after %d queries", query_count)
