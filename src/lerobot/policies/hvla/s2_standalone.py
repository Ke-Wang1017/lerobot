"""Standalone S2 process — keeps VLM hot between S1 restarts.

Usage:
    # Terminal 1: start S2 (stays running)
    python -m lerobot.policies.hvla.s2_standalone \
        --checkpoint ~/.cache/lerobot/converted/soarm-pi05-fast-11999-pytorch/model.safetensors \
        --task "assemble cylinder into ring"

    # Terminal 2: start S1 (connects to existing S2 shared memory)
    python -m lerobot.policies.hvla.launch \
        --s1-type flow --s1-checkpoint outputs/flow_s1_hvla_v7/checkpoint-50000/model.safetensors \
        --task "assemble cylinder into ring" --resize-images 224x224 \
        --attach-s2

    # With subtask decoding + keyboard injection:
    python -m lerobot.policies.hvla.s2_standalone \
        --checkpoint ~/.cache/lerobot/converted/soarm-pi05-fast-11999-pytorch/model.safetensors \
        --task "assemble cylinder into ring" \
        --decode-subtask --inject
    # Press 1-9 to inject captured subtask latent, SPACE to return to normal S2.
"""

import argparse
import contextlib
import logging
import signal
import threading
import time

from lerobot.policies.hvla.ipc import SharedImageBuffer, SharedLatentCache
from lerobot.policies.hvla.logging_utils import setup_process_logging
from lerobot.policies.hvla.s2_process import run_s2

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Standalone S2 VLM process (persistent)")
    parser.add_argument("--checkpoint", required=True, help="Path to Pi0.5 model.safetensors")
    parser.add_argument("--task", required=True, help="High-level task prompt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--s2-image-keys",
        nargs="+",
        default=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "base_1_rgb"],
    )
    parser.add_argument("--decode-subtask", action="store_true")
    parser.add_argument(
        "--inject",
        action="store_true",
        help="Enable keyboard injection of captured subtask latents (requires --decode-subtask)",
    )
    parser.add_argument("--norm-stats", default=None)
    parser.add_argument("--throttle-ms", type=int, default=100)
    args = parser.parse_args()

    setup_process_logging()

    s2_image_keys = tuple(args.s2_image_keys)

    # Create shared memory with well-known names (S1 will attach by name)
    logger.info(
        "Creating shared memory (latent: '%s', images: '%s*')...",
        SharedLatentCache.SHM_NAME,
        SharedImageBuffer.SHM_PREFIX,
    )
    shared_cache = SharedLatentCache(create=True, name=SharedLatentCache.SHM_NAME)
    shared_images = SharedImageBuffer(camera_keys=s2_image_keys, create=True)
    logger.info("Shared memory ready. S1 can connect with --attach-s2")
    logger.info("S2 will stay running until Ctrl+C.")

    stop_event = threading.Event()

    def signal_handler(sig, frame):
        logger.info("Ctrl+C — shutting down S2...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # --- Subtask injection via keyboard (when --inject) ---
    # Number keys (1-9) inject a captured subtask latent.
    # SPACE returns to normal S2 inference.
    # The injection overrides shared_cache.write() in a tight loop.
    inject_latent = None  # when set, injection loop overrides S2
    inject_lock = threading.Lock()
    inject_active = threading.Event()  # when set, run_s2 skips inference

    if args.inject:
        if not args.decode_subtask:
            logger.warning("--inject requires --decode-subtask, enabling it")
            args.decode_subtask = True
        try:
            from pynput import keyboard

            def on_press(key):
                nonlocal inject_latent
                try:
                    # Number keys 1-9: inject corresponding subtask
                    if hasattr(key, "char") and key.char and key.char.isdigit():
                        idx = int(key.char)
                        bank = getattr(run_s2, "subtask_bank", {})
                        order = getattr(run_s2, "subtask_order", [])
                        if 1 <= idx <= len(order):
                            label = order[idx - 1]
                            with inject_lock:
                                inject_latent = bank[label].clone()
                            inject_active.set()
                            logger.info(
                                '>>> INJECT [%d] "%s" (norm=%.1f)', idx, label, inject_latent.norm().item()
                            )
                        else:
                            if order:
                                logger.warning(
                                    "No subtask [%d] — available: %s",
                                    idx,
                                    " | ".join(f"[{i + 1}] {l}" for i, l in enumerate(order)),
                                )
                            else:
                                logger.warning("No subtasks captured yet — wait for S2 to detect some")

                    # SPACE: return to normal S2
                    elif key == keyboard.Key.space:
                        with inject_lock:
                            inject_latent = None
                        inject_active.clear()
                        logger.info(">>> NORMAL MODE (S2 VLM active)")

                except Exception as e:
                    logger.warning("Key handler error: %s", e)

            kb_listener = keyboard.Listener(on_press=on_press)
            kb_listener.start()
            logger.info("Keyboard injection enabled: press 1-9 to inject subtask, SPACE for normal")

        except ImportError:
            logger.warning("pynput not available — keyboard injection disabled")
            kb_listener = None
    else:
        kb_listener = None

    # --- Injection loop: runs in a thread, writes captured latent at 10Hz ---
    if args.inject:

        def injection_loop():
            while not stop_event.is_set():
                with inject_lock:
                    lat = inject_latent
                if lat is not None:
                    shared_cache.write(lat)
                time.sleep(0.1)

        inject_thread = threading.Thread(target=injection_loop, daemon=True)
        inject_thread.start()

    # --- Run S2 VLM ---
    try:
        run_s2(
            checkpoint_path=args.checkpoint,
            shared_cache=shared_cache,
            shared_images=shared_images,
            task=args.task,
            device=args.device,
            image_keys=s2_image_keys,
            decode_subtask=args.decode_subtask,
            norm_stats_path=args.norm_stats,
            stop_event=stop_event,
            throttle_ms=args.throttle_ms,
            inject_active=inject_active if args.inject else None,
        )
    except Exception:
        logger.exception("S2 standalone crashed")
    finally:
        stop_event.set()
        if kb_listener is not None:
            kb_listener.stop()
        logger.info("Cleaning up shared memory...")
        with contextlib.suppress(Exception):
            shared_cache.cleanup()
        with contextlib.suppress(Exception):
            shared_images.cleanup()
        logger.info("S2 standalone shutdown complete.")


if __name__ == "__main__":
    main()
