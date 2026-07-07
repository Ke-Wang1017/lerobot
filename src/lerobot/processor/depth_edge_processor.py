#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry

logger = logging.getLogger(__name__)


def _detect_depth_edges_sobel(
    depth_image: np.ndarray,
    threshold_percentile: int = 90,
    blur_kernel: int = 3,
    blur_sigma: float = 0.5,
    dilation_kernel: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect edges in depth image using Sobel gradients.

    Detects depth discontinuities by computing gradient magnitude and thresholding.
    Uses adaptive percentile-based threshold to handle varying depth ranges.

    Args:
        depth_image: HxW depth in meters (float32 or uint16)
        threshold_percentile: Percentile for adaptive thresholding (85-95)
                            Higher = fewer edges (only strong discontinuities)
        blur_kernel: Gaussian blur kernel size (3, 5, 7)
                    Reduces noise from IR shadows
        blur_sigma: Gaussian blur sigma (0.3-1.0)
        dilation_kernel: Morphological dilation kernel size (0-5)
                        0 = no dilation, larger = thicker edges

    Returns:
        edge_mask: HxW boolean array (True = edge, False = not edge)
        gradient_magnitude: HxW float32 array of gradient magnitudes
    """
    # Convert to float32 if needed
    if depth_image.dtype == np.uint16:
        depth_image = depth_image.astype(np.float32) * 0.001  # mm to meters

    # Identify invalid depth pixels
    invalid_mask = depth_image == 0

    # Apply Gaussian blur to reduce noise
    if blur_kernel > 0:
        depth_smooth = cv2.GaussianBlur(depth_image, (blur_kernel, blur_kernel), blur_sigma)
    else:
        depth_smooth = depth_image

    # Compute Sobel gradients
    grad_x = cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=3)  # Horizontal gradient
    grad_y = cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=3)  # Vertical gradient

    # Compute gradient magnitude: sqrt(grad_x² + grad_y²)
    # cv2.magnitude is a single C-level call that fuses the square+sum+sqrt
    # in one pass over the array. Replaces ``np.sqrt(grad_x**2 + grad_y**2)``
    # which allocated three temporary float32 arrays per frame.
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)

    # Adaptive thresholding using percentile
    # Only compute threshold on valid pixels
    valid_gradients = gradient_magnitude[~invalid_mask]

    if valid_gradients.size == 0:
        # Empty scene - no valid depth
        return np.zeros_like(depth_image, dtype=bool), gradient_magnitude

    # Compute threshold with minimum floor to handle homogeneous scenes.
    # ``np.percentile`` here was costing ~8 ms on 1280×720 depth because it
    # internally fully sorts the valid-gradient array. We get the same value
    # in ~0.5 ms by doing a quickselect on the two ranks bracketing the
    # requested percentile and linearly interpolating — which is exactly
    # what np.percentile does, just without the full sort.
    n_valid = valid_gradients.size
    idx_f = (n_valid - 1) * (threshold_percentile / 100.0)
    idx_lo = int(idx_f)
    idx_hi = min(idx_lo + 1, n_valid - 1)
    weight = idx_f - idx_lo
    partitioned = np.partition(valid_gradients, [idx_lo, idx_hi])
    pct = (1.0 - weight) * partitioned[idx_lo] + weight * partitioned[idx_hi]
    threshold = max(float(pct), 0.01)

    # Create binary edge mask
    edge_mask = gradient_magnitude > threshold

    # Mask out invalid regions
    edge_mask[invalid_mask] = False

    # Optional: Dilate edges for better visibility
    if dilation_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
        edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel) > 0

    return edge_mask, gradient_magnitude


def _create_depth_colored_edge_overlay(
    edge_mask: np.ndarray,
    depth_image: np.ndarray,
    rgb_image: np.ndarray,
    min_depth: float,
    max_depth: float,
    alpha: float,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay depth-colored edges on RGB image, filtering by depth range.

    Colors edges based on their depth value using a colormap, allowing
    distinction between near and far edges. Only edges within [min_depth, max_depth]
    are displayed - edges outside this range are filtered out.

    Args:
        edge_mask: HxW boolean array (True = edge, False = not edge)
        depth_image: HxW depth in meters (float32)
        rgb_image: HxWx3 BGR image (uint8)
        min_depth: Minimum depth in meters (edges below this are filtered out)
        max_depth: Maximum depth in meters (edges above this are filtered out)
        alpha: Overlay transparency (0.0-1.0)
        colormap: OpenCV colormap (default: COLORMAP_JET)
                 JET: blue (far) -> cyan -> green -> yellow -> red (near)

    Returns:
        overlay: HxWx3 BGR image with depth-colored edges (filtered by depth)
    """
    overlay = rgb_image.copy()

    if not edge_mask.any():
        return overlay

    # Filter edges by depth range
    # Strategy: Remove edges that are within OR adjacent to regions outside [min_depth, max_depth]
    # This ensures edges at boundaries with near/far objects are also filtered out

    # Create mask of valid depth range
    valid_depth_mask = (depth_image >= min_depth) & (depth_image <= max_depth)

    # Create mask of regions that are too near or too far
    invalid_depth_mask = ~valid_depth_mask

    # Dilate the invalid regions to include edge pixels adjacent to them
    # This catches edges at boundaries with near/far objects
    if invalid_depth_mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        invalid_depth_dilated = cv2.dilate(invalid_depth_mask.astype(np.uint8), kernel) > 0
    else:
        invalid_depth_dilated = invalid_depth_mask

    # Filter out edges that fall within dilated invalid regions
    filtered_edge_mask = edge_mask & ~invalid_depth_dilated

    if not filtered_edge_mask.any():
        return overlay

    # Get depth values at filtered edge pixels
    edge_depths = depth_image[filtered_edge_mask]

    # Normalize depth to [0, 255] based on specified range
    depth_normalized = (edge_depths - min_depth) / (max_depth - min_depth) * 255
    depth_normalized = np.clip(depth_normalized, 0, 255).astype(np.uint8)

    # Apply colormap to get colors for each edge pixel
    # Create a dummy image with normalized depths
    depth_colors_1d = cv2.applyColorMap(depth_normalized.reshape(-1, 1), colormap)
    edge_colors = depth_colors_1d.reshape(-1, 3)

    # Blend edge colors with RGB image
    overlay[filtered_edge_mask] = (
        overlay[filtered_edge_mask] * (1 - alpha) + edge_colors.astype(np.float32) * alpha
    ).astype(np.uint8)

    return overlay


@ProcessorStepRegistry.register("depth_edge_overlay_processor")
@dataclass
class DepthEdgeOverlayProcessorStep(ObservationProcessorStep):
    """
    Real-time depth edge detection and overlay for RealSense cameras.

    This processor detects depth discontinuities using Sobel gradients
    and overlays colored edges on the RGB image. Edges are colored by depth
    (red=near, blue=far) within a configurable depth range.

    The processor CONSUMES the depth frame: it takes both RGB and depth as input,
    produces RGB with overlaid edges as output, and removes the depth frame from
    observations (depth is only used for visualization, not stored in dataset).

    The processor is fail-safe: if the camera is not RealSense or depth is unavailable,
    it passes through the original RGB image unchanged.

    Attributes:
        camera_key: The camera key to process (e.g., 'top')
        threshold_percentile: Edge sensitivity (85-95, higher = fewer edges)
        blur_kernel: Noise reduction kernel size (1, 3, 5, 7)
        dilation_kernel: Edge thickness (0-5)
        alpha: Edge opacity (0.0-1.0)
        min_depth: Minimum depth in meters for edge filtering
        max_depth: Maximum depth in meters for edge filtering
    """

    camera_key: str = "top"
    threshold_percentile: int = 95
    blur_kernel: int = 3
    dilation_kernel: int = 2
    alpha: float = 0.7
    min_depth: float = 0.3
    max_depth: float = 0.9

    def observation(self, observation: dict) -> dict:
        """Process the specified camera feed with depth edge overlay and consume depth frame."""
        new_observation = dict(observation)

        # Check if both RGB and depth are available
        depth_key = f"{self.camera_key}_depth"
        if self.camera_key not in observation or depth_key not in observation:
            # Fail-safe: camera is not RealSense or depth not available
            logger.debug(
                f"DepthEdgeOverlayProcessor: {self.camera_key} or {depth_key} not in observation, "
                "passing through unchanged"
            )
            return new_observation

        rgb_frame = observation[self.camera_key]  # numpy array (H, W, 3) RGB uint8
        depth_frame = observation[depth_key]  # numpy array (H, W) uint16 in millimeters

        try:
            # Process with depth edge detection
            processed_frame = self.process_frame_with_depth(rgb_frame, depth_frame)
            new_observation[self.camera_key] = processed_frame
        except Exception as e:
            logger.warning(
                f"DepthEdgeOverlayProcessor failed for {self.camera_key}: {e}. Using original frame."
            )
            # Fail-safe: return original frame on error

        # Remove depth frame from observations after processing
        # This processor consumes depth: it uses it for edge detection but doesn't pass it downstream
        del new_observation[depth_key]

        return new_observation

    def process_frame_with_depth(self, rgb_frame: np.ndarray, depth_frame: np.ndarray) -> np.ndarray:
        """
        Apply depth edge detection and overlay on RGB frame.

        Args:
            rgb_frame: RGB image (H, W, 3) uint8
            depth_frame: Depth image (H, W) uint16 in millimeters

        Returns:
            Processed RGB image with depth edges overlaid
        """
        # Convert depth from uint16 (mm) to float32 (meters)
        depth_meters = depth_frame.astype(np.float32) * 0.001

        # Detect depth edges using Sobel gradients
        edge_mask, _ = _detect_depth_edges_sobel(
            depth_meters,
            threshold_percentile=self.threshold_percentile,
            blur_kernel=self.blur_kernel,
            blur_sigma=0.5,
            dilation_kernel=self.dilation_kernel,
        )

        # Convert RGB to BGR for OpenCV processing
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Create depth-colored edge overlay
        overlay_bgr = _create_depth_colored_edge_overlay(
            edge_mask,
            depth_meters,
            bgr_frame,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            alpha=self.alpha,
        )

        # Convert back to RGB
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        return overlay_rgb

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the step for serialization."""
        return {
            "camera_key": self.camera_key,
            "threshold_percentile": self.threshold_percentile,
            "blur_kernel": self.blur_kernel,
            "dilation_kernel": self.dilation_kernel,
            "alpha": self.alpha,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Image processing doesn't change feature structure."""
        return features
