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

from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("top_camera_processor")
@dataclass
class TopCameraProcessorStep(ObservationProcessorStep):
    """Real-time image processing for the top camera.

    Attributes:
        camera_key: The camera key to process (default: 'top')
        brightness_factor: Multiplier for brightness adjustment (default: 1.0)
    """

    camera_key: str = "top"
    brightness_factor: float = 1.0

    def observation(self, observation: dict) -> dict:
        """Process the specified camera feed in real-time."""
        new_observation = dict(observation)

        if self.camera_key in new_observation:
            frame = new_observation[self.camera_key]  # numpy array (H, W, 3) in RGB uint8

            # Your custom processing here
            processed_frame = self.process_frame(frame)

            new_observation[self.camera_key] = processed_frame

        return new_observation

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Your custom image processing logic.

        Args:
            frame: Input image as numpy array (H, W, 3) in RGB format, uint8

        Returns:
            Processed image as numpy array (H, W, 3) in RGB format, uint8
        """
        # Example: Brightness adjustment
        if self.brightness_factor != 1.0:
            processed = np.clip(frame * self.brightness_factor, 0, 255).astype(np.uint8)
            return processed

        # Add your custom processing here:
        # - Edge detection
        # - Object detection overlays
        # - Color filtering
        # - Segmentation
        # - etc.

        return frame

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the step for serialization."""
        return {
            "camera_key": self.camera_key,
            "brightness_factor": self.brightness_factor,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Image processing doesn't change feature structure."""
        return features
