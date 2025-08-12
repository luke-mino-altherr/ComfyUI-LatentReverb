import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any


class LatentReverb(nn.Module):
    """
    A neural reverb that operates in latent space using convolutions and attention
    to create spatial echo/reflection effects on encoded image features.
    """

    def __init__(
        self, channels: int = 4, num_reflections: int = 8, max_delay: int = 16
    ):
        super().__init__()
        self.channels = channels
        self.num_reflections = num_reflections
        self.max_delay = max_delay

        # Learnable reflection parameters
        self.reflection_weights = nn.Parameter(torch.randn(num_reflections) * 0.1)
        self.reflection_delays = nn.Parameter(
            torch.randint(1, max_delay, (num_reflections,)).float()
        )

        # Spatial processing layers
        self.spatial_conv = nn.Conv2d(channels, channels * 2, 3, padding=1)
        self.feedback_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.output_conv = nn.Conv2d(channels * 3, channels, 1)

        # Attention for spatial coherence
        # Ensure num_heads is compatible with channels
        num_heads = min(4, channels) if channels >= 4 else 1
        self.spatial_attention = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )

        # Dampening network
        self.dampen_net = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // 2), 1),
            nn.SiLU(),
            nn.Conv2d(max(1, channels // 2), channels, 1),
            nn.Sigmoid(),
        )

    def create_delay_line(self, x: torch.Tensor, delay: float) -> torch.Tensor:
        """Create delayed version using fractional delay interpolation"""
        b, c, h, w = x.shape

        # Integer and fractional parts
        delay_int = int(delay)
        delay_frac = delay - delay_int

        if delay_int == 0:
            return x

        # Spatial delay using convolution with learned kernels
        kernel_size = min(delay_int * 2 + 1, 7)
        padding = kernel_size // 2

        # Create Gaussian-like delay kernel
        kernel = torch.exp(-torch.linspace(-2, 2, kernel_size).pow(2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, 1).to(x.device)

        # Apply spatial delay
        delayed = F.conv2d(x.view(-1, 1, h, w), kernel, padding=(padding, 0), groups=1)
        delayed = delayed.view(b, c, h, w)

        return delayed

    def forward(
        self,
        x: torch.Tensor,
        wet_mix: float = 0.3,
        feedback: float = 0.4,
        room_size: float = 0.5,
    ) -> torch.Tensor:
        """
        Apply latent reverb effect

        Args:
            x: Input latent tensor [B, C, H, W]
            wet_mix: Dry/wet mix ratio (0=dry, 1=wet)
            feedback: Feedback amount for reflections
            room_size: Controls reflection pattern and delays
        """
        b, c, h, w = x.shape
        dry_signal = x.clone()

        # Scale delays by room size
        scaled_delays = self.reflection_delays * room_size

        # Initialize accumulator
        wet_signal = torch.zeros_like(x)
        feedback_buffer = x * 0.1

        # Process spatial features
        spatial_features = self.spatial_conv(x)

        # Create multiple reflections
        for i in range(self.num_reflections):
            delay = scaled_delays[i]
            weight = torch.sigmoid(self.reflection_weights[i]) * (
                0.8**i
            )  # Natural decay

            # Create delayed reflection
            reflection = self.create_delay_line(feedback_buffer, delay)

            # Apply spatial processing
            reflection = self.feedback_conv(reflection)

            # Add dampening (frequency-dependent decay)
            damping = self.dampen_net(reflection)
            reflection = reflection * damping

            # Accumulate weighted reflection
            wet_signal = wet_signal + reflection * weight

            # Update feedback buffer with some of the reflection
            feedback_buffer = feedback_buffer + reflection * feedback * 0.1

        # Apply spatial attention for coherence
        # Reshape for attention
        wet_flat = wet_signal.view(b, c, -1).transpose(1, 2)  # [B, HW, C]
        attended, _ = self.spatial_attention(wet_flat, wet_flat, wet_flat)
        wet_signal = attended.transpose(1, 2).view(b, c, h, w)

        # Combine with spatial features and output
        combined = torch.cat([spatial_features, wet_signal], dim=1)
        processed_wet = self.output_conv(combined)

        # Final dry/wet mix
        output = dry_signal * (1 - wet_mix) + processed_wet * wet_mix

        return output


class LatentReverbNode:
    """ComfyUI Custom Node for Latent Space Reverb"""

    @classmethod
    def INPUT_TYPES(cls):
        return (
            {
                "required": {
                    "samples": ("LATENT",),
                    "wet_mix": (
                        "FLOAT",
                        {
                            "default": 0.3,
                            "min": 0.0,
                            "max": 1.0,
                            "step": 0.01,
                            "display": "slider",
                        },
                    ),
                    "feedback": (
                        "FLOAT",
                        {
                            "default": 0.4,
                            "min": 0.0,
                            "max": 0.8,
                            "step": 0.01,
                            "display": "slider",
                        },
                    ),
                    "room_size": (
                        "FLOAT",
                        {
                            "default": 0.5,
                            "min": 0.1,
                            "max": 2.0,
                            "step": 0.1,
                            "display": "slider",
                        },
                    ),
                    "num_reflections": (
                        "INT",
                        {"default": 8, "min": 2, "max": 16, "step": 1},
                    ),
                }
            },
        )

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_reverb"
    CATEGORY = "latent/effects"

    def __init__(self):
        self.reverb_cache: Dict[str, LatentReverb] = {}

    def get_reverb_processor(
        self, channels: int, num_reflections: int, device: str
    ) -> LatentReverb:
        """Get or create reverb processor with caching"""
        cache_key = f"{channels}_{num_reflections}_{device}"

        if cache_key not in self.reverb_cache:
            reverb = LatentReverb(
                channels=channels, num_reflections=num_reflections, max_delay=16
            ).to(device)
            self.reverb_cache[cache_key] = reverb

        return self.reverb_cache[cache_key]

    def apply_reverb(
        self,
        samples: Dict[str, torch.Tensor],
        wet_mix: float,
        feedback: float,
        room_size: float,
        num_reflections: int,
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """Apply latent reverb effect"""

        latent = samples["samples"]
        device = latent.device
        channels = latent.shape[1]

        # Get reverb processor
        reverb = self.get_reverb_processor(channels, num_reflections, device)

        # Apply reverb
        with torch.no_grad():
            processed_latent = reverb(
                latent, wet_mix=wet_mix, feedback=feedback, room_size=room_size
            )

        # Return in ComfyUI latent format
        return ({"samples": processed_latent},)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LatentReverb": LatentReverbNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentReverb": "Latent Space Reverb",
}
