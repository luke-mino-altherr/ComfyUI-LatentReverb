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
        self,
        channels: int = 4,
        num_reflections: int = 16,
        max_delay: int = 24,
        decay_rate: float = 0.8,  # Single decay control
    ):
        super().__init__()
        self.channels = channels
        self.num_reflections = num_reflections
        self.max_delay = max_delay
        self.decay_rate = decay_rate

        # Learnable reflection parameters
        self.reflection_weights = nn.Parameter(torch.randn(num_reflections) * 0.1)
        self.reflection_delays = nn.Parameter(
            torch.randint(1, max_delay, (num_reflections,)).float()
        )

        # Simple decay - each reflection decays by decay_rate
        # No complex parameter arrays needed

        # Add diffusion parameters for more realistic reverb
        self.diffusion_strength = nn.Parameter(torch.randn(num_reflections) * 0.2)

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
        """Create delayed version using actual spatial shifting instead of convolution"""
        b, c, h, w = x.shape

        # Convert delay parameter to meaningful pixel shifts
        # Make the shifts more dramatic and visible
        x_shift = int(delay * 2)  # Increased from 4 to 2 for more visible effect
        y_shift = int(delay * 1)  # Increased from 2 to 1 for more visible effect

        if x_shift == 0 and y_shift == 0:
            return x

        # Use torch.roll to actually shift the tensor spatially
        shifted = torch.roll(x, shifts=(y_shift, x_shift), dims=(2, 3))

        # Create fade mask for shifted regions to make the effect more natural
        # This creates a gradual fade at the edges where the roll wraps around
        fade_width = min(
            4, min(h, w) // 8
        )  # Increased fade width for more natural effect

        if fade_width > 0:
            # Create fade mask for height (y-axis)
            if y_shift != 0:
                y_fade = torch.ones(h, device=x.device)
                y_fade[:fade_width] = torch.linspace(
                    0.6, 1.0, fade_width, device=x.device
                )  # More aggressive fade for stronger effect
                y_fade[-fade_width:] = torch.linspace(
                    1.0, 0.6, fade_width, device=x.device
                )
                y_fade = y_fade.view(1, 1, h, 1)
                shifted = shifted * y_fade

            # Create fade mask for width (x-axis)
            if x_shift != 0:
                x_fade = torch.ones(w, device=x.device)
                x_fade[:fade_width] = torch.linspace(
                    0.6, 1.0, fade_width, device=x.device
                )  # More aggressive fade for stronger effect
                x_fade[-fade_width:] = torch.linspace(
                    1.0, 0.6, fade_width, device=x.device
                )
                x_fade = x_fade.view(1, 1, 1, w)
                shifted = shifted * x_fade

        return shifted

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
            feedback: Feedback amount for reflections (0=no feedback, 1=strong feedback)
            room_size: Controls reflection pattern and delays (0.1=small room, 2.0=large room)
        """
        b, c, h, w = x.shape
        dry_signal = x.clone()

        # Scale delays by room size and normalize by number of reflections
        # This ensures consistent visual effect regardless of reflection count
        reflection_scale_factor = (
            8.0 / self.num_reflections
        )  # Normalize by reflection count
        scaled_delays = (
            self.reflection_delays * room_size * 12.0 * reflection_scale_factor
        )

        # Initialize accumulator
        wet_signal = torch.zeros_like(x)
        feedback_buffer = x * 0.1

        # Process spatial features
        spatial_features = self.spatial_conv(x)

        # Create multiple reflections with simple decay
        for i in range(self.num_reflections):
            delay = scaled_delays[i]

            # Simple exponential decay - each reflection gets weaker
            decay_rate = self.decay_rate**i

            # Add diffusion for more realistic reverb
            diffusion = torch.sigmoid(self.diffusion_strength[i])

            # Scale reflection weights based on number of reflections
            # This ensures consistent visual impact regardless of reflection count
            base_weight = torch.sigmoid(self.reflection_weights[i])

            # Scale feedback effect based on number of reflections
            # Fewer reflections = stronger feedback effect to compensate
            feedback_scale = 1.0 + (8.0 / self.num_reflections) * 0.5
            weight = base_weight * decay_rate * (1.0 + feedback * feedback_scale)

            # Create delayed reflection
            reflection = self.create_delay_line(feedback_buffer, delay)

            # Apply spatial processing
            reflection = self.feedback_conv(reflection)

            # Add dampening (frequency-dependent decay)
            damping = self.dampen_net(reflection)
            reflection = reflection * damping

            # Scale reflection strength based on number of reflections
            # Fewer reflections = stronger individual reflections
            reflection_strength = 4.0 * (8.0 / self.num_reflections)
            wet_signal = wet_signal + reflection * weight * reflection_strength

            # Update feedback buffer with reflection-count-aware scaling
            # Scale feedback amount based on number of reflections
            feedback_amount = feedback * 1.2 * (8.0 / self.num_reflections)

            # Apply non-linear feedback scaling that's consistent across reflection counts
            if feedback > 0.8:
                feedback_amount *= 2.0
            feedback_buffer = feedback_buffer + reflection * feedback_amount

            # Add cross-channel feedback with reflection-count-aware scaling
            if feedback > 0.3 and i > 0:
                prev_reflection = self.create_delay_line(feedback_buffer, delay * 0.5)
                cross_feedback_strength = feedback * 0.4 * (8.0 / self.num_reflections)
                feedback_buffer = (
                    feedback_buffer + prev_reflection * cross_feedback_strength
                )

        # Apply spatial attention for coherence
        # Reshape for attention
        wet_flat = wet_signal.view(b, c, -1).transpose(1, 2)  # [B, HW, C]
        attended, _ = self.spatial_attention(wet_flat, wet_flat, wet_flat)
        wet_signal = attended.transpose(1, 2).view(b, c, h, w)

        # Scale blur effect based on number of reflections
        # Fewer reflections = stronger blur to maintain visual impact
        if room_size > 0.3:
            blur_kernel_size = max(3, int(room_size * 3 * (8.0 / self.num_reflections)))
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1  # Ensure odd kernel size
            blur_kernel_size = min(blur_kernel_size, 15)

            # Apply Gaussian blur with room_size and reflection-count dependent sigma
            sigma = room_size * 0.8 * (8.0 / self.num_reflections)

            # Create a simple averaging blur kernel as fallback
            blur_kernel = torch.ones(
                1, 1, blur_kernel_size, blur_kernel_size, device=wet_signal.device
            ) / (blur_kernel_size * blur_kernel_size)
            blur_kernel = blur_kernel.repeat(c, 1, 1, 1)

            # Apply the blur
            wet_signal = F.conv2d(
                wet_signal, blur_kernel, padding=blur_kernel_size // 2, groups=c
            )

        # Scale edge enhancement based on number of reflections
        if feedback > 0.2:
            # Create edge detection kernel
            edge_kernel = torch.tensor(
                [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                dtype=torch.float32,
                device=wet_signal.device,
            ).view(1, 1, 3, 3)
            edge_kernel = edge_kernel.repeat(c, 1, 1, 1)

            # Apply edge detection with feedback and reflection-count dependent strength
            edge_response = F.conv2d(wet_signal, edge_kernel, padding=1, groups=c)
            edge_strength = feedback * 0.2 * (8.0 / self.num_reflections)
            wet_signal = wet_signal + edge_response * edge_strength

        # Scale contrast enhancement based on number of reflections
        if room_size > 0.8:
            # Increase contrast in the wet signal for larger rooms
            mean_val = wet_signal.mean()
            contrast_factor = 1.0 + (room_size - 0.8) * 0.5 * (
                8.0 / self.num_reflections
            )
            wet_signal = (wet_signal - mean_val) * contrast_factor + mean_val

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
        return {
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
                        "description": "Controls the balance between original (dry) and processed (wet) signal. 0.0 = original image only, 1.0 = fully processed image only, 0.3-0.5 = typical reverb mix.",
                    },
                ),
                "feedback": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.5,
                        "step": 0.01,
                        "display": "slider",
                        "description": "Controls how much each reflection feeds back into the system. Higher values create more complex, layered effects. Values above 0.8 create increasingly dramatic and complex patterns. Values above 1.0 create extreme feedback effects. The visual impact is automatically scaled based on the number of reflections.",
                    },
                ),
                "room_size": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.05,
                        "max": 4.0,
                        "step": 0.05,
                        "display": "slider",
                        "description": "Controls the spatial scale of reflections. Small values (0.05-0.3) create subtle shifts, medium values (0.5-1.5) create moderate effects, large values (2.0-4.0) create dramatic spatial effects, blur, and contrast changes. The visual impact is automatically scaled based on the number of reflections.",
                    },
                ),
                "num_reflections": (
                    "INT",
                    {
                        "default": 16,
                        "min": 2,
                        "max": 32,
                        "step": 1,
                        "description": "Number of reflection layers. Higher values create more complex, realistic reverb. Lower values (2-8) create echo-like effects, higher values (16-32) create natural reverb. The system automatically adjusts individual reflection strength to maintain consistent visual impact.",
                    },
                ),
                "decay_rate": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.5,
                        "max": 0.95,
                        "step": 0.01,
                        "display": "slider",
                        "description": "Controls how quickly reflections fade. Higher values (0.9-0.95) create longer reverb tails like a large hall, medium values (0.8-0.85) create natural reverb, lower values (0.5-0.7) create short, tight reverb like a small room. Each reflection gets progressively weaker by this factor.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_reverb"
    CATEGORY = "latent/transform"

    def __init__(self):
        self.reverb_cache: Dict[str, LatentReverb] = {}

    def get_reverb_processor(
        self, channels: int, num_reflections: int, device: str, decay_rate: float
    ) -> LatentReverb:
        """Get or create reverb processor with caching"""
        cache_key = f"{channels}_{num_reflections}_{decay_rate}_{device}"

        if cache_key not in self.reverb_cache:
            reverb = LatentReverb(
                channels=channels,
                num_reflections=num_reflections,
                max_delay=24,
                decay_rate=decay_rate,
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
        decay_rate: float,
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """Apply latent reverb effect"""

        latent = samples["samples"]
        device = latent.device
        channels = latent.shape[1]

        # Get reverb processor
        reverb = self.get_reverb_processor(
            channels, num_reflections, device, decay_rate
        )

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
    "LatentReverb": "Latent Reverb",
}
