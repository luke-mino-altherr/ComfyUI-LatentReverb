#!/usr/bin/env python

"""Tests for `latent_reverb` package."""

import pytest
import torch
import numpy as np
from src.latent_reverb.nodes import LatentReverb, LatentReverbNode


# Test data fixtures
@pytest.fixture
def sample_latent():
    """Create a sample latent tensor for testing."""
    return torch.randn(2, 4, 64, 64)  # [batch, channels, height, width]


@pytest.fixture
def sample_latent_dict():
    """Create a sample latent dictionary in ComfyUI format."""
    return {"samples": torch.randn(2, 4, 64, 64)}


@pytest.fixture
def reverb_model():
    """Create a LatentReverb model instance."""
    return LatentReverb(channels=4, num_reflections=8, max_delay=16)


@pytest.fixture
def reverb_node():
    """Create a LatentReverbNode instance."""
    return LatentReverbNode()


# LatentReverb model tests
class TestLatentReverb:
    """Test the LatentReverb neural network model."""

    def test_initialization(self):
        """Test model initialization with different parameters."""
        model = LatentReverb(channels=4, num_reflections=8, max_delay=16)
        assert model.channels == 4
        assert model.num_reflections == 8
        assert model.max_delay == 16

        # Test parameter shapes
        assert model.reflection_weights.shape == (8,)
        assert model.reflection_delays.shape == (8,)
        assert model.spatial_conv.in_channels == 4
        assert model.spatial_conv.out_channels == 8

    def test_forward_pass(self, reverb_model, sample_latent):
        """Test forward pass with different parameters."""
        # Test basic forward pass
        output = reverb_model(sample_latent)
        assert output.shape == sample_latent.shape
        assert output.dtype == sample_latent.dtype

        # Test with different wet_mix values
        output_dry = reverb_model(sample_latent, wet_mix=0.0)
        output_wet = reverb_model(sample_latent, wet_mix=1.0)

        # Dry output should be close to input
        assert torch.allclose(output_dry, sample_latent, atol=1e-6)
        # Wet output should be different from input
        assert not torch.allclose(output_wet, sample_latent, atol=1e-6)

    def test_parameter_ranges(self, reverb_model, sample_latent):
        """Test behavior with extreme parameter values."""
        # Test feedback extremes
        output_low_feedback = reverb_model(sample_latent, feedback=0.0)
        output_high_feedback = reverb_model(sample_latent, feedback=0.8)

        # Both should produce valid outputs
        assert not torch.isnan(output_low_feedback).any()
        assert not torch.isnan(output_high_feedback).any()

        # Test room size extremes
        output_small_room = reverb_model(sample_latent, room_size=0.1)
        output_large_room = reverb_model(sample_latent, room_size=2.0)

        assert not torch.isnan(output_small_room).any()
        assert not torch.isnan(output_large_room).any()

    def test_delay_line_creation(self, reverb_model):
        """Test the delay line creation method."""
        x = torch.randn(1, 4, 32, 32)

        # Test with zero delay
        delayed_zero = reverb_model.create_delay_line(x, 0.0)
        assert torch.allclose(delayed_zero, x, atol=1e-6)

        # Test with integer delay
        delayed_int = reverb_model.create_delay_line(x, 2.0)
        assert delayed_int.shape == x.shape

        # Test with fractional delay
        delayed_frac = reverb_model.create_delay_line(x, 2.5)
        assert delayed_frac.shape == x.shape

    def test_different_input_sizes(self, reverb_model):
        """Test model with different input tensor sizes."""
        sizes = [(1, 4, 32, 32), (2, 4, 64, 64), (1, 4, 128, 128)]

        for size in sizes:
            x = torch.randn(*size)
            output = reverb_model(x)
            assert output.shape == size
            assert not torch.isnan(output).any()


# LatentReverbNode tests
class TestLatentReverbNode:
    """Test the ComfyUI LatentReverbNode."""

    def test_input_types(self):
        """Test that INPUT_TYPES returns correct structure."""
        input_types = LatentReverbNode.INPUT_TYPES()
        assert "required" in input_types

        required = input_types["required"]
        assert "samples" in required
        assert "wet_mix" in required
        assert "feedback" in required
        assert "room_size" in required
        assert "num_reflections" in required

        # Test wet_mix constraints
        wet_mix_config = required["wet_mix"][1]  # ComfyUI format: (type, config_dict)
        assert wet_mix_config["min"] == 0.0
        assert wet_mix_config["max"] == 1.0
        assert wet_mix_config["default"] == 0.3

    def test_metadata(self):
        """Test node metadata."""
        assert LatentReverbNode.RETURN_TYPES == ("LATENT",)
        assert LatentReverbNode.FUNCTION == "apply_reverb"
        assert LatentReverbNode.CATEGORY == "latent/effects"

    def test_initialization(self, reverb_node):
        """Test node initialization."""
        assert hasattr(reverb_node, "reverb_cache")
        assert isinstance(reverb_node.reverb_cache, dict)

    def test_get_reverb_processor(self, reverb_node):
        """Test reverb processor creation and caching."""
        device = "cpu"
        channels = 4
        num_reflections = 8

        # Get processor
        processor = reverb_node.get_reverb_processor(channels, num_reflections, device)
        assert isinstance(processor, LatentReverb)
        assert processor.channels == channels
        assert processor.num_reflections == num_reflections

        # Test caching
        processor2 = reverb_node.get_reverb_processor(channels, num_reflections, device)
        assert processor is processor2  # Should be the same instance

        # Test different parameters create different processors
        processor_diff = reverb_node.get_reverb_processor(channels, 16, device)
        assert processor is not processor_diff

    def test_apply_reverb(self, reverb_node, sample_latent_dict):
        """Test the main reverb application function."""
        wet_mix = 0.5
        feedback = 0.3
        room_size = 0.7
        num_reflections = 6

        result = reverb_node.apply_reverb(
            sample_latent_dict, wet_mix, feedback, room_size, num_reflections
        )

        # Check return format
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "samples" in result[0]

        # Check output shape
        output_latent = result[0]["samples"]
        assert output_latent.shape == sample_latent_dict["samples"].shape

        # Check that output is different from input (due to reverb)
        assert not torch.allclose(
            output_latent, sample_latent_dict["samples"], atol=1e-6
        )

    def test_apply_reverb_dry(self, reverb_node, sample_latent_dict):
        """Test reverb with dry mix (should be close to input)."""
        result = reverb_node.apply_reverb(
            sample_latent_dict,
            wet_mix=0.0,
            feedback=0.0,
            room_size=0.5,
            num_reflections=8,
        )

        output_latent = result[0]["samples"]
        input_latent = sample_latent_dict["samples"]

        # With wet_mix=0, output should be very close to input
        assert torch.allclose(output_latent, input_latent, atol=1e-5)


# Integration tests
class TestIntegration:
    """Integration tests for the complete system."""

    def test_end_to_end_processing(self, reverb_node, sample_latent_dict):
        """Test complete reverb processing pipeline."""
        # Process with moderate settings
        result = reverb_node.apply_reverb(
            sample_latent_dict,
            wet_mix=0.4,
            feedback=0.3,
            room_size=0.6,
            num_reflections=10,
        )

        # Verify output integrity
        output = result[0]["samples"]
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.requires_grad == False  # Should be inference mode

    def test_different_channel_counts(self, reverb_node):
        """Test processing with different channel counts."""
        channel_counts = [1, 4, 8, 16]

        for channels in channel_counts:
            latent_dict = {"samples": torch.randn(1, channels, 32, 32)}

            result = reverb_node.apply_reverb(latent_dict, 0.3, 0.4, 0.5, 8)

            output = result[0]["samples"]
            assert output.shape[1] == channels
            assert not torch.isnan(output).any()


# Error handling tests
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_wet_mix(self, reverb_node, sample_latent_dict):
        """Test behavior with invalid wet_mix values."""
        # These should not raise errors but clamp values
        result = reverb_node.apply_reverb(
            sample_latent_dict,
            wet_mix=1.5,
            feedback=0.4,
            room_size=0.5,
            num_reflections=8,
        )
        assert result is not None

    def test_empty_latent(self, reverb_node):
        """Test with empty latent tensor."""
        empty_latent = {"samples": torch.empty(0, 4, 64, 64)}

        # Should handle gracefully or raise appropriate error
        with pytest.raises(Exception):
            reverb_node.apply_reverb(empty_latent, 0.3, 0.4, 0.5, 8)


# Performance tests
class TestPerformance:
    """Test performance characteristics."""

    def test_memory_usage(self, reverb_node, sample_latent_dict):
        """Test that memory usage is reasonable."""
        import gc

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        # Process multiple times to check for memory leaks
        for _ in range(5):
            result = reverb_node.apply_reverb(sample_latent_dict, 0.3, 0.4, 0.5, 8)
            del result

        # Force garbage collection
        gc.collect()

    def test_processing_speed(self, reverb_node, sample_latent_dict):
        """Test processing speed is reasonable."""
        import time

        start_time = time.time()
        result = reverb_node.apply_reverb(sample_latent_dict, 0.3, 0.4, 0.5, 8)
        end_time = time.time()

        processing_time = end_time - start_time
        # Should process in reasonable time (adjust threshold as needed)
        assert processing_time < 1.0  # Less than 1 second for small tensor
