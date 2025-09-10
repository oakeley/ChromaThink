"""Tests for colour vector stability (prevent collapse to grey or noise explosion)"""

import pytest
import tensorflow as tf
import numpy as np
from ..layers.cognitive_waveform import CognitiveWaveform
from ..layers.chromatic_resonance import ChromaticResonance
from ..core.spectral_utils import frequency_stability_check
from ..core.colour_utils import prevent_collapse


class TestColourStability:
    """Test suite for colour vector stability"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        batch_size = 8
        input_dim = 64
        return tf.random.normal([batch_size, input_dim], stddev=1.0)
    
    @pytest.fixture
    def cognitive_waveform(self):
        """Create CognitiveWaveform layer for testing"""
        return CognitiveWaveform(
            dimensions=128,
            frequency_range=(0.1, 100.0),
            spectral_normalize=True
        )
    
    @pytest.fixture
    def chromatic_resonance(self):
        """Create ChromaticResonance layer for testing"""
        return ChromaticResonance(
            dimensions=128,
            resonance_depth=5
        )
    
    def test_waveform_prevents_collapse(self, cognitive_waveform, sample_data):
        """Test that CognitiveWaveform prevents colour collapse"""
        # Build the layer
        _ = cognitive_waveform(sample_data)
        
        # Test with multiple inputs that could cause collapse
        uniform_input = tf.ones_like(sample_data) * 0.5  # Uniform grey input
        zero_input = tf.zeros_like(sample_data)  # Zero input
        large_input = tf.ones_like(sample_data) * 100.0  # Large input
        
        outputs = []
        for test_input in [uniform_input, zero_input, large_input]:
            output = cognitive_waveform(test_input)
            outputs.append(output)
            
            # Check stability metrics
            stability = frequency_stability_check(output)
            
            # Variance should be above minimum threshold
            assert stability['variance'] > 1e-4, "Colour vectors collapsed to uniform values"
            
            # Norms should be within reasonable range
            assert stability['mean_norm'] > 0.1, "Mean norm too low (possible collapse)"
            assert stability['max_norm'] < 50.0, "Max norm too high (possible explosion)"
            
            # For complex outputs, check phase diversity
            if output.dtype.is_complex:
                assert stability['phase_variance'] > 1e-3, "Phase collapsed to real values"
                assert 0.1 < stability['spectral_flatness'] < 0.9, "Spectral collapse detected"
    
    def test_resonance_stability(self, chromatic_resonance, sample_data):
        """Test that ChromaticResonance maintains stability"""
        # Convert to complex input
        complex_input = tf.complex(sample_data, sample_data * 0.1)
        
        # Build the layer
        _ = chromatic_resonance(complex_input)
        
        # Test multiple iterations
        current_state = complex_input
        for iteration in range(10):
            current_state = chromatic_resonance(current_state, training=True)
            
            # Check for stability
            stability = frequency_stability_check(current_state)
            
            assert stability['is_stable'], f"Instability detected at iteration {iteration}"
            
            # Check for NaN or Inf values
            assert not tf.reduce_any(tf.math.is_nan(tf.real(current_state))), "NaN in real part"
            assert not tf.reduce_any(tf.math.is_nan(tf.imag(current_state))), "NaN in imaginary part"
            assert not tf.reduce_any(tf.math.is_inf(tf.real(current_state))), "Inf in real part"
            assert not tf.reduce_any(tf.math.is_inf(tf.imag(current_state))), "Inf in imaginary part"
    
    def test_collapse_prevention_methods(self, sample_data):
        """Test different collapse prevention methods"""
        # Create colour vectors that might collapse
        collapsing_vectors = tf.ones([8, 64]) * 0.5 + tf.random.normal([8, 64], stddev=0.01)
        complex_vectors = tf.complex(collapsing_vectors, collapsing_vectors * 0.1)
        
        methods = ['entropy_regularization', 'orthogonal_penalty', 'diversity_loss']
        
        for method in methods:
            # Test with real vectors
            loss_real = prevent_collapse(collapsing_vectors, method=method, strength=0.1)
            assert loss_real > 0, f"Collapse prevention loss should be positive for {method}"
            assert tf.math.is_finite(loss_real), f"Loss should be finite for {method}"
            
            # Test with complex vectors
            loss_complex = prevent_collapse(complex_vectors, method=method, strength=0.1)
            assert loss_complex > 0, f"Complex collapse prevention loss should be positive for {method}"
            assert tf.math.is_finite(loss_complex), f"Complex loss should be finite for {method}"
    
    def test_spectral_normalization(self, sample_data):
        """Test spectral normalization prevents explosion"""
        from ..core.spectral_utils import SpectralNormalizer
        
        normalizer = SpectralNormalizer()
        
        # Create potentially explosive input
        explosive_input = tf.random.normal([8, 64], stddev=10.0)
        
        # Build and apply normalizer
        _ = normalizer(explosive_input)  # Build the layer
        normalized_output = normalizer(explosive_input)
        
        # Check that output is properly normalized
        norms = tf.norm(normalized_output, axis=-1)
        max_norm = tf.reduce_max(norms)
        
        assert max_norm < 10.0, "Spectral normalization failed to contain explosion"
        
        # Check that normalization preserves relative relationships
        input_norms = tf.norm(explosive_input, axis=-1)
        norm_ratios = norms / (input_norms + 1e-8)
        
        # All ratios should be similar (consistent scaling)
        ratio_variance = tf.math.reduce_variance(norm_ratios)
        assert ratio_variance < 0.1, "Spectral normalization applied inconsistently"
    
    def test_frequency_stability_over_time(self, cognitive_waveform, sample_data):
        """Test stability of frequency content over multiple forward passes"""
        # Build layer
        initial_output = cognitive_waveform(sample_data)
        
        # Store initial frequency analysis
        initial_analysis = cognitive_waveform.get_frequency_content(sample_data)
        
        stability_metrics = []
        
        # Multiple forward passes with small variations
        for i in range(20):
            noisy_input = sample_data + tf.random.normal(tf.shape(sample_data), stddev=0.01)
            output = cognitive_waveform(noisy_input)
            
            stability = frequency_stability_check(output)
            stability_metrics.append({
                'variance': stability['variance'],
                'mean_norm': stability['mean_norm'],
                'is_stable': stability['is_stable']
            })
        
        # Check consistency across passes
        variances = [m['variance'] for m in stability_metrics]
        mean_norms = [m['mean_norm'] for m in stability_metrics]
        stable_count = sum(m['is_stable'] for m in stability_metrics)
        
        # Variance should be consistent
        variance_std = tf.math.reduce_std(tf.stack(variances))
        assert variance_std < 0.1, "Frequency variance too inconsistent across passes"
        
        # Mean norms should be stable
        norm_std = tf.math.reduce_std(tf.stack(mean_norms))
        assert norm_std < 1.0, "Mean norms too inconsistent across passes"
        
        # Most passes should be stable
        assert stable_count >= 18, "Too many unstable passes detected"
    
    def test_gradient_stability(self, cognitive_waveform, sample_data):
        """Test that gradients remain stable during training"""
        # Build layer
        _ = cognitive_waveform(sample_data)
        
        # Create a simple loss function
        with tf.GradientTape() as tape:
            output = cognitive_waveform(sample_data, training=True)
            loss = tf.reduce_mean(tf.abs(output))
        
        # Calculate gradients
        gradients = tape.gradient(loss, cognitive_waveform.trainable_variables)
        
        # Check gradient properties
        for i, grad in enumerate(gradients):
            if grad is not None:
                # No NaN or Inf gradients
                assert not tf.reduce_any(tf.math.is_nan(grad)), f"NaN gradient in variable {i}"
                assert not tf.reduce_any(tf.math.is_inf(grad)), f"Inf gradient in variable {i}"
                
                # Reasonable gradient magnitudes
                grad_norm = tf.norm(grad)
                assert grad_norm > 1e-8, f"Gradient too small (vanishing) in variable {i}"
                assert grad_norm < 100.0, f"Gradient too large (exploding) in variable {i}"
    
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("input_dim", [32, 64, 128])
    def test_stability_across_batch_sizes(self, batch_size, input_dim):
        """Test stability across different batch sizes and dimensions"""
        # Create layer and data
        layer = CognitiveWaveform(dimensions=64, spectral_normalize=True)
        data = tf.random.normal([batch_size, input_dim])
        
        # Forward pass
        output = layer(data)
        
        # Check stability
        stability = frequency_stability_check(output)
        
        assert stability['is_stable'], f"Instability with batch_size={batch_size}, input_dim={input_dim}"
        
        # Batch size shouldn't affect per-sample stability much
        per_sample_variance = tf.math.reduce_variance(output, axis=-1)
        variance_consistency = tf.math.reduce_std(per_sample_variance)
        
        assert variance_consistency < 0.5, "Per-sample variance too inconsistent across batch"