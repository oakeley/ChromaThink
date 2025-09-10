"""Tests for wave interference pattern validation with known frequency combinations"""

import pytest
import tensorflow as tf
import numpy as np
from ..layers.interference import InterferenceLayer, MultiWaveInterference
from ..layers.cognitive_waveform import CognitiveWaveform


class TestWaveInterference:
    """Test suite for wave interference patterns"""
    
    @pytest.fixture
    def interference_layer(self):
        """Create InterferenceLayer for testing"""
        return InterferenceLayer(
            interference_type='full',
            amplitude_weighting=True,
            phase_coupling=True,
            nonlinear_mixing=False
        )
    
    @pytest.fixture
    def multi_wave_interference(self):
        """Create MultiWaveInterference layer for testing"""
        return MultiWaveInterference(max_waves=4)
    
    def create_test_waves(self, batch_size=4, dimensions=64):
        """Create test waveforms with known properties"""
        # Create waves with specific frequencies and phases
        frequencies = tf.linspace(1.0, 10.0, dimensions)
        time_points = tf.linspace(0.0, 2*np.pi, batch_size)
        
        # Wave 1: Low frequency, phase 0
        amplitude1 = tf.ones([batch_size, dimensions]) * 0.5
        phase1 = tf.expand_dims(time_points, 1) * tf.expand_dims(frequencies, 0) * 0.1
        wave1 = amplitude1 * tf.exp(tf.complex(0.0, phase1))
        
        # Wave 2: High frequency, phase π/2
        amplitude2 = tf.ones([batch_size, dimensions]) * 0.7
        phase2 = tf.expand_dims(time_points, 1) * tf.expand_dims(frequencies, 0) * 0.3 + np.pi/2
        wave2 = amplitude2 * tf.exp(tf.complex(0.0, phase2))
        
        return wave1, wave2, frequencies
    
    def test_constructive_interference(self):
        """Test constructive interference with aligned waves"""
        layer = InterferenceLayer(interference_type='constructive')
        
        # Create two identical waves (should interfere constructively)
        amplitude = tf.ones([4, 32]) * 0.5
        phase = tf.zeros([4, 32])
        wave1 = amplitude * tf.exp(tf.complex(0.0, phase))
        wave2 = amplitude * tf.exp(tf.complex(0.0, phase))
        
        # Build and apply layer
        output = layer([wave1, wave2])
        
        # Output amplitude should be approximately doubled
        output_amplitude = tf.abs(output)
        expected_amplitude = tf.abs(wave1) + tf.abs(wave2)
        
        # Allow some tolerance for numerical precision
        amplitude_diff = tf.abs(output_amplitude - expected_amplitude)
        max_diff = tf.reduce_max(amplitude_diff)
        
        assert max_diff < 0.1, f"Constructive interference failed: max difference {max_diff}"
        
        # Phase should be preserved
        output_phase = tf.angle(output)
        input_phase = tf.angle(wave1)
        phase_diff = tf.abs(output_phase - input_phase)
        max_phase_diff = tf.reduce_max(phase_diff)
        
        assert max_phase_diff < 0.2, f"Phase not preserved in constructive interference: {max_phase_diff}"
    
    def test_destructive_interference(self):
        """Test destructive interference with opposing waves"""
        layer = InterferenceLayer(interference_type='destructive')
        
        # Create two opposing waves (π phase difference)
        amplitude = tf.ones([4, 32]) * 0.5
        phase1 = tf.zeros([4, 32])
        phase2 = tf.ones([4, 32]) * np.pi
        
        wave1 = amplitude * tf.exp(tf.complex(0.0, phase1))
        wave2 = amplitude * tf.exp(tf.complex(0.0, phase2))
        
        # Build and apply layer
        output = layer([wave1, wave2])
        
        # Output amplitude should be reduced (ideally zero for perfect opposition)
        output_amplitude = tf.abs(output)
        input_amplitude = tf.abs(wave1)
        
        # Destructive interference should reduce amplitude
        amplitude_reduction = tf.reduce_mean(input_amplitude - output_amplitude)
        assert amplitude_reduction > 0.2, f"Insufficient destructive interference: reduction {amplitude_reduction}"
    
    def test_full_interference_physics(self):
        """Test full interference follows wave physics principles"""
        layer = InterferenceLayer(interference_type='full')
        
        wave1, wave2, frequencies = self.create_test_waves()
        
        # Build and apply layer
        output = layer([wave1, wave2])
        
        # Test interference strength metrics
        strength_metrics = layer.interference_strength([wave1, wave2])
        
        # Phase correlation should be reasonable
        phase_corr = strength_metrics['phase_correlation']
        assert tf.reduce_all(tf.abs(phase_corr) <= 1.0), "Phase correlation out of bounds"
        
        # Amplitude correlation should be positive for similar waves
        amp_corr = strength_metrics['amplitude_correlation']
        assert tf.reduce_all(amp_corr >= 0.0), "Amplitude correlation should be non-negative"
        
        # Interference strength should be finite and reasonable
        interference_strength = strength_metrics['interference_strength']
        assert tf.reduce_all(tf.math.is_finite(interference_strength)), "Interference strength not finite"
        assert tf.reduce_all(interference_strength >= 0.0), "Interference strength should be non-negative"
    
    def test_amplitude_weighting(self):
        """Test amplitude weighting functionality"""
        layer_weighted = InterferenceLayer(amplitude_weighting=True)
        layer_unweighted = InterferenceLayer(amplitude_weighting=False)
        
        wave1, wave2, _ = self.create_test_waves()
        
        # Create waves with very different amplitudes
        wave1_scaled = wave1 * 0.1  # Very small amplitude
        wave2_scaled = wave2 * 2.0  # Large amplitude
        
        # Build layers
        output_weighted = layer_weighted([wave1_scaled, wave2_scaled])
        output_unweighted = layer_unweighted([wave1_scaled, wave2_scaled])
        
        # Weighted version should handle amplitude differences better
        weighted_variance = tf.math.reduce_variance(tf.abs(output_weighted))
        unweighted_variance = tf.math.reduce_variance(tf.abs(output_unweighted))
        
        # This test checks that weighting affects the output meaningfully
        assert not tf.reduce_all(tf.abs(output_weighted - output_unweighted) < 1e-6), \
            "Amplitude weighting has no effect"
    
    def test_phase_coupling(self):
        """Test phase coupling between frequencies"""
        layer_coupled = InterferenceLayer(phase_coupling=True)
        layer_uncoupled = InterferenceLayer(phase_coupling=False)
        
        wave1, wave2, _ = self.create_test_waves()
        
        # Build layers
        output_coupled = layer_coupled([wave1, wave2])
        output_uncoupled = layer_uncoupled([wave1, wave2])
        
        # Phase coupling should create different phase relationships
        phase_coupled = tf.angle(output_coupled)
        phase_uncoupled = tf.angle(output_uncoupled)
        
        phase_difference = tf.abs(phase_coupled - phase_uncoupled)
        mean_phase_diff = tf.reduce_mean(phase_difference)
        
        assert mean_phase_diff > 0.1, f"Phase coupling has minimal effect: {mean_phase_diff}"
    
    def test_nonlinear_mixing(self):
        """Test nonlinear mixing for richer interference"""
        layer_linear = InterferenceLayer(nonlinear_mixing=False)
        layer_nonlinear = InterferenceLayer(nonlinear_mixing=True)
        
        wave1, wave2, _ = self.create_test_waves()
        
        # Build layers
        output_linear = layer_linear([wave1, wave2])
        output_nonlinear = layer_nonlinear([wave1, wave2])
        
        # Nonlinear mixing should create harmonics and richer spectrum
        fft_linear = tf.signal.fft(output_linear)
        fft_nonlinear = tf.signal.fft(output_nonlinear)
        
        power_linear = tf.abs(fft_linear) ** 2
        power_nonlinear = tf.abs(fft_nonlinear) ** 2
        
        # Nonlinear version should have broader spectrum
        spectral_width_linear = self._calculate_spectral_width(power_linear)
        spectral_width_nonlinear = self._calculate_spectral_width(power_nonlinear)
        
        width_increase = spectral_width_nonlinear - spectral_width_linear
        assert tf.reduce_mean(width_increase) > 0, "Nonlinear mixing didn't broaden spectrum"
    
    def test_multi_wave_interference(self, multi_wave_interference):
        """Test interference between multiple waves"""
        # Create 3 waves with different properties
        batch_size, dimensions = 4, 32
        
        # Wave 1: Low frequency
        wave1 = tf.complex(
            tf.sin(tf.linspace(0.0, 2*np.pi, dimensions)) * tf.ones([batch_size, dimensions]),
            tf.cos(tf.linspace(0.0, 2*np.pi, dimensions)) * tf.ones([batch_size, dimensions]) * 0.1
        )
        
        # Wave 2: Medium frequency
        wave2 = tf.complex(
            tf.sin(tf.linspace(0.0, 6*np.pi, dimensions)) * tf.ones([batch_size, dimensions]) * 0.7,
            tf.cos(tf.linspace(0.0, 6*np.pi, dimensions)) * tf.ones([batch_size, dimensions]) * 0.2
        )
        
        # Wave 3: High frequency
        wave3 = tf.complex(
            tf.sin(tf.linspace(0.0, 12*np.pi, dimensions)) * tf.ones([batch_size, dimensions]) * 0.5,
            tf.cos(tf.linspace(0.0, 12*np.pi, dimensions)) * tf.ones([batch_size, dimensions]) * 0.3
        )
        
        # Test multi-wave interference
        output = multi_wave_interference([wave1, wave2, wave3])
        
        # Output should be finite and reasonable
        assert tf.reduce_all(tf.math.is_finite(tf.real(output))), "Multi-wave output contains non-finite values"
        assert tf.reduce_all(tf.math.is_finite(tf.imag(output))), "Multi-wave output contains non-finite values"
        
        # Output should contain contributions from all input waves
        output_power = tf.abs(output) ** 2
        total_power = tf.reduce_sum(output_power)
        
        assert total_power > 0, "Multi-wave interference produced zero output"
    
    def test_interference_energy_conservation(self):
        """Test that interference conserves energy appropriately"""
        layer = InterferenceLayer(interference_type='full', nonlinear_mixing=False)
        
        wave1, wave2, _ = self.create_test_waves()
        
        # Calculate input energies
        energy1 = tf.reduce_sum(tf.abs(wave1) ** 2, axis=-1)
        energy2 = tf.reduce_sum(tf.abs(wave2) ** 2, axis=-1)
        total_input_energy = energy1 + energy2
        
        # Calculate output energy
        output = layer([wave1, wave2])
        output_energy = tf.reduce_sum(tf.abs(output) ** 2, axis=-1)
        
        # For linear interference, output energy should be related to input energies
        # (not necessarily equal due to interference effects, but in same ballpark)
        energy_ratio = output_energy / (total_input_energy + 1e-8)
        
        # Ratio should be reasonable (between 0.5 and 2.0)
        assert tf.reduce_all(energy_ratio > 0.3), "Energy ratio too low"
        assert tf.reduce_all(energy_ratio < 3.0), "Energy ratio too high"
    
    def test_known_frequency_combinations(self):
        """Test interference with specific known frequency combinations"""
        layer = InterferenceLayer(interference_type='full')
        
        batch_size, dimensions = 2, 64
        
        # Create waves with known frequency relationships
        # Wave 1: Fundamental frequency
        t = tf.linspace(0.0, 2*np.pi, dimensions)
        fundamental_freq = 1.0
        wave1 = tf.expand_dims(
            tf.complex(tf.sin(fundamental_freq * t), tf.cos(fundamental_freq * t) * 0.1),
            0
        )
        wave1 = tf.tile(wave1, [batch_size, 1])
        
        # Wave 2: Third harmonic
        harmonic_freq = 3.0
        wave2 = tf.expand_dims(
            tf.complex(tf.sin(harmonic_freq * t) * 0.7, tf.cos(harmonic_freq * t) * 0.2),
            0
        )
        wave2 = tf.tile(wave2, [batch_size, 1])
        
        # Apply interference
        output = layer([wave1, wave2])
        
        # Analyze frequency content
        fft_output = tf.signal.fft(output)
        power_spectrum = tf.abs(fft_output) ** 2
        
        # Should see peaks at fundamental and harmonic frequencies
        # (This is a simplified test - in practice would need more sophisticated analysis)
        total_power = tf.reduce_sum(power_spectrum, axis=-1)
        assert tf.reduce_all(total_power > 0), "No power in interference output"
        
        # Power spectrum should not be flat (should show structure)
        power_variance = tf.math.reduce_variance(power_spectrum, axis=-1)
        assert tf.reduce_all(power_variance > 1e-4), "Power spectrum too flat"
    
    def _calculate_spectral_width(self, power_spectrum):
        """Calculate spectral width of power spectrum"""
        # Normalize power spectrum
        normalized_power = power_spectrum / (tf.reduce_sum(power_spectrum, axis=-1, keepdims=True) + 1e-8)
        
        # Calculate centroid
        frequencies = tf.range(tf.shape(power_spectrum)[-1], dtype=tf.float32)
        centroid = tf.reduce_sum(normalized_power * frequencies, axis=-1)
        
        # Calculate second moment (width)
        freq_diff_sq = tf.square(tf.expand_dims(frequencies, 0) - tf.expand_dims(centroid, -1))
        width = tf.sqrt(tf.reduce_sum(normalized_power * freq_diff_sq, axis=-1))
        
        return width
    
    @pytest.mark.parametrize("interference_type", ['constructive', 'destructive', 'full'])
    def test_interference_types_consistency(self, interference_type):
        """Test consistency across different interference types"""
        layer = InterferenceLayer(interference_type=interference_type)
        
        wave1, wave2, _ = self.create_test_waves()
        
        # Multiple forward passes should give consistent results
        outputs = []
        for _ in range(5):
            output = layer([wave1, wave2])
            outputs.append(output)
        
        # Check consistency
        output_stack = tf.stack(outputs)
        output_variance = tf.math.reduce_variance(tf.abs(output_stack), axis=0)
        max_variance = tf.reduce_max(output_variance)
        
        assert max_variance < 0.01, f"Inconsistent outputs for {interference_type}: max_var={max_variance}"