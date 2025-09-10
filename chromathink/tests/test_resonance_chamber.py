"""Tests for resonance chamber convergence testing for stable standing waves"""

import pytest
import tensorflow as tf
import numpy as np
from ..layers.resonance_chamber import ResonanceChamber
from ..layers.cognitive_waveform import CognitiveWaveform


class TestResonanceChamber:
    """Test suite for resonance chamber standing wave patterns"""
    
    @pytest.fixture
    def resonance_chamber(self):
        """Create ResonanceChamber for testing"""
        return ResonanceChamber(
            dimensions=128,
            chamber_length=1.0,
            boundary_conditions='mixed',
            num_modes=16,
            quality_factor=50.0
        )
    
    @pytest.fixture
    def sample_waveform(self):
        """Create sample complex waveform"""
        batch_size, dimensions = 4, 128
        amplitude = tf.random.uniform([batch_size, dimensions], 0.1, 1.0)
        phase = tf.random.uniform([batch_size, dimensions], 0, 2*np.pi)
        return amplitude * tf.exp(tf.complex(0.0, phase))
    
    def test_chamber_initialization(self, resonance_chamber):
        """Test that resonance chamber initializes correctly"""
        # Check that resonant frequencies are calculated
        assert hasattr(resonance_chamber, 'resonant_frequencies')
        
        # Frequencies should be positive and ordered
        frequencies = resonance_chamber.resonant_frequencies
        assert tf.reduce_all(frequencies > 0), "Resonant frequencies should be positive"
        
        # Check mode shapes exist
        assert hasattr(resonance_chamber, 'mode_shapes')
        mode_shapes = resonance_chamber.mode_shapes
        
        # Mode shapes should be normalized
        norms = tf.norm(mode_shapes, axis=0)
        assert tf.reduce_all(tf.abs(norms - 1.0) < 0.1), "Mode shapes should be approximately normalized"
    
    def test_standing_wave_convergence(self, resonance_chamber, sample_waveform):
        """Test that standing waves converge to stable patterns"""
        # Build the chamber
        initial_output = resonance_chamber(sample_waveform)
        
        convergence_history = []
        current_input = sample_waveform
        
        # Test convergence over multiple iterations
        for iteration in range(10):
            output = resonance_chamber(current_input, training=False)
            
            # Use output as input for next iteration (feedback)
            current_input = output * 0.9 + sample_waveform * 0.1  # Add some driving
            
            # Analyze standing wave characteristics
            analysis = resonance_chamber.get_standing_wave_analysis(current_input)
            
            convergence_history.append({
                'mode_amplitudes': analysis['mode_amplitudes'],
                'swr': analysis['standing_wave_ratio'],
                'node_count': analysis['node_count']
            })
        
        # Check convergence: later iterations should be more similar
        early_modes = convergence_history[2]['mode_amplitudes']  # Skip first few for stability
        late_modes = convergence_history[-1]['mode_amplitudes']
        
        mode_change = tf.reduce_mean(tf.abs(late_modes - early_modes))
        assert mode_change < 0.5, f"Standing waves not converging: change={mode_change}"
        
        # Standing wave ratio should stabilize
        swr_values = [h['swr'] for h in convergence_history]
        swr_variance = tf.math.reduce_variance(tf.stack(swr_values[-3:]))  # Last 3 iterations
        assert tf.reduce_mean(swr_variance) < 1.0, "Standing wave ratio not stabilizing"
    
    def test_boundary_conditions(self):
        """Test different boundary conditions produce appropriate standing waves"""
        boundary_types = ['rigid', 'free', 'mixed', 'absorbing']
        sample_input = tf.complex(
            tf.random.normal([2, 64]),
            tf.random.normal([2, 64]) * 0.1
        )
        
        chambers = {}
        outputs = {}
        
        for boundary_type in boundary_types:
            chamber = ResonanceChamber(
                dimensions=64,
                boundary_conditions=boundary_type,
                num_modes=8
            )
            output = chamber(sample_input)
            
            chambers[boundary_type] = chamber
            outputs[boundary_type] = output
            
            # Check that output is reasonable
            assert tf.reduce_all(tf.math.is_finite(tf.real(output))), f"Non-finite output for {boundary_type}"
            assert tf.reduce_all(tf.math.is_finite(tf.imag(output))), f"Non-finite output for {boundary_type}"
        
        # Different boundary conditions should produce different outputs
        for i, type1 in enumerate(boundary_types):
            for type2 in boundary_types[i+1:]:
                output_diff = tf.reduce_mean(tf.abs(outputs[type1] - outputs[type2]))
                assert output_diff > 0.01, f"Boundary conditions {type1} and {type2} produce too similar outputs"
    
    def test_resonant_mode_excitation(self, resonance_chamber, sample_waveform):
        """Test that specific frequencies excite appropriate resonant modes"""
        # Get resonant frequencies
        resonant_freqs = resonance_chamber.resonant_frequencies
        
        # Create input that matches a specific resonant frequency
        target_mode = 3  # Test mode 3
        target_freq = resonant_freqs[target_mode]
        
        # Create sinusoidal input at target frequency
        dimensions = 128
        positions = tf.linspace(0.0, 1.0, dimensions)
        
        # Approximate sinusoidal excitation
        excitation_pattern = tf.sin(target_freq * 2 * np.pi * positions)
        excitation_input = tf.complex(
            tf.expand_dims(excitation_pattern, 0) * tf.ones([2, dimensions]),
            tf.expand_dims(excitation_pattern, 0) * tf.ones([2, dimensions]) * 0.1
        )
        
        # Apply to chamber
        output = resonance_chamber(excitation_input)
        analysis = resonance_chamber.get_standing_wave_analysis(excitation_input)
        
        # Target mode should be more strongly excited
        mode_amplitudes = analysis['mode_amplitudes']
        target_mode_amp = mode_amplitudes[:, target_mode]
        other_mode_amps = tf.concat([
            mode_amplitudes[:, :target_mode],
            mode_amplitudes[:, target_mode+1:]
        ], axis=1)
        
        mean_target_amp = tf.reduce_mean(target_mode_amp)
        mean_other_amps = tf.reduce_mean(other_mode_amps)
        
        # Target mode should be more excited (allowing some tolerance)
        excitation_ratio = mean_target_amp / (mean_other_amps + 1e-8)
        assert excitation_ratio > 1.2, f"Target mode not preferentially excited: ratio={excitation_ratio}"
    
    def test_quality_factor_effects(self):
        """Test that quality factor affects resonance behavior"""
        sample_input = tf.complex(
            tf.random.normal([2, 64]),
            tf.random.normal([2, 64]) * 0.1
        )
        
        low_q_chamber = ResonanceChamber(
            dimensions=64,
            quality_factor=10.0,
            num_modes=8
        )
        
        high_q_chamber = ResonanceChamber(
            dimensions=64,
            quality_factor=100.0,
            num_modes=8
        )
        
        low_q_output = low_q_chamber(sample_input)
        high_q_output = high_q_chamber(sample_input)
        
        # Analyze frequency selectivity
        low_q_analysis = low_q_chamber.get_standing_wave_analysis(sample_input)
        high_q_analysis = high_q_chamber.get_standing_wave_analysis(sample_input)
        
        # High Q should have sharper resonances (more selective)
        low_q_modes = low_q_analysis['mode_amplitudes']
        high_q_modes = high_q_analysis['mode_amplitudes']
        
        # Calculate selectivity as ratio of max to mean mode amplitude
        low_q_selectivity = tf.reduce_max(low_q_modes, axis=1) / (tf.reduce_mean(low_q_modes, axis=1) + 1e-8)
        high_q_selectivity = tf.reduce_max(high_q_modes, axis=1) / (tf.reduce_mean(high_q_modes, axis=1) + 1e-8)
        
        mean_low_selectivity = tf.reduce_mean(low_q_selectivity)
        mean_high_selectivity = tf.reduce_mean(high_q_selectivity)
        
        assert mean_high_selectivity > mean_low_selectivity, \
            f"High Q not more selective: low={mean_low_selectivity}, high={mean_high_selectivity}"
    
    def test_chamber_memory_persistence(self, resonance_chamber, sample_waveform):
        """Test that chamber memory persists and affects future responses"""
        # Build chamber
        _ = resonance_chamber(sample_waveform, training=True)
        
        # Get initial memory state
        initial_memory = tf.identity(resonance_chamber.standing_wave_memory)
        
        # Apply several training iterations
        for _ in range(5):
            _ = resonance_chamber(sample_waveform, training=True)
        
        # Check that memory has changed
        final_memory = resonance_chamber.standing_wave_memory
        memory_change = tf.reduce_mean(tf.abs(final_memory - initial_memory))
        
        assert memory_change > 1e-4, f"Chamber memory not updating: change={memory_change}"
        
        # Reset chamber and verify memory is cleared
        resonance_chamber.reset_chamber()
        reset_memory = resonance_chamber.standing_wave_memory
        
        memory_after_reset = tf.reduce_mean(tf.abs(reset_memory))
        assert memory_after_reset < 1e-6, f"Chamber memory not reset properly: {memory_after_reset}"
    
    def test_standing_wave_node_antinode_patterns(self, resonance_chamber, sample_waveform):
        """Test that standing waves show proper node and antinode patterns"""
        output = resonance_chamber(sample_waveform)
        analysis = resonance_chamber.get_standing_wave_analysis(sample_waveform)
        
        node_count = analysis['node_count']
        antinode_count = analysis['antinode_count']
        swr = analysis['standing_wave_ratio']
        
        # Should have some nodes and antinodes for standing wave pattern
        mean_nodes = tf.reduce_mean(node_count)
        mean_antinodes = tf.reduce_mean(antinode_count)
        
        assert mean_nodes > 1, f"Too few nodes detected: {mean_nodes}"
        assert mean_antinodes > 1, f"Too few antinodes detected: {mean_antinodes}"
        
        # Standing wave ratio should be > 1 (indicating standing wave presence)
        mean_swr = tf.reduce_mean(swr)
        assert mean_swr > 1.1, f"Standing wave ratio too low: {mean_swr}"
        
        # SWR shouldn't be extremely high (would indicate numerical issues)
        assert mean_swr < 100, f"Standing wave ratio too high: {mean_swr}"
    
    def test_mode_coupling_effects(self, resonance_chamber, sample_waveform):
        """Test that mode coupling creates nonlinear interactions"""
        # Apply chamber multiple times to see coupling effects
        output1 = resonance_chamber(sample_waveform)
        
        # Use output as new input (feedback creates coupling)
        output2 = resonance_chamber(output1 * 0.5 + sample_waveform * 0.5)
        
        # Analyze mode content
        analysis1 = resonance_chamber.get_standing_wave_analysis(sample_waveform)
        analysis2 = resonance_chamber.get_standing_wave_analysis(output1 * 0.5 + sample_waveform * 0.5)
        
        modes1 = analysis1['mode_amplitudes']
        modes2 = analysis2['mode_amplitudes']
        
        # Mode distribution should change due to coupling
        mode_distribution_change = tf.reduce_mean(tf.abs(modes2 - modes1))
        assert mode_distribution_change > 0.01, f"Mode coupling has minimal effect: {mode_distribution_change}"
        
        # Should see energy transfer between modes
        total_energy1 = tf.reduce_sum(modes1**2, axis=1)
        total_energy2 = tf.reduce_sum(modes2**2, axis=1)
        
        # Total energy change should be reasonable (not too extreme)
        energy_ratio = total_energy2 / (total_energy1 + 1e-8)
        assert tf.reduce_all(energy_ratio > 0.3), "Too much energy loss in coupling"
        assert tf.reduce_all(energy_ratio < 3.0), "Too much energy gain in coupling"
    
    def test_frequency_dependent_damping(self, resonance_chamber, sample_waveform):
        """Test that frequency-dependent damping works correctly"""
        output = resonance_chamber(sample_waveform)
        
        # Analyze frequency content
        fft_input = tf.signal.fft(sample_waveform)
        fft_output = tf.signal.fft(output)
        
        power_input = tf.abs(fft_input) ** 2
        power_output = tf.abs(fft_output) ** 2
        
        # Calculate power ratio as function of frequency
        power_ratio = power_output / (power_input + 1e-8)
        
        # Higher frequencies should generally be more damped
        # (This is a simplified test - actual behavior depends on mode structure)
        low_freq_power = tf.reduce_mean(power_ratio[:, :16])  # First 16 frequencies
        high_freq_power = tf.reduce_mean(power_ratio[:, -16:])  # Last 16 frequencies
        
        # Allow for some variation, but generally expect more damping at high freq
        damping_ratio = high_freq_power / (low_freq_power + 1e-8)
        assert tf.reduce_mean(damping_ratio) < 2.0, "High frequencies not sufficiently damped"
    
    @pytest.mark.parametrize("num_modes", [4, 8, 16, 32])
    def test_mode_count_effects(self, num_modes):
        """Test effect of different numbers of modes"""
        sample_input = tf.complex(
            tf.random.normal([2, 64]),
            tf.random.normal([2, 64]) * 0.1
        )
        
        chamber = ResonanceChamber(
            dimensions=64,
            num_modes=num_modes
        )
        
        output = chamber(sample_input)
        analysis = chamber.get_standing_wave_analysis(sample_input)
        
        # Should have the requested number of modes
        mode_amplitudes = analysis['mode_amplitudes']
        assert mode_amplitudes.shape[-1] == num_modes, f"Wrong number of modes: expected {num_modes}, got {mode_amplitudes.shape[-1]}"
        
        # Output should be reasonable regardless of mode count
        assert tf.reduce_all(tf.math.is_finite(tf.real(output))), f"Non-finite output with {num_modes} modes"
        assert tf.reduce_all(tf.math.is_finite(tf.imag(output))), f"Non-finite output with {num_modes} modes"
        
        # More modes should generally allow for richer patterns
        output_complexity = tf.math.reduce_std(tf.abs(output))
        assert output_complexity > 0.01, f"Output too uniform with {num_modes} modes"
    
    def test_chamber_length_scaling(self):
        """Test that chamber length affects resonant frequencies appropriately"""
        sample_input = tf.complex(
            tf.random.normal([2, 64]),
            tf.random.normal([2, 64]) * 0.1
        )
        
        short_chamber = ResonanceChamber(
            dimensions=64,
            chamber_length=0.5,
            num_modes=8
        )
        
        long_chamber = ResonanceChamber(
            dimensions=64,
            chamber_length=2.0,
            num_modes=8
        )
        
        # Resonant frequencies should scale inversely with length
        short_freqs = short_chamber.resonant_frequencies
        long_freqs = long_chamber.resonant_frequencies
        
        # For most boundary conditions, f ∝ 1/L
        expected_ratio = 2.0 / 0.5  # long_length / short_length
        actual_ratio = tf.reduce_mean(short_freqs) / tf.reduce_mean(long_freqs)
        
        # Allow some tolerance for discretization effects
        assert 2.0 < actual_ratio < 6.0, f"Frequency scaling incorrect: expected ~4, got {actual_ratio}"