"""Tests for chromatic mixing verification against theoretical colour dynamics"""

import pytest
import tensorflow as tf
import numpy as np
from ..layers.chromatic_resonance import ChromaticResonance
from ..core.colour_utils import colour_distance, colour_interpolation, prevent_collapse


class TestChromaticMixing:
    """Test suite for chromatic colour mixing dynamics"""
    
    @pytest.fixture
    def chromatic_resonance(self):
        """Create ChromaticResonance layer for testing"""
        return ChromaticResonance(
            dimensions=64,
            resonance_depth=5,
            harmonic_orders=[1, 2, 3],
            nonlinearity='tanh'
        )
    
    @pytest.fixture
    def colour_waveforms(self):
        """Create test colour waveforms"""
        batch_size, dimensions = 4, 64
        
        # Create "primary" colours in frequency space
        red_freq = 1.0
        green_freq = 2.0
        blue_freq = 3.0
        
        t = tf.linspace(0.0, 2*np.pi, dimensions)
        
        # Red: Low frequency, high amplitude
        red_colour = tf.complex(
            tf.sin(red_freq * t) * tf.ones([batch_size, dimensions]),
            tf.cos(red_freq * t) * tf.ones([batch_size, dimensions]) * 0.1
        )
        
        # Green: Medium frequency, medium amplitude
        green_colour = tf.complex(
            tf.sin(green_freq * t) * tf.ones([batch_size, dimensions]) * 0.7,
            tf.cos(green_freq * t) * tf.ones([batch_size, dimensions]) * 0.2
        )
        
        # Blue: High frequency, varying amplitude
        blue_colour = tf.complex(
            tf.sin(blue_freq * t) * tf.ones([batch_size, dimensions]) * 0.5,
            tf.cos(blue_freq * t) * tf.ones([batch_size, dimensions]) * 0.3
        )
        
        return red_colour, green_colour, blue_colour
    
    def test_colour_mixing_commutativity(self, chromatic_resonance, colour_waveforms):
        """Test that colour mixing is approximately commutative"""
        red, green, _ = colour_waveforms
        
        # Build layer
        _ = chromatic_resonance(red)
        
        # Mix red then green
        red_output = chromatic_resonance(red)
        red_green_output = chromatic_resonance([red_output, green])
        
        # Mix green then red
        green_output = chromatic_resonance(green)
        green_red_output = chromatic_resonance([green_output, red])
        
        # Should be approximately commutative
        mixing_difference = colour_distance(red_green_output, green_red_output, metric='spectral')
        max_difference = tf.reduce_max(mixing_difference)
        
        # Allow some tolerance due to nonlinear effects
        assert max_difference < 1.0, f"Colour mixing not commutative: max_diff={max_difference}"
    
    def test_colour_mixing_additivity(self, chromatic_resonance, colour_waveforms):
        """Test additive properties of colour mixing"""
        red, green, blue = colour_waveforms
        
        # Build layer
        _ = chromatic_resonance(red)
        
        # Mix pairs
        red_green = chromatic_resonance([red, green])
        green_blue = chromatic_resonance([green, blue])
        red_blue = chromatic_resonance([red, blue])
        
        # Mix all three
        all_three = chromatic_resonance([red, green, blue])
        
        # The three-way mix should contain aspects of all pairwise mixes
        # (This is a qualitative test - exact additivity may not hold due to nonlinearity)
        
        dist_to_rg = colour_distance(all_three, red_green, metric='spectral')
        dist_to_gb = colour_distance(all_three, green_blue, metric='spectral')
        dist_to_rb = colour_distance(all_three, red_blue, metric='spectral')
        
        # Three-way mix should be "between" pairwise mixes
        min_pairwise_dist = tf.minimum(tf.minimum(dist_to_rg, dist_to_gb), dist_to_rb)
        max_pairwise_dist = tf.maximum(tf.maximum(dist_to_rg, dist_to_gb), dist_to_rb)
        
        # Some relationship should exist
        mean_min_dist = tf.reduce_mean(min_pairwise_dist)
        mean_max_dist = tf.reduce_mean(max_pairwise_dist)
        
        assert mean_min_dist < mean_max_dist, "Colour mixing lacks expected relationships"
    
    def test_harmonic_generation(self, chromatic_resonance, colour_waveforms):
        """Test that harmonic generation creates expected frequency content"""
        red, _, _ = colour_waveforms
        
        # Build layer
        _ = chromatic_resonance(red)
        
        # Apply resonance to generate harmonics
        resonated = chromatic_resonance(red)
        
        # Analyze frequency content
        fft_input = tf.signal.fft(red)
        fft_output = tf.signal.fft(resonated)
        
        power_input = tf.abs(fft_input) ** 2
        power_output = tf.abs(fft_output) ** 2
        
        # Output should have richer harmonic content
        # Calculate spectral centroid (center of mass of spectrum)
        frequencies = tf.range(tf.shape(power_input)[-1], dtype=tf.float32)
        
        centroid_input = tf.reduce_sum(power_input * frequencies, axis=-1) / (tf.reduce_sum(power_input, axis=-1) + 1e-8)
        centroid_output = tf.reduce_sum(power_output * frequencies, axis=-1) / (tf.reduce_sum(power_output, axis=-1) + 1e-8)
        
        # Calculate spectral spread (width of spectrum)
        spread_input = tf.sqrt(
            tf.reduce_sum(power_input * tf.square(frequencies - tf.expand_dims(centroid_input, -1)), axis=-1) /
            (tf.reduce_sum(power_input, axis=-1) + 1e-8)
        )
        spread_output = tf.sqrt(
            tf.reduce_sum(power_output * tf.square(frequencies - tf.expand_dims(centroid_output, -1)), axis=-1) /
            (tf.reduce_sum(power_output, axis=-1) + 1e-8)
        )
        
        # Output should generally have broader spectrum due to harmonics
        mean_spread_increase = tf.reduce_mean(spread_output - spread_input)
        assert mean_spread_increase > 0, f"Harmonic generation not broadening spectrum: {mean_spread_increase}"
    
    def test_resonance_memory_effects(self, chromatic_resonance, colour_waveforms):
        """Test that resonance memory affects colour mixing"""
        red, green, _ = colour_waveforms
        
        # Build layer
        _ = chromatic_resonance(red, training=True)
        
        # Get initial memory state
        initial_memory = tf.identity(chromatic_resonance.resonance_memory)
        
        # Apply red colour multiple times to build up memory
        for _ in range(5):
            _ = chromatic_resonance(red, training=True)
        
        red_memory = tf.identity(chromatic_resonance.resonance_memory)
        
        # Now apply green - should be affected by red memory
        green_with_red_memory = chromatic_resonance(green, training=False)
        
        # Reset memory and apply green again
        chromatic_resonance.reset_memory()
        green_without_memory = chromatic_resonance(green, training=False)
        
        # The two green outputs should be different due to memory effects
        memory_effect = colour_distance(green_with_red_memory, green_without_memory, metric='spectral')
        mean_effect = tf.reduce_mean(memory_effect)
        
        assert mean_effect > 0.1, f"Resonance memory has minimal effect: {mean_effect}"
    
    def test_nonlinear_colour_interactions(self, chromatic_resonance):
        """Test nonlinear interactions between colours"""
        batch_size, dimensions = 2, 64
        
        # Create complementary colours (opposite phases)
        base_pattern = tf.sin(tf.linspace(0.0, 4*np.pi, dimensions))
        
        colour1 = tf.complex(
            tf.expand_dims(base_pattern, 0) * tf.ones([batch_size, dimensions]),
            tf.expand_dims(base_pattern, 0) * tf.ones([batch_size, dimensions]) * 0.1
        )
        
        colour2 = tf.complex(
            tf.expand_dims(-base_pattern, 0) * tf.ones([batch_size, dimensions]),  # Opposite phase
            tf.expand_dims(-base_pattern, 0) * tf.ones([batch_size, dimensions]) * 0.1
        )
        
        # Build layer
        _ = chromatic_resonance(colour1)
        
        # Mix complementary colours
        mixed = chromatic_resonance([colour1, colour2])
        
        # Should not cancel completely due to nonlinear effects
        mixed_power = tf.reduce_sum(tf.abs(mixed) ** 2, axis=-1)
        input_power = tf.reduce_sum(tf.abs(colour1) ** 2, axis=-1)
        
        power_ratio = mixed_power / (input_power + 1e-8)
        
        # Some power should remain due to nonlinear mixing
        assert tf.reduce_all(power_ratio > 0.1), "Complementary colours cancelled too completely"
        
        # But shouldn't be too large either
        assert tf.reduce_all(power_ratio < 2.0), "Nonlinear mixing creating too much power"
    
    def test_colour_interpolation_consistency(self, colour_waveforms):
        """Test colour interpolation methods for consistency"""
        red, green, blue = colour_waveforms
        
        methods = ['linear', 'spherical', 'frequency']
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for method in methods:
            interpolated_colours = []
            
            for alpha in alphas:
                interp_colour = colour_interpolation(red, green, alpha, method=method)
                interpolated_colours.append(interp_colour)
            
            # Check boundary conditions
            start_diff = colour_distance(interpolated_colours[0], red, metric='spectral')
            end_diff = colour_distance(interpolated_colours[-1], green, metric='spectral')
            
            assert tf.reduce_max(start_diff) < 0.1, f"Interpolation start point wrong for {method}"
            assert tf.reduce_max(end_diff) < 0.1, f"Interpolation end point wrong for {method}"
            
            # Check monotonicity (distance to red should generally increase)
            distances_to_red = [
                colour_distance(interp, red, metric='spectral') 
                for interp in interpolated_colours
            ]
            
            # Should be generally increasing (allowing some tolerance)
            for i in range(len(distances_to_red) - 1):
                current_dist = tf.reduce_mean(distances_to_red[i])
                next_dist = tf.reduce_mean(distances_to_red[i + 1])
                
                # Allow some deviation but general trend should be increasing
                assert next_dist >= current_dist - 0.2, f"Interpolation not monotonic for {method} at step {i}"
    
    def test_colour_mixing_conservation_laws(self, chromatic_resonance, colour_waveforms):
        """Test conservation properties in colour mixing"""
        red, green, _ = colour_waveforms
        
        # Build layer
        _ = chromatic_resonance(red)
        
        # Calculate input properties
        red_power = tf.reduce_sum(tf.abs(red) ** 2, axis=-1)
        green_power = tf.reduce_sum(tf.abs(green) ** 2, axis=-1)
        total_input_power = red_power + green_power
        
        red_centroid = self._calculate_spectral_centroid(red)
        green_centroid = self._calculate_spectral_centroid(green)
        
        # Mix colours
        mixed = chromatic_resonance([red, green])
        
        # Calculate output properties
        mixed_power = tf.reduce_sum(tf.abs(mixed) ** 2, axis=-1)
        mixed_centroid = self._calculate_spectral_centroid(mixed)
        
        # Power conservation (allowing for nonlinear effects)
        power_ratio = mixed_power / (total_input_power + 1e-8)
        assert tf.reduce_all(power_ratio > 0.3), "Too much power lost in mixing"
        assert tf.reduce_all(power_ratio < 3.0), "Too much power gained in mixing"
        
        # Spectral centroid should be influenced by both inputs
        min_input_centroid = tf.minimum(red_centroid, green_centroid)
        max_input_centroid = tf.maximum(red_centroid, green_centroid)
        
        # Mixed centroid should generally be between input centroids (allowing some tolerance)
        centroid_in_range = tf.logical_and(
            mixed_centroid >= min_input_centroid - 5.0,
            mixed_centroid <= max_input_centroid + 5.0
        )
        
        assert tf.reduce_mean(tf.cast(centroid_in_range, tf.float32)) > 0.7, \
            "Mixed centroid outside expected range too often"
    
    def test_colour_mixing_with_different_amplitudes(self, chromatic_resonance):
        """Test colour mixing with very different amplitude scales"""
        batch_size, dimensions = 2, 64
        
        # Create colours with very different amplitudes
        base_pattern = tf.sin(tf.linspace(0.0, 2*np.pi, dimensions))
        
        weak_colour = tf.complex(
            tf.expand_dims(base_pattern, 0) * tf.ones([batch_size, dimensions]) * 0.01,  # Very weak
            tf.expand_dims(base_pattern, 0) * tf.ones([batch_size, dimensions]) * 0.001
        )
        
        strong_colour = tf.complex(
            tf.expand_dims(base_pattern * 2, 0) * tf.ones([batch_size, dimensions]) * 2.0,  # Very strong
            tf.expand_dims(base_pattern * 2, 0) * tf.ones([batch_size, dimensions]) * 0.2
        )
        
        # Build layer
        _ = chromatic_resonance(weak_colour)
        
        # Mix colours with different amplitudes
        mixed = chromatic_resonance([weak_colour, strong_colour])
        
        # Mixed result should not be dominated entirely by strong colour
        weak_power = tf.reduce_sum(tf.abs(weak_colour) ** 2, axis=-1)
        strong_power = tf.reduce_sum(tf.abs(strong_colour) ** 2, axis=-1)
        mixed_power = tf.reduce_sum(tf.abs(mixed) ** 2, axis=-1)
        
        # Check that weak colour had some influence
        power_from_weak = mixed_power - strong_power
        weak_influence_ratio = power_from_weak / (weak_power + 1e-8)
        
        # Should have some influence (nonlinear mixing can amplify weak signals)
        assert tf.reduce_mean(weak_influence_ratio) > -0.5, "Weak colour has no influence in mixing"
    
    def test_temporal_colour_evolution(self, chromatic_resonance, colour_waveforms):
        """Test evolution of colours over multiple mixing steps"""
        red, green, blue = colour_waveforms
        
        # Build layer
        _ = chromatic_resonance(red, training=True)
        
        # Evolve colour through multiple steps
        current_colour = red
        evolution_history = [current_colour]
        
        for step in range(10):
            # Mix current colour with green (driving force)
            mixed = chromatic_resonance([current_colour, green * 0.1], training=True)
            
            # Evolve (feedback with some fresh input)
            current_colour = mixed * 0.9 + red * 0.1
            evolution_history.append(current_colour)
        
        # Analyze evolution trajectory
        distances_from_start = [
            colour_distance(colour, red, metric='spectral')
            for colour in evolution_history
        ]
        
        # Should show some evolution away from starting point
        final_distance = tf.reduce_mean(distances_from_start[-1])
        initial_distance = tf.reduce_mean(distances_from_start[0])  # Should be 0
        
        assert final_distance > 0.1, f"Insufficient colour evolution: {final_distance}"
        
        # Evolution should be gradual (not chaotic)
        step_changes = [
            tf.reduce_mean(colour_distance(evolution_history[i+1], evolution_history[i], metric='spectral'))
            for i in range(len(evolution_history) - 1)
        ]
        
        max_step_change = max(step_changes)
        assert max_step_change < 1.0, f"Evolution too chaotic: max_step_change={max_step_change}"
    
    def _calculate_spectral_centroid(self, waveform):
        """Calculate spectral centroid of waveform"""
        power_spectrum = tf.abs(tf.signal.fft(waveform)) ** 2
        frequencies = tf.range(tf.shape(power_spectrum)[-1], dtype=tf.float32)
        
        centroid = tf.reduce_sum(power_spectrum * frequencies, axis=-1) / (
            tf.reduce_sum(power_spectrum, axis=-1) + 1e-8
        )
        
        return centroid
    
    @pytest.mark.parametrize("resonance_depth", [3, 5, 7, 10])
    def test_resonance_depth_effects(self, resonance_depth, colour_waveforms):
        """Test effect of different resonance depths on mixing"""
        red, green, _ = colour_waveforms
        
        layer = ChromaticResonance(
            dimensions=64,
            resonance_depth=resonance_depth,
            harmonic_orders=[1, 2, 3]
        )
        
        # Build and apply
        mixed = layer([red, green])
        
        # Deeper resonance should create more complex patterns
        complexity = tf.math.reduce_std(tf.abs(mixed), axis=-1)
        mean_complexity = tf.reduce_mean(complexity)
        
        # Should have reasonable complexity
        assert mean_complexity > 0.01, f"Insufficient complexity with depth {resonance_depth}"
        
        # Shouldn't be completely chaotic either
        assert mean_complexity < 2.0, f"Excessive complexity with depth {resonance_depth}"
    
    def test_harmonic_order_effects(self, colour_waveforms):
        """Test effect of different harmonic orders"""
        red, green, _ = colour_waveforms
        
        # Test different harmonic configurations
        harmonic_configs = [
            [1],              # Fundamental only
            [1, 2],           # Fundamental + 2nd harmonic
            [1, 2, 3],        # Up to 3rd harmonic
            [1, 2, 3, 5, 7]   # Multiple harmonics including non-integer ratios
        ]
        
        results = {}
        
        for harmonics in harmonic_configs:
            layer = ChromaticResonance(
                dimensions=64,
                resonance_depth=5,
                harmonic_orders=harmonics
            )
            
            mixed = layer([red, green])
            results[str(harmonics)] = mixed
            
            # Check that result is reasonable
            assert tf.reduce_all(tf.math.is_finite(tf.real(mixed))), f"Non-finite result with harmonics {harmonics}"
            assert tf.reduce_all(tf.math.is_finite(tf.imag(mixed))), f"Non-finite result with harmonics {harmonics}"
        
        # Different harmonic orders should produce different results
        configs = list(results.keys())
        for i in range(len(configs)):
            for j in range(i + 1, len(configs)):
                diff = colour_distance(results[configs[i]], results[configs[j]], metric='spectral')
                mean_diff = tf.reduce_mean(diff)
                
                assert mean_diff > 0.05, f"Harmonic configs {configs[i]} and {configs[j]} too similar: {mean_diff}"