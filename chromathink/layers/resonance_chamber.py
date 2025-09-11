"""ResonanceChamber creating standing wave patterns through reflection and frequency-dependent damping"""

import tensorflow as tf
import numpy as np
from ..core.colour_utils import prevent_collapse
from .chromatic_resonance import ChromaticResonance


class ResonanceChamber(tf.keras.layers.Layer):
    """
    Creates standing wave patterns through wave reflection and interference.
    Models a cognitive resonance chamber where thoughts establish stable patterns
    through multiple reflections and frequency-dependent damping.
    """
    
    def __init__(self,
                 dimensions,
                 chamber_length=1.0,
                 reflection_coefficients=None,
                 boundary_conditions='mixed',  # 'rigid', 'free', 'mixed', 'absorbing'
                 num_modes=None,
                 damping_factor=0.01,
                 excitation_frequency_range=(1.0, 100.0),
                 quality_factor=100.0,
                 **kwargs):
        """
        Initialize ResonanceChamber
        
        Args:
            dimensions: Number of frequency dimensions
            chamber_length: Virtual length of resonance chamber
            reflection_coefficients: Reflection coefficients for boundaries
            boundary_conditions: Type of boundary conditions
            num_modes: Number of resonant modes to model
            damping_factor: Base damping factor
            excitation_frequency_range: Range of excitation frequencies
            quality_factor: Quality factor for resonances
        """
        super().__init__(**kwargs)
        
        self.dimensions = dimensions
        self.chamber_length = chamber_length
        self.boundary_conditions = boundary_conditions
        self.num_modes = num_modes or dimensions // 4
        self.damping_factor = damping_factor
        self.excitation_frequency_range = excitation_frequency_range
        self.quality_factor = quality_factor
        
        # Set default reflection coefficients based on boundary conditions
        if reflection_coefficients is None:
            if boundary_conditions == 'rigid':
                self.reflection_coeffs = [1.0, 1.0]  # Both ends reflect with same phase
            elif boundary_conditions == 'free':
                self.reflection_coeffs = [-1.0, -1.0]  # Both ends reflect with phase flip
            elif boundary_conditions == 'mixed':
                self.reflection_coeffs = [1.0, -1.0]  # One rigid, one free
            elif boundary_conditions == 'absorbing':
                self.reflection_coeffs = [0.8, 0.8]  # Partial absorption
            else:
                self.reflection_coeffs = [0.9, 0.9]
        else:
            self.reflection_coeffs = reflection_coefficients
        
        # Create chromatic resonance layer for nonlinear effects
        self.chromatic_resonance = ChromaticResonance(
            dimensions=dimensions,
            resonance_depth=5,
            harmonic_orders=[1, 2, 3]
        )
    
    def build(self, input_shape):
        super().build(input_shape)
        
        # Calculate resonant frequencies for the chamber
        self.resonant_frequencies = self._calculate_resonant_frequencies()
        
        # Mode shapes for standing wave patterns
        self.mode_shapes = self._calculate_mode_shapes()
        
        # Coupling strengths between modes
        self.mode_coupling = self.add_weight(
            name='mode_coupling',
            shape=(self.num_modes, self.num_modes),
            initializer='orthogonal',
            trainable=True
        )
        
        # Quality factors for each mode (learnable)
        self.mode_q_factors = self.add_weight(
            name='mode_q_factors',
            shape=(self.num_modes,),
            initializer=tf.initializers.Constant(self.quality_factor),
            constraint=tf.keras.constraints.NonNeg(),
            trainable=True
        )
        
        # Excitation weights (how strongly each input frequency excites each mode)
        self.excitation_matrix = self.add_weight(
            name='excitation_matrix',
            shape=(self.dimensions, self.num_modes),
            initializer='random_normal',
            trainable=True
        )
        
        # Reflection phase shifts (learnable)
        self.reflection_phases = self.add_weight(
            name='reflection_phases',
            shape=(2, self.num_modes),  # Two boundaries
            initializer='random_uniform',
            trainable=True
        )
        
        # Standing wave amplitude tracking
        self.standing_wave_memory = self.add_weight(
            name='standing_wave_memory',
            shape=(self.num_modes,),
            initializer='zeros',
            trainable=False
        )
    
    def _calculate_resonant_frequencies(self):
        """Calculate resonant frequencies for the chamber"""
        if self.boundary_conditions == 'rigid':
            # Both ends fixed: f_n = n * c / (2L)
            modes = tf.range(1, self.num_modes + 1, dtype=tf.float32)
            frequencies = modes / (2.0 * self.chamber_length)
        elif self.boundary_conditions == 'free':
            # Both ends free: f_n = n * c / (2L)  
            modes = tf.range(1, self.num_modes + 1, dtype=tf.float32)
            frequencies = modes / (2.0 * self.chamber_length)
        elif self.boundary_conditions == 'mixed':
            # One fixed, one free: f_n = (2n-1) * c / (4L)
            modes = tf.range(1, self.num_modes + 1, dtype=tf.float32)
            frequencies = (2.0 * modes - 1.0) / (4.0 * self.chamber_length)
        else:  # absorbing or custom
            # Slightly shifted frequencies due to absorption
            modes = tf.range(1, self.num_modes + 1, dtype=tf.float32)
            frequencies = modes / (2.0 * self.chamber_length)
            frequencies = frequencies * (1.0 + 0.1 * tf.random.normal([self.num_modes], stddev=0.01))
        
        # Scale to desired frequency range
        min_freq, max_freq = self.excitation_frequency_range
        frequencies = min_freq + frequencies * (max_freq - min_freq) / tf.reduce_max(frequencies)
        
        return frequencies
    
    def _calculate_mode_shapes(self):
        """Calculate spatial mode shapes for standing waves"""
        # Position along chamber (normalized to [0, 1])
        positions = tf.linspace(0.0, 1.0, self.dimensions)
        positions = tf.expand_dims(positions, 1)  # Shape: [dimensions, 1]
        
        # Mode numbers
        modes = tf.range(1, self.num_modes + 1, dtype=tf.float32)
        modes = tf.expand_dims(modes, 0)  # Shape: [1, num_modes]
        
        if self.boundary_conditions == 'rigid':
            # sin(n*pi*x/L) for both ends fixed
            mode_shapes = tf.sin(modes * np.pi * positions)
        elif self.boundary_conditions == 'free':
            # cos(n*pi*x/L) for both ends free
            mode_shapes = tf.cos(modes * np.pi * positions)
        elif self.boundary_conditions == 'mixed':
            # sin((2n-1)*pi*x/(2L)) for mixed boundary
            mode_shapes = tf.sin((2.0 * modes - 1.0) * np.pi * positions / 2.0)
        else:
            # Default to sine modes with small random perturbations
            mode_shapes = tf.sin(modes * np.pi * positions)
            perturbation = 0.1 * tf.random.normal([self.dimensions, self.num_modes])
            mode_shapes = mode_shapes + perturbation
        
        # Normalize mode shapes
        mode_shapes = tf.nn.l2_normalize(mode_shapes, axis=0)
        
        return mode_shapes
    
    def call(self, inputs, training=None):
        """
        Create standing wave patterns in the resonance chamber
        
        Args:
            inputs: Complex waveform input or list of waveform inputs
            training: Training mode flag
            
        Returns:
            Standing wave pattern output
        """
        
        # Handle multiple inputs by combining them
        if isinstance(inputs, (list, tuple)):
            # Ensure all inputs have compatible shapes
            processed_inputs = []
            target_ndim = max(len(inp.shape) for inp in inputs)
            
            for inp in inputs:
                # Expand dimensions to match the highest-dimensional input
                while len(inp.shape) < target_ndim:
                    inp = tf.expand_dims(inp, axis=1)
                processed_inputs.append(inp)
            
            # Stack inputs along a new axis and then sum them for interference
            stacked_inputs = tf.stack(processed_inputs, axis=0)
            combined_input = tf.reduce_sum(stacked_inputs, axis=0)
        else:
            combined_input = inputs
        
        # Extract amplitude and phase from input
        input_amplitude = tf.math.abs(combined_input)
        input_phase = tf.math.angle(combined_input)
        
        # Calculate excitation of each mode
        mode_excitations = tf.matmul(input_amplitude, self.excitation_matrix)
        
        # Apply resonant frequency response
        frequency_response = self._calculate_frequency_response(combined_input)
        excited_modes = mode_excitations * frequency_response
        
        # Create standing wave patterns
        standing_waves = self._create_standing_waves(excited_modes, input_phase)
        
        # Apply mode coupling for nonlinear interactions
        coupled_waves = self._apply_mode_coupling(standing_waves)
        
        # Add reflection effects
        reflected_waves = self._apply_reflections(coupled_waves)
        
        # Apply chromatic resonance for richer harmonics
        resonated_waves = self.chromatic_resonance(reflected_waves)
        
        # Update standing wave memory
        if training:
            current_amplitudes = tf.reduce_mean(tf.math.abs(reflected_waves), axis=0)
            new_memory = 0.9 * self.standing_wave_memory + 0.1 * current_amplitudes
            self.standing_wave_memory.assign(new_memory)
        
        # Apply frequency-dependent damping
        damped_output = self._apply_chamber_damping(resonated_waves)
        
        # Prevent collapse during training
        if training:
            collapse_loss = prevent_collapse(
                damped_output, method='diversity_loss', strength=0.05
            )
            self.add_loss(collapse_loss)
        
        return damped_output
    
    def _calculate_frequency_response(self, inputs):
        """Calculate frequency response of resonant modes"""
        input_frequencies = tf.linspace(
            self.excitation_frequency_range[0],
            self.excitation_frequency_range[1], 
            self.dimensions
        )
        
        # Calculate response for each mode
        responses = []
        for i in range(self.num_modes):
            resonant_freq = self.resonant_frequencies[i]
            q_factor = self.mode_q_factors[i]
            
            # Lorentzian response curve
            frequency_diff = input_frequencies - resonant_freq
            denominator = 1.0 + tf.square(2.0 * q_factor * frequency_diff / resonant_freq)
            response = 1.0 / denominator
            
            responses.append(response)
        
        # Stack and transpose to match excitation matrix
        return tf.transpose(tf.stack(responses))
    
    def _create_standing_waves(self, mode_amplitudes, input_phase):
        """Create standing wave patterns from mode excitations"""
        batch_size = tf.shape(mode_amplitudes)[0]
        
        # Expand mode shapes for batch processing
        expanded_modes = tf.expand_dims(self.mode_shapes, 0)
        expanded_modes = tf.tile(expanded_modes, [batch_size, 1, 1])
        
        # Expand mode amplitudes
        expanded_amplitudes = tf.expand_dims(mode_amplitudes, 1)
        
        # Create standing wave amplitudes
        standing_amplitudes = tf.reduce_sum(
            expanded_modes * expanded_amplitudes, axis=-1
        )
        
        # Combine with input phase information
        # Use average phase across frequencies for each mode
        mode_phases = tf.matmul(input_phase, self.excitation_matrix / tf.reduce_sum(self.excitation_matrix, axis=0, keepdims=True))
        
        # Create phase patterns
        expanded_phases = tf.expand_dims(mode_phases, 1)
        standing_phases = tf.reduce_sum(
            expanded_modes * expanded_phases, axis=-1
        )
        
        # Combine into complex standing wave
        standing_waves = tf.cast(standing_amplitudes, tf.complex64) * tf.exp(tf.complex(0.0, standing_phases))
        
        return standing_waves
    
    def _apply_mode_coupling(self, standing_waves):
        """Apply coupling between different modes"""
        # Convert to real for matrix multiplication
        real_part = tf.math.real(standing_waves)
        imag_part = tf.math.imag(standing_waves)
        
        # Project to mode space
        mode_real = tf.matmul(real_part, self.mode_shapes)
        mode_imag = tf.matmul(imag_part, self.mode_shapes)
        
        # Apply coupling matrix
        coupled_real = tf.matmul(mode_real, self.mode_coupling)
        coupled_imag = tf.matmul(mode_imag, self.mode_coupling)
        
        # Project back to spatial domain
        spatial_real = tf.matmul(coupled_real, self.mode_shapes, transpose_b=True)
        spatial_imag = tf.matmul(coupled_imag, self.mode_shapes, transpose_b=True)
        
        return tf.complex(spatial_real, spatial_imag)
    
    def _apply_reflections(self, waves):
        """Apply boundary reflections with phase shifts"""
        real_part = tf.math.real(waves)
        imag_part = tf.math.imag(waves)
        
        # Project to mode space for reflection processing
        mode_real = tf.matmul(real_part, self.mode_shapes)
        mode_imag = tf.matmul(imag_part, self.mode_shapes)
        
        # Apply reflection coefficients and phase shifts
        reflected_real = mode_real
        reflected_imag = mode_imag
        
        for boundary in range(2):  # Two boundaries
            refl_coeff = self.reflection_coeffs[boundary]
            phase_shifts = self.reflection_phases[boundary]
            
            # Apply reflection coefficient
            reflected_real = reflected_real * refl_coeff
            reflected_imag = reflected_imag * refl_coeff
            
            # Apply phase shift
            cos_phase = tf.cos(phase_shifts)
            sin_phase = tf.sin(phase_shifts)
            
            new_real = reflected_real * cos_phase - reflected_imag * sin_phase
            new_imag = reflected_real * sin_phase + reflected_imag * cos_phase
            
            reflected_real = new_real
            reflected_imag = new_imag
        
        # Project back to spatial domain
        spatial_real = tf.matmul(reflected_real, self.mode_shapes, transpose_b=True)
        spatial_imag = tf.matmul(reflected_imag, self.mode_shapes, transpose_b=True)
        
        return tf.complex(spatial_real, spatial_imag)
    
    def _apply_chamber_damping(self, waves):
        """Apply frequency-dependent damping"""
        # Higher frequency modes are damped more
        frequency_weights = tf.exp(-tf.linspace(0.0, 3.0, self.dimensions) * self.damping_factor)
        
        # Apply damping
        damping_tensor = tf.cast(frequency_weights, waves.dtype)
        
        return waves * damping_tensor
    
    def get_standing_wave_analysis(self, inputs):
        """
        Analyze standing wave characteristics
        
        Args:
            inputs: Input waveforms
            
        Returns:
            Dictionary with standing wave metrics
        """
        output = self.call(inputs, training=False)
        
        # Modal decomposition
        output_real = tf.math.real(output)
        output_imag = tf.math.imag(output)
        
        mode_amplitudes_real = tf.matmul(output_real, self.mode_shapes)
        mode_amplitudes_imag = tf.matmul(output_imag, self.mode_shapes)
        mode_amplitudes = tf.sqrt(mode_amplitudes_real**2 + mode_amplitudes_imag**2)
        
        # Node locations (zeros in standing wave)
        amplitude_magnitude = tf.math.abs(output)
        threshold = 0.1 * tf.reduce_max(amplitude_magnitude, axis=-1, keepdims=True)
        nodes = tf.cast(amplitude_magnitude < threshold, tf.float32)
        node_count = tf.reduce_sum(nodes, axis=-1)
        
        # Antinode locations (maxima in standing wave)
        local_maxima = self._find_local_maxima(amplitude_magnitude)
        antinode_count = tf.reduce_sum(local_maxima, axis=-1)
        
        # Standing wave ratio (SWR)
        max_amplitude = tf.reduce_max(amplitude_magnitude, axis=-1)
        min_amplitude = tf.reduce_min(amplitude_magnitude + 1e-8, axis=-1)
        swr = max_amplitude / min_amplitude
        
        return {
            'mode_amplitudes': mode_amplitudes,
            'node_count': node_count,
            'antinode_count': antinode_count,
            'standing_wave_ratio': swr,
            'resonant_frequencies': self.resonant_frequencies,
            'chamber_memory': self.standing_wave_memory
        }
    
    def _find_local_maxima(self, signal):
        """Find local maxima in 1D signal"""
        # Simple local maximum detection
        padded = tf.pad(signal, [[0, 0], [1, 1]], mode='REFLECT')
        left_neighbor = padded[:, :-2]
        right_neighbor = padded[:, 2:]
        center = padded[:, 1:-1]
        
        is_maximum = tf.logical_and(center > left_neighbor, center > right_neighbor)
        return tf.cast(is_maximum, tf.float32)
    
    def reset_chamber(self):
        """Reset chamber memory and standing wave patterns"""
        self.standing_wave_memory.assign(tf.zeros_like(self.standing_wave_memory))
        if hasattr(self.chromatic_resonance, 'reset_memory'):
            self.chromatic_resonance.reset_memory()
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dimensions': self.dimensions,
            'chamber_length': self.chamber_length,
            'reflection_coefficients': self.reflection_coeffs,
            'boundary_conditions': self.boundary_conditions,
            'num_modes': self.num_modes,
            'damping_factor': self.damping_factor,
            'excitation_frequency_range': self.excitation_frequency_range,
            'quality_factor': self.quality_factor
        })
        return config