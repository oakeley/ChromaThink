"""ChromaticResonance layers for nonlinear colour mixing with learned interference patterns"""

import tensorflow as tf
import numpy as np
from ..core.colour_utils import prevent_collapse, colour_distance
from .interference import InterferenceLayer


class ChromaticResonance(tf.keras.layers.Layer):
    """
    Implements nonlinear colour mixing through resonance patterns.
    Thoughts interact through interference creating complex chromatic harmonics.
    """
    
    def __init__(self,
                 dimensions,
                 resonance_depth=7,
                 coupling_strength=1.0,
                 damping_profile='frequency_dependent',
                 nonlinearity='tanh',
                 memory_decay=0.9,
                 harmonic_orders=[1, 2, 3, 5],
                 **kwargs):
        """
        Initialize ChromaticResonance layer
        
        Args:
            dimensions: Number of colour frequency dimensions
            resonance_depth: Number of resonance iterations
            coupling_strength: Strength of wave coupling
            damping_profile: Type of damping ('uniform', 'frequency_dependent', 'learnable')
            nonlinearity: Nonlinear activation function
            memory_decay: Decay factor for resonance memory
            harmonic_orders: Harmonic orders to generate
        """
        super().__init__(**kwargs)
        
        self.dimensions = dimensions
        self.resonance_depth = resonance_depth
        self.coupling_strength = coupling_strength
        self.damping_profile = damping_profile
        self.nonlinearity = nonlinearity
        self.memory_decay = memory_decay
        self.harmonic_orders = harmonic_orders
        
        # Create interference layer for wave interactions
        self.interference_layer = InterferenceLayer(
            interference_type='full',
            amplitude_weighting=True,
            phase_coupling=True,
            nonlinear_mixing=True
        )
    
    def build(self, input_shape):
        super().build(input_shape)
        
        # Coupling matrix for wave transformations
        self.coupling_matrix = self.add_weight(
            name='coupling_matrix',
            shape=(self.dimensions, self.dimensions),
            initializer='orthogonal',
            trainable=True
        )
        
        # Damping profile
        if self.damping_profile == 'frequency_dependent':
            # Higher frequencies damped more (like real physical systems)
            frequencies = tf.exp(tf.linspace(0.0, 3.0, self.dimensions))
            damping = 0.1 / (1.0 + frequencies)
            self.damping_coeffs = tf.Variable(
                damping, trainable=False, name='damping_coeffs'
            )
        elif self.damping_profile == 'learnable':
            self.damping_coeffs = self.add_weight(
                name='damping_coeffs',
                shape=(self.dimensions,),
                initializer=tf.initializers.Constant(0.1),
                constraint=tf.keras.constraints.NonNeg(),
                trainable=True
            )
        else:  # uniform
            self.damping_coeffs = tf.Variable(
                tf.fill([self.dimensions], 0.1), 
                trainable=False, name='damping_coeffs'
            )
        
        # Harmonic generation matrices
        self.harmonic_matrices = {}
        for order in self.harmonic_orders:
            self.harmonic_matrices[order] = self.add_weight(
                name=f'harmonic_matrix_{order}',
                shape=(self.dimensions, self.dimensions),
                initializer='orthogonal',
                trainable=True
            )
        
        # Nonlinear mixing parameters
        self.mixing_scale = self.add_weight(
            name='mixing_scale',
            shape=(self.dimensions,),
            initializer='ones',
            trainable=True
        )
        
        self.mixing_bias = self.add_weight(
            name='mixing_bias',
            shape=(self.dimensions,),
            initializer='zeros',
            trainable=True
        )
        
        # Resonance memory for maintaining standing wave patterns
        self.resonance_memory = self.add_weight(
            name='resonance_memory',
            shape=(self.dimensions,),
            initializer='zeros',
            trainable=False
        )
    
    def call(self, inputs, training=None):
        """
        Apply chromatic resonance to colour waveforms
        
        Args:
            inputs: Complex waveform tensor or list of waveforms
            training: Training mode flag
            
        Returns:
            Resonated colour waveform
        """
        
        # Handle single input or multiple inputs
        if isinstance(inputs, list):
            if len(inputs) == 1:
                wave_input = inputs[0]
            else:
                # Combine multiple inputs through interference
                wave_input = inputs[0]
                for other_wave in inputs[1:]:
                    wave_input = self.interference_layer([wave_input, other_wave])
        else:
            wave_input = inputs
        
        # Initialize chamber state
        chamber_state = wave_input
        resonance_history = []
        
        # Iterative resonance process
        for depth in range(self.resonance_depth):
            # Apply coupling transformation
            transformed = self._apply_coupling(chamber_state)
            
            # Generate harmonics
            harmonic_contributions = self._generate_harmonics(chamber_state)
            
            # Combine with harmonics
            with_harmonics = transformed + harmonic_contributions
            
            # Create interference with original wave
            if depth > 0:
                interference = self.interference_layer([chamber_state, with_harmonics])
            else:
                interference = with_harmonics
            
            # Apply nonlinear activation
            nonlinear_response = self._apply_nonlinearity(interference)
            
            # Apply damping
            damped = self._apply_damping(nonlinear_response, depth)
            
            # Update chamber state
            chamber_state = damped
            resonance_history.append(chamber_state)
        
        # Create final superposition of all resonances
        final_resonance = self._create_superposition(resonance_history)
        
        # Update resonance memory (exponential moving average)
        if training:
            current_memory = tf.reduce_mean(tf.math.abs(final_resonance), axis=0)
            new_memory = (self.memory_decay * self.resonance_memory + 
                         (1 - self.memory_decay) * current_memory)
            self.resonance_memory.assign(new_memory)
        
        # Add collapse prevention regularization
        if training:
            collapse_loss = prevent_collapse(
                final_resonance, method='entropy_regularization', strength=0.05
            )
            self.add_loss(collapse_loss)
        
        return final_resonance
    
    def _apply_coupling(self, wave):
        """Apply coupling matrix transformation"""
        return tf.matmul(wave, tf.cast(self.coupling_matrix, wave.dtype)) * self.coupling_strength
    
    def _generate_harmonics(self, wave):
        """Generate harmonic frequencies of the input wave"""
        harmonics = []
        
        for order in self.harmonic_orders:
            # Apply harmonic transformation matrix
            harmonic_matrix = tf.cast(self.harmonic_matrices[order], wave.dtype)
            harmonic_wave = tf.matmul(wave, harmonic_matrix)
            
            # Generate harmonic by raising to power (in complex domain)
            if order == 1:
                harmonic = harmonic_wave
            elif order == 2:
                harmonic = harmonic_wave * tf.math.conj(harmonic_wave)
            elif order == 3:
                harmonic = harmonic_wave * tf.math.conj(harmonic_wave) * harmonic_wave
            else:
                # General case using magnitude and phase
                magnitude = tf.math.abs(harmonic_wave)
                phase = tf.math.angle(harmonic_wave)
                
                # Scale magnitude and multiply phase
                harmonic_mag = tf.pow(magnitude, 1.0/order)  # Prevent explosion
                harmonic_phase = phase * tf.cast(order, tf.float32)
                
                harmonic = harmonic_mag * tf.exp(tf.complex(0.0, harmonic_phase))
            
            # Weight by harmonic strength (higher orders typically weaker)
            weight = 1.0 / (order * order)
            harmonics.append(harmonic * weight)
        
        # Combine all harmonics
        if harmonics:
            return tf.reduce_sum(tf.stack(harmonics), axis=0)
        else:
            return tf.zeros_like(wave)
    
    def _apply_nonlinearity(self, wave):
        """Apply nonlinear activation to complex wave"""
        real_part = tf.math.real(wave)
        imag_part = tf.math.imag(wave)
        
        # Scale and bias
        scaled_real = real_part * self.mixing_scale + self.mixing_bias
        scaled_imag = imag_part * self.mixing_scale + self.mixing_bias
        
        # Apply nonlinearity
        if self.nonlinearity == 'tanh':
            nonlinear_real = tf.nn.tanh(scaled_real)
            nonlinear_imag = tf.nn.tanh(scaled_imag)
        elif self.nonlinearity == 'sigmoid':
            nonlinear_real = tf.nn.sigmoid(scaled_real) * 2 - 1  # Center around 0
            nonlinear_imag = tf.nn.sigmoid(scaled_imag) * 2 - 1
        elif self.nonlinearity == 'swish':
            nonlinear_real = scaled_real * tf.nn.sigmoid(scaled_real)
            nonlinear_imag = scaled_imag * tf.nn.sigmoid(scaled_imag)
        elif self.nonlinearity == 'gelu':
            nonlinear_real = tf.nn.gelu(scaled_real)
            nonlinear_imag = tf.nn.gelu(scaled_imag)
        else:  # linear
            nonlinear_real = scaled_real
            nonlinear_imag = scaled_imag
        
        return tf.complex(nonlinear_real, nonlinear_imag)
    
    def _apply_damping(self, wave, depth):
        """Apply frequency-dependent damping"""
        # Damping increases with depth (like energy dissipation)
        depth_factor = tf.exp(-self.damping_coeffs * tf.cast(depth, tf.float32))
        damping_tensor = tf.cast(depth_factor, wave.dtype)
        
        return wave * damping_tensor
    
    def _create_superposition(self, resonance_history):
        """Create superposition of all resonance states"""
        if not resonance_history:
            raise ValueError("Empty resonance history")
        
        # Weight recent resonances more heavily
        weights = tf.exp(-tf.linspace(0.0, 2.0, len(resonance_history)))
        weights = weights / tf.reduce_sum(weights)
        
        # Weighted sum of resonances
        weighted_resonances = []
        for i, resonance in enumerate(resonance_history):
            weighted_resonances.append(resonance * tf.cast(weights[i], resonance.dtype))
        
        return tf.reduce_sum(tf.stack(weighted_resonances), axis=0)
    
    def get_resonance_metrics(self, inputs):
        """
        Analyze resonance characteristics
        
        Args:
            inputs: Input waveforms
            
        Returns:
            Dictionary with resonance metrics
        """
        output = self.call(inputs, training=False)
        
        # Quality factor (sharpness of resonance)
        magnitude = tf.math.abs(output)
        peak_magnitude = tf.reduce_max(magnitude, axis=-1)
        mean_magnitude = tf.reduce_mean(magnitude, axis=-1)
        q_factor = peak_magnitude / (mean_magnitude + 1e-8)
        
        # Resonance stability (consistency of memory)
        stability = 1.0 / (1.0 + tf.reduce_std(self.resonance_memory))
        
        # Harmonic richness
        fft_output = tf.signal.fft(output)
        power_spectrum = tf.math.abs(fft_output) ** 2
        total_power = tf.reduce_sum(power_spectrum, axis=-1)
        fundamental_power = power_spectrum[..., :self.dimensions//4]  # Lower frequencies
        harmonic_ratio = tf.reduce_sum(fundamental_power, axis=-1) / (total_power + 1e-8)
        
        return {
            'q_factor': q_factor,
            'stability': stability,
            'harmonic_ratio': harmonic_ratio,
            'resonance_memory': self.resonance_memory
        }
    
    def reset_memory(self):
        """Reset resonance memory to initial state"""
        self.resonance_memory.assign(tf.zeros_like(self.resonance_memory))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dimensions': self.dimensions,
            'resonance_depth': self.resonance_depth,
            'coupling_strength': self.coupling_strength,
            'damping_profile': self.damping_profile,
            'nonlinearity': self.nonlinearity,
            'memory_decay': self.memory_decay,
            'harmonic_orders': self.harmonic_orders
        })
        return config