"""InterferenceLayer implementing wave interference with amplitude weighting and phase calculations"""

import tensorflow as tf
import numpy as np
from ..core.colour_utils import prevent_collapse


class InterferenceLayer(tf.keras.layers.Layer):
    """
    Implements wave interference patterns between colour waveforms.
    Models constructive and destructive interference with amplitude weighting
    and precise phase calculations.
    """
    
    def __init__(self,
                 interference_type='full',  # 'constructive', 'destructive', 'full'
                 amplitude_weighting=True,
                 phase_coupling=True,
                 nonlinear_mixing=False,
                 collapse_prevention='entropy_regularization',
                 **kwargs):
        """
        Initialize InterferenceLayer
        
        Args:
            interference_type: Type of interference to model
            amplitude_weighting: Whether to weight by amplitude
            phase_coupling: Enable phase coupling between waves
            nonlinear_mixing: Apply nonlinear mixing functions
            collapse_prevention: Method to prevent colour collapse
        """
        super().__init__(**kwargs)
        
        self.interference_type = interference_type
        self.amplitude_weighting = amplitude_weighting
        self.phase_coupling = phase_coupling
        self.nonlinear_mixing = nonlinear_mixing
        self.collapse_prevention = collapse_prevention
    
    def build(self, input_shape):
        super().build(input_shape)
        
        if len(input_shape) != 2:
            raise ValueError("InterferenceLayer expects exactly 2 inputs")
        
        dim1, dim2 = input_shape[0][-1], input_shape[1][-1]
        
        if dim1 != dim2:
            raise ValueError(f"Input dimensions must match: {dim1} vs {dim2}")
        
        self.dimensions = dim1
        
        if self.amplitude_weighting:
            # Learnable amplitude weighting coefficients
            self.amplitude_weights = self.add_weight(
                name='amplitude_weights',
                shape=(self.dimensions,),
                initializer='ones',
                trainable=True
            )
        
        if self.phase_coupling:
            # Phase coupling matrix for cross-frequency interactions
            self.phase_coupling_matrix = self.add_weight(
                name='phase_coupling',
                shape=(self.dimensions, self.dimensions),
                initializer='orthogonal',
                trainable=True
            )
        
        if self.nonlinear_mixing:
            # Nonlinear mixing parameters
            self.mixing_weights = self.add_weight(
                name='mixing_weights',
                shape=(3, self.dimensions),  # For quadratic, cubic, and cross terms
                initializer='random_normal',
                trainable=True
            )
    
    def call(self, inputs, training=None):
        """
        Apply wave interference between two colour waveforms
        
        Args:
            inputs: List of two complex waveform tensors
            training: Training mode flag
            
        Returns:
            Interfered waveform tensor
        """
        wave1, wave2 = inputs
        
        # Extract amplitude and phase components
        amp1, phase1 = tf.math.abs(wave1), tf.math.angle(wave1)
        amp2, phase2 = tf.math.abs(wave2), tf.math.angle(wave2)
        
        # Apply amplitude weighting if enabled
        if self.amplitude_weighting:
            weighted_amp1 = amp1 * self.amplitude_weights
            weighted_amp2 = amp2 * self.amplitude_weights
        else:
            weighted_amp1, weighted_amp2 = amp1, amp2
        
        # Phase coupling for cross-frequency interactions
        if self.phase_coupling:
            # Ensure phase tensors have batch dimension for matmul
            if len(phase1.shape) == 1:
                phase1_expanded = tf.expand_dims(phase1, 0)
                phase2_expanded = tf.expand_dims(phase2, 0)
            else:
                phase1_expanded = phase1
                phase2_expanded = phase2
            
            # Transform phases through coupling matrix
            coupled_phase1_expanded = tf.matmul(phase1_expanded, self.phase_coupling_matrix)
            coupled_phase2_expanded = tf.matmul(phase2_expanded, self.phase_coupling_matrix)
            
            # Remove batch dimension if it was added
            if len(phase1.shape) == 1:
                coupled_phase1 = tf.squeeze(coupled_phase1_expanded, 0)
                coupled_phase2 = tf.squeeze(coupled_phase2_expanded, 0)
            else:
                coupled_phase1 = coupled_phase1_expanded
                coupled_phase2 = coupled_phase2_expanded
        else:
            coupled_phase1, coupled_phase2 = phase1, phase2
        
        # Calculate interference based on type
        if self.interference_type == 'constructive':
            interfered = self._constructive_interference(
                weighted_amp1, weighted_amp2, coupled_phase1, coupled_phase2
            )
        elif self.interference_type == 'destructive':
            interfered = self._destructive_interference(
                weighted_amp1, weighted_amp2, coupled_phase1, coupled_phase2
            )
        else:  # 'full' interference
            interfered = self._full_interference(
                weighted_amp1, weighted_amp2, coupled_phase1, coupled_phase2
            )
        
        # Apply nonlinear mixing if enabled
        if self.nonlinear_mixing:
            interfered = self._apply_nonlinear_mixing(interfered, wave1, wave2)
        
        # Add collapse prevention regularization during training
        if training and self.collapse_prevention:
            collapse_loss = prevent_collapse(
                interfered, method=self.collapse_prevention, strength=0.1
            )
            self.add_loss(collapse_loss)
        
        return interfered
    
    def _constructive_interference(self, amp1, amp2, phase1, phase2):
        """Calculate constructive interference pattern"""
        # Phases aligned for maximum constructive interference
        phase_diff = phase1 - phase2
        alignment_factor = tf.cos(phase_diff)
        
        # Enhanced amplitude where waves align
        combined_amp = tf.sqrt(
            amp1**2 + amp2**2 + 2 * amp1 * amp2 * tf.math.abs(alignment_factor)
        )
        
        # Average phase weighted by amplitudes
        total_amp = amp1 + amp2 + 1e-8
        combined_phase = (amp1 * phase1 + amp2 * phase2) / total_amp
        
        return tf.cast(combined_amp, tf.complex64) * tf.exp(tf.complex(0.0, combined_phase))
    
    def _destructive_interference(self, amp1, amp2, phase1, phase2):
        """Calculate destructive interference pattern"""
        # Phase difference for destructive interference
        phase_diff = phase1 - phase2
        opposition_factor = tf.sin(phase_diff)
        
        # Reduced amplitude where waves oppose
        combined_amp = tf.sqrt(
            tf.nn.relu(amp1**2 + amp2**2 - 2 * amp1 * amp2 * tf.math.abs(opposition_factor))
        )
        
        # Phase determined by stronger wave
        stronger_mask = tf.cast(amp1 > amp2, tf.float32)
        combined_phase = stronger_mask * phase1 + (1 - stronger_mask) * phase2
        
        return tf.cast(combined_amp, tf.complex64) * tf.exp(tf.complex(0.0, combined_phase))
    
    def _full_interference(self, amp1, amp2, phase1, phase2):
        """Calculate full interference with both constructive and destructive components"""
        # Standard wave interference formula
        phase_diff = phase1 - phase2
        
        # Combined amplitude considering interference
        combined_amp = tf.sqrt(
            amp1**2 + amp2**2 + 2 * amp1 * amp2 * tf.cos(phase_diff)
        )
        
        # Combined phase using complex vector addition
        real_part = amp1 * tf.cos(phase1) + amp2 * tf.cos(phase2)
        imag_part = amp1 * tf.sin(phase1) + amp2 * tf.sin(phase2)
        
        combined_phase = tf.atan2(imag_part, real_part)
        
        return tf.cast(combined_amp, tf.complex64) * tf.exp(tf.complex(0.0, combined_phase))
    
    def _apply_nonlinear_mixing(self, linear_result, wave1, wave2):
        """Apply nonlinear mixing for richer interference patterns"""
        # Extract real and imaginary parts
        linear_real = tf.math.real(linear_result)
        linear_imag = tf.math.imag(linear_result)
        
        wave1_real, wave1_imag = tf.math.real(wave1), tf.math.imag(wave1)
        wave2_real, wave2_imag = tf.math.real(wave2), tf.math.imag(wave2)
        
        # Quadratic terms
        quad_real = self.mixing_weights[0] * (wave1_real * wave2_real - wave1_imag * wave2_imag)
        quad_imag = self.mixing_weights[0] * (wave1_real * wave2_imag + wave1_imag * wave2_real)
        
        # Cubic terms  
        cubic_real = self.mixing_weights[1] * (
            wave1_real * (wave1_real**2 - 3 * wave1_imag**2) * 
            wave2_real * (wave2_real**2 - 3 * wave2_imag**2)
        )
        cubic_imag = self.mixing_weights[1] * (
            wave1_imag * (3 * wave1_real**2 - wave1_imag**2) * 
            wave2_imag * (3 * wave2_real**2 - wave2_imag**2)
        )
        
        # Cross terms
        cross_real = self.mixing_weights[2] * (wave1_real**2 * wave2_real - wave1_imag**2 * wave2_imag)
        cross_imag = self.mixing_weights[2] * (wave1_real**2 * wave2_imag + wave1_imag**2 * wave2_real)
        
        # Combine all terms
        total_real = linear_real + quad_real + cubic_real + cross_real
        total_imag = linear_imag + quad_imag + cubic_imag + cross_imag
        
        return tf.complex(total_real, total_imag)
    
    def interference_strength(self, inputs):
        """
        Calculate interference strength between two waveforms
        
        Args:
            inputs: List of two complex waveform tensors
            
        Returns:
            Interference strength metrics
        """
        wave1, wave2 = inputs
        
        # Phase correlation
        phase1, phase2 = tf.math.angle(wave1), tf.math.angle(wave2)
        phase_corr = tf.reduce_mean(tf.cos(phase1 - phase2), axis=-1)
        
        # Amplitude correlation
        amp1, amp2 = tf.math.abs(wave1), tf.math.abs(wave2)
        amp_corr = tf.reduce_sum(amp1 * amp2, axis=-1) / (
            tf.sqrt(tf.reduce_sum(amp1**2, axis=-1)) * 
            tf.sqrt(tf.reduce_sum(amp2**2, axis=-1)) + 1e-8
        )
        
        # Overall interference strength
        interference_strength = tf.math.abs(phase_corr) * amp_corr
        
        return {
            'phase_correlation': phase_corr,
            'amplitude_correlation': amp_corr,
            'interference_strength': interference_strength
        }
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'interference_type': self.interference_type,
            'amplitude_weighting': self.amplitude_weighting,
            'phase_coupling': self.phase_coupling,
            'nonlinear_mixing': self.nonlinear_mixing,
            'collapse_prevention': self.collapse_prevention
        })
        return config


class MultiWaveInterference(tf.keras.layers.Layer):
    """
    Handles interference between multiple waveforms simultaneously
    """
    
    def __init__(self, max_waves=4, **kwargs):
        """
        Initialize multi-wave interference layer
        
        Args:
            max_waves: Maximum number of waves to interfere
        """
        super().__init__(**kwargs)
        self.max_waves = max_waves
        
        # Create pairwise interference layers
        self.pairwise_layers = []
        for i in range(max_waves):
            for j in range(i + 1, max_waves):
                self.pairwise_layers.append(
                    InterferenceLayer(name=f'interference_{i}_{j}')
                )
    
    def call(self, inputs):
        """
        Apply interference between all pairs of input waves
        
        Args:
            inputs: List of waveform tensors
            
        Returns:
            Combined interference result
        """
        if len(inputs) > self.max_waves:
            raise ValueError(f"Too many input waves: {len(inputs)} > {self.max_waves}")
        
        # Calculate all pairwise interferences
        interferences = []
        layer_idx = 0
        
        for i in range(len(inputs)):
            for j in range(i + 1, len(inputs)):
                interference = self.pairwise_layers[layer_idx]([inputs[i], inputs[j]])
                interferences.append(interference)
                layer_idx += 1
        
        if not interferences:
            # Single wave case
            return inputs[0]
        
        # Combine all interferences
        # Simple averaging - could be made more sophisticated
        combined = tf.reduce_mean(tf.stack(interferences), axis=0)
        
        return combined