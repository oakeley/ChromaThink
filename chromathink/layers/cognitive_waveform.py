"""CognitiveWaveform layer for transforming inputs to N-dimensional frequency domain representations"""

import tensorflow as tf
import numpy as np
from ..core.spectral_utils import create_stable_frequencies, SpectralNormalizer


class CognitiveWaveform(tf.keras.layers.Layer):
    """
    Transforms any input into waveform representation across N-dimensional colour space.
    Each thought is encoded as a superposition of frequencies with amplitude and phase.
    """
    
    def __init__(self, 
                 dimensions=512,
                 frequency_range=(0.001, 1000.0),
                 use_fft=True,
                 spectral_normalize=True,
                 initialization='orthogonal',
                 **kwargs):
        """
        Initialize CognitiveWaveform layer
        
        Args:
            dimensions: Number of frequency dimensions
            frequency_range: (min_freq, max_freq) for frequency basis
            use_fft: Whether to use FFT for frequency domain transformation
            spectral_normalize: Apply spectral normalization to prevent collapse
            initialization: Weight initialization scheme
        """
        super().__init__(**kwargs)
        
        self.dimensions = dimensions
        self.frequency_range = frequency_range
        self.use_fft = use_fft
        self.spectral_normalize = spectral_normalize
        self.initialization = initialization
        
        # Create stable frequency basis
        self.frequencies = create_stable_frequencies(
            dimensions, frequency_range, distribution='log'
        )
        
        if spectral_normalize:
            self.spectral_normalizer = SpectralNormalizer()
    
    def build(self, input_shape):
        super().build(input_shape)
        
        input_dim = input_shape[-1]
        
        # Projection matrices for amplitude and phase
        self.amplitude_projection = self.add_weight(
            name='amplitude_projection',
            shape=(input_dim, self.dimensions),
            initializer=self.initialization,
            trainable=True
        )
        
        self.phase_projection = self.add_weight(
            name='phase_projection', 
            shape=(input_dim, self.dimensions),
            initializer=self.initialization,
            trainable=True
        )
        
        # Learnable frequency modulation
        self.frequency_modulation = self.add_weight(
            name='frequency_modulation',
            shape=(self.dimensions,),
            initializer='ones',
            trainable=True
        )
        
        # Nonlinear activation parameters
        self.amplitude_scale = self.add_weight(
            name='amplitude_scale',
            shape=(self.dimensions,),
            initializer='ones',
            trainable=True
        )
        
        self.phase_scale = self.add_weight(
            name='phase_scale', 
            shape=(self.dimensions,),
            initializer='ones',
            trainable=True
        )
    
    def call(self, inputs, training=None):
        """
        Transform input into complex waveform representation
        
        Args:
            inputs: Input tensor of shape (..., input_dim)
            training: Training mode flag
            
        Returns:
            Complex waveform tensor of shape (..., dimensions)
        """
        
        # Project to amplitude and phase spaces
        amplitude_raw = tf.matmul(inputs, self.amplitude_projection)
        phase_raw = tf.matmul(inputs, self.phase_projection)
        
        # Apply nonlinear activations
        # Amplitude should be positive
        amplitude = tf.nn.softplus(amplitude_raw * self.amplitude_scale)
        
        # Phase should be in [0, 2π]
        phase = tf.nn.sigmoid(phase_raw * self.phase_scale) * 2 * np.pi
        
        # Create complex waveform
        waveform = amplitude * tf.exp(tf.complex(tf.constant(0.0), phase))
        
        # Apply frequency modulation
        modulated_frequencies = self.frequencies * self.frequency_modulation
        frequency_terms = tf.cast(modulated_frequencies, tf.complex64)
        
        # Frequency domain representation
        if self.use_fft:
            # Use FFT for efficient frequency domain transformation
            waveform_fft = tf.signal.fft(waveform)
            modulated = waveform_fft * frequency_terms
            # Return to time domain if needed, or keep in frequency domain
            result = modulated
        else:
            # Direct frequency modulation
            result = waveform * frequency_terms
        
        # Apply spectral normalization if enabled
        if self.spectral_normalize and hasattr(self, 'spectral_normalizer'):
            # Convert complex to real for normalization
            real_part = tf.real(result)
            imag_part = tf.imag(result)
            
            normalized_real = self.spectral_normalizer(real_part)
            normalized_imag = self.spectral_normalizer(imag_part)
            
            result = tf.complex(normalized_real, normalized_imag)
        
        return result
    
    def get_frequency_content(self, inputs):
        """
        Analyze frequency content of the waveform representation
        
        Args:
            inputs: Input tensor
            
        Returns:
            Dictionary with frequency analysis
        """
        waveform = self.call(inputs, training=False)
        
        # Extract magnitude and phase
        magnitude = tf.abs(waveform)
        phase = tf.angle(waveform)
        
        # Power spectral density
        power = tf.square(magnitude)
        
        # Dominant frequencies
        dominant_indices = tf.argmax(power, axis=-1)
        dominant_frequencies = tf.gather(self.frequencies, dominant_indices)
        
        # Spectral centroid (center of mass of spectrum)
        frequency_weights = tf.cast(self.frequencies, tf.float32)
        spectral_centroid = tf.reduce_sum(
            power * frequency_weights, axis=-1
        ) / (tf.reduce_sum(power, axis=-1) + 1e-8)
        
        # Spectral bandwidth
        spectral_spread = tf.sqrt(
            tf.reduce_sum(
                power * tf.square(frequency_weights - tf.expand_dims(spectral_centroid, -1)),
                axis=-1
            ) / (tf.reduce_sum(power, axis=-1) + 1e-8)
        )
        
        return {
            'magnitude': magnitude,
            'phase': phase,
            'power': power,
            'dominant_frequencies': dominant_frequencies,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_spread
        }
    
    def visualize_spectrum(self, inputs, sample_index=0):
        """
        Create visualization data for frequency spectrum
        
        Args:
            inputs: Input tensor
            sample_index: Which sample to visualize
            
        Returns:
            Dictionary with visualization data
        """
        analysis = self.get_frequency_content(inputs)
        
        return {
            'frequencies': self.frequencies.numpy(),
            'magnitude': analysis['magnitude'][sample_index].numpy(),
            'phase': analysis['phase'][sample_index].numpy(),
            'power': analysis['power'][sample_index].numpy(),
        }
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dimensions': self.dimensions,
            'frequency_range': self.frequency_range,
            'use_fft': self.use_fft,
            'spectral_normalize': self.spectral_normalize,
            'initialization': self.initialization
        })
        return config