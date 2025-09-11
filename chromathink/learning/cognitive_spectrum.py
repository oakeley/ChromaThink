"""
CognitiveSpectrum: Core colour-based cognition system

Transforms thoughts into frequency domain representations across N-dimensional colour space.
"""

import tensorflow as tf
import numpy as np
from ..layers.cognitive_waveform import CognitiveWaveform
from ..layers.chromatic_resonance import ChromaticResonance
from ..core.spectral_utils import SpectralNormalizer


class CognitiveSpectrum(tf.keras.Model):
    """
    Core cognitive system that processes all thought as colour waveforms.
    Unlike token-based models, this system thinks entirely in frequency space.
    """
    
    def __init__(self, 
                 spectrum_dims=512,
                 resonance_depth=7,
                 harmonic_orders=[1, 2, 3, 5, 8],
                 frequency_range=(0.1, 100.0),
                 name='cognitive_spectrum'):
        super().__init__(name=name)
        
        self.spectrum_dims = spectrum_dims
        self.resonance_depth = resonance_depth
        self.harmonic_orders = harmonic_orders
        
        # Core cognitive waveform layer
        self.waveform_layer = CognitiveWaveform(
            dimensions=spectrum_dims,
            frequency_range=frequency_range,
            use_fft=True,
            spectral_normalize=True
        )
        
        # Deep resonance for complex thought
        self.resonance_layers = []
        for depth in range(resonance_depth):
            self.resonance_layers.append(
                ChromaticResonance(
                    dimensions=spectrum_dims,
                    resonance_depth=3,
                    harmonic_orders=harmonic_orders,
                    memory_decay=0.95 ** depth  # Deeper layers have longer memory
                )
            )
        
        # Spectral normalization to prevent collapse
        self.spectral_normalizer = SpectralNormalizer(
            power_iterations=3,
            epsilon=1e-6
        )
        
        # Cognitive state memory
        self.cognitive_memory = None
        
    def call(self, inputs, training=None):
        """
        Process input through cognitive spectrum.
        
        Args:
            inputs: Can be any tensor - text embeddings, sensory data, or existing colour states
            training: Training mode flag
        """
        
        # Transform input to colour waveform
        colour_state = self.waveform_layer(inputs, training=training)
        
        # Apply simple normalization (spectral normalizer disabled for compatibility)
        # colour_state = self.spectral_normalizer(colour_state, training=training)
        colour_state = tf.nn.l2_normalize(colour_state, axis=-1)
        
        # Process through resonance layers
        for i, resonance_layer in enumerate(self.resonance_layers):
            # Mix with previous cognitive memory for continuity
            if self.cognitive_memory is not None:
                # Blend current state with memory
                memory_influence = 0.1 * (0.8 ** i)  # Diminishing memory influence
                colour_state = (1 - memory_influence) * colour_state + \
                              memory_influence * self.cognitive_memory
            
            colour_state = resonance_layer(colour_state, training=training)
        
        # Update cognitive memory
        if training:
            self.cognitive_memory = tf.stop_gradient(colour_state)
        
        return colour_state
    
    def think(self, concept, intensity=1.0):
        """
        Think about a concept, returning the colour representation.
        
        Args:
            concept: Input concept (can be text, embeddings, or colour state)
            intensity: Thinking intensity (0.0 to 2.0)
        """
        
        # Amplify input based on thinking intensity
        if tf.is_tensor(concept):
            amplified_concept = concept * intensity
        else:
            # Convert to tensor if needed
            amplified_concept = tf.constant(concept, dtype=tf.float32) * intensity
        
        return self(amplified_concept, training=False)
    
    def wonder(self, uncertainty=0.5):
        """
        Generate a state of wonder/curiosity.
        
        Args:
            uncertainty: Level of uncertainty/randomness (0.0 to 1.0)
        """
        
        # Create noisy input to represent uncertainty
        noise_shape = [1, self.spectrum_dims]
        wondering_input = tf.random.normal(noise_shape) * uncertainty
        
        # Add structured patterns to represent curiosity
        frequencies = tf.linspace(0.1, 10.0, self.spectrum_dims // 4)
        curiosity_pattern = tf.sin(2 * np.pi * frequencies * uncertainty)
        curiosity_pattern = tf.tile(tf.expand_dims(curiosity_pattern, 0), [1, 4])
        
        wondering_state = wondering_input + 0.3 * curiosity_pattern
        
        return self(wondering_state, training=False)
    
    def get_cognitive_metrics(self):
        """
        Return metrics about current cognitive state.
        """
        if self.cognitive_memory is None:
            return {
                'memory_complexity': 0.0,
                'spectral_entropy': 0.0,
                'dominant_frequencies': []
            }
        
        # Calculate complexity as spectral diversity
        fft_memory = tf.signal.fft(tf.cast(self.cognitive_memory, tf.complex64))
        power_spectrum = tf.math.abs(fft_memory) ** 2
        
        # Spectral entropy
        normalized_spectrum = power_spectrum / tf.reduce_sum(power_spectrum, axis=-1, keepdims=True)
        spectral_entropy = -tf.reduce_sum(
            normalized_spectrum * tf.math.log(normalized_spectrum + 1e-8), 
            axis=-1
        )
        
        # Find dominant frequencies
        top_k = 5
        _, top_indices = tf.nn.top_k(power_spectrum[0], k=top_k)
        dominant_frequencies = tf.gather(
            tf.linspace(0.0, float(self.spectrum_dims // 2), self.spectrum_dims),
            top_indices
        )
        
        return {
            'memory_complexity': float(tf.reduce_mean(spectral_entropy)),
            'spectral_entropy': float(tf.reduce_mean(spectral_entropy)),
            'dominant_frequencies': dominant_frequencies.numpy().tolist(),
            'memory_magnitude': float(tf.reduce_mean(tf.math.abs(self.cognitive_memory)))
        }
    
    def reset_memory(self):
        """Reset cognitive memory (useful for starting fresh conversations)."""
        self.cognitive_memory = None
        
        # Reset memory in all resonance layers
        for layer in self.resonance_layers:
            if hasattr(layer, 'reset_memory'):
                layer.reset_memory()