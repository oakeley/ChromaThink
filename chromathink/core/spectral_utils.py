"""Spectral normalization and frequency domain utilities for preventing colour collapse"""

import tensorflow as tf
import numpy as np


class SpectralNormalizer(tf.keras.layers.Layer):
    """
    Applies spectral normalization to prevent colour vectors from collapsing
    to grey or exploding to noise. Maintains diversity in frequency domain.
    """
    
    def __init__(self, power_iterations=1, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.power_iterations = power_iterations
        self.epsilon = epsilon
        
    def build(self, input_shape):
        super().build(input_shape)
        
        # Initialize singular vector for power iteration
        self.u = self.add_weight(
            name='u',
            shape=(1, input_shape[-1]),
            initializer='random_normal',
            trainable=False
        )
        
    def call(self, inputs):
        """Apply spectral normalization to colour vectors"""
        
        # Reshape for matrix operations
        batch_size = tf.shape(inputs)[0]
        features = inputs.shape[-1]
        
        # Flatten to (batch, features)
        w_mat = tf.reshape(inputs, (batch_size, features))
        
        # Power iteration to find largest singular value
        u_hat = self.u
        for _ in range(self.power_iterations):
            v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, w_mat, transpose_b=True))
            u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w_mat))
            
        # Update stored u vector
        self.u.assign(u_hat)
        
        # Calculate spectral norm
        sigma = tf.reduce_sum(u_hat * tf.matmul(u_hat, w_mat))
        
        # Normalize by spectral norm with epsilon for stability
        w_normalized = w_mat / (sigma + self.epsilon)
        
        # Reshape back to original shape
        return tf.reshape(w_normalized, tf.shape(inputs))


def frequency_stability_check(colour_vectors, min_variance=1e-4, max_norm=10.0):
    """
    Check if colour vectors maintain stable frequency characteristics
    
    Args:
        colour_vectors: Complex tensor of colour representations
        min_variance: Minimum variance to prevent collapse
        max_norm: Maximum norm to prevent explosion
        
    Returns:
        Dict with stability metrics
    """
    
    # Convert to real representation for analysis
    if colour_vectors.dtype.is_complex:
        real_part = tf.real(colour_vectors)
        imag_part = tf.imag(colour_vectors)
        combined = tf.concat([real_part, imag_part], axis=-1)
    else:
        combined = colour_vectors
    
    # Calculate variance across frequency dimensions
    variance = tf.reduce_mean(tf.math.reduce_variance(combined, axis=0))
    
    # Calculate norms
    norms = tf.norm(combined, axis=-1)
    mean_norm = tf.reduce_mean(norms)
    max_norm_actual = tf.reduce_max(norms)
    
    # Frequency domain analysis
    if colour_vectors.dtype.is_complex:
        magnitudes = tf.abs(colour_vectors)
        phases = tf.angle(colour_vectors)
        
        # Check for phase diversity (prevents collapse to real values)
        phase_variance = tf.reduce_mean(tf.math.reduce_variance(phases, axis=0))
        
        # Spectral flatness (measure of noise vs tone-like)
        geometric_mean = tf.exp(tf.reduce_mean(tf.math.log(magnitudes + 1e-8), axis=-1))
        arithmetic_mean = tf.reduce_mean(magnitudes, axis=-1)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-8)
        
        return {
            'variance': variance,
            'mean_norm': mean_norm,
            'max_norm': max_norm_actual,
            'phase_variance': phase_variance,
            'spectral_flatness': tf.reduce_mean(spectral_flatness),
            'is_stable': tf.logical_and(
                tf.logical_and(variance > min_variance, max_norm_actual < max_norm),
                phase_variance > 1e-3
            )
        }
    else:
        return {
            'variance': variance,
            'mean_norm': mean_norm,
            'max_norm': max_norm_actual,
            'is_stable': tf.logical_and(variance > min_variance, max_norm_actual < max_norm)
        }


def create_stable_frequencies(dimensions, frequency_range=(0.001, 1000.0), distribution='log'):
    """
    Create a stable frequency basis for colour representation
    
    Args:
        dimensions: Number of frequency components
        frequency_range: (min_freq, max_freq) range
        distribution: 'log' for logarithmic spacing, 'linear' for linear
        
    Returns:
        Frequency tensor
    """
    
    min_freq, max_freq = frequency_range
    
    if distribution == 'log':
        # Logarithmic spacing for better coverage
        frequencies = tf.exp(tf.linspace(
            tf.math.log(min_freq),
            tf.math.log(max_freq), 
            dimensions
        ))
    else:
        # Linear spacing
        frequencies = tf.linspace(min_freq, max_freq, dimensions)
    
    # Add small random perturbations to break symmetries
    perturbations = tf.random.normal(
        [dimensions], 
        mean=0.0, 
        stddev=0.01 * (max_freq - min_freq) / dimensions
    )
    
    return frequencies + perturbations