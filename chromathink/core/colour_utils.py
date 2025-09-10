"""Colour space utilities for preventing collapse and measuring colour distances"""

import tensorflow as tf
import numpy as np


def prevent_collapse(colour_vectors, method='entropy_regularization', strength=0.1):
    """
    Prevent colour vectors from collapsing to uniform grey or single colours
    
    Args:
        colour_vectors: Tensor of colour representations
        method: Prevention method ('entropy_regularization', 'orthogonal_penalty', 'diversity_loss')
        strength: Regularization strength
        
    Returns:
        Regularization loss to add to main loss
    """
    
    if method == 'entropy_regularization':
        # Encourage high entropy in colour distribution
        if colour_vectors.dtype.is_complex:
            magnitudes = tf.abs(colour_vectors)
            normalized = tf.nn.softmax(magnitudes, axis=-1)
        else:
            normalized = tf.nn.softmax(colour_vectors, axis=-1)
            
        # Calculate entropy
        entropy = -tf.reduce_sum(normalized * tf.math.log(normalized + 1e-8), axis=-1)
        max_entropy = tf.math.log(tf.cast(tf.shape(colour_vectors)[-1], tf.float32))
        
        # Penalty for low entropy (collapse)
        collapse_penalty = tf.reduce_mean(tf.nn.relu(max_entropy * 0.5 - entropy))
        
        return strength * collapse_penalty
        
    elif method == 'orthogonal_penalty':
        # Encourage orthogonality between different colour vectors
        batch_size = tf.shape(colour_vectors)[0]
        
        if colour_vectors.dtype.is_complex:
            # For complex vectors, use Hermitian inner product
            gram_matrix = tf.matmul(colour_vectors, colour_vectors, adjoint_b=True)
        else:
            gram_matrix = tf.matmul(colour_vectors, colour_vectors, transpose_b=True)
            
        # Ideal would be identity matrix
        identity = tf.eye(batch_size, dtype=gram_matrix.dtype)
        orthogonal_penalty = tf.reduce_mean(tf.square(gram_matrix - identity))
        
        return strength * orthogonal_penalty
        
    elif method == 'diversity_loss':
        # Encourage diversity by penalizing similar colour vectors
        if colour_vectors.dtype.is_complex:
            vectors = tf.concat([tf.real(colour_vectors), tf.imag(colour_vectors)], axis=-1)
        else:
            vectors = colour_vectors
            
        # Normalize vectors
        normalized = tf.nn.l2_normalize(vectors, axis=-1)
        
        # Calculate pairwise similarities
        similarities = tf.matmul(normalized, normalized, transpose_b=True)
        
        # Penalize high off-diagonal similarities
        batch_size = tf.shape(similarities)[0]
        mask = 1.0 - tf.eye(batch_size)
        similarity_penalty = tf.reduce_mean(tf.square(similarities * mask))
        
        return strength * similarity_penalty
        
    else:
        raise ValueError(f"Unknown collapse prevention method: {method}")


def colour_distance(colour1, colour2, metric='spectral'):
    """
    Calculate distance between colour representations in frequency domain
    
    Args:
        colour1, colour2: Colour vectors to compare
        metric: Distance metric ('spectral', 'phase', 'wasserstein')
        
    Returns:
        Distance tensor
    """
    
    if metric == 'spectral':
        # Spectral distance based on magnitude and phase differences
        if colour1.dtype.is_complex and colour2.dtype.is_complex:
            mag1, phase1 = tf.abs(colour1), tf.angle(colour1)
            mag2, phase2 = tf.abs(colour2), tf.angle(colour2)
            
            # Magnitude difference
            mag_diff = tf.reduce_mean(tf.square(mag1 - mag2), axis=-1)
            
            # Phase difference (handle wrapping)
            phase_diff = tf.angle(tf.exp(1j * (phase1 - phase2)))
            phase_dist = tf.reduce_mean(tf.square(phase_diff), axis=-1)
            
            return tf.sqrt(mag_diff + phase_dist)
        else:
            # For real vectors, use Euclidean distance
            return tf.norm(colour1 - colour2, axis=-1)
            
    elif metric == 'phase':
        # Phase-only distance for complex vectors
        if colour1.dtype.is_complex and colour2.dtype.is_complex:
            phase1, phase2 = tf.angle(colour1), tf.angle(colour2)
            phase_diff = tf.angle(tf.exp(1j * (phase1 - phase2)))
            return tf.reduce_mean(tf.square(phase_diff), axis=-1)
        else:
            raise ValueError("Phase distance only available for complex vectors")
            
    elif metric == 'wasserstein':
        # Approximate Wasserstein distance for colour distributions
        if colour1.dtype.is_complex:
            dist1 = tf.nn.softmax(tf.abs(colour1), axis=-1)
            dist2 = tf.nn.softmax(tf.abs(colour2), axis=-1)
        else:
            dist1 = tf.nn.softmax(colour1, axis=-1)
            dist2 = tf.nn.softmax(colour2, axis=-1)
            
        # Cumulative distributions
        cum1 = tf.cumsum(dist1, axis=-1)
        cum2 = tf.cumsum(dist2, axis=-1)
        
        # L2 Wasserstein approximation
        return tf.reduce_mean(tf.square(cum1 - cum2), axis=-1)
        
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def colour_interpolation(colour1, colour2, alpha, method='linear'):
    """
    Interpolate between two colour representations
    
    Args:
        colour1, colour2: Start and end colours
        alpha: Interpolation parameter (0 to 1)
        method: Interpolation method ('linear', 'spherical', 'frequency')
        
    Returns:
        Interpolated colour
    """
    
    if method == 'linear':
        # Simple linear interpolation
        return (1 - alpha) * colour1 + alpha * colour2
        
    elif method == 'spherical':
        # Spherical linear interpolation (SLERP)
        if colour1.dtype.is_complex and colour2.dtype.is_complex:
            # Normalize to unit circle
            norm1 = colour1 / (tf.abs(colour1) + 1e-8)
            norm2 = colour2 / (tf.abs(colour2) + 1e-8)
            
            # Calculate angle between them
            dot_product = tf.real(norm1 * tf.conj(norm2))
            dot_product = tf.clip_by_value(dot_product, -1.0, 1.0)
            omega = tf.acos(tf.abs(dot_product))
            
            # SLERP formula
            sin_omega = tf.sin(omega)
            interp = (tf.sin((1-alpha) * omega) * norm1 + tf.sin(alpha * omega) * norm2) / (sin_omega + 1e-8)
            
            # Interpolate magnitudes linearly
            mag1, mag2 = tf.abs(colour1), tf.abs(colour2)
            interp_mag = (1 - alpha) * mag1 + alpha * mag2
            
            return interp * interp_mag
        else:
            # For real vectors, normalize and interpolate
            norm1 = tf.nn.l2_normalize(colour1, axis=-1)
            norm2 = tf.nn.l2_normalize(colour2, axis=-1)
            
            dot_product = tf.reduce_sum(norm1 * norm2, axis=-1, keepdims=True)
            dot_product = tf.clip_by_value(dot_product, -1.0, 1.0)
            omega = tf.acos(tf.abs(dot_product))
            
            sin_omega = tf.sin(omega)
            interp = (tf.sin((1-alpha) * omega) * norm1 + tf.sin(alpha * omega) * norm2) / (sin_omega + 1e-8)
            
            # Interpolate norms
            norm_1 = tf.norm(colour1, axis=-1, keepdims=True)
            norm_2 = tf.norm(colour2, axis=-1, keepdims=True)
            interp_norm = (1 - alpha) * norm_1 + alpha * norm_2
            
            return interp * interp_norm
            
    elif method == 'frequency':
        # Interpolate in frequency domain with proper phase handling
        if colour1.dtype.is_complex and colour2.dtype.is_complex:
            mag1, phase1 = tf.abs(colour1), tf.angle(colour1)
            mag2, phase2 = tf.abs(colour2), tf.angle(colour2)
            
            # Interpolate magnitudes logarithmically
            log_mag1 = tf.math.log(mag1 + 1e-8)
            log_mag2 = tf.math.log(mag2 + 1e-8)
            interp_log_mag = (1 - alpha) * log_mag1 + alpha * log_mag2
            interp_mag = tf.exp(interp_log_mag)
            
            # Interpolate phases (shortest path on circle)
            phase_diff = tf.angle(tf.exp(1j * (phase2 - phase1)))
            interp_phase = phase1 + alpha * phase_diff
            
            return interp_mag * tf.exp(1j * interp_phase)
        else:
            # Fall back to linear for real vectors
            return (1 - alpha) * colour1 + alpha * colour2
            
    else:
        raise ValueError(f"Unknown interpolation method: {method}")