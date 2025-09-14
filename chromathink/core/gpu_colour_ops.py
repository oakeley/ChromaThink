"""
GPU Accelerated Colour Mathematics for ChromaThink

GPUs excel at ray tracing and optical mathematics. This module leverages CUDA
to accelerate colour interference, resonance, and wave propagation calculations
for real-time colour-based thinking.

Key optimizations:
- Parallel colour interference calculations
- Vectorized resonance chamber processing  
- Fast Fourier transforms for spectral analysis
- Memory-efficient colour pattern storage
"""

import tensorflow as tf
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging


class GPUColourAccelerator:
    """
    GPU-accelerated colour mathematics optimized for ChromaThink's cognitive operations.
    
    This class provides highly optimized implementations of colour operations
    that leverage GPU parallelism for real-time colour-based thinking.
    """
    
    def __init__(self, spectrum_dims: int = 512, use_mixed_precision: bool = True):
        self.spectrum_dims = spectrum_dims
        self.use_mixed_precision = use_mixed_precision
        self.logger = logging.getLogger("GPUColourAccelerator")
        
        # Setup GPU configuration
        self._configure_gpu()
        
        # Pre-compile common operations for better performance
        self._compile_operations()
        
        self.logger.info(f"GPU colour accelerator initialized for {spectrum_dims} dimensions")
    
    def _configure_gpu(self):
        """Configure GPU settings for optimal colour mathematics performance."""
        
        # Enable GPU memory growth to prevent OOM
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Enable mixed precision for faster computation
                if self.use_mixed_precision:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    self.logger.info("Mixed precision enabled for GPU acceleration")
                
                self.logger.info(f"GPU acceleration configured for {len(gpus)} GPUs")
                
            except RuntimeError as e:
                self.logger.warning(f"GPU configuration failed: {e}")
        else:
            self.logger.info("No GPUs detected - using CPU with optimizations")
    
    def _compile_operations(self):
        """Pre-compile TensorFlow operations for better performance."""
        
        # Compile interference operation
        @tf.function(reduce_retracing=True)
        def _compiled_interference(wave1, wave2):
            return self._gpu_interference_kernel(wave1, wave2)
        
        # Compile resonance operation  
        @tf.function(reduce_retracing=True)
        def _compiled_resonance(waves, resonance_matrix):
            return self._gpu_resonance_kernel(waves, resonance_matrix)
        
        # Compile spectral analysis
        @tf.function(reduce_retracing=True)
        def _compiled_spectral_analysis(colours):
            return self._gpu_spectral_kernel(colours)
        
        self._compiled_interference = _compiled_interference
        self._compiled_resonance = _compiled_resonance
        self._compiled_spectral_analysis = _compiled_spectral_analysis
        
        self.logger.info("GPU operations compiled for optimal performance")
    
    @tf.function(reduce_retracing=True)
    def accelerated_colour_interference(self, 
                                      colours: List[tf.Tensor],
                                      interference_type: str = 'adaptive') -> tf.Tensor:
        """
        GPU-accelerated colour interference for multiple colour patterns.
        
        This leverages GPU parallelism to compute interference between
        multiple colour waves simultaneously.
        """
        
        if len(colours) < 2:
            return colours[0] if colours else tf.zeros(self.spectrum_dims, dtype=tf.complex64)
        
        # Stack colours for parallel processing
        colour_stack = tf.stack(colours)
        
        # Parallel interference computation
        result = self._parallel_interference_reduction(colour_stack, interference_type)
        
        return result
    
    @tf.function(reduce_retracing=True)
    def _parallel_interference_reduction(self, 
                                       colour_stack: tf.Tensor,
                                       interference_type: str) -> tf.Tensor:
        """Parallel reduction of colour interference using GPU."""
        
        # Use tf.scan for parallel reduction
        def interference_step(acc, current_colour):
            return self._gpu_interference_kernel(acc, current_colour)
        
        # Initialize with first colour
        initial_colour = colour_stack[0]
        remaining_colours = colour_stack[1:]
        
        # Parallel scan reduction
        result = tf.scan(
            interference_step,
            remaining_colours,
            initializer=initial_colour,
            parallel_iterations=10  # Allow parallel execution
        )
        
        # Return final result
        return result[-1]
    
    @tf.function(reduce_retracing=True)
    def _gpu_interference_kernel(self, wave1: tf.Tensor, wave2: tf.Tensor) -> tf.Tensor:
        """
        Core GPU kernel for colour wave interference.

        This implements the physics of wave interference optimized for GPU execution.
        """

        # Ensure inputs are complex64
        wave1 = tf.cast(wave1, tf.complex64)
        wave2 = tf.cast(wave2, tf.complex64)

        # Extract amplitude and phase
        amp1 = tf.abs(wave1)
        amp2 = tf.abs(wave2)
        phase1 = tf.math.angle(wave1)
        phase2 = tf.math.angle(wave2)

        # Phase difference
        phase_diff = phase1 - phase2

        # Interference amplitude (vectorized across all frequencies)
        interference_amp = tf.sqrt(
            amp1**2 + amp2**2 + 2*amp1*amp2*tf.cos(phase_diff)
        )

        # Interference phase (vectorized)
        interference_phase = tf.atan2(
            amp1*tf.sin(phase1) + amp2*tf.sin(phase2),
            amp1*tf.cos(phase1) + amp2*tf.cos(phase2)
        )

        # Construct complex result - ensure proper complex multiplication
        # Convert amplitude to complex64 for proper dtype matching
        complex_amplitude = tf.cast(interference_amp, tf.complex64)
        phase_factor = tf.exp(tf.complex(tf.cast(0.0, tf.float32), interference_phase))
        result = complex_amplitude * phase_factor

        return result
    
    @tf.function(reduce_retracing=True)
    def accelerated_resonance_processing(self,
                                       colour_input: tf.Tensor,
                                       resonance_matrices: List[tf.Tensor],
                                       num_iterations: int = 3) -> tf.Tensor:
        """
        GPU-accelerated resonance chamber processing.
        
        Processes colour through multiple resonance chambers in parallel.
        """
        
        # Stack resonance matrices for batch processing
        if resonance_matrices:
            matrix_stack = tf.stack(resonance_matrices)
            
            # Parallel resonance processing
            result = self._batch_resonance_processing(
                colour_input, matrix_stack, num_iterations
            )
        else:
            # No resonance matrices - return input
            result = colour_input
        
        return result
    
    @tf.function(reduce_retracing=True)
    def _batch_resonance_processing(self,
                                  colour_input: tf.Tensor,
                                  matrix_stack: tf.Tensor,
                                  num_iterations: int) -> tf.Tensor:
        """Batch process colour through multiple resonance matrices."""
        
        # Expand input for batch processing
        batch_input = tf.expand_dims(colour_input, 0)
        
        # Apply each resonance matrix
        def apply_resonance(matrix):
            return self._single_resonance_step(batch_input, matrix, num_iterations)
        
        # Map over all matrices
        resonance_results = tf.map_fn(
            apply_resonance,
            matrix_stack,
            fn_output_signature=tf.TensorSpec([None, self.spectrum_dims], dtype=tf.complex64),
            parallel_iterations=10
        )
        
        # Combine results through interference
        combined_result = self._parallel_interference_reduction(
            resonance_results, 'adaptive'
        )
        
        return combined_result
    
    @tf.function(reduce_retracing=True)
    def _single_resonance_step(self,
                             colour_input: tf.Tensor,
                             resonance_matrix: tf.Tensor,
                             num_iterations: int) -> tf.Tensor:
        """Single resonance step with iterative processing."""

        # Ensure input is complex64
        state = tf.cast(colour_input, tf.complex64)
        resonance_matrix = tf.cast(resonance_matrix, tf.float32)

        for i in tf.range(num_iterations):
            # Apply resonance transformation
            real_part = tf.linalg.matvec(resonance_matrix, tf.real(state))
            imag_part = tf.linalg.matvec(resonance_matrix, tf.imag(state))

            transformed = tf.complex(real_part, imag_part)

            # Mix with original (partial feedback) - ensure complex64 arithmetic
            feedback_strength = tf.cast(0.3 * tf.cast(i + 1, tf.float32) / tf.cast(num_iterations, tf.float32), tf.complex64)
            one_minus_feedback = tf.cast(1.0, tf.complex64) - feedback_strength
            state = one_minus_feedback * state + feedback_strength * transformed

            # Normalize to prevent explosion
            state = self._gpu_normalize_colour(state)

        return state
    
    @tf.function(reduce_retracing=True)
    def accelerated_spectral_analysis(self, colours: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        GPU-accelerated spectral analysis of colour patterns.
        
        Computes FFT, power spectrum, dominant frequencies, and complexity metrics
        in parallel across multiple colour patterns.
        """
        
        # Ensure colours are complex
        if colours.dtype != tf.complex64:
            colours = tf.cast(colours, tf.complex64)
        
        # Batch FFT computation
        fft_colours = tf.signal.fft(colours)
        
        # Power spectrum
        power_spectrum = tf.abs(fft_colours)**2
        
        # Spectral entropy (complexity measure)
        normalized_power = power_spectrum / (tf.reduce_sum(power_spectrum, axis=-1, keepdims=True) + 1e-8)
        spectral_entropy = -tf.reduce_sum(
            normalized_power * tf.math.log(normalized_power + 1e-8),
            axis=-1
        )
        
        # Dominant frequencies (top 5 for each colour)
        _, top_freq_indices = tf.nn.top_k(power_spectrum, k=5)
        
        # Phase coherence
        phase_coherence = tf.abs(tf.reduce_mean(
            tf.exp(tf.complex(0.0, tf.angle(colours))), axis=-1
        ))
        
        return {
            'fft_spectrum': fft_colours,
            'power_spectrum': power_spectrum,
            'spectral_entropy': spectral_entropy,
            'dominant_frequencies': top_freq_indices,
            'phase_coherence': phase_coherence
        }
    
    @tf.function(reduce_retracing=True)
    def accelerated_memory_search(self,
                                query_colour: tf.Tensor,
                                memory_colours: tf.Tensor,
                                memory_strengths: tf.Tensor,
                                top_k: int = 5) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        GPU-accelerated memory search using parallel similarity computation.
        
        Finds the most similar colours in memory using GPU parallelism.
        """
        
        # Expand query for broadcasting
        query_expanded = tf.expand_dims(query_colour, 0)
        
        # Parallel similarity computation
        similarities = self._batch_colour_similarity(query_expanded, memory_colours)
        
        # Weight by memory strengths
        weighted_similarities = similarities * memory_strengths
        
        # Get top-k most similar
        top_similarities, top_indices = tf.nn.top_k(weighted_similarities, k=top_k)
        
        return top_similarities, top_indices
    
    @tf.function(reduce_retracing=True)  
    def _batch_colour_similarity(self,
                               query: tf.Tensor,
                               memories: tf.Tensor) -> tf.Tensor:
        """Compute colour similarity in parallel across all memories."""
        
        # Complex dot product similarity
        complex_similarities = tf.reduce_sum(
            query * tf.math.conj(memories), axis=-1
        )
        
        # Magnitude gives similarity strength
        similarities = tf.abs(complex_similarities)
        
        return similarities
    
    @tf.function(reduce_retracing=True)
    def _gpu_normalize_colour(self, colour: tf.Tensor) -> tf.Tensor:
        """GPU-optimized colour normalization."""

        # Ensure colour is complex64
        colour = tf.cast(colour, tf.complex64)

        # Compute magnitude
        magnitude = tf.abs(colour)
        max_mag = tf.reduce_max(magnitude)

        # Prevent explosion - ensure complex division
        colour = tf.cond(
            max_mag > 10.0,
            lambda: colour / tf.cast(max_mag / 2.0, tf.complex64),
            lambda: colour
        )

        # Prevent collapse - ensure complex multiplication
        mean_mag = tf.reduce_mean(tf.abs(colour))
        colour = tf.cond(
            mean_mag < 0.01,
            lambda: colour * tf.cast(0.1 / (mean_mag + 1e-8), tf.complex64),
            lambda: colour
        )

        return colour
    
    def create_optimized_resonance_matrix(self, 
                                        matrix_size: int,
                                        resonance_strength: float = 0.1) -> tf.Tensor:
        """
        Create resonance matrix optimized for GPU computation.
        
        The matrix structure is designed for efficient GPU matrix operations.
        """
        
        # Create base matrix with structure that promotes GPU cache efficiency
        indices = tf.range(matrix_size, dtype=tf.float32)
        
        # Create circulant matrix structure (good for FFT-based operations)  
        i, j = tf.meshgrid(indices, indices, indexing='ij')
        
        # Distance-based coupling with periodic boundary conditions
        distance = tf.minimum(
            tf.abs(i - j),
            matrix_size - tf.abs(i - j)
        )
        
        # Exponential coupling decay
        coupling = resonance_strength * tf.exp(-distance / (matrix_size / 4))
        
        # Add small diagonal for numerical stability
        coupling = coupling + 0.01 * tf.eye(matrix_size)
        
        return coupling
    
    def benchmark_gpu_performance(self, test_size: int = 1000) -> Dict[str, float]:
        """
        Benchmark GPU performance for colour operations.
        
        Returns timing information for different operations.
        """
        
        self.logger.info(f"Benchmarking GPU performance with {test_size} test colours")
        
        # Create test data
        test_colours = tf.random.normal([test_size, self.spectrum_dims], dtype=tf.float32)
        test_colours = tf.cast(test_colours, tf.complex64)
        
        # Benchmark interference
        start_time = tf.timestamp()
        interference_result = self.accelerated_colour_interference(
            [test_colours[i] for i in range(min(10, test_size))]
        )
        interference_time = tf.timestamp() - start_time
        
        # Benchmark spectral analysis  
        start_time = tf.timestamp()
        spectral_results = self.accelerated_spectral_analysis(test_colours[:100])
        spectral_time = tf.timestamp() - start_time
        
        # Benchmark memory search
        start_time = tf.timestamp()
        query = test_colours[0]
        memory_colours = test_colours[1:501]  # 500 memories
        memory_strengths = tf.ones(500)
        similarities, indices = self.accelerated_memory_search(
            query, memory_colours, memory_strengths
        )
        memory_search_time = tf.timestamp() - start_time
        
        # Benchmark resonance
        resonance_matrix = self.create_optimized_resonance_matrix(self.spectrum_dims)
        start_time = tf.timestamp()
        resonance_result = self.accelerated_resonance_processing(
            test_colours[0], [resonance_matrix], num_iterations=5
        )
        resonance_time = tf.timestamp() - start_time
        
        results = {
            'colour_interference_ms': float(interference_time * 1000),
            'spectral_analysis_ms': float(spectral_time * 1000),
            'memory_search_ms': float(memory_search_time * 1000),
            'resonance_processing_ms': float(resonance_time * 1000)
        }
        
        self.logger.info(f"GPU benchmark results: {results}")
        
        return results


# Factory function for GPU accelerator
def create_gpu_accelerator(spectrum_dims: int = 512,
                         mixed_precision: bool = False) -> GPUColourAccelerator:
    """
    Create GPU colour accelerator with optimal settings.
    
    Args:
        spectrum_dims: Number of colour spectrum dimensions
        mixed_precision: Enable mixed precision for faster computation
        
    Returns:
        Configured GPU colour accelerator
    """
    
    return GPUColourAccelerator(
        spectrum_dims=spectrum_dims,
        use_mixed_precision=mixed_precision
    )