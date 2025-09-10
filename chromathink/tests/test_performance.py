"""Performance benchmarks on GPU for high-dimensional colour spaces"""

import pytest
import tensorflow as tf
import numpy as np
import time
from ..layers.cognitive_waveform import CognitiveWaveform
from ..layers.interference import InterferenceLayer
from ..layers.chromatic_resonance import ChromaticResonance
from ..layers.resonance_chamber import ResonanceChamber


class TestPerformance:
    """Test suite for GPU performance benchmarks"""
    
    @pytest.fixture(scope="class")
    def gpu_available(self):
        """Check if GPU is available for testing"""
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            pytest.skip("No GPU available for performance testing")
        return len(gpus) > 0
    
    def setup_method(self):
        """Setup for each test method"""
        # Clear any existing graphs and reset default graph
        tf.keras.backend.clear_session()
        
        # Enable memory growth to avoid OOM issues
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    
    def benchmark_function(self, func, warmup_runs=3, benchmark_runs=10):
        """Benchmark a function with warmup and multiple runs"""
        # Warmup runs
        for _ in range(warmup_runs):
            result = func()
            if hasattr(result, 'numpy'):
                _ = result.numpy()  # Force evaluation
        
        # Benchmark runs
        times = []
        for _ in range(benchmark_runs):
            start_time = time.time()
            result = func()
            if hasattr(result, 'numpy'):
                _ = result.numpy()  # Force evaluation
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'times': times
        }
    
    @pytest.mark.skipif(not tf.config.list_physical_devices('GPU'), reason="GPU not available")
    def test_cognitive_waveform_performance(self, gpu_available):
        """Benchmark CognitiveWaveform layer performance"""
        dimensions = [128, 256, 512, 1024]
        batch_sizes = [1, 8, 32, 64]
        
        results = {}
        
        for dim in dimensions:
            for batch_size in batch_sizes:
                layer = CognitiveWaveform(
                    dimensions=dim,
                    use_fft=True,
                    spectral_normalize=True
                )
                
                # Create test data
                input_data = tf.random.normal([batch_size, dim])
                
                # Build layer
                _ = layer(input_data)
                
                # Benchmark forward pass
                def forward_pass():
                    return layer(input_data, training=False)
                
                benchmark_result = self.benchmark_function(forward_pass)
                
                key = f"dim_{dim}_batch_{batch_size}"
                results[key] = benchmark_result
                
                # Performance assertions
                mean_time = benchmark_result['mean_time']
                
                # Should complete within reasonable time
                max_allowed_time = 0.1  # 100ms for forward pass
                assert mean_time < max_allowed_time, \
                    f"CognitiveWaveform too slow: {mean_time:.4f}s > {max_allowed_time}s for {key}"
                
                # Variance shouldn't be too high (consistent performance)
                cv = benchmark_result['std_time'] / mean_time if mean_time > 0 else 0
                assert cv < 0.5, f"Performance too variable for {key}: CV={cv:.3f}"
        
        # Print results for analysis
        print("\nCognitiveWaveform Performance Results:")
        for key, result in results.items():
            print(f"{key}: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")
    
    @pytest.mark.skipif(not tf.config.list_physical_devices('GPU'), reason="GPU not available")
    def test_interference_layer_performance(self, gpu_available):
        """Benchmark InterferenceLayer performance"""
        dimensions = [64, 128, 256, 512]
        batch_sizes = [1, 8, 32]
        
        results = {}
        
        for dim in dimensions:
            for batch_size in batch_sizes:
                layer = InterferenceLayer(
                    interference_type='full',
                    amplitude_weighting=True,
                    phase_coupling=True,
                    nonlinear_mixing=True
                )
                
                # Create test data
                wave1 = tf.complex(
                    tf.random.normal([batch_size, dim]),
                    tf.random.normal([batch_size, dim]) * 0.1
                )
                wave2 = tf.complex(
                    tf.random.normal([batch_size, dim]),
                    tf.random.normal([batch_size, dim]) * 0.1
                )
                
                # Build layer
                _ = layer([wave1, wave2])
                
                # Benchmark forward pass
                def forward_pass():
                    return layer([wave1, wave2], training=False)
                
                benchmark_result = self.benchmark_function(forward_pass)
                
                key = f"dim_{dim}_batch_{batch_size}"
                results[key] = benchmark_result
                
                # Performance assertions
                mean_time = benchmark_result['mean_time']
                max_allowed_time = 0.15  # 150ms for interference calculation
                
                assert mean_time < max_allowed_time, \
                    f"InterferenceLayer too slow: {mean_time:.4f}s > {max_allowed_time}s for {key}"
        
        print("\nInterferenceLayer Performance Results:")
        for key, result in results.items():
            print(f"{key}: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")
    
    @pytest.mark.skipif(not tf.config.list_physical_devices('GPU'), reason="GPU not available")
    def test_chromatic_resonance_performance(self, gpu_available):
        """Benchmark ChromaticResonance layer performance"""
        dimensions = [64, 128, 256]
        resonance_depths = [3, 5, 7]
        batch_size = 8
        
        results = {}
        
        for dim in dimensions:
            for depth in resonance_depths:
                layer = ChromaticResonance(
                    dimensions=dim,
                    resonance_depth=depth,
                    harmonic_orders=[1, 2, 3]
                )
                
                # Create test data
                input_wave = tf.complex(
                    tf.random.normal([batch_size, dim]),
                    tf.random.normal([batch_size, dim]) * 0.1
                )
                
                # Build layer
                _ = layer(input_wave)
                
                # Benchmark forward pass
                def forward_pass():
                    return layer(input_wave, training=False)
                
                benchmark_result = self.benchmark_function(forward_pass)
                
                key = f"dim_{dim}_depth_{depth}"
                results[key] = benchmark_result
                
                # Performance assertions - should scale with depth
                mean_time = benchmark_result['mean_time']
                max_allowed_time = 0.05 * depth  # Scale with resonance depth
                
                assert mean_time < max_allowed_time, \
                    f"ChromaticResonance too slow: {mean_time:.4f}s > {max_allowed_time}s for {key}"
        
        print("\nChromaticResonance Performance Results:")
        for key, result in results.items():
            print(f"{key}: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")
    
    @pytest.mark.skipif(not tf.config.list_physical_devices('GPU'), reason="GPU not available")
    def test_resonance_chamber_performance(self, gpu_available):
        """Benchmark ResonanceChamber performance"""
        dimensions = [64, 128, 256]
        num_modes_list = [8, 16, 32]
        batch_size = 4
        
        results = {}
        
        for dim in dimensions:
            for num_modes in num_modes_list:
                layer = ResonanceChamber(
                    dimensions=dim,
                    num_modes=num_modes,
                    quality_factor=50.0
                )
                
                # Create test data
                input_wave = tf.complex(
                    tf.random.normal([batch_size, dim]),
                    tf.random.normal([batch_size, dim]) * 0.1
                )
                
                # Build layer
                _ = layer(input_wave)
                
                # Benchmark forward pass
                def forward_pass():
                    return layer(input_wave, training=False)
                
                benchmark_result = self.benchmark_function(forward_pass)
                
                key = f"dim_{dim}_modes_{num_modes}"
                results[key] = benchmark_result
                
                # Performance assertions
                mean_time = benchmark_result['mean_time']
                max_allowed_time = 0.2  # 200ms for chamber calculation
                
                assert mean_time < max_allowed_time, \
                    f"ResonanceChamber too slow: {mean_time:.4f}s > {max_allowed_time}s for {key}"
        
        print("\nResonanceChamber Performance Results:")
        for key, result in results.items():
            print(f"{key}: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")
    
    @pytest.mark.skipif(not tf.config.list_physical_devices('GPU'), reason="GPU not available")
    def test_memory_usage(self, gpu_available):
        """Test memory usage with large dimensions"""
        # Test progressively larger dimensions to check memory scaling
        dimensions = [256, 512, 1024, 2048]
        batch_size = 16
        
        for dim in dimensions:
            try:
                # Create a complex pipeline
                waveform_layer = CognitiveWaveform(dimensions=dim, spectral_normalize=True)
                resonance_layer = ChromaticResonance(dimensions=dim, resonance_depth=3)
                
                # Create test data
                input_data = tf.random.normal([batch_size, dim])
                
                # Forward pass through pipeline
                waveform = waveform_layer(input_data)
                resonated = resonance_layer(waveform)
                
                # Force computation
                result = tf.reduce_mean(tf.abs(resonated)).numpy()
                
                print(f"Dimension {dim}: Memory usage OK, result={result:.4f}")
                
            except tf.errors.ResourceExhaustedError:
                print(f"Dimension {dim}: Out of memory - this is expected for large dimensions")
                break
            except Exception as e:
                pytest.fail(f"Unexpected error at dimension {dim}: {e}")
    
    @pytest.mark.skipif(not tf.config.list_physical_devices('GPU'), reason="GPU not available")
    def test_gradient_computation_performance(self, gpu_available):
        """Benchmark gradient computation performance"""
        dimensions = [128, 256, 512]
        batch_size = 8
        
        for dim in dimensions:
            # Create model
            waveform_layer = CognitiveWaveform(dimensions=dim)
            resonance_layer = ChromaticResonance(dimensions=dim, resonance_depth=5)
            
            # Create test data and target
            input_data = tf.random.normal([batch_size, dim])
            target = tf.random.normal([batch_size, dim])
            
            # Build layers
            _ = waveform_layer(input_data)
            waveform = waveform_layer(input_data)
            _ = resonance_layer(waveform)
            
            # Benchmark gradient computation
            def compute_gradients():
                with tf.GradientTape() as tape:
                    waveform = waveform_layer(input_data, training=True)
                    output = resonance_layer(waveform, training=True)
                    
                    # Simple loss
                    loss = tf.reduce_mean(tf.square(tf.real(output) - target))
                
                # Compute gradients
                all_vars = waveform_layer.trainable_variables + resonance_layer.trainable_variables
                gradients = tape.gradient(loss, all_vars)
                
                return gradients
            
            benchmark_result = self.benchmark_function(compute_gradients, warmup_runs=2, benchmark_runs=5)
            
            mean_time = benchmark_result['mean_time']
            max_allowed_time = 0.5  # 500ms for gradient computation
            
            assert mean_time < max_allowed_time, \
                f"Gradient computation too slow for dim {dim}: {mean_time:.4f}s > {max_allowed_time}s"
            
            print(f"Gradient computation dim {dim}: {mean_time:.4f}s ± {benchmark_result['std_time']:.4f}s")
    
    @pytest.mark.skipif(not tf.config.list_physical_devices('GPU'), reason="GPU not available")
    def test_batch_size_scaling(self, gpu_available):
        """Test performance scaling with batch size"""
        dimensions = 256
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        layer = ChromaticResonance(dimensions=dimensions, resonance_depth=5)
        
        results = []
        
        for batch_size in batch_sizes:
            input_wave = tf.complex(
                tf.random.normal([batch_size, dimensions]),
                tf.random.normal([batch_size, dimensions]) * 0.1
            )
            
            # Build layer if needed
            if not layer.built:
                _ = layer(input_wave)
            
            # Benchmark
            def forward_pass():
                return layer(input_wave, training=False)
            
            benchmark_result = self.benchmark_function(forward_pass, warmup_runs=2, benchmark_runs=5)
            
            results.append({
                'batch_size': batch_size,
                'mean_time': benchmark_result['mean_time'],
                'throughput': batch_size / benchmark_result['mean_time']  # samples/second
            })
        
        # Check scaling properties
        print("\nBatch Size Scaling Results:")
        for result in results:
            print(f"Batch {result['batch_size']}: {result['mean_time']:.4f}s, {result['throughput']:.1f} samples/s")
        
        # Throughput should generally increase with batch size (up to a point)
        throughputs = [r['throughput'] for r in results]
        
        # At least the throughput with batch_size=32 should be higher than batch_size=1
        if len(throughputs) >= 6:  # Ensure we have enough data points
            assert throughputs[5] > throughputs[0] * 2, "Insufficient throughput improvement with batching"
    
    @pytest.mark.skipif(not tf.config.list_physical_devices('GPU'), reason="GPU not available")
    def test_mixed_precision_performance(self, gpu_available):
        """Test performance with mixed precision training"""
        if not hasattr(tf.keras.mixed_precision, 'set_global_policy'):
            pytest.skip("Mixed precision not available in this TensorFlow version")
        
        dimensions = 512
        batch_size = 16
        
        # Test with float32 (default)
        tf.keras.mixed_precision.set_global_policy('float32')
        
        layer_fp32 = ChromaticResonance(dimensions=dimensions, resonance_depth=5)
        input_data = tf.complex(
            tf.random.normal([batch_size, dimensions]),
            tf.random.normal([batch_size, dimensions]) * 0.1
        )
        
        # Build and benchmark
        _ = layer_fp32(input_data)
        
        def forward_pass_fp32():
            return layer_fp32(input_data, training=False)
        
        fp32_result = self.benchmark_function(forward_pass_fp32, warmup_runs=2, benchmark_runs=5)
        
        # Test with mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        layer_mixed = ChromaticResonance(dimensions=dimensions, resonance_depth=5)
        
        # Build and benchmark
        _ = layer_mixed(input_data)
        
        def forward_pass_mixed():
            return layer_mixed(input_data, training=False)
        
        mixed_result = self.benchmark_function(forward_pass_mixed, warmup_runs=2, benchmark_runs=5)
        
        # Reset policy
        tf.keras.mixed_precision.set_global_policy('float32')
        
        # Mixed precision should be faster or at least not significantly slower
        speedup = fp32_result['mean_time'] / mixed_result['mean_time']
        
        print(f"Mixed precision speedup: {speedup:.2f}x")
        print(f"FP32: {fp32_result['mean_time']:.4f}s")
        print(f"Mixed: {mixed_result['mean_time']:.4f}s")
        
        # Should not be significantly slower
        assert speedup > 0.8, f"Mixed precision too slow: {speedup:.2f}x speedup"
    
    @pytest.mark.skipif(not tf.config.list_physical_devices('GPU'), reason="GPU not available")
    def test_concurrent_operations(self, gpu_available):
        """Test performance with concurrent operations"""
        dimensions = 256
        batch_size = 8
        
        # Create multiple layers
        layers = [
            ChromaticResonance(dimensions=dimensions, resonance_depth=3),
            ChromaticResonance(dimensions=dimensions, resonance_depth=3),
            ChromaticResonance(dimensions=dimensions, resonance_depth=3)
        ]
        
        # Create test data
        inputs = [
            tf.complex(tf.random.normal([batch_size, dimensions]), tf.random.normal([batch_size, dimensions]) * 0.1)
            for _ in range(3)
        ]
        
        # Build layers
        for i, layer in enumerate(layers):
            _ = layer(inputs[i])
        
        # Sequential execution
        def sequential_execution():
            results = []
            for i, layer in enumerate(layers):
                result = layer(inputs[i], training=False)
                results.append(result)
            return results
        
        sequential_result = self.benchmark_function(sequential_execution, warmup_runs=2, benchmark_runs=5)
        
        # Note: True parallel execution is complex in TensorFlow
        # This test mainly ensures that multiple operations don't interfere catastrophically
        
        mean_time = sequential_result['mean_time']
        max_allowed_time = 0.3  # 300ms for three operations
        
        assert mean_time < max_allowed_time, \
            f"Concurrent operations too slow: {mean_time:.4f}s > {max_allowed_time}s"
        
        print(f"Sequential execution of 3 layers: {mean_time:.4f}s ± {sequential_result['std_time']:.4f}s")