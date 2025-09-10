"""Benchmarking and profiling utilities for performance optimization"""

import tensorflow as tf
import numpy as np
import time
import psutil
import os
from typing import Dict, Any, Callable, Optional, List
import matplotlib.pyplot as plt
from contextlib import contextmanager


def benchmark_layer(layer: tf.keras.layers.Layer,
                   input_data: tf.Tensor,
                   num_warmup: int = 3,
                   num_runs: int = 10,
                   training: bool = False) -> Dict[str, float]:
    """
    Benchmark a TensorFlow layer's performance
    
    Args:
        layer: Layer to benchmark
        input_data: Input tensor(s) for the layer
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        training: Whether to run in training mode
        
    Returns:
        Dictionary with timing statistics
    """
    # Ensure layer is built
    if not layer.built:
        _ = layer(input_data)
    
    # Warmup runs
    for _ in range(num_warmup):
        if isinstance(input_data, list):
            _ = layer(input_data, training=training)
        else:
            _ = layer(input_data, training=training)
    
    # Benchmark runs
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        if isinstance(input_data, list):
            result = layer(input_data, training=training)
        else:
            result = layer(input_data, training=training)
            
        # Ensure computation is complete
        if hasattr(result, 'numpy'):
            _ = result.numpy()
        elif isinstance(result, list):
            for r in result:
                if hasattr(r, 'numpy'):
                    _ = r.numpy()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Calculate statistics
    times = np.array(times)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'median_time': np.median(times),
        'percentile_95': np.percentile(times, 95),
        'times': times.tolist()
    }


@contextmanager
def profile_memory_usage():
    """
    Context manager to profile memory usage
    
    Yields:
        Dictionary with memory statistics during execution
    """
    process = psutil.Process(os.getpid())
    
    # Get initial memory info
    initial_memory = process.memory_info()
    peak_memory = initial_memory.rss
    memory_samples = [initial_memory.rss]
    
    class MemoryProfiler:
        def __init__(self):
            self.peak_memory = peak_memory
            self.memory_samples = memory_samples
            
        def sample(self):
            current_memory = process.memory_info().rss
            self.memory_samples.append(current_memory)
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
                
        def get_stats(self):
            return {
                'initial_memory_mb': initial_memory.rss / 1024 / 1024,
                'peak_memory_mb': self.peak_memory / 1024 / 1024,
                'final_memory_mb': self.memory_samples[-1] / 1024 / 1024,
                'memory_increase_mb': (self.memory_samples[-1] - initial_memory.rss) / 1024 / 1024,
                'peak_increase_mb': (self.peak_memory - initial_memory.rss) / 1024 / 1024,
                'memory_samples': [m / 1024 / 1024 for m in self.memory_samples]  # Convert to MB
            }
    
    profiler = MemoryProfiler()
    
    try:
        yield profiler
    finally:
        # Final sample
        profiler.sample()


def compare_implementations(implementations: Dict[str, Callable],
                           input_data: tf.Tensor,
                           num_runs: int = 10) -> Dict[str, Dict[str, float]]:
    """
    Compare performance of different implementations
    
    Args:
        implementations: Dictionary mapping names to callable implementations
        input_data: Input data for all implementations
        num_runs: Number of runs for each implementation
        
    Returns:
        Dictionary with benchmark results for each implementation
    """
    results = {}
    
    for name, impl in implementations.items():
        print(f"Benchmarking {name}...")
        
        # Create a wrapper function for benchmarking
        def wrapper():
            return impl(input_data)
        
        # Time the implementation
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            result = wrapper()
            
            # Ensure computation is complete
            if hasattr(result, 'numpy'):
                _ = result.numpy()
            elif isinstance(result, (list, tuple)):
                for r in result:
                    if hasattr(r, 'numpy'):
                        _ = r.numpy()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        times = np.array(times)
        results[name] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times)
        }
    
    # Calculate speedup ratios
    if len(results) > 1:
        baseline_time = min(r['mean_time'] for r in results.values())
        for name, result in results.items():
            result['speedup'] = baseline_time / result['mean_time']
    
    return results


def benchmark_gradient_computation(model: tf.keras.Model,
                                  input_data: tf.Tensor,
                                  target_data: tf.Tensor,
                                  loss_fn: Callable,
                                  num_runs: int = 5) -> Dict[str, float]:
    """
    Benchmark gradient computation performance
    
    Args:
        model: Model to benchmark
        input_data: Input tensor
        target_data: Target tensor for loss computation
        loss_fn: Loss function
        num_runs: Number of runs
        
    Returns:
        Timing statistics for gradient computation
    """
    times = []
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        with tf.GradientTape() as tape:
            predictions = model(input_data, training=True)
            loss = loss_fn(target_data, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Ensure gradients are computed
        for grad in gradients:
            if grad is not None:
                _ = grad.numpy()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }


def profile_layer_operations(layer: tf.keras.layers.Layer,
                           input_data: tf.Tensor,
                           operation_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Profile individual operations within a layer
    
    Args:
        layer: Layer to profile
        input_data: Input data
        operation_names: Names of operations to track
        
    Returns:
        Profiling results
    """
    # This is a simplified profiler - TensorFlow Profiler would be more comprehensive
    
    with tf.profiler.experimental.Profile('logdir'):
        if isinstance(input_data, list):
            result = layer(input_data, training=True)
        else:
            result = layer(input_data, training=True)
    
    # Basic timing of different aspects
    results = {}
    
    # Time just the forward pass
    start_time = time.perf_counter()
    if isinstance(input_data, list):
        _ = layer(input_data, training=False)
    else:
        _ = layer(input_data, training=False)
    forward_time = time.perf_counter() - start_time
    
    results['forward_pass_time'] = forward_time
    
    # Time forward pass with training=True (includes additional ops)
    start_time = time.perf_counter()
    if isinstance(input_data, list):
        _ = layer(input_data, training=True)
    else:
        _ = layer(input_data, training=True)
    training_time = time.perf_counter() - start_time
    
    results['training_pass_time'] = training_time
    results['training_overhead'] = training_time - forward_time
    
    return results


def plot_benchmark_results(results: Dict[str, Dict[str, float]],
                          metric: str = 'mean_time',
                          title: str = "Performance Comparison",
                          save_path: Optional[str] = None):
    """
    Plot benchmark results
    
    Args:
        results: Results from compare_implementations
        metric: Metric to plot ('mean_time', 'speedup', etc.)
        title: Plot title
        save_path: Optional path to save plot
    """
    names = list(results.keys())
    values = [results[name][metric] for name in names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom')
    
    plt.title(title)
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Color bars based on performance (green = better)
    if metric == 'mean_time':
        # Lower is better for time
        min_val = min(values)
        colors = ['green' if v == min_val else 'orange' for v in values]
    elif metric == 'speedup':
        # Higher is better for speedup
        max_val = max(values)
        colors = ['green' if v == max_val else 'orange' for v in values]
    else:
        colors = ['blue'] * len(values)
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def benchmark_batch_scaling(layer: tf.keras.layers.Layer,
                           input_shape: tuple,
                           batch_sizes: List[int],
                           dtype: tf.DType = tf.float32) -> Dict[int, Dict[str, float]]:
    """
    Benchmark how performance scales with batch size
    
    Args:
        layer: Layer to benchmark
        input_shape: Shape of input (without batch dimension)
        batch_sizes: List of batch sizes to test
        dtype: Data type for inputs
        
    Returns:
        Dictionary mapping batch sizes to benchmark results
    """
    results = {}
    
    for batch_size in batch_sizes:
        print(f"Testing batch size {batch_size}...")
        
        # Create input data
        full_shape = (batch_size,) + input_shape
        if dtype.is_complex:
            input_data = tf.complex(
                tf.random.normal(full_shape, dtype=tf.float32),
                tf.random.normal(full_shape, dtype=tf.float32)
            )
        else:
            input_data = tf.random.normal(full_shape, dtype=dtype)
        
        # Benchmark this batch size
        benchmark_result = benchmark_layer(layer, input_data, num_runs=5)
        
        # Calculate throughput (samples per second)
        throughput = batch_size / benchmark_result['mean_time']
        benchmark_result['throughput'] = throughput
        benchmark_result['batch_size'] = batch_size
        
        results[batch_size] = benchmark_result
    
    return results


def plot_batch_scaling(scaling_results: Dict[int, Dict[str, float]],
                      title: str = "Batch Size Scaling",
                      save_path: Optional[str] = None):
    """
    Plot batch scaling results
    
    Args:
        scaling_results: Results from benchmark_batch_scaling
        title: Plot title
        save_path: Optional path to save plot
    """
    batch_sizes = sorted(scaling_results.keys())
    mean_times = [scaling_results[bs]['mean_time'] for bs in batch_sizes]
    throughputs = [scaling_results[bs]['throughput'] for bs in batch_sizes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Time vs batch size
    ax1.plot(batch_sizes, mean_times, 'o-', color='blue', markersize=8)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Mean Time (seconds)')
    ax1.set_title('Execution Time vs Batch Size')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Throughput vs batch size
    ax2.plot(batch_sizes, throughputs, 'o-', color='green', markersize=8)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (samples/second)')
    ax2.set_title('Throughput vs Batch Size')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


class PerformanceMonitor:
    """
    Real-time performance monitoring during training
    """
    
    def __init__(self):
        self.metrics = {
            'step_times': [],
            'memory_usage': [],
            'gpu_memory': [],
            'step_numbers': []
        }
        self.step_counter = 0
        
    def start_step(self):
        """Start timing a training step"""
        self.step_start_time = time.perf_counter()
        
    def end_step(self):
        """End timing a training step and record metrics"""
        step_time = time.perf_counter() - self.step_start_time
        
        # Record metrics
        self.metrics['step_times'].append(step_time)
        self.metrics['step_numbers'].append(self.step_counter)
        
        # Memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.metrics['memory_usage'].append(memory_mb)
        
        # GPU memory if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                gpu_memory_mb = gpu_memory['current'] / 1024 / 1024
                self.metrics['gpu_memory'].append(gpu_memory_mb)
            except:
                self.metrics['gpu_memory'].append(0)
        else:
            self.metrics['gpu_memory'].append(0)
            
        self.step_counter += 1
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        if not self.metrics['step_times']:
            return {}
            
        recent_times = self.metrics['step_times'][-10:]  # Last 10 steps
        
        return {
            'mean_step_time': np.mean(recent_times),
            'steps_per_second': 1.0 / np.mean(recent_times),
            'current_memory_mb': self.metrics['memory_usage'][-1],
            'current_gpu_memory_mb': self.metrics['gpu_memory'][-1],
            'total_steps': self.step_counter
        }
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot collected metrics"""
        if not self.metrics['step_times']:
            print("No metrics to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        steps = self.metrics['step_numbers']
        
        # Step times
        axes[0, 0].plot(steps, self.metrics['step_times'], 'b-', alpha=0.7)
        axes[0, 0].set_title('Step Times')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory usage
        axes[0, 1].plot(steps, self.metrics['memory_usage'], 'g-', alpha=0.7)
        axes[0, 1].set_title('CPU Memory Usage')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # GPU memory
        if any(gpu_mem > 0 for gpu_mem in self.metrics['gpu_memory']):
            axes[1, 0].plot(steps, self.metrics['gpu_memory'], 'r-', alpha=0.7)
            axes[1, 0].set_title('GPU Memory Usage')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Memory (MB)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Throughput (steps per second)
        if len(self.metrics['step_times']) > 1:
            window_size = min(10, len(self.metrics['step_times']))
            throughputs = []
            for i in range(window_size - 1, len(self.metrics['step_times'])):
                recent_times = self.metrics['step_times'][i-window_size+1:i+1]
                throughput = 1.0 / np.mean(recent_times)
                throughputs.append(throughput)
            
            throughput_steps = steps[window_size-1:]
            axes[1, 1].plot(throughput_steps, throughputs, 'purple', alpha=0.7)
            axes[1, 1].set_title('Throughput (Steps/Second)')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Steps/Second')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()