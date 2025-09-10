"""Advanced examples demonstrating ChromaThink's capabilities"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from chromathink.layers.cognitive_waveform import CognitiveWaveform
from chromathink.layers.interference import InterferenceLayer, MultiWaveInterference  
from chromathink.layers.chromatic_resonance import ChromaticResonance
from chromathink.layers.resonance_chamber import ResonanceChamber
from chromathink.core.colour_utils import colour_interpolation, colour_distance
from chromathink.utils.visualization import ColourSpaceVisualizer
from chromathink.utils.benchmarking import PerformanceMonitor, benchmark_batch_scaling


class ChromaThinkModel(tf.keras.Model):
    """Complete ChromaThink model for sequence processing"""
    
    def __init__(self, 
                 input_dim: int,
                 colour_dimensions: int = 256,
                 resonance_depth: int = 7,
                 num_modes: int = 32,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.colour_dimensions = colour_dimensions
        
        # Core layers
        self.waveform_layer = CognitiveWaveform(
            dimensions=colour_dimensions,
            frequency_range=(0.01, 1000.0),
            spectral_normalize=True
        )
        
        self.resonance_layer = ChromaticResonance(
            dimensions=colour_dimensions,
            resonance_depth=resonance_depth,
            harmonic_orders=[1, 2, 3, 5, 7],
            nonlinearity='gelu'
        )
        
        self.chamber_layer = ResonanceChamber(
            dimensions=colour_dimensions,
            num_modes=num_modes,
            boundary_conditions='mixed',
            quality_factor=75.0
        )
        
        self.output_projection = tf.keras.layers.Dense(input_dim)
    
    def call(self, inputs, training=None):
        """Forward pass through ChromaThink pipeline"""
        # Transform to colour space
        colours = self.waveform_layer(inputs, training=training)
        
        # Apply chromatic resonance
        resonated = self.resonance_layer(colours, training=training)
        
        # Create standing waves
        standing_waves = self.chamber_layer(resonated, training=training)
        
        # Project back to output space
        # Take real part for output projection
        real_output = tf.concat([tf.real(standing_waves), tf.imag(standing_waves)], axis=-1)
        output = self.output_projection(real_output)
        
        return output, standing_waves


def creative_colour_mixing_example():
    """Demonstrate creative colour mixing and interpolation"""
    print("=== Creative Colour Mixing ===")
    
    # Create "emotion" colours
    emotions = {
        'joy': tf.complex(
            tf.ones([1, 64]) * 0.8 * tf.sin(tf.linspace(0, 2*np.pi, 64)),
            tf.ones([1, 64]) * 0.2 * tf.cos(tf.linspace(0, 2*np.pi, 64))
        ),
        'melancholy': tf.complex(
            tf.ones([1, 64]) * 0.4 * tf.sin(tf.linspace(0, np.pi, 64)),
            tf.ones([1, 64]) * 0.6 * tf.cos(tf.linspace(0, 3*np.pi, 64))
        ),
        'excitement': tf.complex(
            tf.ones([1, 64]) * tf.random.uniform([64], 0.3, 1.0),
            tf.ones([1, 64]) * tf.random.uniform([64], 0.1, 0.5)
        )
    }
    
    # Create resonance layer for mixing
    mixer = ChromaticResonance(dimensions=64, resonance_depth=5)
    
    # Mix emotions in different proportions
    mixtures = {}
    
    # Joy + Melancholy = Bittersweet
    joy_melancholy = mixer([emotions['joy'], emotions['melancholy']])
    mixtures['bittersweet'] = joy_melancholy
    
    # Joy + Excitement = Euphoria  
    joy_excitement = mixer([emotions['joy'], emotions['excitement']])
    mixtures['euphoria'] = joy_excitement
    
    # All three = Complex emotion
    all_emotions = MultiWaveInterference(max_waves=3)
    complex_emotion = all_emotions(list(emotions.values()))
    mixtures['complex'] = complex_emotion
    
    # Visualize emotion space
    visualizer = ColourSpaceVisualizer()
    
    plt.figure(figsize=(15, 10))
    
    plot_idx = 1
    for name, colour in {**emotions, **mixtures}.items():
        plt.subplot(2, 3, plot_idx)
        
        magnitude = tf.abs(colour[0]).numpy()
        phase = tf.angle(colour[0]).numpy()
        
        # Create polar plot
        theta = np.linspace(0, 2*np.pi, len(magnitude))
        ax = plt.subplot(2, 3, plot_idx, projection='polar')
        ax.plot(theta, magnitude, label='Magnitude')
        ax.fill_between(theta, magnitude, alpha=0.3)
        ax.set_title(name.title())
        
        plot_idx += 1
        if plot_idx > 6:
            break
    
    plt.tight_layout()
    plt.suptitle("Emotion Colour Space", fontsize=16, y=1.02)
    plt.show()
    
    # Demonstrate colour interpolation paths
    alphas = np.linspace(0, 1, 11)
    joy_to_melancholy_path = []
    
    for alpha in alphas:
        interpolated = colour_interpolation(
            emotions['joy'], emotions['melancholy'], 
            alpha, method='frequency'
        )
        joy_to_melancholy_path.append(interpolated)
    
    # Visualize interpolation path
    visualizer.plot_resonance_evolution(
        joy_to_melancholy_path, 
        title="Joy → Melancholy Interpolation Path"
    )
    
    return mixtures


def multimodal_synthesis_example():
    """Demonstrate cross-modal synthesis capabilities"""
    print("\n=== Multimodal Synthesis ===")
    
    # Simulate different modalities
    batch_size = 4
    
    # Visual features (e.g., from CNN)
    visual_features = tf.random.normal([batch_size, 64], seed=1)
    
    # Audio features (e.g., from mel spectrogram)
    audio_features = tf.random.normal([batch_size, 64], seed=2) * 1.5
    
    # Text features (e.g., from transformer)
    text_features = tf.random.normal([batch_size, 64], seed=3) * 0.8
    
    # Create modality-specific waveform encoders
    visual_encoder = CognitiveWaveform(
        dimensions=128, 
        frequency_range=(1.0, 100.0),  # Mid-range frequencies
        name='visual_encoder'
    )
    
    audio_encoder = CognitiveWaveform(
        dimensions=128,
        frequency_range=(0.1, 1000.0),  # Wide frequency range
        name='audio_encoder'
    )
    
    text_encoder = CognitiveWaveform(
        dimensions=128,
        frequency_range=(0.01, 10.0),  # Low frequencies
        name='text_encoder'
    )
    
    # Encode each modality
    visual_colours = visual_encoder(visual_features)
    audio_colours = audio_encoder(audio_features)  
    text_colours = text_encoder(text_features)
    
    print(f"Visual colours power: {tf.reduce_mean(tf.abs(visual_colours)**2):.3f}")
    print(f"Audio colours power: {tf.reduce_mean(tf.abs(audio_colours)**2):.3f}")
    print(f"Text colours power: {tf.reduce_mean(tf.abs(text_colours)**2):.3f}")
    
    # Cross-modal synthesis through interference
    multimodal_mixer = MultiWaveInterference(max_waves=3)
    
    # Different combinations
    visual_audio = InterferenceLayer(name='visual_audio')([visual_colours, audio_colours])
    visual_text = InterferenceLayer(name='visual_text')([visual_colours, text_colours])
    audio_text = InterferenceLayer(name='audio_text')([audio_colours, text_colours])
    
    # All three modalities
    all_modalities = multimodal_mixer([visual_colours, audio_colours, text_colours])
    
    # Analyze cross-modal distances
    modalities = {
        'Visual': visual_colours,
        'Audio': audio_colours,
        'Text': text_colours,
        'Visual+Audio': visual_audio,
        'Visual+Text': visual_text,
        'Audio+Text': audio_text,
        'All Three': all_modalities
    }
    
    # Create distance matrix
    names = list(modalities.keys())
    n_modalities = len(names)
    distance_matrix = np.zeros((n_modalities, n_modalities))
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i != j:
                dist = colour_distance(
                    modalities[name1], modalities[name2], 
                    metric='spectral'
                )
                distance_matrix[i, j] = tf.reduce_mean(dist).numpy()
    
    # Visualize distance matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap='viridis')
    plt.colorbar(label='Spectral Distance')
    plt.xticks(range(n_modalities), names, rotation=45)
    plt.yticks(range(n_modalities), names)
    plt.title('Cross-Modal Colour Distance Matrix')
    plt.tight_layout()
    plt.show()
    
    # Show spectral signatures
    plt.figure(figsize=(15, 8))
    
    for i, (name, colour) in enumerate(modalities.items()):
        plt.subplot(2, 4, i+1)
        
        # Power spectrum
        power = tf.abs(colour[0])**2
        freqs = np.linspace(0.01, 1000, len(power))
        
        plt.semilogy(freqs, power.numpy())
        plt.title(f'{name} Spectrum')
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.grid(True, alpha=0.3)
        
        if i >= 7:  # Limit to 8 subplots
            break
    
    plt.tight_layout()
    plt.suptitle("Multimodal Spectral Signatures", fontsize=14, y=1.02)
    plt.show()
    
    return modalities


def adaptive_resonance_example():
    """Demonstrate adaptive resonance and memory formation"""
    print("\n=== Adaptive Resonance ===")
    
    # Create a sequence of related "thoughts"
    sequence_length = 10
    dimensions = 64
    
    # Generate evolving pattern (like a melody or narrative)
    base_frequency = 2.0
    base_pattern = tf.sin(tf.linspace(0, base_frequency * 2 * np.pi, dimensions))
    
    thought_sequence = []
    for i in range(sequence_length):
        # Gradually shift frequency and add noise
        freq_shift = 1.0 + 0.1 * i
        amplitude_var = 0.8 + 0.2 * np.sin(i * 0.5)
        
        pattern = amplitude_var * tf.sin(tf.linspace(0, base_frequency * freq_shift * 2 * np.pi, dimensions))
        noise = tf.random.normal([dimensions], stddev=0.1)
        
        thought = tf.complex(
            tf.expand_dims(pattern + noise, 0),
            tf.expand_dims(pattern * 0.2 + noise * 0.1, 0)
        )
        
        thought_sequence.append(thought)
    
    # Create adaptive resonance system
    adaptive_resonance = ChromaticResonance(
        dimensions=dimensions,
        resonance_depth=8,
        memory_decay=0.95,  # Strong memory
        harmonic_orders=[1, 2, 3, 5]
    )
    
    # Process sequence and track memory formation
    memory_evolution = []
    output_evolution = []
    
    for i, thought in enumerate(thought_sequence):
        print(f"Processing thought {i+1}/{sequence_length}")
        
        # Apply resonance (training=True to update memory)
        resonated = adaptive_resonance(thought, training=True)
        output_evolution.append(resonated)
        
        # Record memory state
        memory_state = tf.identity(adaptive_resonance.resonance_memory)
        memory_evolution.append(memory_state.numpy())
        
        # Get resonance metrics
        metrics = adaptive_resonance.get_resonance_metrics(thought)
        print(f"  Q-factor: {metrics['q_factor'][0]:.2f}, Stability: {metrics['stability']:.3f}")
    
    # Analyze memory formation
    memory_matrix = np.stack(memory_evolution)
    
    plt.figure(figsize=(15, 10))
    
    # Memory evolution heatmap
    plt.subplot(2, 3, 1)
    plt.imshow(memory_matrix.T, aspect='auto', cmap='plasma')
    plt.colorbar(label='Memory Strength')
    plt.xlabel('Time Step')
    plt.ylabel('Frequency Bin')
    plt.title('Memory Evolution')
    
    # Memory growth over time
    plt.subplot(2, 3, 2)
    memory_totals = np.sum(memory_matrix, axis=1)
    plt.plot(memory_totals, 'o-', markersize=6)
    plt.xlabel('Time Step')
    plt.ylabel('Total Memory')
    plt.title('Memory Accumulation')
    plt.grid(True, alpha=0.3)
    
    # Output power evolution
    plt.subplot(2, 3, 3)
    output_powers = [tf.reduce_mean(tf.abs(out)**2).numpy() for out in output_evolution]
    plt.plot(output_powers, 's-', color='orange', markersize=6)
    plt.xlabel('Time Step')
    plt.ylabel('Output Power')
    plt.title('Output Power Evolution')
    plt.grid(True, alpha=0.3)
    
    # Resonance patterns
    for i in [0, len(output_evolution)//2, -1]:
        plt.subplot(2, 3, 4 + i if i >= 0 else 6)
        
        output = output_evolution[i]
        magnitude = tf.abs(output[0]).numpy()
        
        plt.plot(magnitude, linewidth=2)
        plt.title(f'Resonance Pattern (Step {i+1 if i >= 0 else len(output_evolution)})')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Magnitude')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle("Adaptive Resonance and Memory Formation", fontsize=14, y=1.02)
    plt.show()
    
    # Test memory recall
    print("\n--- Testing Memory Recall ---")
    
    # Present early pattern again
    recall_test = thought_sequence[1]  # Second thought
    recalled_output = adaptive_resonance(recall_test, training=False)
    
    # Compare with original response
    original_output = output_evolution[1]
    
    recall_similarity = colour_distance(recalled_output, original_output, metric='spectral')
    print(f"Recall similarity: {tf.reduce_mean(recall_similarity):.3f} (lower = more similar)")
    
    return output_evolution, memory_evolution


def performance_optimization_example():
    """Demonstrate performance optimization techniques"""
    print("\n=== Performance Optimization ===")
    
    # Test different configurations
    configurations = {
        'Small Fast': {
            'dimensions': 128,
            'resonance_depth': 3,
            'num_modes': 8,
            'spectral_normalize': True
        },
        'Medium Balanced': {
            'dimensions': 256, 
            'resonance_depth': 5,
            'num_modes': 16,
            'spectral_normalize': True
        },
        'Large Quality': {
            'dimensions': 512,
            'resonance_depth': 7,
            'num_modes': 32,
            'spectral_normalize': True
        }
    }
    
    batch_sizes = [1, 4, 16, 32]
    results = {}
    
    for config_name, config in configurations.items():
        print(f"\nTesting {config_name} configuration...")
        
        # Create model
        model = ChromaThinkModel(
            input_dim=64,
            colour_dimensions=config['dimensions'],
            resonance_depth=config['resonance_depth'],
            num_modes=config['num_modes']
        )
        
        # Benchmark batch scaling
        input_shape = (64,)  # Excluding batch dimension
        scaling_results = benchmark_batch_scaling(
            model, input_shape, batch_sizes, dtype=tf.float32
        )
        
        results[config_name] = scaling_results
        
        # Print summary
        for batch_size in batch_sizes:
            result = scaling_results[batch_size]
            print(f"  Batch {batch_size}: {result['mean_time']:.3f}s, "
                  f"{result['throughput']:.1f} samples/s")
    
    # Visualize performance comparison
    plt.figure(figsize=(15, 5))
    
    # Throughput comparison
    plt.subplot(1, 3, 1)
    for config_name, config_results in results.items():
        batch_sizes_list = sorted(config_results.keys())
        throughputs = [config_results[bs]['throughput'] for bs in batch_sizes_list]
        plt.plot(batch_sizes_list, throughputs, 'o-', label=config_name, markersize=6)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (samples/s)')
    plt.title('Throughput vs Batch Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    
    # Time comparison
    plt.subplot(1, 3, 2)
    for config_name, config_results in results.items():
        batch_sizes_list = sorted(config_results.keys())
        times = [config_results[bs]['mean_time'] for bs in batch_sizes_list]
        plt.plot(batch_sizes_list, times, 's-', label=config_name, markersize=6)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Mean Time (seconds)')
    plt.title('Execution Time vs Batch Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    # Efficiency (samples per second per parameter)
    plt.subplot(1, 3, 3)
    
    # Estimate parameter counts (rough approximation)
    param_estimates = {
        'Small Fast': 128**2 * 3,    # Rough estimate
        'Medium Balanced': 256**2 * 3,
        'Large Quality': 512**2 * 3
    }
    
    for config_name, config_results in results.items():
        batch_32_throughput = config_results[32]['throughput']
        efficiency = batch_32_throughput / param_estimates[config_name] * 1e6  # Scale for readability
        plt.bar(config_name, efficiency)
    
    plt.ylabel('Efficiency (samples/s/Mparam)')
    plt.title('Model Efficiency (Batch=32)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def real_time_processing_demo():
    """Demonstrate real-time processing capabilities"""
    print("\n=== Real-Time Processing Demo ===")
    
    # Create optimized model for real-time processing
    realtime_model = ChromaThinkModel(
        input_dim=32,
        colour_dimensions=128,  # Smaller for speed
        resonance_depth=3,     # Reduced depth
        num_modes=8            # Fewer modes
    )
    
    # Simulate streaming data
    stream_length = 50
    batch_size = 1  # Real-time = single samples
    
    performance_monitor = PerformanceMonitor()
    
    outputs = []
    latencies = []
    
    print(f"Processing {stream_length} real-time samples...")
    
    for i in range(stream_length):
        # Generate streaming input (e.g., audio frame, sensor reading)
        stream_input = tf.random.normal([batch_size, 32], seed=i)
        
        # Monitor performance
        performance_monitor.start_step()
        
        # Process sample
        output, colours = realtime_model(stream_input, training=False)
        
        # End monitoring
        performance_monitor.end_step()
        
        outputs.append(output)
        
        # Track latency
        current_stats = performance_monitor.get_current_stats()
        if current_stats:
            latency_ms = current_stats['mean_step_time'] * 1000
            latencies.append(latency_ms)
        
        # Print progress
        if (i + 1) % 10 == 0:
            if current_stats:
                print(f"  Processed {i+1}/{stream_length} samples, "
                      f"avg latency: {latency_ms:.1f}ms, "
                      f"throughput: {current_stats['steps_per_second']:.1f} samples/s")
    
    # Analyze real-time performance
    final_stats = performance_monitor.get_current_stats()
    print(f"\nReal-time Performance Summary:")
    print(f"  Mean latency: {np.mean(latencies):.1f}ms ± {np.std(latencies):.1f}ms")
    print(f"  Max latency: {np.max(latencies):.1f}ms")
    print(f"  Throughput: {final_stats['steps_per_second']:.1f} samples/s")
    
    # Check if real-time capable (e.g., for 44.1kHz audio = 22.7ms per frame)
    target_latency_ms = 20  # Real-time threshold
    realtime_capable = np.percentile(latencies, 95) < target_latency_ms
    
    print(f"  Real-time capable (95%ile < {target_latency_ms}ms): {realtime_capable}")
    
    # Visualize performance
    performance_monitor.plot_metrics()
    
    return outputs, performance_monitor


if __name__ == "__main__":
    # Set up TensorFlow for optimal performance
    tf.random.set_seed(42)
    
    # Enable mixed precision if available
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")
    except:
        print("Mixed precision not available")
    
    try:
        # Run advanced examples
        print("Running Advanced ChromaThink Examples...")
        
        # Creative applications
        emotion_mixtures = creative_colour_mixing_example()
        multimodal_features = multimodal_synthesis_example()
        
        # Adaptive and learning examples
        resonance_outputs, memory_states = adaptive_resonance_example()
        
        # Performance optimization
        perf_results = performance_optimization_example()
        
        # Real-time demo
        realtime_outputs, monitor = real_time_processing_demo()
        
        print("\n=== All Advanced Examples Completed Successfully! ===")
        
    except Exception as e:
        print(f"Error in advanced examples: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Reset mixed precision
        tf.keras.mixed_precision.set_global_policy('float32')