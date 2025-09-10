"""Basic usage examples for ChromaThink colour-wave neural architecture"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from chromathink.layers.cognitive_waveform import CognitiveWaveform
from chromathink.layers.interference import InterferenceLayer
from chromathink.layers.chromatic_resonance import ChromaticResonance
from chromathink.layers.resonance_chamber import ResonanceChamber
from chromathink.utils.visualization import ColourSpaceVisualizer, plot_frequency_spectrum
from chromathink.utils.benchmarking import benchmark_layer


def basic_waveform_example():
    """Basic example of transforming input to colour waveform"""
    print("=== Basic Waveform Transformation ===")
    
    # Create some sample input data (could be text embeddings, image patches, etc.)
    batch_size = 4
    input_dim = 64
    sample_input = tf.random.normal([batch_size, input_dim], seed=42)
    
    # Create cognitive waveform layer
    waveform_layer = CognitiveWaveform(
        dimensions=128,
        frequency_range=(0.1, 100.0),
        use_fft=True,
        spectral_normalize=True
    )
    
    # Transform input to colour waveform
    colour_waveform = waveform_layer(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Colour waveform shape: {colour_waveform.shape}")
    print(f"Waveform dtype: {colour_waveform.dtype}")
    
    # Analyze frequency content
    analysis = waveform_layer.get_frequency_content(sample_input)
    print(f"Spectral centroid: {analysis['spectral_centroid'][0]:.2f}")
    print(f"Spectral bandwidth: {analysis['spectral_bandwidth'][0]:.2f}")
    print(f"Dominant frequency: {analysis['dominant_frequencies'][0]:.2f}")
    
    # Visualize the result
    visualizer = ColourSpaceVisualizer()
    visualizer.plot_colour_field(colour_waveform, sample_index=0, title="Basic Colour Waveform")
    
    return colour_waveform


def interference_example():
    """Example of wave interference between two colour waveforms"""
    print("\n=== Wave Interference Example ===")
    
    # Create two different colour waveforms
    batch_size = 2
    dimensions = 64
    
    # "Red" colour - low frequency, high amplitude
    red_amplitude = tf.ones([batch_size, dimensions]) * 0.8
    red_phase = tf.linspace(0, 2*np.pi, dimensions) * tf.ones([batch_size, dimensions])
    red_wave = red_amplitude * tf.exp(tf.complex(0.0, red_phase))
    
    # "Blue" colour - high frequency, medium amplitude
    blue_amplitude = tf.ones([batch_size, dimensions]) * 0.5
    blue_phase = tf.linspace(0, 6*np.pi, dimensions) * tf.ones([batch_size, dimensions])
    blue_wave = blue_amplitude * tf.exp(tf.complex(0.0, blue_phase))
    
    print(f"Red wave power: {tf.reduce_mean(tf.abs(red_wave)**2):.3f}")
    print(f"Blue wave power: {tf.reduce_mean(tf.abs(blue_wave)**2):.3f}")
    
    # Create interference layer
    interference_layer = InterferenceLayer(
        interference_type='full',
        amplitude_weighting=True,
        phase_coupling=True,
        nonlinear_mixing=True
    )
    
    # Apply interference
    mixed_colour = interference_layer([red_wave, blue_wave])
    
    print(f"Mixed colour power: {tf.reduce_mean(tf.abs(mixed_colour)**2):.3f}")
    
    # Analyze interference strength
    strength_metrics = interference_layer.interference_strength([red_wave, blue_wave])
    print(f"Phase correlation: {strength_metrics['phase_correlation'][0]:.3f}")
    print(f"Amplitude correlation: {strength_metrics['amplitude_correlation'][0]:.3f}")
    print(f"Interference strength: {strength_metrics['interference_strength'][0]:.3f}")
    
    # Visualize interference
    visualizer = ColourSpaceVisualizer()
    visualizer.plot_interference_pattern(
        red_wave, blue_wave, mixed_colour,
        title="Red-Blue Wave Interference"
    )
    
    return mixed_colour


def chromatic_resonance_example():
    """Example of chromatic resonance for colour evolution"""
    print("\n=== Chromatic Resonance Example ===")
    
    # Create initial colour
    batch_size = 2
    dimensions = 64
    
    initial_colour = tf.complex(
        tf.random.normal([batch_size, dimensions], seed=42),
        tf.random.normal([batch_size, dimensions], seed=43) * 0.2
    )
    
    # Create resonance layer
    resonance_layer = ChromaticResonance(
        dimensions=dimensions,
        resonance_depth=7,
        harmonic_orders=[1, 2, 3, 5],
        nonlinearity='tanh'
    )
    
    # Evolve colour through resonance
    evolved_colour = resonance_layer(initial_colour, training=True)
    
    print(f"Initial colour power: {tf.reduce_mean(tf.abs(initial_colour)**2):.3f}")
    print(f"Evolved colour power: {tf.reduce_mean(tf.abs(evolved_colour)**2):.3f}")
    
    # Analyze resonance characteristics
    metrics = resonance_layer.get_resonance_metrics(initial_colour)
    print(f"Q-factor: {metrics['q_factor'][0]:.2f}")
    print(f"Stability: {metrics['stability']:.3f}")
    print(f"Harmonic ratio: {metrics['harmonic_ratio'][0]:.3f}")
    
    # Show evolution over multiple steps
    evolution_history = [initial_colour]
    current_colour = initial_colour
    
    for step in range(5):
        current_colour = resonance_layer(current_colour, training=True)
        evolution_history.append(current_colour)
    
    # Visualize evolution
    visualizer = ColourSpaceVisualizer()
    visualizer.plot_resonance_evolution(evolution_history, title="Colour Evolution Through Resonance")
    
    return evolved_colour


def resonance_chamber_example():
    """Example of standing wave patterns in resonance chamber"""
    print("\n=== Resonance Chamber Example ===")
    
    # Create excitation signal
    batch_size = 2
    dimensions = 128
    
    # Create a signal that should excite specific modes
    excitation = tf.complex(
        tf.sin(tf.linspace(0, 4*np.pi, dimensions)) * tf.ones([batch_size, dimensions]),
        tf.cos(tf.linspace(0, 4*np.pi, dimensions)) * tf.ones([batch_size, dimensions]) * 0.1
    )
    
    # Create resonance chamber
    chamber = ResonanceChamber(
        dimensions=dimensions,
        chamber_length=1.0,
        boundary_conditions='mixed',
        num_modes=16,
        quality_factor=50.0
    )
    
    # Create standing waves
    standing_waves = chamber(excitation, training=True)
    
    print(f"Input signal power: {tf.reduce_mean(tf.abs(excitation)**2):.3f}")
    print(f"Standing wave power: {tf.reduce_mean(tf.abs(standing_waves)**2):.3f}")
    
    # Analyze standing wave characteristics
    analysis = chamber.get_standing_wave_analysis(excitation)
    
    print(f"Number of modes: {len(analysis['mode_amplitudes'][0])}")
    print(f"Node count: {analysis['node_count'][0]:.0f}")
    print(f"Antinode count: {analysis['antinode_count'][0]:.0f}")
    print(f"Standing wave ratio: {analysis['standing_wave_ratio'][0]:.2f}")
    
    # Visualize standing wave analysis
    visualizer = ColourSpaceVisualizer()
    visualizer.plot_standing_wave_analysis(analysis, title="Resonance Chamber Analysis")
    
    # Show resonant frequencies
    resonant_freqs = analysis['resonant_frequencies'].numpy()
    mode_amps = analysis['mode_amplitudes'][0].numpy()
    
    plt.figure(figsize=(10, 6))
    plt.stem(resonant_freqs, mode_amps, basefmt='none')
    plt.xlabel('Resonant Frequency')
    plt.ylabel('Mode Amplitude')
    plt.title('Resonant Modes in Chamber')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return standing_waves


def complete_pipeline_example():
    """Complete pipeline combining all components"""
    print("\n=== Complete ChromaThink Pipeline ===")
    
    # Input data (e.g., from text encoder, image patches, etc.)
    batch_size = 4
    input_dim = 32
    input_data = tf.random.normal([batch_size, input_dim], seed=42)
    
    print(f"Processing {batch_size} samples with {input_dim} input features")
    
    # Step 1: Transform to cognitive waveforms
    waveform_layer = CognitiveWaveform(dimensions=64, spectral_normalize=True)
    waveforms = waveform_layer(input_data)
    print(f"Step 1 - Waveform generation: {waveforms.shape}")
    
    # Step 2: Apply chromatic resonance
    resonance_layer = ChromaticResonance(dimensions=64, resonance_depth=5)
    resonated = resonance_layer(waveforms)
    print(f"Step 2 - Chromatic resonance: {resonated.shape}")
    
    # Step 3: Create standing waves in chamber
    chamber = ResonanceChamber(dimensions=64, num_modes=12)
    standing_waves = chamber(resonated)
    print(f"Step 3 - Resonance chamber: {standing_waves.shape}")
    
    # Step 4: Mix with interference (simulate multiple thoughts)
    interference_layer = InterferenceLayer(interference_type='full')
    
    # Split batch into two groups for interference
    wave_group1 = standing_waves[:batch_size//2]
    wave_group2 = standing_waves[batch_size//2:]
    
    final_output = interference_layer([wave_group1, wave_group2])
    print(f"Step 4 - Final interference: {final_output.shape}")
    
    # Analyze final output
    final_power = tf.reduce_mean(tf.abs(final_output)**2)
    print(f"Final output power: {final_power:.3f}")
    
    # Visualize the complete transformation
    visualizer = ColourSpaceVisualizer()
    
    plt.figure(figsize=(15, 10))
    
    # Show progression through pipeline
    stages = [
        ("Input (as waveform)", waveform_layer(input_data[:1])),
        ("After Resonance", resonance_layer(waveforms[:1])),  
        ("After Chamber", standing_waves[:1]),
        ("Final Output", final_output[:1])
    ]
    
    for i, (title, data) in enumerate(stages):
        plt.subplot(2, 2, i+1)
        magnitude = tf.abs(data[0]).numpy()
        plt.plot(magnitude)
        plt.title(title)
        plt.xlabel('Frequency Bin')
        plt.ylabel('Magnitude')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle("ChromaThink Pipeline Progression", fontsize=16, y=1.02)
    plt.show()
    
    return final_output


def benchmark_performance():
    """Benchmark performance of different components"""
    print("\n=== Performance Benchmarking ===")
    
    # Test data
    batch_size = 8
    dimensions = 256
    input_data = tf.random.normal([batch_size, dimensions//4])
    complex_data = tf.complex(
        tf.random.normal([batch_size, dimensions]),
        tf.random.normal([batch_size, dimensions]) * 0.1
    )
    
    # Components to benchmark
    components = {
        "CognitiveWaveform": CognitiveWaveform(dimensions=dimensions),
        "ChromaticResonance": ChromaticResonance(dimensions=dimensions, resonance_depth=5),
        "ResonanceChamber": ResonanceChamber(dimensions=dimensions, num_modes=32)
    }
    
    # Benchmark each component
    for name, component in components.items():
        print(f"\nBenchmarking {name}...")
        
        # Use appropriate input data
        if name == "CognitiveWaveform":
            test_data = input_data
        else:
            test_data = complex_data
        
        # Benchmark
        results = benchmark_layer(component, test_data, num_runs=10)
        
        print(f"  Mean time: {results['mean_time']:.4f}s ± {results['std_time']:.4f}s")
        print(f"  Min time: {results['min_time']:.4f}s")
        print(f"  Max time: {results['max_time']:.4f}s")
        print(f"  Throughput: {batch_size/results['mean_time']:.1f} samples/second")


if __name__ == "__main__":
    # Set up TensorFlow
    tf.random.set_seed(42)
    
    # Run examples
    try:
        # Basic examples
        waveform_output = basic_waveform_example()
        mixed_output = interference_example()
        evolved_output = chromatic_resonance_example()
        standing_waves = resonance_chamber_example()
        
        # Complete pipeline
        pipeline_output = complete_pipeline_example()
        
        # Performance benchmarking
        benchmark_performance()
        
        print("\n=== All Examples Completed Successfully! ===")
        
    except Exception as e:
        print(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()