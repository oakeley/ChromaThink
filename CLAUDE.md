# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChromaThink is a TensorFlow implementation of an experimental neural network architecture that processes thoughts as colours and waveforms rather than discrete tokens. The system explores whether continuous colour-frequency representations might better capture cognitive processes than traditional language-based models.

## Installation and Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install ChromaThink package
pip install -e .

# Run tests
pytest

# Run examples
python examples/basic_usage.py
python examples/advanced_examples.py
```

## Core Architecture Components

### CognitiveWaveform Layer (`chromathink/layers/cognitive_waveform.py`)
- Transforms any input into waveform representation across N-dimensional colour space
- Uses FFT for efficient frequency domain transformation
- Includes spectral normalization to prevent colour vector collapse
- Key parameters: `dimensions`, `frequency_range`, `use_fft`, `spectral_normalize`

### InterferenceLayer (`chromathink/layers/interference.py`)
- Implements wave interference patterns with amplitude weighting and phase calculations
- Supports constructive, destructive, and full interference types
- Includes nonlinear mixing for richer interference patterns
- Key parameters: `interference_type`, `amplitude_weighting`, `phase_coupling`, `nonlinear_mixing`

### ChromaticResonance Layer (`chromathink/layers/chromatic_resonance.py`)
- Implements nonlinear colour mixing through resonance patterns
- Generates harmonics and applies frequency-dependent damping
- Maintains resonance memory for adaptive behavior
- Key parameters: `dimensions`, `resonance_depth`, `harmonic_orders`, `memory_decay`

### ResonanceChamber Layer (`chromathink/layers/resonance_chamber.py`)
- Creates standing wave patterns through reflection and interference
- Models different boundary conditions (rigid, free, mixed, absorbing)
- Supports modal analysis and quality factor control
- Key parameters: `dimensions`, `num_modes`, `boundary_conditions`, `quality_factor`

## Testing Framework

The project includes comprehensive tests organized by functionality:

- **Colour Stability Tests** (`test_colour_stability.py`): Prevent collapse to grey or noise explosion
- **Wave Interference Tests** (`test_wave_interference.py`): Validate interference patterns with known frequencies
- **Resonance Chamber Tests** (`test_resonance_chamber.py`): Test standing wave convergence
- **Chromatic Mixing Tests** (`test_chromatic_mixing.py`): Verify theoretical colour dynamics
- **Performance Tests** (`test_performance.py`): GPU benchmarks and optimization

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m stability          # Stability tests only
pytest -m performance        # Performance tests only
pytest -m gpu                # GPU-specific tests only
pytest -m slow               # Slow/comprehensive tests

# Run tests with coverage
pytest --cov=chromathink
```

## Performance Optimization

### GPU Requirements
- CUDA-compatible GPU recommended for optimal performance
- Mixed precision training available via `tf.keras.mixed_precision.set_global_policy('mixed_float16')`
- Memory growth enabled to prevent OOM issues

### Benchmarking Tools
Use the benchmarking utilities in `chromathink/utils/benchmarking.py`:

```python
from chromathink.utils.benchmarking import benchmark_layer, benchmark_batch_scaling

# Benchmark individual layers
results = benchmark_layer(layer, input_data, num_runs=10)

# Test batch size scaling
scaling_results = benchmark_batch_scaling(layer, input_shape, batch_sizes)
```

### Real-time Processing
For real-time applications:
- Use smaller dimensions (128-256)
- Reduce resonance depth (3-5)
- Fewer resonance modes (8-16)
- Target latency <20ms for audio applications

## Key Utilities

### Visualization (`chromathink/utils/visualization.py`)
- `ColourSpaceVisualizer`: Comprehensive colour field visualization
- `plot_frequency_spectrum()`: Frequency domain analysis
- `plot_interference_patterns()`: Wave interference visualization

### Core Utilities (`chromathink/core/`)
- `spectral_utils.py`: Spectral normalization and frequency stability
- `colour_utils.py`: Colour distance metrics and interpolation methods

## Development Guidelines

### Architecture Principles
1. **Colour Representation**: All thought is represented as complex waveforms in frequency domain
2. **Wave Interference**: Understanding emerges from interference patterns, not token prediction
3. **Resonance**: Standing wave patterns create stable memory and processing states
4. **Spectral Stability**: Prevent collapse to uniform grey or noise explosion

### Performance Considerations
1. **Complex Operations**: Extensive use of complex number arithmetic - ensure TensorFlow complex support
2. **Memory Usage**: High-dimensional frequency spaces require significant GPU memory
3. **Numerical Stability**: Use spectral normalization and frequency stability checks
4. **Gradient Flow**: Complex gradients can be unstable - monitor gradient norms

### Common Patterns

#### Creating a Basic Processing Pipeline
```python
# Input transformation
waveform_layer = CognitiveWaveform(dimensions=256, spectral_normalize=True)
colours = waveform_layer(inputs)

# Resonance processing
resonance_layer = ChromaticResonance(dimensions=256, resonance_depth=5)
resonated = resonance_layer(colours)

# Standing wave formation
chamber = ResonanceChamber(dimensions=256, num_modes=32)
standing_waves = chamber(resonated)
```

#### Interference Between Multiple Inputs
```python
# Create interference layer
interference = InterferenceLayer(interference_type='full', nonlinear_mixing=True)

# Apply to multiple waveforms
result = interference([wave1, wave2])

# For multiple waves
multi_interference = MultiWaveInterference(max_waves=4)
result = multi_interference([wave1, wave2, wave3, wave4])
```

## Troubleshooting

### Common Issues

1. **NaN/Inf Values**: Usually caused by unstable complex operations
   - Enable spectral normalization
   - Use frequency stability checks
   - Reduce learning rates

2. **Memory Issues**: High-dimensional colour spaces are memory-intensive
   - Reduce dimensions or batch size
   - Enable GPU memory growth
   - Use gradient checkpointing

3. **Slow Performance**: Complex operations are computationally expensive
   - Use GPU with CUDA support
   - Enable mixed precision training
   - Optimize hyperparameters for your hardware

4. **Colour Collapse**: Waveforms converging to uniform values
   - Enable collapse prevention methods
   - Use entropy regularization
   - Increase spectral diversity

### Debug Tools
```python
from chromathink.core.spectral_utils import frequency_stability_check
from chromathink.core.colour_utils import prevent_collapse

# Check stability
stability = frequency_stability_check(colour_vectors)
print(f"Stable: {stability['is_stable']}")

# Add collapse prevention
loss = prevent_collapse(colour_vectors, method='entropy_regularization')
```