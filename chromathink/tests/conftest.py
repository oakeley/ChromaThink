"""Pytest configuration and shared fixtures for ChromaThink tests"""

import pytest
import tensorflow as tf
import numpy as np
import os


def pytest_configure(config):
    """Configure pytest settings"""
    # Suppress TensorFlow warnings during testing
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')


@pytest.fixture(scope="session", autouse=True)
def setup_tensorflow():
    """Setup TensorFlow for testing"""
    # Enable memory growth for GPU to avoid OOM issues
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)


@pytest.fixture(scope="function", autouse=True)
def reset_tensorflow_state():
    """Reset TensorFlow state between tests"""
    # Clear the default graph and reset the random state
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)


@pytest.fixture
def sample_batch_data():
    """Generate sample batch data for testing"""
    batch_size = 4
    input_dim = 32
    return tf.random.normal([batch_size, input_dim], seed=42)


@pytest.fixture
def sample_complex_data():
    """Generate sample complex waveform data"""
    batch_size = 4
    dimensions = 32
    real_part = tf.random.normal([batch_size, dimensions], seed=42)
    imag_part = tf.random.normal([batch_size, dimensions], seed=43) * 0.1
    return tf.complex(real_part, imag_part)


@pytest.fixture
def small_dimensions():
    """Small dimensions for fast testing"""
    return 32


@pytest.fixture
def medium_dimensions():
    """Medium dimensions for more thorough testing"""
    return 64


@pytest.fixture
def large_dimensions():
    """Large dimensions for stress testing"""
    return 128


@pytest.fixture
def gpu_available():
    """Check if GPU is available"""
    return len(tf.config.list_physical_devices('GPU')) > 0


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if no GPU is available"""
    if not tf.config.list_physical_devices('GPU'):
        pytest.skip("GPU not available")


# Parametrized fixtures for testing different configurations
@pytest.fixture(params=[32, 64, 128])
def various_dimensions(request):
    """Test with various dimension sizes"""
    return request.param


@pytest.fixture(params=[1, 4, 8])
def various_batch_sizes(request):
    """Test with various batch sizes"""
    return request.param


@pytest.fixture(params=['tanh', 'sigmoid', 'swish', 'gelu'])
def various_activations(request):
    """Test with various activation functions"""
    return request.param


@pytest.fixture(params=[3, 5, 7])
def various_resonance_depths(request):
    """Test with various resonance depths"""
    return request.param


# Tolerance fixtures for numerical comparisons
@pytest.fixture
def tight_tolerance():
    """Tight tolerance for precise numerical tests"""
    return 1e-6


@pytest.fixture
def loose_tolerance():
    """Loose tolerance for approximate tests"""
    return 1e-3


@pytest.fixture
def very_loose_tolerance():
    """Very loose tolerance for stochastic tests"""
    return 1e-1


# Custom markers for categorizing tests
def pytest_collection_modifyitems(config, items):
    """Add custom markers to tests"""
    for item in items:
        # Mark performance tests
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)
        
        # Mark GPU-only tests
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)
        
        # Mark slow tests
        if "large_dimensions" in str(item.fixturenames) or "stress" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark stability tests
        if "stability" in item.name or "collapse" in item.name:
            item.add_marker(pytest.mark.stability)


# Register custom markers
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "performance: mark test as performance/benchmark test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "stability: mark test as stability test"
    )