# ChromaThink Testing Guide

## 🚀 **Quick Test Commands**

### Basic System Tests

```bash
# Test basic ChromaThink functionality
python examples/developmental_learning_minimal_demo.py

# Expected: ChromaThink generates color-based questions and learns through dialogue
# Success indicators:
# - "ChromaThink asks: Why does this seem important?"  
# - "Resonance: 0.156, Development: 0.009"
# - "Final development stage: 0.026"
```

### Real Apertus Bootstrap Test

```bash
# Test real 8B parameter model integration  
python bootstrap_chromathink.py --demo

# Expected results:
# ✅ "Loaded Apertus model with 8053338176 parameters"
# ✅ "Extracted colour patterns from 10000 token embeddings"
# ✅ "Analysis complete: 10000 tokens, 12 attention patterns, 8 transformation patterns"
# ✅ "Bootstrap complete!"
# ✅ Enhanced question formation: "How do these elements relate?"
```

### Component Tests

```bash
# Test individual components
python -c "
from chromathink.learning.development_minimal import DevelopmentalLearningMinimal
chromathink = DevelopmentalLearningMinimal(spectrum_dims=128)
print('✅ ChromaThink initialized')

wonder = chromathink.curiosity.generate_wonder()  
print(f'✅ Wonder state generated: magnitude {abs(wonder.numpy()).max():.3f}')

question = chromathink.curiosity.formulate_question(wonder)
print(f'✅ Question formulated: type {question[\"question_type\"]}')
"
```

### Memory System Test

```bash
# Test chromatic memory system
python -c "
from chromathink.learning.development_minimal import DevelopmentalLearningMinimal
import tensorflow as tf

chromathink = DevelopmentalLearningMinimal(spectrum_dims=128)

# Create test colors
color1 = tf.random.normal([1, 128], dtype=tf.complex64)
color2 = tf.random.normal([1, 128], dtype=tf.complex64)

# Store association
chromathink.colour_memory.store_association(color1, color2, strength=0.8)
print(f'✅ Memory stored. Total memories: {len(chromathink.colour_memory.episodic_memory)}')
"
```

## 🧪 **Expected Outputs & Verification**

### Successful Minimal Demo Output
```
🎨 ChromaThink Developmental Learning Demo (Minimal)
ChromaThink wonders: Mag: 0.77, Peak: 2.69, Dom: 118
ChromaThink asks: What is the meaning behind this?
Teacher responds: Let me explain simply: I love how you're connecting different ideas together.
Resonance: 0.156, Development: 0.005

Final Learning Metrics:
Development Stage: 0.026
Total Memories: 5  
Questions Asked: 5
✨ Demo complete! ChromaThink has learned through colour-based dialogue.
```

### Successful Bootstrap Demo Output
```
🎨 ChromaThink Bootstrap Demo
Found 4 safetensor files - using real Apertus model
Loaded Apertus model with 8053338176 parameters
Extracted colour patterns from 10000 token embeddings (subset of 131072 total)

Bootstrap Results:
vocabulary_transfer: success
attention_transfer: success  
transformation_transfer: success
memory_creation: success
success: True

Pre-bootstrap question: "How can I understand this better?"
Post-bootstrap question: "How do these elements relate?"

Live Learning Session:
ChromaThink asks: How should I powerfully interpret this?
Teacher responds: That's a question even experts find challenging.
Resonance: 0.138, Development: 0.014

✅ Bootstrap Demo Completed Successfully!
```

## 🔧 **Troubleshooting**

### Common Issues

**1. Missing Dependencies**
```bash
# Error: "No module named 'safetensors'" 
pip install safetensors accelerate

# Error: "No module named 'transformers'"
pip install transformers>=4.30.0
```

**2. Memory Issues**
```bash
# If running out of memory during bootstrap:
# The system automatically limits processing to 10K tokens and 12+8 layers
# This should work on systems with 16GB+ RAM

# For lower memory systems, edit bootstrap_chromathink.py:
# Change max_tokens = min(5000, embed_weights.shape[0])  # Reduce from 10000
```

**3. Model Path Issues**
```bash
# If Apertus model not found:
# Check the model exists at: ~/.cache/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509/
# If not, the system will automatically use mock patterns

# To download Apertus model:
python -c "from transformers import AutoModel; AutoModel.from_pretrained('swiss-ai/Apertus-8B-Instruct-2509')"
```

**4. TensorFlow Complex Number Issues**
```bash
# Error: "complex64 not supported"
# This should be automatically handled, but if issues persist:
export TF_ENABLE_ONEDNN_OPTS=0  # Disable oneDNN optimizations
```

## 🎯 **Performance Benchmarks**

### Expected Performance (CPU-only)

- **Basic ChromaThink Initialization**: ~1 second
- **Minimal Demo (5 exchanges)**: ~10 seconds  
- **Bootstrap with Real Apertus**: ~2-3 minutes
- **Memory per process**: 4-8GB RAM, 2-4GB for model

### Verification Commands

```bash
# Time the basic system
time python -c "
from chromathink.learning.development_minimal import DevelopmentalLearningMinimal
DevelopmentalLearningMinimal(spectrum_dims=128)
print('Initialized successfully')
"

# Time the bootstrap (should complete in under 5 minutes)
time python bootstrap_chromathink.py --demo
```

## 🧠 **Understanding the Outputs**

### Color State Analysis
```
Dominant frequencies: [82, 81, 83, 80, 60]  # Most active frequency components
Amplitude range: [0.007, 1.949]             # Color intensity range  
Phase distribution: mean=1.374, std=1.558   # Color relationship patterns
Spectral entropy: 4.565                     # Information complexity
Magnitude: 8.494                            # Overall color strength
```

### Learning Metrics  
```
Resonance: 0.156        # How well question and response colors matched
Development: 0.005      # Progress in chromatic cognitive development
Memory count: 5         # Number of stored color associations
```

### Bootstrap Success Indicators
- ✅ Real model loading: "8053338176 parameters"
- ✅ Token processing: "10000 token embeddings"  
- ✅ Pattern extraction: "12 attention patterns, 8 transformation patterns"
- ✅ Knowledge transfer: Questions change quality post-bootstrap
- ✅ Learning capability: Successful dialogue exchanges

## 🔍 **Debug Mode**

For detailed debugging, modify the bootstrap script:

```python
# In bootstrap_chromathink.py, change logging level:
logging.basicConfig(level=logging.DEBUG)  # Instead of INFO

# This will show:
# - Individual layer processing  
# - Tensor shape transformations
# - Memory allocation details
# - Color pattern statistics
```

This testing guide ensures you can verify that ChromaThink's revolutionary color-based cognition and real 8B parameter model integration are working correctly!