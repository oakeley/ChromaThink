"""
Create a real integration with Apertus that:
1. Loads the actual model from the four safetensors files
2. Analyses how Apertus encodes semantic relationships in its weight matrices
3. Translates these patterns into colour frequencies for ChromaThink

The key insight: Apertus's attention patterns and embedding spaces contain
implicit "colours" of meaning that we can extract and translate.
"""

import torch
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from safetensors import safe_open
from safetensors.torch import load_file
import warnings
import pickle
import hashlib

# Handle safetensors import gracefully
try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    warnings.warn("safetensors not available. Install with: pip install safetensors")

# Handle transformers import gracefully  
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not fully available")


class ApertusWeightTranslator:
    """
    Read Apertus's weights and translate linguistic patterns to colour space.
    We're essentially performing synaesthesia on the model's knowledge.
    """
    
    def __init__(self, 
                 apertus_path: str = "models/apertus",
                 spectrum_dims: int = 512,
                 use_mock: bool = False,
                 max_tokens: int = None,  # Auto-detect vocab size by default
                 max_attention_layers: int = 12,
                 max_mlp_layers: int = 8,
                 extract_full_vocab: bool = True,  # Full vocab by default
                 force_rebuild: bool = False):  # Force rebuild cache
        
        self.apertus_path = Path(apertus_path)
        self.spectrum_dims = spectrum_dims
        self.use_mock = use_mock
        self.max_tokens = max_tokens
        self.max_attention_layers = max_attention_layers
        self.max_mlp_layers = max_mlp_layers
        self.extract_full_vocab = extract_full_vocab
        self.force_rebuild = force_rebuild
        
        # Auto-detect vocabulary size if not specified
        self.actual_vocab_size = None
        
        # Big colour model components
        self.big_colour_model = None
        self.concept_encoder = None
        self.waveform_decoder = None
        
        # Setup logging to track question formation
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ChromaThink.Bootstrap")
        
        if self.use_mock:
            self.logger.info("Using mock Apertus model (use_mock=True)")
            self.create_mock_apertus()
        elif not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available - install with: pip install transformers")
        elif not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors library not available - install with: pip install safetensors")
        else:
            # Load the actual Apertus model from safetensors files
            self.load_apertus_from_safetensors()
        
        # Auto-detect vocabulary size from model
        if not self.use_mock:
            self._detect_vocabulary_size()
        
        # Check for cached weights first
        self.weight_patterns = self._load_or_create_weight_patterns()
        
    def create_mock_apertus(self):
        """Create a mock Apertus model for testing/development"""
        self.logger.info("Creating mock Apertus model...")
        
        # Create mock tokenizer
        class MockTokenizer:
            def __init__(self):
                self.vocab_size = 32000
                self.vocab = {f"token_{i}": i for i in range(self.vocab_size)}
                self.vocab.update({
                    "what": 1000, "why": 1001, "how": 1002, "when": 1003, "where": 1004, "who": 1005,
                    "think": 2000, "feel": 2001, "know": 2002, "understand": 2003, "learn": 2004,
                    "colour": 3000, "sound": 3001, "light": 3002, "pattern": 3003, "meaning": 3004
                })
            
            def encode(self, text, add_special_tokens=False):
                words = text.lower().split()
                return [self.vocab.get(word, 0) for word in words]
        
        # Create mock model
        class MockModel:
            def __init__(self, vocab_size, hidden_size):
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size
                
                # Create mock embeddings
                self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
                
                # Create mock attention layers
                self.attention_layers = []
                for i in range(4):  # 4 layers for mock
                    layer = {
                        'q_proj': torch.nn.Linear(hidden_size, hidden_size),
                        'k_proj': torch.nn.Linear(hidden_size, hidden_size),
                        'v_proj': torch.nn.Linear(hidden_size, hidden_size),
                        'mlp_gate': torch.nn.Linear(hidden_size, hidden_size * 4),
                        'mlp_up': torch.nn.Linear(hidden_size, hidden_size * 4),
                        'mlp_down': torch.nn.Linear(hidden_size * 4, hidden_size)
                    }
                    self.attention_layers.append(layer)
                
            def named_parameters(self):
                """Mock named_parameters for weight analysis"""
                params = []
                
                # Embedding weights
                params.append(('embed_tokens.weight', self.embed_tokens.weight))
                
                # Layer weights
                for i, layer in enumerate(self.attention_layers):
                    for name, module in layer.items():
                        params.append((f'layers.{i}.self_attn.{name}.weight', module.weight))
                        if hasattr(module, 'bias') and module.bias is not None:
                            params.append((f'layers.{i}.self_attn.{name}.bias', module.bias))
                
                return params
        
        self.tokenizer = MockTokenizer()
        self.model = MockModel(vocab_size=32000, hidden_size=512)
        self.logger.info("Mock Apertus model created successfully")
        
    def _detect_vocabulary_size(self):
        """Auto-detect the actual vocabulary size from model weights."""
        try:
            if hasattr(self, 'model_weights') and self.model_weights:
                # Find embedding weights in the loaded weights
                embed_key = None
                for key in self.model_weights.keys():
                    if 'embed_tokens.weight' in key:
                        embed_key = key
                        break
                
                if embed_key:
                    embed_tensor = self.model_weights[embed_key]
                    self.actual_vocab_size = embed_tensor.shape[0]
                    self.logger.info(f"Auto-detected vocabulary size: {self.actual_vocab_size:,} tokens")
                    return
            
            # Fallback: Load from config.json
            config_path = self.apertus_path / "config.json"
            if config_path.exists():
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.actual_vocab_size = config.get('vocab_size', 131072)
                self.logger.info(f"Vocabulary size from config: {self.actual_vocab_size:,} tokens")
            else:
                self.actual_vocab_size = 131072  # Default fallback
                self.logger.warning(f"Could not detect vocab size, using default: {self.actual_vocab_size:,}")
                
        except Exception as e:
            self.logger.warning(f"Failed to detect vocab size: {e}, using default: 131072")
            self.actual_vocab_size = 131072
        
    def load_apertus_from_safetensors(self):
        """
        Load the actual Apertus model from the downloaded safetensors files.
        Handle the sharded model files properly.
        """
        
        self.logger.info("Loading Apertus from safetensors files...")
        
        # Dynamically find all safetensor files
        safetensor_files = list(self.apertus_path.glob("model-*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No model-*.safetensors files found in {self.apertus_path}")
        
        # Sort files numerically by their part number
        safetensor_files.sort(key=lambda x: int(x.stem.split('-')[1]))
        
        for file in safetensor_files:
            self.logger.info(f"Found: {file}")
        
        # Load the tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/DialoGPT-medium"  # Fallback tokenizer
            )
            self.logger.info("Loaded tokenizer")
        except Exception as e:
            self.logger.warning(f"Could not load tokenizer: {e}")
            # Create basic tokenizer
            class BasicTokenizer:
                def __init__(self):
                    self.vocab_size = 50000
                def encode(self, text, add_special_tokens=False):
                    return [hash(word) % self.vocab_size for word in text.split()]
            self.tokenizer = BasicTokenizer()
        
        # Load model weights directly from safetensors without transformers
        try:
            self.model_weights = self._load_weights_from_safetensors()
            self.logger.info(f"Loaded Apertus weights directly from safetensors")
            self.model = None  # We don't need the actual model, just the weights
        except Exception as e:
            self.logger.error(f"Failed to load weights: {e}")
            raise e
    
    def _load_weights_from_safetensors(self):
        """Load weights directly from safetensors files without using transformers"""
        
        weights = {}
        
        # Load index to know which weights are in which file
        import json
        index_path = self.apertus_path / "model.safetensors.index.json"
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data["weight_map"]
        
        # Load all safetensor files
        loaded_files = {}
        for filename in set(weight_map.values()):
            file_path = self.apertus_path / filename
            self.logger.info(f"Loading weights from: {filename}")
            loaded_files[filename] = load_file(str(file_path))
        
        # Extract the weights we need (embeddings and key transformer weights)
        for weight_name, filename in weight_map.items():
            if any(key in weight_name for key in ["embed_tokens", "lm_head", "layers.0.", "layers.1.", "layers.2."]):
                weights[weight_name] = loaded_files[filename][weight_name]
        
        self.logger.info(f"Loaded {len(weights)} weight tensors for analysis")
        return weights
    
    def _get_cache_path(self):
        """Generate cache path based on model weights hash."""
        cache_dir = Path("models/chromathink/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create hash of model files for cache key
        hash_md5 = hashlib.md5()
        safetensor_files = list(self.apertus_path.glob("model-*.safetensors"))
        safetensor_files.sort()  # Consistent ordering for hash
        for file_path in safetensor_files:
            if file_path.exists():
                hash_md5.update(str(file_path.stat().st_mtime).encode())
                
        cache_key = hash_md5.hexdigest()[:16]
        return cache_dir / f"big_colour_model_{cache_key}.pkl"
    
    def _load_or_create_weight_patterns(self):
        """Load cached weight patterns or create new ones."""
        
        cache_path = self._get_cache_path()
        
        # Try to load from cache
        if cache_path.exists() and not self.force_rebuild:
            try:
                self.logger.info(f"Loading cached Big Colour Model from {cache_path}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}, rebuilding...")
        
        # Create new patterns
        if self.force_rebuild:
            self.logger.info("Force rebuilding Big Colour Model (ignoring cache)")
        self.logger.info("Creating new Big Colour Model (this may take a while for 131k tokens)")
        patterns = self.analyse_apertus_weights()
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(patterns, f)
            self.logger.info(f"Big Colour Model cached to {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
        
        return patterns
        
    def analyse_apertus_weights(self):
        """
        Analyse Apertus's weight matrices to understand its semantic encoding.
        We're looking for patterns that can translate to colour frequencies.
        """
        
        self.logger.info("Analysing Apertus weight patterns...")
        
        patterns = {}
        
        # Extract embedding layer patterns from direct weights
        embed_weights = None
        if self.model is None and self.model_weights:
            # Working with direct safetensor weights
            if "model.embed_tokens.weight" in self.model_weights:
                embed_tensor = self.model_weights["model.embed_tokens.weight"]
                # Convert bfloat16 to float32 for numpy compatibility
                if embed_tensor.dtype == torch.bfloat16:
                    embed_tensor = embed_tensor.to(torch.float32)
                embed_weights = embed_tensor.cpu().numpy()
            elif "embed_tokens.weight" in self.model_weights:
                embed_tensor = self.model_weights["embed_tokens.weight"]
                if embed_tensor.dtype == torch.bfloat16:
                    embed_tensor = embed_tensor.to(torch.float32)
                embed_weights = embed_tensor.cpu().numpy()
        elif self.model is not None:
            # Fallback to model-based extraction
            if hasattr(self.model, 'embed_tokens'):
                embed_weights = self.model.embed_tokens.weight.detach().cpu().numpy()
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                embed_weights = self.model.model.embed_tokens.weight.detach().cpu().numpy()
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
                embed_weights = self.model.transformer.wte.weight.detach().cpu().numpy()
        
        if embed_weights is not None:
            # Determine how many tokens to process using auto-detected size
            actual_vocab_size = embed_weights.shape[0]
            
            if self.extract_full_vocab:
                max_tokens = actual_vocab_size
                self.logger.info(f"Full vocabulary mode: processing all {max_tokens:,} tokens")
            elif self.max_tokens is not None:
                max_tokens = min(self.max_tokens, actual_vocab_size)
                self.logger.info(f"Limited mode: processing {max_tokens:,} tokens (subset of {actual_vocab_size:,} total)")
            else:
                # No max_tokens specified and not full vocab - use auto-detected size
                max_tokens = actual_vocab_size
                self.logger.info(f"Auto-detected mode: processing all {max_tokens:,} tokens")
            
            embed_subset = embed_weights[:max_tokens]
            
            # For large vocabularies, process in batches to manage memory
            if max_tokens > 50000:
                self.logger.info("Processing large vocabulary in batches for memory efficiency...")
                patterns['token_colours'] = self.process_embeddings_in_batches(embed_subset)
            else:
                patterns['token_colours'] = self.weights_to_colour_spectrum(embed_subset)
                
            self.logger.info(f"Successfully extracted colour patterns from {max_tokens} token embeddings")
        else:
            self.logger.warning("Could not find embedding weights, creating synthetic patterns")
            # Create synthetic token colours
            patterns['token_colours'] = self.create_synthetic_token_colours()
        
        # Extract attention patterns (these encode relationships)
        attention_patterns = []
        attention_count = 0
        
        # Work with direct weights if model is None
        if self.model is None and self.model_weights:
            for name, weight_tensor in self.model_weights.items():
                if 'self_attn' in name and 'weight' in name and attention_count < self.max_attention_layers:
                    # Convert bfloat16 to float32 for numpy compatibility
                    if weight_tensor.dtype == torch.bfloat16:
                        weight_tensor = weight_tensor.to(torch.float32)
                    weight = weight_tensor.cpu().numpy()
                    
                    # Attention patterns become interference patterns in colour space
                    colour_pattern = self.attention_to_interference(weight)
                    attention_patterns.append(colour_pattern)
                    attention_count += 1
                    
                    self.logger.debug(f"Processed attention layer {attention_count}: {name}, shape: {weight.shape}")
        elif self.model is not None:
            for name, param in self.model.named_parameters():
                if 'self_attn' in name and 'weight' in name and attention_count < self.max_attention_layers:
                    weight = param.detach().cpu().numpy()
                    
                    # Attention patterns become interference patterns in colour space
                    colour_pattern = self.attention_to_interference(weight)
                    attention_patterns.append(colour_pattern)
                    attention_count += 1
                    
                    self.logger.debug(f"Processed attention layer {attention_count}: {name}, shape: {weight.shape}")
        
        if not attention_patterns:
            self.logger.warning("No attention patterns found, creating synthetic ones")
            attention_patterns = [self.create_synthetic_attention_pattern() for _ in range(4)]
        
        patterns['attention_colours'] = np.array(attention_patterns)
        
        # Extract feed-forward patterns (these encode transformations)
        mlp_patterns = []
        mlp_count = 0
        
        # Work with direct weights if model is None
        if self.model is None and self.model_weights:
            for name, weight_tensor in self.model_weights.items():
                if 'mlp' in name and 'weight' in name and mlp_count < self.max_mlp_layers:
                    # Convert bfloat16 to float32 for numpy compatibility
                    if weight_tensor.dtype == torch.bfloat16:
                        weight_tensor = weight_tensor.to(torch.float32)
                    weight = weight_tensor.cpu().numpy()
                    
                    # MLP weights become colour transformation matrices
                    colour_transform = self.mlp_to_colour_transform(weight)
                    mlp_patterns.append(colour_transform)
                    mlp_count += 1
                    
                    self.logger.debug(f"Processed MLP layer {mlp_count}: {name}, shape: {weight.shape}")
        elif self.model is not None:
            for name, param in self.model.named_parameters():
                if 'mlp' in name and 'weight' in name and mlp_count < self.max_mlp_layers:
                    weight = param.detach().cpu().numpy()
                    
                    # MLP weights become colour transformation matrices
                    colour_transform = self.mlp_to_colour_transform(weight)
                    mlp_patterns.append(colour_transform)
                    mlp_count += 1
                    
                    self.logger.debug(f"Processed MLP layer {mlp_count}: {name}, shape: {weight.shape}")
        
        if not mlp_patterns:
            self.logger.warning("No MLP patterns found, creating synthetic ones")
            mlp_patterns = [self.create_synthetic_mlp_pattern() for _ in range(4)]
        
        patterns['transformation_colours'] = np.array(mlp_patterns)
        
        self.logger.info(f"Analysis complete: {len(patterns['token_colours'])} tokens, "
                        f"{len(patterns['attention_colours'])} attention patterns, "
                        f"{len(patterns['transformation_colours'])} transformation patterns")
        
        return patterns
    
    def create_synthetic_token_colours(self):
        """Create synthetic token colours for testing"""
        np.random.seed(42)  # Deterministic
        vocab_size = getattr(self.tokenizer, 'vocab_size', 32000)
        
        # Create meaningful patterns for important tokens
        colours = []
        for i in range(min(vocab_size, 10000)):  # Limit for memory
            # Create token-specific frequency pattern
            base_freq = (i % 100) / 100.0 * 2 * np.pi
            pattern = np.zeros(self.spectrum_dims, dtype=complex)
            
            # Add harmonics based on token properties
            for h in range(1, 6):  # 5 harmonics
                amplitude = 1.0 / h
                phase = base_freq * h + np.random.random() * 2 * np.pi
                freq_idx = (h * i) % self.spectrum_dims
                pattern[freq_idx] = amplitude * np.exp(1j * phase)
            
            colours.append(pattern)
        
        return np.array(colours)
    
    def create_synthetic_attention_pattern(self):
        """Create synthetic attention pattern"""
        np.random.seed(42)
        pattern = np.random.randn(self.spectrum_dims) + 1j * np.random.randn(self.spectrum_dims)
        return pattern / np.linalg.norm(pattern)
    
    def create_synthetic_mlp_pattern(self):
        """Create synthetic MLP pattern"""
        np.random.seed(42)
        pattern = np.random.randn(self.spectrum_dims) + 1j * np.random.randn(self.spectrum_dims)
        return pattern / np.linalg.norm(pattern)
    
    def weights_to_colour_spectrum(self, weight_matrix):
        """
        Convert weight matrix to high-nuance colour spectrum representation.
        Extracts frequency, wavelength, intensity, and harmonic components for light cognition.
        """
        
        # Handle different weight matrix sizes
        if len(weight_matrix.shape) == 1:
            weight_matrix = weight_matrix.reshape(1, -1)
        
        # Convert to float64 for higher precision FFT operations
        weight_matrix = weight_matrix.astype(np.float64)
        
        # Multi-scale spectral analysis for maximum nuance
        colours = []
        
        for i, weights in enumerate(weight_matrix):
            # 1. Primary FFT for fundamental frequencies
            fft_primary = np.fft.fft(weights, n=self.spectrum_dims)
            
            # 2. Window the signal to reduce spectral leakage
            windowed_weights = weights * np.hanning(len(weights))
            fft_windowed = np.fft.fft(windowed_weights, n=self.spectrum_dims)
            
            # 3. Extract multiple spectral characteristics
            amplitudes = np.abs(fft_primary)
            phases = np.angle(fft_primary)
            
            # 4. Calculate spectral features for light cognition
            # - Spectral centroid (brightness/color temperature)
            freqs = np.fft.fftfreq(self.spectrum_dims)
            spectral_centroid = np.sum(freqs * amplitudes) / (np.sum(amplitudes) + 1e-10)
            
            # - Spectral rolloff (intensity distribution)
            energy_cumsum = np.cumsum(amplitudes**2)
            total_energy = energy_cumsum[-1]
            rolloff_85 = np.where(energy_cumsum >= 0.85 * total_energy)[0]
            spectral_rolloff = rolloff_85[0] if len(rolloff_85) > 0 else len(amplitudes) - 1
            
            # - Spectral flux (dynamic change)
            spectral_flux = np.sum(np.diff(amplitudes)**2)
            
            # 5. Harmonic analysis for deeper color structure
            harmonics = []
            for h in range(1, min(8, self.spectrum_dims // 8)):  # Up to 7 harmonics
                harmonic_indices = np.arange(h, self.spectrum_dims, h)[:self.spectrum_dims//h]
                harmonic_energy = np.mean(amplitudes[harmonic_indices])
                harmonics.append(harmonic_energy)
            
            # 6. Combine spectral features into rich color representation
            # Base color from amplitude + phase
            base_color = amplitudes * np.exp(1j * phases)
            
            # Modulate with spectral characteristics
            color_temperature_factor = np.exp(spectral_centroid * 2j)  # Affects hue
            intensity_factor = np.exp(-spectral_rolloff / self.spectrum_dims)  # Affects saturation
            dynamic_factor = np.tanh(spectral_flux / 10.0)  # Affects brightness
            
            # Apply harmonic enrichment
            harmonic_modulation = np.ones(self.spectrum_dims, dtype=complex)
            for h, harmonic_energy in enumerate(harmonics, 1):
                harmonic_freq = np.arange(self.spectrum_dims) * h / self.spectrum_dims
                harmonic_modulation += harmonic_energy * 0.1 * np.exp(2j * np.pi * harmonic_freq)
            
            # Final rich color representation
            enriched_color = (base_color * color_temperature_factor * intensity_factor * 
                            dynamic_factor * harmonic_modulation)
            
            # Normalize to maintain stability while preserving nuance
            max_amplitude = np.max(np.abs(enriched_color))
            if max_amplitude > 0:
                enriched_color = enriched_color / max_amplitude
            
            colours.append(enriched_color)
        
        return np.array(colours)
    
    def attention_to_interference(self, attention_weights):
        """
        Convert attention patterns to rich interference patterns in colour space.
        Models how concepts interfere constructively/destructively with enhanced spectral detail.
        """
        
        # Reshape if needed
        if len(attention_weights.shape) > 2:
            attention_weights = attention_weights.reshape(
                attention_weights.shape[0], -1
            )
        
        # Handle single dimension
        if len(attention_weights.shape) == 1:
            attention_weights = attention_weights.reshape(1, -1)
        
        # Convert to float64 for higher precision analysis
        attention_weights = attention_weights.astype(np.float64)
        
        # Multi-modal interference analysis
        interference_pattern = np.zeros(self.spectrum_dims, dtype=complex)
        
        try:
            # 1. SVD for primary interference modes
            U, S, Vt = np.linalg.svd(attention_weights, full_matrices=False)
            
            # 2. Extract dominant patterns with phase relationships
            num_modes = min(self.spectrum_dims, len(S), 32)  # Limit to most significant modes
            
            for mode_idx in range(num_modes):
                # Mode strength and frequency
                mode_strength = S[mode_idx]
                mode_vector = Vt[mode_idx]
                
                # Convert to frequency domain for interference
                mode_fft = np.fft.fft(mode_vector, n=self.spectrum_dims)
                
                # Calculate interference frequency based on mode characteristics
                mode_freq = (mode_idx + 1) / num_modes  # Normalized frequency
                
                # 3. Model different types of interference
                # - Constructive interference (in-phase)
                constructive = mode_strength * np.abs(mode_fft) * np.exp(1j * np.angle(mode_fft))
                
                # - Destructive interference (phase-shifted)
                destructive = mode_strength * 0.3 * np.abs(mode_fft) * np.exp(1j * (np.angle(mode_fft) + np.pi))
                
                # - Partial interference (frequency-dependent phase)
                freq_indices = np.arange(self.spectrum_dims)
                phase_shift = 2 * np.pi * mode_freq * freq_indices / self.spectrum_dims
                partial = mode_strength * 0.5 * np.abs(mode_fft) * np.exp(1j * (np.angle(mode_fft) + phase_shift))
                
                # Weight interference types by mode significance
                mode_weight = mode_strength / (S[0] + 1e-10)  # Normalize by strongest mode
                
                # Combine interference types
                combined_interference = (
                    mode_weight * constructive +
                    (1 - mode_weight) * 0.6 * destructive +
                    0.4 * partial
                )
                
                # Add to total interference pattern
                interference_pattern += combined_interference
                
            # 4. Add spatial interference effects
            # Model how attention patterns create standing waves
            spatial_freq = np.arange(self.spectrum_dims) / self.spectrum_dims * 2 * np.pi
            
            # Create beating patterns between different modes
            for i in range(min(3, num_modes)):
                for j in range(i+1, min(3, num_modes)):
                    beat_freq = abs(S[i] - S[j]) / (S[0] + 1e-10)
                    beat_pattern = 0.2 * np.sin(spatial_freq * beat_freq) * np.exp(1j * spatial_freq * beat_freq)
                    interference_pattern += beat_pattern
            
        except np.linalg.LinAlgError:
            # Fallback: Create synthetic interference from weight statistics
            self.logger.warning("SVD failed, using statistical interference model")
            
            # Use weight statistics to create interference
            weight_mean = np.mean(attention_weights, axis=1, keepdims=True)
            weight_std = np.std(attention_weights, axis=1, keepdims=True)
            
            # Create interference based on statistical properties
            freqs = np.arange(self.spectrum_dims) / self.spectrum_dims * 2 * np.pi
            for i, (mean, std) in enumerate(zip(weight_mean.flatten(), weight_std.flatten())):
                # Interference frequency based on statistics
                interference_freq = (mean + 1) * (i + 1)
                amplitude = std + 0.1
                phase = np.angle(mean + 1j * std)
                
                wave = amplitude * np.exp(1j * (freqs * interference_freq + phase))
                interference_pattern += wave
        
        # Normalize while preserving relative phase relationships
        max_amplitude = np.max(np.abs(interference_pattern))
        if max_amplitude > 0:
            interference_pattern = interference_pattern / max_amplitude
            
        return interference_pattern
    
    def mlp_to_colour_transform(self, mlp_weights):
        """
        Convert MLP weights to colour transformation matrices.
        These define how colours evolve through thought.
        """
        
        # Ensure correct shape
        if len(mlp_weights.shape) > 2:
            mlp_weights = mlp_weights.reshape(mlp_weights.shape[0], -1)
        elif len(mlp_weights.shape) == 1:
            mlp_weights = mlp_weights.reshape(1, -1)
        
        # Sample or interpolate to spectrum dimensions
        if mlp_weights.shape[1] > self.spectrum_dims:
            # Downsample through averaging
            indices = np.linspace(0, mlp_weights.shape[1]-1, self.spectrum_dims, dtype=int)
            sampled = mlp_weights[:, indices]
        else:
            # Upsample through interpolation
            sampled = np.zeros((mlp_weights.shape[0], self.spectrum_dims))
            for i in range(mlp_weights.shape[0]):
                sampled[i] = np.interp(
                    np.linspace(0, mlp_weights.shape[1]-1, self.spectrum_dims),
                    np.arange(mlp_weights.shape[1]),
                    mlp_weights[i]
                )
        
        # Convert to colour transform
        colour_transform = self.weights_to_colour_spectrum(sampled)
        
        # Return first row if multiple rows
        return colour_transform[0] if len(colour_transform) > 0 else np.zeros(self.spectrum_dims, dtype=complex)
    
    def process_embeddings_in_batches(self, embeddings, batch_size=10000):
        """
        Process large embedding matrices in batches to manage memory usage.
        """
        
        total_tokens = embeddings.shape[0]
        all_colours = []
        
        self.logger.info(f"Processing {total_tokens} embeddings in batches of {batch_size}")
        
        for i in range(0, total_tokens, batch_size):
            end_idx = min(i + batch_size, total_tokens)
            batch = embeddings[i:end_idx]
            
            self.logger.info(f"Processing batch {i//batch_size + 1}: tokens {i} to {end_idx-1}")
            
            # Process this batch
            batch_colours = self.weights_to_colour_spectrum(batch)
            all_colours.append(batch_colours)
            
            # Clear memory
            del batch
            
        # Concatenate all batches
        result = np.vstack(all_colours)
        self.logger.info(f"Completed batch processing: {result.shape[0]} token colours generated")
        
        return result
    
    def build_big_colour_model(self):
        """Build the comprehensive colour model for encoding/decoding concepts to waveforms."""
        
        self.logger.info("Building Big Colour Model from weight patterns...")
        
        if not hasattr(self, 'weight_patterns') or not self.weight_patterns:
            self.logger.warning("No weight patterns available, building from synthetic data")
            self.weight_patterns = self.analyse_apertus_weights()
        
        # Build comprehensive concept encoder
        self.concept_encoder = ConceptEncoder(
            token_colours=self.weight_patterns['token_colours'],
            attention_patterns=self.weight_patterns['attention_colours'],
            spectrum_dims=self.spectrum_dims
        )
        
        # Build waveform decoder
        self.waveform_decoder = WaveformDecoder(
            transformation_patterns=self.weight_patterns['transformation_colours'],
            spectrum_dims=self.spectrum_dims
        )
        
        # Integrate into big colour model
        self.big_colour_model = BigColourModel(
            encoder=self.concept_encoder,
            decoder=self.waveform_decoder,
            spectrum_dims=self.spectrum_dims
        )
        
        vocab_size = len(self.weight_patterns['token_colours'])
        self.logger.info(f"Big Colour Model built successfully: {vocab_size} token patterns, "
                        f"{len(self.weight_patterns['attention_colours'])} attention patterns, "
                        f"{len(self.weight_patterns['transformation_colours'])} transformation patterns")
        
        return self.big_colour_model
    
    def encode_concept_to_waveform(self, text: str) -> np.ndarray:
        """Encode text concept to colour waveform using the big colour model."""
        
        if self.big_colour_model is None:
            self.build_big_colour_model()
        
        return self.big_colour_model.encode_concept(text)
    
    def decode_waveform_to_concept(self, waveform: np.ndarray, num_concepts: int = 5) -> list:
        """Decode colour waveform back to concept descriptors."""
        
        if self.big_colour_model is None:
            self.build_big_colour_model()
        
        return self.big_colour_model.decode_waveform(waveform, num_concepts)


class ConceptEncoder:
    """Encodes text concepts into colour waveforms using extracted token patterns."""
    
    def __init__(self, token_colours, attention_patterns, spectrum_dims=512):
        self.token_colours = token_colours
        self.attention_patterns = attention_patterns
        self.spectrum_dims = spectrum_dims
        
        # Create word-to-colour lookup for fast encoding
        self.word_colour_map = self._build_word_colour_map()
    
    def _build_word_colour_map(self):
        """Build mapping from common words to colour patterns."""
        
        # Common English words and their approximate token indices
        common_words = {
            'the': 1, 'of': 2, 'to': 3, 'and': 4, 'a': 5, 'in': 6, 'is': 7, 'it': 8, 'you': 9, 'that': 10,
            'he': 11, 'was': 12, 'for': 13, 'on': 14, 'are': 15, 'as': 16, 'with': 17, 'his': 18, 'they': 19, 'i': 20,
            'at': 21, 'be': 22, 'this': 23, 'have': 24, 'from': 25, 'or': 26, 'one': 27, 'had': 28, 'by': 29, 'word': 30,
            'what': 50, 'why': 51, 'how': 52, 'when': 53, 'where': 54, 'who': 55,
            'think': 100, 'feel': 101, 'know': 102, 'understand': 103, 'learn': 104, 'see': 105, 'hear': 106,
            'colour': 200, 'color': 201, 'sound': 202, 'light': 203, 'pattern': 204, 'meaning': 205, 'frequency': 206,
            'wave': 207, 'interference': 208, 'resonance': 209, 'spectrum': 210, 'amplitude': 211, 'phase': 212
        }
        
        word_map = {}
        for word, approx_idx in common_words.items():
            if approx_idx < len(self.token_colours):
                word_map[word] = self.token_colours[approx_idx]
            else:
                # Create synthetic colour for missing words
                word_hash = hash(word) % len(self.token_colours)
                word_map[word] = self.token_colours[word_hash]
        
        return word_map
    
    def encode_concept(self, text: str) -> np.ndarray:
        """Encode text to colour waveform by combining token patterns."""
        
        words = text.lower().split()
        if not words:
            return np.zeros(self.spectrum_dims, dtype=complex)
        
        # Start with first word's colour
        combined_waveform = self._get_word_colour(words[0]).copy()
        
        # Add interference from other words
        for word in words[1:]:
            word_colour = self._get_word_colour(word)
            combined_waveform = self._colour_interference(combined_waveform, word_colour)
        
        # Apply attention patterns for context
        if len(self.attention_patterns) > 0:
            attention_idx = hash(text) % len(self.attention_patterns)
            attention_pattern = self.attention_patterns[attention_idx]
            combined_waveform = self._apply_attention(combined_waveform, attention_pattern)
        
        return combined_waveform
    
    def _get_word_colour(self, word: str) -> np.ndarray:
        """Get colour pattern for a word."""
        
        if word in self.word_colour_map:
            return self.word_colour_map[word]
        
        # Generate colour from character patterns for unknown words
        char_sum = sum(ord(c) for c in word)
        token_idx = char_sum % len(self.token_colours)
        return self.token_colours[token_idx]
    
    def _colour_interference(self, colour1: np.ndarray, colour2: np.ndarray) -> np.ndarray:
        """Combine two colour patterns through interference."""
        
        # Extract amplitude and phase
        amp1, phase1 = np.abs(colour1), np.angle(colour1)
        amp2, phase2 = np.abs(colour2), np.angle(colour2)
        
        # Wave interference
        phase_diff = phase1 - phase2
        result_amp = np.sqrt(amp1**2 + amp2**2 + 2*amp1*amp2*np.cos(phase_diff))
        result_phase = np.arctan2(
            amp1*np.sin(phase1) + amp2*np.sin(phase2),
            amp1*np.cos(phase1) + amp2*np.cos(phase2)
        )
        
        return result_amp * np.exp(1j * result_phase)
    
    def _apply_attention(self, waveform: np.ndarray, attention_pattern: np.ndarray) -> np.ndarray:
        """Apply attention pattern to modulate waveform."""
        
        # Ensure attention pattern matches waveform size
        if len(attention_pattern) != len(waveform):
            attention_pattern = np.interp(
                np.linspace(0, len(attention_pattern)-1, len(waveform)),
                np.arange(len(attention_pattern)),
                attention_pattern
            )
        
        # Modulate waveform with attention
        return waveform * (1.0 + 0.3 * attention_pattern)


class WaveformDecoder:
    """Decodes colour waveforms back to concept descriptors."""
    
    def __init__(self, transformation_patterns, spectrum_dims=512):
        self.transformation_patterns = transformation_patterns
        self.spectrum_dims = spectrum_dims
        
        # Concept descriptors based on spectral properties
        self.frequency_concepts = self._build_frequency_concepts()
    
    def _build_frequency_concepts(self):
        """Build mapping from frequency ranges to concept descriptors."""
        
        concepts = {}
        
        # Low frequencies (0-20%) - foundational concepts
        concepts['foundation'] = [
            'fundamental', 'basic', 'core', 'essential', 'primary', 'root',
            'underlying', 'foundational', 'elemental', 'primitive'
        ]
        
        # Mid-low frequencies (20-40%) - structural concepts  
        concepts['structure'] = [
            'structured', 'organized', 'systematic', 'logical', 'ordered',
            'hierarchical', 'methodical', 'coherent', 'unified', 'integrated'
        ]
        
        # Mid frequencies (40-60%) - relational concepts
        concepts['connection'] = [
            'connected', 'related', 'linked', 'associated', 'correlated',
            'interdependent', 'intertwined', 'networked', 'bound', 'joined'
        ]
        
        # Mid-high frequencies (60-80%) - nuanced concepts
        concepts['nuance'] = [
            'nuanced', 'subtle', 'refined', 'sophisticated', 'complex',
            'intricate', 'layered', 'multifaceted', 'detailed', 'elaborate'
        ]
        
        # High frequencies (80-100%) - emergent concepts
        concepts['emergence'] = [
            'emergent', 'novel', 'innovative', 'creative', 'unique',
            'original', 'unprecedented', 'revolutionary', 'transformative', 'visionary'
        ]
        
        return concepts
    
    def decode_waveform(self, waveform: np.ndarray, num_concepts: int = 5) -> list:
        """Decode waveform to concept descriptors based on spectral analysis."""
        
        # Calculate spectral properties
        amplitudes = np.abs(waveform)
        phases = np.angle(waveform)
        
        # Find dominant frequencies
        dominant_freqs = np.argsort(amplitudes)[-num_concepts:][::-1]
        
        concepts = []
        
        for freq_idx in dominant_freqs:
            # Determine frequency band
            freq_ratio = freq_idx / len(waveform)
            
            if freq_ratio < 0.2:
                band = 'foundation'
            elif freq_ratio < 0.4:
                band = 'structure'
            elif freq_ratio < 0.6:
                band = 'connection'
            elif freq_ratio < 0.8:
                band = 'nuance'
            else:
                band = 'emergence'
            
            # Select concept based on amplitude and phase
            concept_list = self.frequency_concepts[band]
            amp_val = amplitudes[freq_idx]
            phase_val = phases[freq_idx]
            
            # Use phase to select from concept list
            concept_idx = int((phase_val + np.pi) / (2*np.pi) * len(concept_list))
            concept_idx = min(concept_idx, len(concept_list) - 1)
            
            concept = concept_list[concept_idx]
            concepts.append((concept, amp_val, freq_idx))
        
        return concepts


class BigColourModel:
    """Comprehensive colour model for encoding/decoding concepts to/from waveforms."""
    
    def __init__(self, encoder, decoder, spectrum_dims=512):
        self.encoder = encoder
        self.decoder = decoder
        self.spectrum_dims = spectrum_dims
        
        # Track encoding/decoding statistics
        self.encoding_count = 0
        self.decoding_count = 0
    
    def encode_concept(self, text: str) -> np.ndarray:
        """Encode text concept to colour waveform."""
        self.encoding_count += 1
        return self.encoder.encode_concept(text)
    
    def decode_waveform(self, waveform: np.ndarray, num_concepts: int = 5) -> list:
        """Decode colour waveform to concept descriptors."""
        self.decoding_count += 1
        return self.decoder.decode_waveform(waveform, num_concepts)
    
    def get_statistics(self) -> dict:
        """Get model usage statistics."""
        return {
            'encoding_count': self.encoding_count,
            'decoding_count': self.decoding_count,
            'spectrum_dims': self.spectrum_dims,
            'vocab_size': len(self.encoder.token_colours),
            'attention_patterns': len(self.encoder.attention_patterns),
            'transformation_patterns': len(self.decoder.transformation_patterns)
        }