"""
ChromaThinkCore: Pure colour-based thinking engine

CRITICAL: ChromaThink has NO knowledge of text or language. 
It thinks entirely in colour space through interference and resonance patterns.
All responses emerge from colour dynamics, never pre-programmed text.

The thinking process:
1. Receives concept colours (from ApertusTranslator)
2. Processes through interference and resonance
3. Generates response colours (for ApertusTranslator to synthesise)

ChromaThink is the mind, Apertus is the translator.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from ..layers.chromatic_resonance import ChromaticResonance
from ..layers.resonance_chamber import ResonanceChamber
from ..layers.interference import InterferenceLayer
from ..core.colour_utils import colour_distance


class InterferenceEngine:
    """
    Handles interference between colour patterns for complex thought processes.
    This is where understanding emerges from wave interactions.
    """
    
    def __init__(self, spectrum_dims: int = 512):
        self.spectrum_dims = spectrum_dims
        self.logger = logging.getLogger("InterferenceEngine")
        
        # Different types of interference layers for various thought processes
        self.constructive_interference = InterferenceLayer(
            interference_type='constructive',
            amplitude_weighting=True,
            phase_coupling=True,
            nonlinear_mixing=False
        )
        
        self.destructive_interference = InterferenceLayer(
            interference_type='destructive',
            amplitude_weighting=True,
            phase_coupling=True,
            nonlinear_mixing=False
        )
        
        self.full_interference = InterferenceLayer(
            interference_type='full',
            amplitude_weighting=True,
            phase_coupling=True,
            nonlinear_mixing=True
        )
    
    def interfere_colours(self, 
                         colour1: tf.Tensor, 
                         colour2: tf.Tensor, 
                         interference_type: str = 'adaptive') -> tf.Tensor:
        """
        Create interference between two colour patterns.
        This is how ChromaThink combines concepts.
        """
        
        # Ensure compatible shapes
        colour1, colour2 = self._ensure_compatible_shapes(colour1, colour2)
        
        if interference_type == 'constructive':
            # Concepts reinforce each other
            return self.constructive_interference([colour1, colour2])
        
        elif interference_type == 'destructive':
            # Concepts oppose each other (useful for contradictions)
            return self.destructive_interference([colour1, colour2])
        
        elif interference_type == 'full':
            # Complex interference patterns
            return self.full_interference([colour1, colour2])
        
        elif interference_type == 'adaptive':
            # Let natural interference occur based on colour properties
            return self._adaptive_interference(colour1, colour2)
        
        else:
            raise ValueError(f"Unknown interference type: {interference_type}")
    
    def multi_colour_interference(self, colours: List[tf.Tensor]) -> tf.Tensor:
        """
        Create interference patterns between multiple colours simultaneously.
        This is how ChromaThink handles complex multi-concept thinking.
        """
        
        if len(colours) < 2:
            return colours[0] if colours else tf.zeros(self.spectrum_dims, dtype=tf.complex64)
        
        # Start with first two colours
        result = self.interfere_colours(colours[0], colours[1], 'adaptive')
        
        # Progressively add more colours
        for colour in colours[2:]:
            result = self.interfere_colours(result, colour, 'adaptive')
        
        return result
    
    def _adaptive_interference(self, colour1: tf.Tensor, colour2: tf.Tensor) -> tf.Tensor:
        """
        Adaptive interference that chooses the pattern based on colour properties.
        """
        
        # Calculate colour properties with consistent dtypes
        amp1 = tf.cast(tf.abs(colour1), tf.float32)
        amp2 = tf.cast(tf.abs(colour2), tf.float32)
        phase1 = tf.cast(tf.math.angle(colour1), tf.float32)
        phase2 = tf.cast(tf.math.angle(colour2), tf.float32)
        
        # Calculate phase differences
        phase_diff = phase1 - phase2
        
        # Natural interference calculation
        # This follows the physics of wave interference
        combined_amplitude = tf.sqrt(
            amp1**2 + amp2**2 + 2*amp1*amp2*tf.cos(phase_diff)
        )
        
        # Phase is weighted average
        combined_phase = tf.atan2(
            amp1*tf.sin(phase1) + amp2*tf.sin(phase2),
            amp1*tf.cos(phase1) + amp2*tf.cos(phase2)
        )
        
        # Create complex result with consistent dtypes
        result = tf.cast(combined_amplitude, tf.complex64) * tf.exp(tf.complex(tf.cast(0.0, tf.float32), tf.cast(combined_phase, tf.float32)))
        
        return result
    
    def _ensure_compatible_shapes(self, colour1: tf.Tensor, colour2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Ensure two colour tensors have compatible shapes for interference."""
        
        # Get shapes
        shape1 = tf.shape(colour1)
        shape2 = tf.shape(colour2)
        
        # If different batch dimensions, expand the smaller one
        if len(colour1.shape) != len(colour2.shape):
            if len(colour1.shape) < len(colour2.shape):
                colour1 = tf.expand_dims(colour1, 0)
            else:
                colour2 = tf.expand_dims(colour2, 0)
        
        # If different spectrum dimensions, truncate or pad
        spectrum_dim1 = colour1.shape[-1]
        spectrum_dim2 = colour2.shape[-1]
        
        if spectrum_dim1 != spectrum_dim2:
            target_dim = min(spectrum_dim1, spectrum_dim2)
            colour1 = colour1[..., :target_dim]
            colour2 = colour2[..., :target_dim]
        
        return colour1, colour2


class PureColourMemory:
    """
    Pure colour-based memory with NO text storage.
    Memories are stored as colour patterns and retrieved through resonance.
    """
    
    def __init__(self, spectrum_dims: int = 512, capacity: int = 10000, gpu_accelerator=None):
        self.spectrum_dims = spectrum_dims
        self.capacity = capacity
        self.gpu_accelerator = gpu_accelerator
        self.logger = logging.getLogger("PureColourMemory")
        
        # Memory storage as pure colour patterns
        self.memory_colours = tf.Variable(
            tf.zeros([capacity, spectrum_dims], dtype=tf.complex64),
            trainable=False,
            name='memory_colours'
        )
        
        # Memory strengths (how important each memory is)
        self.memory_strengths = tf.Variable(
            tf.zeros([capacity], dtype=tf.float32),
            trainable=False,
            name='memory_strengths'
        )
        
        # Current memory index (circular buffer)
        self.memory_index = tf.Variable(0, trainable=False, name='memory_index')
        
        # Memory access counts for strengthening frequently accessed memories
        self.access_counts = tf.Variable(
            tf.zeros([capacity], dtype=tf.int32),
            trainable=False,
            name='access_counts'
        )
    
    def store(self, colour_pattern: tf.Tensor, strength: float = 1.0):
        """
        Store a colour pattern in memory.
        """
        
        # Ensure correct shape and dtype
        if len(colour_pattern.shape) > 1:
            colour_pattern = tf.squeeze(colour_pattern)
        
        if colour_pattern.dtype != tf.complex64:
            colour_pattern = tf.cast(colour_pattern, tf.complex64)
        
        # Pad or truncate to spectrum dimensions
        current_size = tf.shape(colour_pattern)[0]
        if current_size < self.spectrum_dims:
            # Pad with zeros
            padding = tf.zeros([self.spectrum_dims - current_size], dtype=tf.complex64)
            colour_pattern = tf.concat([colour_pattern, padding], axis=0)
        else:
            # Truncate
            colour_pattern = colour_pattern[:self.spectrum_dims]
        
        # Store in circular buffer
        idx = self.memory_index % self.capacity
        self.memory_colours[idx].assign(colour_pattern)
        self.memory_strengths[idx].assign(strength)
        self.access_counts[idx].assign(0)  # Reset access count
        
        # Increment index
        self.memory_index.assign(self.memory_index + 1)
        
        self.logger.debug(f"Stored colour memory at index {idx} with strength {strength}")
    
    def recall(self, query_colour: tf.Tensor, num_memories: int = 5) -> List[Tuple[float, tf.Tensor, int]]:
        """
        Recall memories that resonate with the query colour.
        Returns list of (resonance_strength, memory_colour, memory_index).
        """
        
        # Ensure query has correct shape and dtype
        if len(query_colour.shape) > 1:
            query_colour = tf.squeeze(query_colour)
        
        if query_colour.dtype != tf.complex64:
            query_colour = tf.cast(query_colour, tf.complex64)
        
        # Pad or truncate query to match memory dimensions
        current_size = tf.shape(query_colour)[0]
        if current_size < self.spectrum_dims:
            padding = tf.zeros([self.spectrum_dims - current_size], dtype=tf.complex64)
            query_colour = tf.concat([query_colour, padding], axis=0)
        else:
            query_colour = query_colour[:self.spectrum_dims]
        
        # Calculate how many memories we actually have
        num_stored = tf.minimum(self.memory_index, self.capacity)
        
        if num_stored == 0:
            return []
        
        # Use GPU accelerated memory search if available
        if self.gpu_accelerator is not None and int(num_stored) > 10:
            try:
                stored_memories = self.memory_colours[:int(num_stored)]
                stored_strengths = self.memory_strengths[:int(num_stored)]

                similarities, indices = self.gpu_accelerator.accelerated_memory_search(
                    query_colour, stored_memories, stored_strengths, top_k=min(num_memories, int(num_stored))
                )

                # Convert to expected format
                top_memories = []
                for i in range(len(similarities)):
                    idx = int(indices[i])
                    similarity = float(similarities[i])
                    memory = self.memory_colours[idx]
                    top_memories.append((similarity, memory, idx))

                    # Update access count
                    self.access_counts[idx].assign(self.access_counts[idx] + 1)

                return top_memories

            except Exception as e:
                self.logger.warning(f"GPU memory search failed, falling back to CPU: {e}")

        # Fallback to CPU implementation
        # Calculate resonance with all stored memories
        resonances = []

        for i in range(int(num_stored)):
            memory = self.memory_colours[i]

            # Calculate complex resonance (colour similarity)
            resonance = tf.abs(tf.reduce_sum(query_colour * tf.math.conj(memory)))

            # Weight by memory strength and access history
            strength_weight = self.memory_strengths[i]
            access_weight = 1.0 + tf.cast(self.access_counts[i], tf.float32) * 0.1

            weighted_resonance = resonance * strength_weight * access_weight

            resonances.append((float(weighted_resonance), memory, i))
        
        # Sort by resonance strength and return top memories
        resonances.sort(key=lambda x: x[0], reverse=True)
        top_memories = resonances[:num_memories]
        
        # Update access counts for retrieved memories
        for _, memory, idx in top_memories:
            self.access_counts[idx].assign(self.access_counts[idx] + 1)
        
        self.logger.debug(f"Recalled {len(top_memories)} memories for query")
        
        return top_memories


class ChromaThinkCore:
    """
    The pure colour-based thinking engine.
    NO pre-programmed text responses exist here.
    All thinking happens through colour interference and resonance.
    """
    
    def __init__(self, spectrum_dims: int = 512, num_resonance_chambers: int = 5, gpu_accelerator=None):
        self.spectrum_dims = spectrum_dims
        self.num_resonance_chambers = num_resonance_chambers
        self.gpu_accelerator = gpu_accelerator
        self.logger = logging.getLogger("ChromaThinkCore")
        
        # Resonance chambers for different types of thinking
        self.resonance_chambers = []
        for i in range(num_resonance_chambers):
            chamber = ResonanceChamber(
                dimensions=spectrum_dims,
                num_modes=32 + i * 8,  # Increasing complexity
                boundary_conditions='mixed',
                quality_factor=0.8 + i * 0.03  # Increasing quality
            )
            self.resonance_chambers.append(chamber)
        
        # Deep chromatic resonance layers
        self.chromatic_layers = []
        for i in range(3):  # 3 deep layers
            layer = ChromaticResonance(
                dimensions=spectrum_dims,
                resonance_depth=5 + i * 2,
                harmonic_orders=[1, 2, 3, 5, 8, 13],  # Fibonacci harmonics
                memory_decay=0.9 - i * 0.1
            )
            self.chromatic_layers.append(layer)
        
        # Interference engine for complex thought
        self.interference_engine = InterferenceEngine(spectrum_dims)
        
        # Pure colour memory (no text)
        self.colour_memory = PureColourMemory(spectrum_dims, capacity=10000, gpu_accelerator=gpu_accelerator)
        
        # Current cognitive state (working memory)
        self.cognitive_state = tf.Variable(
            tf.zeros([spectrum_dims], dtype=tf.complex64),
            trainable=False,
            name='cognitive_state'
        )
        
        self.logger.info(f"ChromaThinkCore initialized with {spectrum_dims} dimensions and {num_resonance_chambers} resonance chambers")
    
    def think_in_colour_complex(self, 
                               concept_colours: Dict[str, List[tf.Tensor]], 
                               context_colour: Optional[tf.Tensor] = None,
                               thinking_intensity: float = 1.0) -> tf.Tensor:
        """
        Complex colour-based thinking process for structured input.
        
        Args:
            concept_colours: Dict with 'concepts', 'intent', 'context' colour lists
            context_colour: Optional additional context from previous thinking
            thinking_intensity: How intensely to think (0.5 to 2.0)
        
        Returns:
            Response colour pattern representing the thought
        """
        
        self.logger.info("Beginning complex colour-based thinking process")
        
        # Step 1: Combine input concepts into unified colour pattern
        thought_pattern = self._combine_concept_colours(concept_colours)
        
        # Step 2: Add context if available
        if context_colour is not None:
            thought_pattern = self.interference_engine.interfere_colours(
                thought_pattern, context_colour, 'constructive'
            )
        
        # Step 3: Blend with current cognitive state
        if tf.reduce_sum(tf.abs(self.cognitive_state)) > 0:
            thought_pattern = self.interference_engine.interfere_colours(
                thought_pattern, self.cognitive_state, 'adaptive'
            )
        
        # Step 4: Recall relevant memories and interfere with them
        memory_resonances = self.colour_memory.recall(thought_pattern, num_memories=7)
        for resonance_strength, memory_colour, memory_idx in memory_resonances:
            # Weight by resonance strength
            if resonance_strength > 0.1:  # Only use strongly resonating memories
                weighted_memory = memory_colour * resonance_strength * 0.3
                thought_pattern = self.interference_engine.interfere_colours(
                    thought_pattern, weighted_memory, 'adaptive'
                )
        
        # Step 5: Process through resonance chambers (this is where deep thinking happens)
        for i, chamber in enumerate(self.resonance_chambers):
            # Apply thinking intensity
            intensity_factor = thinking_intensity * (0.8 + i * 0.05)
            amplified_pattern = thought_pattern * intensity_factor
            
            # Resonate through chamber
            thought_pattern = chamber(tf.expand_dims(amplified_pattern, 0))[0]
            
            self.logger.debug(f"After resonance chamber {i}: amplitude={tf.reduce_mean(tf.abs(thought_pattern)):.4f}")
        
        # Step 6: Process through chromatic resonance layers (harmonic analysis)
        for layer in self.chromatic_layers:
            thought_pattern = layer(tf.expand_dims(thought_pattern, 0))[0]
        
        # Step 7: Generate curiosity/response through interference with itself
        # This creates self-reflection and questioning
        curiosity_pattern = self.interference_engine.interfere_colours(
            thought_pattern, 
            tf.roll(thought_pattern, shift=self.spectrum_dims//4, axis=0),  # Phase-shifted self
            'adaptive'
        )
        
        # Step 8: Create final response by interfering thought with curiosity
        response_colour = self.interference_engine.interfere_colours(
            thought_pattern, curiosity_pattern, 'full'
        )
        
        # Step 9: Normalize and stabilize
        response_colour = self._stabilize_colour_pattern(response_colour)
        
        # Step 10: Store this thought experience in memory
        self.colour_memory.store(response_colour, strength=thinking_intensity)
        
        # Step 11: Update cognitive state
        self.cognitive_state.assign(
            0.7 * self.cognitive_state + 0.3 * response_colour
        )
        
        self.logger.info(f"Colour thinking complete. Response amplitude: {tf.reduce_mean(tf.abs(response_colour)):.4f}")
        
        return response_colour
    
    def encode_concept_to_colour(self, concept: str) -> tf.Tensor:
        """
        Encode a concept string to colour without pre-programmed mappings.
        Uses deterministic hash-based frequency generation.
        """
        
        # Generate deterministic colour from concept hash
        concept_hash = hash(concept) % (2**32)
        np.random.seed(concept_hash)
        
        # Create complex frequency pattern
        frequencies = np.random.exponential(scale=20, size=self.spectrum_dims)
        phases = np.random.uniform(0, 2*np.pi, size=self.spectrum_dims)
        amplitudes = 1.0 / (1.0 + frequencies)  # Higher frequency = lower amplitude
        
        # Create complex colour
        colour = amplitudes * np.exp(1j * phases)
        colour_tensor = tf.constant(colour, dtype=tf.complex64)
        
        return colour_tensor
    
    def _combine_concept_colours(self, concept_colours: Dict[str, List[tf.Tensor]]) -> tf.Tensor:
        """
        Combine multiple concept colours into unified thought pattern.
        """
        
        all_colours = []
        weights = {'intent': 0.4, 'concepts': 0.4, 'context': 0.2}  # Intent and concepts most important
        
        for concept_type, colour_list in concept_colours.items():
            weight = weights.get(concept_type, 0.1)
            
            if isinstance(colour_list, list) and colour_list:
                # Average multiple colours of same type
                colour_tensors = []
                for colour in colour_list:
                    if isinstance(colour, str):
                        colour = self.encode_concept_to_colour(colour)
                    colour_tensors.append(colour)
                
                if colour_tensors:
                    avg_colour = tf.reduce_mean(tf.stack(colour_tensors), axis=0)
                    weighted_colour = avg_colour * weight
                    all_colours.append(weighted_colour)
        
        if not all_colours:
            # Return neutral wonder state
            return tf.complex(
                tf.random.normal([self.spectrum_dims]),
                tf.random.normal([self.spectrum_dims])
            ) * 0.1
        
        # Use interference engine to combine all colours
        combined = self.interference_engine.multi_colour_interference(all_colours)
        
        return combined
    
    def _stabilize_colour_pattern(self, colour: tf.Tensor) -> tf.Tensor:
        """
        Stabilize colour pattern to prevent explosion or collapse.
        """
        
        # Calculate current amplitude
        amplitude = tf.abs(colour)
        max_amplitude = tf.reduce_max(amplitude)
        mean_amplitude = tf.reduce_mean(amplitude)
        
        # Prevent explosion (amplitude too high)
        if max_amplitude > 10.0:
            scalar_factor = tf.cast(2.0 / max_amplitude, tf.complex64)
            colour = colour * scalar_factor
        
        # Prevent collapse (amplitude too low)
        elif mean_amplitude < 0.01:
            scalar_factor = tf.cast(0.1 / (mean_amplitude + 1e-8), tf.complex64)
            colour = colour * scalar_factor
        
        # Add small amount of noise to prevent exact zeros
        real_noise = tf.random.normal(tf.shape(colour), stddev=0.001, dtype=tf.float32)
        imag_noise = tf.random.normal(tf.shape(colour), stddev=0.001, dtype=tf.float32)
        noise = tf.complex(real_noise, imag_noise)
        colour = colour + noise
        
        return colour
    
    def get_cognitive_metrics(self) -> Dict:
        """
        Return metrics about current cognitive state.
        """
        
        state_amplitude = tf.reduce_mean(tf.abs(self.cognitive_state))
        state_complexity = self._calculate_spectral_entropy(self.cognitive_state)
        
        # Memory metrics
        num_memories = min(int(self.colour_memory.memory_index), self.colour_memory.capacity)
        avg_memory_strength = tf.reduce_mean(self.colour_memory.memory_strengths[:num_memories]) if num_memories > 0 else 0.0
        
        return {
            'cognitive_amplitude': float(state_amplitude),
            'cognitive_complexity': float(state_complexity),
            'memory_count': num_memories,
            'average_memory_strength': float(avg_memory_strength),
            'spectrum_dimensions': self.spectrum_dims,
            'resonance_chambers': len(self.resonance_chambers)
        }
    
    def _calculate_spectral_entropy(self, colour: tf.Tensor) -> float:
        """Calculate spectral entropy as measure of complexity."""
        
        # Handle ResourceVariable by getting its value
        if hasattr(colour, 'value'):
            colour = colour.value()
        
        power = tf.abs(colour) ** 2
        power_norm = power / (tf.reduce_sum(power) + 1e-8)
        
        entropy = -tf.reduce_sum(power_norm * tf.math.log(power_norm + 1e-8))
        max_entropy = tf.math.log(tf.cast(tf.shape(colour)[0], tf.float32))
        
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0
    
    def reset_cognitive_state(self):
        """Reset cognitive state to neutral."""
        self.cognitive_state.assign(tf.zeros_like(self.cognitive_state))
        self.logger.info("Cognitive state reset")
    
    def think_in_colour(self, input_colour: tf.Tensor, memory_context: List[tf.Tensor] = None) -> tf.Tensor:
        """
        Simplified colour thinking method for integration with BigColourIntegration.
        
        Args:
            input_colour: Input colour pattern to think about
            memory_context: List of recent colour memories for context
        
        Returns:
            Response colour pattern from pure colour thinking
        """
        
        self.logger.info("Pure colour thinking process started")
        
        # Ensure correct format
        if len(input_colour.shape) > 1:
            input_colour = tf.squeeze(input_colour)
        if input_colour.dtype != tf.complex64:
            input_colour = tf.cast(input_colour, tf.complex64)
        
        # Start with input colour
        thought_pattern = input_colour
        
        # Add memory context through interference
        if memory_context:
            for memory_colour in memory_context[-3:]:  # Use last 3 memories
                if len(memory_colour.shape) > 1:
                    memory_colour = tf.squeeze(memory_colour)
                if memory_colour.dtype != tf.complex64:
                    memory_colour = tf.cast(memory_colour, tf.complex64)
                
                # Interfere with memory
                thought_pattern = self.colour_interference(thought_pattern, memory_colour)
        
        # Process through resonance chambers with proper tensor handling
        for i, chamber in enumerate(self.resonance_chambers[:3]):  # Use first 3 chambers
            try:
                # Ensure correct input format for resonance chamber
                if len(thought_pattern.shape) == 1:
                    expanded = tf.expand_dims(thought_pattern, 0)
                else:
                    expanded = thought_pattern
                
                # Process through chamber with error handling
                resonated = chamber(expanded)
                
                # Ensure correct output shape - force to 1D
                thought_pattern = tf.squeeze(resonated)
                if len(thought_pattern.shape) > 1:
                    # If still 2D, take the first row or flatten
                    if thought_pattern.shape[0] == 1:
                        thought_pattern = thought_pattern[0]
                    else:
                        # Take mean across first dimension to collapse to 1D
                        thought_pattern = tf.reduce_mean(thought_pattern, axis=0)
                    
                self.logger.debug(f"Resonance chamber {i}: input={expanded.shape}, output={resonated.shape}")
                
            except Exception as e:
                self.logger.warning(f"Resonance chamber {i} failed: {e}, continuing without resonance")
                # Continue with the pattern as-is if resonance fails
                pass
        
        # Add curiosity through self-interference  
        shifted_pattern = tf.roll(thought_pattern, shift=self.spectrum_dims//4, axis=0)
        curiosity_pattern = self.colour_interference(thought_pattern, shifted_pattern)
        
        # Final response through interference
        response_colour = self.colour_interference(thought_pattern, curiosity_pattern)
        
        # Stabilize response
        response_colour = self._stabilize_colour_pattern(response_colour)
        
        # Ensure response is exactly 1D with correct dimensions
        response_colour = tf.squeeze(response_colour)
        if len(response_colour.shape) > 1:
            # Force to 1D if still multi-dimensional
            if response_colour.shape[0] == 1:
                response_colour = response_colour[0]
            else:
                response_colour = tf.reduce_mean(response_colour, axis=0)
        
        # Ensure correct size
        if response_colour.shape[0] != self.spectrum_dims:
            # Interpolate or pad to correct size
            if response_colour.shape[0] > self.spectrum_dims:
                # Downsample
                indices = tf.cast(tf.linspace(0, tf.cast(response_colour.shape[0]-1, tf.float32), self.spectrum_dims), tf.int32)
                response_colour = tf.gather(response_colour, indices)
            else:
                # Pad with zeros
                padding = self.spectrum_dims - response_colour.shape[0]
                response_colour = tf.concat([response_colour, tf.zeros(padding, dtype=response_colour.dtype)], axis=0)
        
        self.logger.debug(f"Final response shape: {response_colour.shape}")
        self.colour_memory.store(response_colour, strength=1.0)
        
        # Update cognitive state
        self.cognitive_state.assign(0.8 * self.cognitive_state + 0.2 * response_colour)
        
        self.logger.info("Pure colour thinking complete")
        return response_colour
    
    def colour_interference(self, colour1: tf.Tensor, colour2: tf.Tensor) -> tf.Tensor:
        """
        Simple colour interference for wave interaction.
        Uses GPU acceleration when available.
        """

        # Use GPU accelerated interference if available
        if self.gpu_accelerator is not None:
            try:
                return self.gpu_accelerator.accelerated_colour_interference([colour1, colour2])
            except Exception as e:
                self.logger.warning(f"GPU acceleration failed, falling back to CPU: {e}")

        # Fallback to CPU implementation
        # Extract amplitude and phase
        amp1 = tf.abs(colour1)
        amp2 = tf.abs(colour2)
        phase1 = tf.math.angle(colour1)
        phase2 = tf.math.angle(colour2)

        # Wave interference calculation
        phase_diff = phase1 - phase2
        result_amp = tf.sqrt(amp1**2 + amp2**2 + 2*amp1*amp2*tf.cos(phase_diff))

        # Phase calculation
        result_phase = tf.atan2(
            amp1*tf.sin(phase1) + amp2*tf.sin(phase2),
            amp1*tf.cos(phase1) + amp2*tf.cos(phase2)
        )

        # Combine amplitude and phase
        result = tf.cast(result_amp, tf.complex64) * tf.exp(tf.complex(0.0, result_phase))

        return result
    
    def apply_resonance(self, colour: tf.Tensor, depth: int = 3) -> tf.Tensor:
        """
        Apply resonance transformation for dream-like processing.
        """
        
        processed_colour = colour
        
        # Apply multiple resonance layers
        for i in range(min(depth, len(self.chromatic_layers))):
            expanded = tf.expand_dims(processed_colour, 0)
            resonated = self.chromatic_layers[i](expanded)
            processed_colour = tf.squeeze(resonated)
        
        return processed_colour