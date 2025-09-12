"""
IntegratedChromaThink: Complete system integrating Apertus translation with ChromaThink thinking

This is the main interface that implements true colour-based thinking:

Complete Pipeline:
1. Human text (any language) -> Apertus extracts concepts
2. Concepts encoded as colours -> ChromaThink processes in pure colour space
3. Colour pattern output -> Apertus synthesises back to same language
4. Human text response

Apertus is the translator, ChromaThink is the mind.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from .apertus_translator import ApertusTranslator
from .chromathink_core import ChromaThinkCore


class IntegratedChromaThink:
    """
    Complete ChromaThink system with pure colour-based cognition.
    
    This system ensures that:
    - Apertus ONLY translates (never generates responses)
    - ChromaThink ONLY thinks in colour (never processes text)
    - All responses emerge from colour interference patterns
    - Language is preserved (input language = output language)
    """
    
    def __init__(self, 
                 apertus_path: str = "models/apertus",
                 spectrum_dims: int = 512,
                 num_resonance_chambers: int = 5,
                 use_mock_apertus: bool = False,
                 gpu_acceleration: bool = True):
        
        self.spectrum_dims = spectrum_dims
        self.logger = logging.getLogger("IntegratedChromaThink")
        
        # Initialize GPU acceleration if requested
        if gpu_acceleration:
            self._setup_gpu_acceleration()
        
        # Initialize Apertus translator (ONLY for translation)
        self.translator = ApertusTranslator(
            apertus_path=apertus_path,
            spectrum_dims=spectrum_dims,
            use_mock=use_mock_apertus
        )
        
        # Initialize ChromaThink core (ONLY for thinking)
        self.thinker = ChromaThinkCore(
            spectrum_dims=spectrum_dims,
            num_resonance_chambers=num_resonance_chambers
        )
        
        # Conversation history (for context)
        self.conversation_context = []
        self.max_context_length = 5
        
        # Performance metrics
        self.metrics = {
            'total_conversations': 0,
            'average_thinking_time': 0.0,
            'colour_complexity_history': [],
            'language_distribution': {}
        }
        
        self.logger.info("IntegratedChromaThink system ready - pure colour cognition enabled")
    
    def _setup_gpu_acceleration(self):
        """Setup GPU acceleration for colour mathematics."""
        
        # Enable GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set mixed precision for faster computation
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                
                self.logger.info(f"GPU acceleration enabled with {len(gpus)} GPUs")
            except RuntimeError as e:
                self.logger.warning(f"GPU setup failed: {e}")
        else:
            self.logger.info("No GPUs detected, using CPU")
    
    def process_conversation(self, 
                           human_input: str, 
                           thinking_intensity: float = 1.0,
                           use_conversation_context: bool = True) -> str:
        """
        Process a complete conversation turn using pure colour thinking.
        
        Args:
            human_input: Text input from human (any language)
            thinking_intensity: How intensely ChromaThink should think (0.5-2.0)
            use_conversation_context: Whether to use previous conversation context
        
        Returns:
            Text response in the same language as input
        """
        
        self.logger.info(f"Processing conversation input: {human_input[:50]}...")
        
        start_time = tf.timestamp()
        
        # Step 1: Extract concepts using Apertus (TRANSLATION ONLY)
        self.logger.info("Step 1: Extracting concepts from human language")
        concepts, source_language = self.translator.extract_concepts(human_input)
        
        self.logger.info(f"Detected language: {source_language}")
        self.logger.debug(f"Extracted concepts: {concepts}")
        
        # Update language distribution metrics
        self.metrics['language_distribution'][source_language] = \
            self.metrics['language_distribution'].get(source_language, 0) + 1
        
        # Step 2: Convert concepts to colours
        self.logger.info("Step 2: Converting concepts to colour representations")
        concept_colours = self._concepts_to_colours(concepts)
        
        # Step 3: Add conversation context if requested
        context_colour = None
        if use_conversation_context and self.conversation_context:
            context_colour = self._get_context_colour()
        
        # Step 4: Think in pure colour space (NO TEXT PROCESSING)
        self.logger.info("Step 4: Pure colour-based thinking process")
        
        # Convert concept colours to a single input colour pattern
        input_colour = self._combine_concept_colours(concept_colours)
        
        # Add context if available
        if context_colour is not None:
            # Simple interference to combine input with context
            input_colour = self._simple_colour_interference(input_colour, context_colour)
        
        # Get memory context in the correct format
        memory_context = []
        if hasattr(self.thinker, 'colour_memory'):
            memory_recall = self.thinker.colour_memory.recall(input_colour, num_memories=3)
            # Extract just the memory tensors from the tuples
            memory_context = [memory_tensor for _, memory_tensor, _ in memory_recall]
        
        # Use the updated think_in_colour method
        response_colour = self.thinker.think_in_colour(
            input_colour=input_colour,
            memory_context=memory_context
        )
        
        # Step 5: Convert colour response to concept descriptors
        self.logger.info("Step 5: Converting colour response to concept descriptors")
        response_descriptors = self.translator.colour_to_concept_descriptors(response_colour)
        
        self.logger.debug(f"Generated descriptors: {response_descriptors}")
        
        # Step 6: Synthesise response in original language (TRANSLATION ONLY)
        self.logger.info(f"Step 6: Synthesising response in {source_language}")
        response_text = self.translator.synthesise_response(
            colour_descriptors=response_descriptors,
            target_language=source_language,
            original_concepts=concepts
        )
        
        # Step 7: Update conversation context
        self._update_conversation_context(human_input, response_text, response_colour)
        
        # Step 8: Update metrics
        thinking_time = float(tf.timestamp() - start_time)
        self._update_metrics(thinking_time, response_colour)
        
        self.logger.info(f"Conversation processing complete in {thinking_time:.3f}s")
        
        return response_text
    
    def _concepts_to_colours(self, concepts: Dict[str, List[str]]) -> Dict[str, List[tf.Tensor]]:
        """
        Convert concept strings to colour tensors.
        """
        
        concept_colours = {}
        
        for concept_type, concept_list in concepts.items():
            colours = []
            for concept in concept_list:
                colour = self.thinker.encode_concept_to_colour(concept)
                colours.append(colour)
            concept_colours[concept_type] = colours
        
        return concept_colours
    
    def _get_context_colour(self) -> tf.Tensor:
        """
        Create context colour from recent conversation history.
        """
        
        if not self.conversation_context:
            return None
        
        # Get recent response colours
        recent_colours = [
            entry['response_colour'] for entry in self.conversation_context[-3:]
            if entry['response_colour'] is not None
        ]
        
        if not recent_colours:
            return None
        
        # Combine recent colours through interference
        context_colour = self.thinker.interference_engine.multi_colour_interference(recent_colours)
        
        # Reduce intensity (context should be subtle)
        context_colour = context_colour * 0.3
        
        return context_colour
    
    def _update_conversation_context(self, 
                                   human_input: str, 
                                   response_text: str, 
                                   response_colour: tf.Tensor):
        """
        Update conversation context for future interactions.
        """
        
        context_entry = {
            'human_input': human_input,
            'response_text': response_text,
            'response_colour': tf.identity(response_colour),  # Store copy
            'timestamp': tf.timestamp()
        }
        
        self.conversation_context.append(context_entry)
        
        # Maintain context length limit
        if len(self.conversation_context) > self.max_context_length:
            self.conversation_context.pop(0)
    
    def _update_metrics(self, thinking_time: float, response_colour: tf.Tensor):
        """Update performance and cognitive metrics."""
        
        self.metrics['total_conversations'] += 1
        
        # Update average thinking time
        prev_avg = self.metrics['average_thinking_time']
        count = self.metrics['total_conversations']
        self.metrics['average_thinking_time'] = (prev_avg * (count - 1) + thinking_time) / count
        
        # Track colour complexity
        complexity = self.thinker._calculate_spectral_entropy(response_colour)
        self.metrics['colour_complexity_history'].append(float(complexity))
        
        # Keep only recent complexity history
        if len(self.metrics['colour_complexity_history']) > 100:
            self.metrics['colour_complexity_history'].pop(0)
    
    def think_about(self, 
                   concept: str, 
                   intensity: float = 1.0) -> tf.Tensor:
        """
        Make ChromaThink think about a specific concept in pure colour.
        Returns the colour representation of the thought.
        """
        
        self.logger.info(f"Thinking about concept: {concept}")
        
        # Encode concept to colour
        concept_colour = self.thinker.encode_concept_to_colour(concept)
        
        # Create concept dictionary
        concept_colours = {
            'concepts': [concept_colour],
            'intent': [self.thinker.encode_concept_to_colour('understanding')],
            'context': [self.thinker.encode_concept_to_colour('contemplation')]
        }
        
        # Think about it
        thought_colour = self.thinker.think_in_colour(
            concept_colours=concept_colours,
            thinking_intensity=intensity
        )
        
        return thought_colour
    
    def dream(self, dream_intensity: float = 0.7, num_dream_cycles: int = 5) -> List[tf.Tensor]:
        """
        Let ChromaThink dream - process memories through colour interference.
        This is memory consolidation through colour dynamics.
        """
        
        self.logger.info(f"Beginning dream sequence with {num_dream_cycles} cycles")
        
        dream_colours = []
        
        for cycle in range(num_dream_cycles):
            # Get random memories from colour memory
            num_memories = min(5, int(self.thinker.colour_memory.memory_index))
            if num_memories == 0:
                break
            
            # Sample random memories
            memory_indices = np.random.choice(num_memories, size=min(3, num_memories), replace=False)
            dream_memories = [
                self.thinker.colour_memory.memory_colours[idx] 
                for idx in memory_indices
            ]
            
            # Create dream through interference
            dream_colour = self.thinker.interference_engine.multi_colour_interference(dream_memories)
            
            # Apply dream intensity
            dream_colour = dream_colour * dream_intensity
            
            # Process through one resonance chamber
            dream_colour = self.thinker.resonance_chambers[cycle % len(self.thinker.resonance_chambers)](
                tf.expand_dims(dream_colour, 0)
            )[0]
            
            dream_colours.append(dream_colour)
            
            # Store dream as new memory
            self.thinker.colour_memory.store(dream_colour, strength=dream_intensity * 0.5)
            
            self.logger.debug(f"Dream cycle {cycle + 1}: amplitude={tf.reduce_mean(tf.abs(dream_colour)):.4f}")
        
        self.logger.info(f"Dream sequence complete - {len(dream_colours)} dream colours generated")
        
        return dream_colours
    
    def get_system_status(self) -> Dict:
        """
        Get comprehensive system status and metrics.
        """
        
        # Get cognitive metrics from ChromaThink core
        cognitive_metrics = self.thinker.get_cognitive_metrics()
        
        # Combine with system metrics
        status = {
            'system_type': 'IntegratedChromaThink',
            'spectrum_dimensions': self.spectrum_dims,
            'cognitive': cognitive_metrics,
            'conversation': {
                'total_conversations': self.metrics['total_conversations'],
                'average_thinking_time': self.metrics['average_thinking_time'],
                'context_length': len(self.conversation_context),
                'language_distribution': self.metrics['language_distribution']
            },
            'colour_complexity': {
                'recent_average': np.mean(self.metrics['colour_complexity_history'][-10:]) if self.metrics['colour_complexity_history'] else 0.0,
                'overall_average': np.mean(self.metrics['colour_complexity_history']) if self.metrics['colour_complexity_history'] else 0.0,
                'history_length': len(self.metrics['colour_complexity_history'])
            }
        }
        
        return status
    
    def reset_system(self, preserve_language_patterns: bool = True):
        """
        Reset the system while optionally preserving learned language patterns.
        """
        
        self.logger.info("Resetting IntegratedChromaThink system")
        
        # Reset ChromaThink cognitive state
        self.thinker.reset_cognitive_state()
        
        # Clear conversation context
        self.conversation_context.clear()
        
        # Reset metrics (but preserve language distribution if requested)
        language_dist = self.metrics['language_distribution'] if preserve_language_patterns else {}
        
        self.metrics = {
            'total_conversations': 0,
            'average_thinking_time': 0.0,
            'colour_complexity_history': [],
            'language_distribution': language_dist
        }
        
        self.logger.info("System reset complete")
    
    def save_system_state(self, save_path: str):
        """
        Save the current system state for later restoration.
        """
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save cognitive state
        cognitive_state_path = save_path / "cognitive_state.npy"
        np.save(cognitive_state_path, self.thinker.cognitive_state.numpy())
        
        # Save colour memory (non-zero memories only)
        memory_count = min(int(self.thinker.colour_memory.memory_index), self.thinker.colour_memory.capacity)
        if memory_count > 0:
            memory_colours = self.thinker.colour_memory.memory_colours[:memory_count].numpy()
            memory_strengths = self.thinker.colour_memory.memory_strengths[:memory_count].numpy()
            
            np.save(save_path / "memory_colours.npy", memory_colours)
            np.save(save_path / "memory_strengths.npy", memory_strengths)
        
        # Save metrics
        import json
        with open(save_path / "metrics.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = self.metrics.copy()
            serializable_metrics['colour_complexity_history'] = [
                float(x) for x in self.metrics['colour_complexity_history']
            ]
            json.dump(serializable_metrics, f, indent=2)
        
        self.logger.info(f"System state saved to {save_path}")
    
    def load_system_state(self, load_path: str):
        """
        Load previously saved system state.
        """
        
        load_path = Path(load_path)
        
        if not load_path.exists():
            self.logger.warning(f"Load path {load_path} does not exist")
            return
        
        try:
            # Load cognitive state
            cognitive_state_path = load_path / "cognitive_state.npy"
            if cognitive_state_path.exists():
                cognitive_state = np.load(cognitive_state_path)
                self.thinker.cognitive_state.assign(tf.constant(cognitive_state, dtype=tf.complex64))
            
            # Load colour memory
            memory_colours_path = load_path / "memory_colours.npy"
            memory_strengths_path = load_path / "memory_strengths.npy"
            
            if memory_colours_path.exists() and memory_strengths_path.exists():
                memory_colours = np.load(memory_colours_path)
                memory_strengths = np.load(memory_strengths_path)
                
                # Restore memories
                for i, (colour, strength) in enumerate(zip(memory_colours, memory_strengths)):
                    if i < self.thinker.colour_memory.capacity:
                        self.thinker.colour_memory.memory_colours[i].assign(tf.constant(colour, dtype=tf.complex64))
                        self.thinker.colour_memory.memory_strengths[i].assign(float(strength))
                
                self.thinker.colour_memory.memory_index.assign(len(memory_colours))
            
            # Load metrics
            metrics_path = load_path / "metrics.json"
            if metrics_path.exists():
                import json
                with open(metrics_path, 'r') as f:
                    self.metrics.update(json.load(f))
            
            self.logger.info(f"System state loaded from {load_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load system state: {e}")
    
    def _combine_concept_colours(self, concept_colours: Dict[str, List[tf.Tensor]]) -> tf.Tensor:
        """
        Combine multiple concept colours into a single input colour pattern.
        """
        
        all_colours = []
        weights = {'intent': 0.4, 'concepts': 0.4, 'context': 0.2}
        
        for concept_type, colour_list in concept_colours.items():
            weight = weights.get(concept_type, 0.1)
            
            if isinstance(colour_list, list) and colour_list:
                # Average multiple colours of same type
                stacked_colours = tf.stack(colour_list)
                avg_colour = tf.reduce_mean(stacked_colours, axis=0)
                weighted_colour = avg_colour * weight
                all_colours.append(weighted_colour)
        
        if not all_colours:
            # Return neutral pattern if no colours
            return tf.complex(
                tf.random.normal([self.spectrum_dims]),
                tf.random.normal([self.spectrum_dims])
            ) * 0.1
        
        # Combine all colours through simple addition
        combined = tf.reduce_sum(tf.stack(all_colours), axis=0)
        
        # Normalize to prevent explosion
        max_amplitude = tf.reduce_max(tf.abs(combined))
        if max_amplitude > 2.0:
            combined = combined / (max_amplitude / 1.0)
        
        return combined
    
    def _simple_colour_interference(self, colour1: tf.Tensor, colour2: tf.Tensor) -> tf.Tensor:
        """
        Simple colour interference for combining patterns.
        """
        
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


# Factory function for easy instantiation
def create_integrated_chromathink(
    apertus_path: str = "models/apertus",
    spectrum_dims: int = 512,
    gpu_acceleration: bool = True,
    use_mock_apertus: bool = False
) -> IntegratedChromaThink:
    """
    Factory function to create IntegratedChromaThink system with recommended settings.
    
    Args:
        apertus_path: Path to Apertus model files
        spectrum_dims: Number of colour spectrum dimensions
        gpu_acceleration: Enable GPU acceleration for colour mathematics
        use_mock_apertus: Use mock Apertus translator (for testing)
    
    Returns:
        Configured IntegratedChromaThink system
    """
    
    return IntegratedChromaThink(
        apertus_path=apertus_path,
        spectrum_dims=spectrum_dims,
        num_resonance_chambers=5,
        use_mock_apertus=use_mock_apertus,
        gpu_acceleration=gpu_acceleration
    )