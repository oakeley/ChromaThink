"""
Big Colour Model Integration with ChromaThink Core

This module integrates the comprehensive colour model with ChromaThink's
existing architecture, creating a bridge between human language and
pure colour-based cognition through 131k token analysis.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

from ..bootstrap.apertus_integration import ApertusWeightTranslator
from .chromathink_core import ChromaThinkCore
from .apertus_translator import ApertusTranslator


class BigColourChromatThink:
    """
    Enhanced ChromaThink system with Big Colour Model integration.
    
    This system:
    1. Loads all 131k tokens from Apertus safetensors
    2. Creates comprehensive concept-to-waveform mappings
    3. Maintains pure colour-based thinking while bridging to human language
    4. NO pre-programmed responses - all from colour dynamics
    """
    
    def __init__(self, 
                 apertus_path: str = "models/apertus",
                 spectrum_dims: int = 512,
                 use_mock: bool = False,
                 force_rebuild: bool = False,
                 gpu_acceleration: bool = True):
        
        self.spectrum_dims = spectrum_dims
        self.logger = logging.getLogger("BigColourChromatThink")
        
        # Initialize components
        self.weight_translator = None
        self.big_colour_model = None
        self.chromathink_core = None
        self.language_bridge = None
        self.gpu_accelerator = None
        
        # Colour memory system
        self.colour_memories = []
        self.conversation_context = []
        
        self.logger.info("Initializing Big Colour ChromaThink System...")
        self._initialize_system(apertus_path, use_mock, force_rebuild, gpu_acceleration)
    
    def _initialize_system(self, apertus_path: str, use_mock: bool, force_rebuild: bool, gpu_acceleration: bool):
        """Initialize all system components."""
        
        # 1. Initialize GPU acceleration if requested
        if gpu_acceleration:
            self.logger.info("Initializing GPU acceleration for ray-traced colour mathematics...")
            try:
                from .gpu_colour_ops import create_gpu_accelerator
                # Disable mixed precision to avoid dtype conflicts in ResonanceChamber
                self.gpu_accelerator = create_gpu_accelerator(
                    spectrum_dims=self.spectrum_dims, 
                    mixed_precision=False
                )
                self.logger.info("GPU acceleration initialized successfully")
            except Exception as e:
                self.logger.warning(f"GPU acceleration failed: {e}, falling back to CPU")
                self.gpu_accelerator = None
        
        # 2. Build Big Colour Model from safetensors
        self.logger.info("Building Big Colour Model from Apertus safetensors...")
        self.weight_translator = ApertusWeightTranslator(
            apertus_path=apertus_path,
            spectrum_dims=self.spectrum_dims,
            use_mock=use_mock,
            extract_full_vocab=True,  # All 131k tokens
            max_tokens=131072
        )
        
        # Set force_rebuild flag if requested
        if force_rebuild:
            self.weight_translator.force_rebuild = True
        
        self.big_colour_model = self.weight_translator.build_big_colour_model()
        
        # 3. Initialize pure colour thinking core
        self.logger.info("Initializing pure colour thinking core...")
        self.chromathink_core = ChromaThinkCore(spectrum_dims=self.spectrum_dims)
        
        # 4. Create language bridge (translation only, no thinking)
        self.logger.info("Creating language bridge...")
        self.language_bridge = LanguageBridge(
            big_colour_model=self.big_colour_model,
            spectrum_dims=self.spectrum_dims,
            apertus_path=apertus_path
        )
        
        # Log initialization success
        stats = self.big_colour_model.get_statistics()
        self.logger.info(f"System initialized successfully:")
        self.logger.info(f"  Vocabulary: {stats['vocab_size']:,} tokens")
        self.logger.info(f"  Attention patterns: {stats['attention_patterns']}")
        self.logger.info(f"  Spectrum dimensions: {stats['spectrum_dims']}")
    
    def think_about(self, human_input: str, intensity: float = 1.0) -> str:
        """
        Process human input through pure colour thinking.
        
        Flow:
        1. Human text → Big Colour Model → Colour waveform
        2. Colour waveform → ChromaThink Core → Pure colour thinking
        3. Colour response → Big Colour Model → Human language
        """
        
        self.logger.info(f"Processing: '{human_input[:50]}...' (intensity: {intensity})")
        
        # Step 1: Convert human language to colour waveform
        input_colour = self.language_bridge.text_to_colour(human_input)
        self.logger.debug(f"Input converted to colour: shape={input_colour.shape}, energy={np.sum(np.abs(input_colour)**2):.3f}")
        
        # Step 2: Apply thinking intensity
        thought_colour = input_colour * intensity
        
        # Step 3: Pure colour thinking (NO language involved)
        response_colour = self.chromathink_core.think_in_colour(
            thought_colour, 
            memory_context=self.colour_memories[-3:] if self.colour_memories else []
        )
        
        # Step 4: Store colour memory
        self.colour_memories.append(response_colour)
        
        # Step 5: Convert colour response to human language
        human_response = self.language_bridge.colour_to_text(response_colour)
        
        # Step 6: Add conversation context
        self.conversation_context.append((human_input, human_response))
        
        # Show colour analysis
        self._log_colour_analysis(input_colour, response_colour)
        
        return human_response
    
    def learn_from_example(self, input_text: str, teaching_text: str, learning_intensity: float = 1.5):
        """
        Learn from a human explanation or correction.
        
        This strengthens the colour associations between concepts mentioned in the teaching.
        The learning happens through colour interference reinforcement.
        """
        
        self.logger.info(f"Learning from example: '{teaching_text[:50]}...'")
        
        # Step 1: Convert both input and teaching to colour patterns
        input_colour = self.language_bridge.text_to_colour(input_text)
        teaching_colour = self.language_bridge.text_to_colour(teaching_text)
        
        # Step 2: Create constructive interference between question and answer
        # This strengthens the connection in colour space
        learning_pattern = self.chromathink_core.colour_interference(
            input_colour, teaching_colour
        )
        
        # Step 3: Apply learning intensity to strengthen the pattern
        reinforced_pattern = learning_pattern * learning_intensity
        
        # Step 4: Store as a strong memory that will influence future thinking
        self.colour_memories.append(reinforced_pattern)
        
        # Step 5: Update conversation context with learning
        self.conversation_context.append((input_text, f"LEARNED: {teaching_text}"))
        
        # Step 6: Log learning progress
        self.logger.info(f"Learning pattern stored with energy: {np.sum(np.abs(reinforced_pattern)**2):.3f}")
        
        return f"Learning integrated into colour memory. Pattern strengthened with {len(self.colour_memories)} total memories."
    
    def _log_colour_analysis(self, input_colour: np.ndarray, response_colour: np.ndarray):
        """Log detailed colour analysis for transparency."""
        
        # Input analysis
        input_amp = np.abs(input_colour)
        input_dominant = np.argmax(input_amp)
        input_energy = np.sum(input_amp**2)
        
        # Response analysis  
        response_amp = np.abs(response_colour)
        response_dominant = np.argmax(response_amp)
        response_energy = np.sum(response_amp**2)
        
        self.logger.info("Colour Analysis:")
        self.logger.info(f"  Input: dominant_freq={input_dominant}, energy={input_energy:.3f}")
        self.logger.info(f"  Response: dominant_freq={response_dominant}, energy={response_energy:.3f}")
        
        # Show interference pattern
        interference_strength = np.abs(np.vdot(input_colour, response_colour))
        self.logger.info(f"  Interference strength: {interference_strength:.3f}")
    
    def get_colour_spectrum(self, text: str) -> Dict:
        """Get detailed colour spectrum analysis for visualization."""
        
        colour = self.language_bridge.text_to_colour(text)
        amplitude = np.abs(colour)
        phase = np.angle(colour)
        
        # Find top frequencies
        top_freqs = np.argsort(amplitude)[-10:][::-1]
        
        return {
            'amplitudes': amplitude.tolist(),
            'phases': phase.tolist(),
            'dominant_frequencies': top_freqs.tolist(),
            'total_energy': float(np.sum(amplitude**2)),
            'spectral_centroid': float(np.average(range(len(amplitude)), weights=amplitude)),
            'colour_descriptors': self.big_colour_model.decode_waveform(colour, num_concepts=5)
        }
    
    def dream_consolidation(self, num_memories: int = 10) -> str:
        """
        Perform memory consolidation through colour interference.
        Combines recent colour memories to strengthen patterns.
        """
        
        if len(self.colour_memories) < 2:
            return "Not enough colour memories for consolidation."
        
        self.logger.info(f"Dream consolidation: processing {min(num_memories, len(self.colour_memories))} memories")
        
        # Take recent memories
        recent_memories = self.colour_memories[-num_memories:]
        
        # Start with first memory
        consolidated_colour = recent_memories[0].copy()
        
        # Interfere with other memories
        for memory in recent_memories[1:]:
            consolidated_colour = self.chromathink_core.colour_interference(consolidated_colour, memory)
        
        # Apply dream-like transformations
        dream_colour = self.chromathink_core.apply_resonance(consolidated_colour, depth=3)
        
        # Store consolidated memory
        self.colour_memories.append(dream_colour)
        
        # Convert to human language
        dream_description = self.language_bridge.colour_to_text(dream_colour)
        
        return f"Dream consolidation complete: {dream_description}"
    
    def get_system_statistics(self) -> Dict:
        """Get comprehensive system statistics."""
        
        model_stats = self.big_colour_model.get_statistics()
        
        return {
            'big_colour_model': model_stats,
            'colour_memories': len(self.colour_memories),
            'conversations': len(self.conversation_context),
            'spectrum_dimensions': self.spectrum_dims,
            'thinking_architecture': 'Pure Colour Cognition',
            'language_bridge': 'Big Colour Model (131k tokens)',
            'memory_energy': float(np.sum([np.sum(np.abs(mem)**2) for mem in self.colour_memories])) if self.colour_memories else 0.0
        }


class LanguageBridge:
    """
    Bridge between human language and colour space using Big Colour Model.
    ONLY handles translation - NO thinking or response generation.
    """
    
    def __init__(self, big_colour_model, spectrum_dims=512, apertus_path: str = "models/apertus"):
        self.big_colour_model = big_colour_model
        self.spectrum_dims = spectrum_dims
        self.logger = logging.getLogger("LanguageBridge")
        
        # Initialize Apertus translator for synthesis (force CPU to leave GPU for ChromaThink)
        from .apertus_translator import ApertusTranslator
        self.apertus_translator = ApertusTranslator(apertus_path, device='cpu')
    
    def text_to_colour(self, text: str) -> np.ndarray:
        """Convert human text to colour waveform using Big Colour Model."""
        
        # Use the comprehensive concept encoder
        colour_waveform = self.big_colour_model.encode_concept(text)
        
        # Ensure correct format for ChromaThink
        if len(colour_waveform) != self.spectrum_dims:
            # Interpolate to correct size
            colour_waveform = np.interp(
                np.linspace(0, len(colour_waveform)-1, self.spectrum_dims),
                np.arange(len(colour_waveform)),
                colour_waveform
            )
        
        return colour_waveform.astype(np.complex64)
    
    def colour_to_text(self, colour_waveform: np.ndarray) -> str:
        """Convert colour waveform to human language using Big Colour Model + Apertus.
        
        Flow:
        1. Decode colour waveform to resonant concepts from 131k vocabulary
        2. Use those concepts as semantic guidance for Apertus text generation
        3. Generate coherent response using Apertus's language capabilities
        4. Result: Intelligent text that emerged from colour mathematics
        """
        
        # Step 1: Decode waveform to the most resonant concepts
        concept_data = self.big_colour_model.decode_waveform(colour_waveform, num_concepts=15)
        
        if not concept_data:
            # Fallback: extract concepts from raw frequency analysis
            dominant_frequencies = np.argsort(np.abs(colour_waveform))[-10:][::-1]
            # Map frequencies to vocabulary indices 
            concept_tokens = [f"freq_{freq}" for freq in dominant_frequencies[:5]]
            concept_prompt = "Concepts from colour frequencies: " + " ".join(concept_tokens)
        else:
            # Extract the strongest resonant concepts from colour interference
            concepts = [item[0] for item in concept_data]
            amplitudes = [item[1] for item in concept_data]
            
            # Build semantic prompt based on colour strength
            primary_concepts = []
            for i, concept in enumerate(concepts[:8]):  # Use top 8 concepts
                amplitude = amplitudes[i]
                if amplitude > 0.05:  # Resonance threshold
                    primary_concepts.append(concept)
            
            # Create semantic guidance prompt for Apertus
            concept_prompt = "Key concepts from colour analysis: " + ", ".join(primary_concepts)
        
        # Step 2: Use Apertus for actual text synthesis from concepts
        # This ensures NO template responses - all text emerges from Apertus
        
        if not concept_data:
            # Fallback: use frequency analysis for concept extraction
            return self.apertus_translator.synthesise_from_frequency_pattern(colour_waveform)
        
        # Extract the strongest concepts for Apertus synthesis
        concepts = [item[0] for item in concept_data[:5]]  # Top 5 concepts
        
        # Use Apertus to synthesize response from these concepts
        response = self.apertus_translator.synthesise_from_concepts(
            concepts=concepts,
            colour_guidance=colour_waveform  # Provide colour pattern as guidance
        )
        
        return response


def create_big_colour_chromathink(apertus_path: str = "models/apertus", 
                                  use_mock: bool = False,
                                  force_rebuild: bool = False) -> BigColourChromatThink:
    """
    Factory function to create a fully initialized Big Colour ChromaThink system.
    """
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("BigColourFactory")
    
    logger.info("Creating Big Colour ChromaThink system...")
    
    system = BigColourChromatThink(
        apertus_path=apertus_path,
        spectrum_dims=512,
        use_mock=use_mock,
        force_rebuild=force_rebuild
    )
    
    logger.info("Big Colour ChromaThink system created successfully!")
    return system