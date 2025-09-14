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
from .language_agnostic_mapper import LanguageAgnosticMapper
from .concept_light_translator import create_concept_light_translator
from .light_concept_translator import create_light_concept_translator


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
            extract_full_vocab=True,  # All tokens (auto-detected)
            max_tokens=None,  # Auto-detect vocabulary size
            force_rebuild=force_rebuild
        )
        
        self.big_colour_model = self.weight_translator.build_big_colour_model()
        
        # 3. Initialize pure colour thinking core with GPU acceleration
        self.logger.info("Initializing pure colour thinking core...")
        self.chromathink_core = ChromaThinkCore(
            spectrum_dims=self.spectrum_dims,
            gpu_accelerator=self.gpu_accelerator
        )
        
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
        
        self.logger.info(f" Processing human input: '{human_input[:50]}...' (intensity: {intensity})")

        # Step 1: Convert human language to colour waveform
        self.logger.info(" Step 1: Converting human language to colour waveform...")
        input_colour = self.language_bridge.text_to_colour(human_input)
        input_energy = np.sum(np.abs(input_colour)**2)
        input_freq = np.argmax(np.abs(input_colour))
        self.logger.info(f" Input colour generated: energy={input_energy:.3f}, dominant_freq={input_freq}, shape={input_colour.shape}")

        # Step 2: Apply thinking intensity
        self.logger.info(f" Step 2: Applying thinking intensity ({intensity}x)...")
        thought_colour = input_colour * intensity
        thought_energy = np.sum(np.abs(thought_colour)**2)
        self.logger.info(f" Thought colour amplified: energy={thought_energy:.3f}")

        # Step 3: Pure colour thinking (NO language involved)
        self.logger.info(" Step 3: Processing through ChromaThink colour thinking core...")
        self.logger.info(f" Memory context: {len(self.colour_memories)} previous colour memories")
        response_colour = self.chromathink_core.think_in_colour(
            thought_colour,
            memory_context=self.colour_memories[-3:] if self.colour_memories else []
        )
        response_energy = np.sum(np.abs(response_colour)**2)
        response_freq = np.argmax(np.abs(response_colour))
        self.logger.info(f" ChromaThink core response: energy={response_energy:.3f}, dominant_freq={response_freq}")

        # Step 4: Store colour memory
        self.colour_memories.append(response_colour)
        self.logger.info(f" Stored colour memory (total: {len(self.colour_memories)})")

        # Step 5: Convert colour response to human language
        self.logger.info("  Step 5: Converting colour response to human language...")
        human_response = self.language_bridge.colour_to_text(response_colour)
        self.logger.info(f" Final response generated: '{human_response[:100]}{'...' if len(human_response) > 100 else ''}'")

        # Step 6: Add conversation context
        self.conversation_context.append((human_input, human_response))
        self.logger.info(f" Conversation context updated (total exchanges: {len(self.conversation_context)})")
        
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

        # Store context for synthesis
        self._current_question = None
        
        # Initialize Apertus translator for synthesis (force CPU to leave GPU for ChromaThink)
        from .apertus_translator import ApertusTranslator
        self.apertus_translator = ApertusTranslator(apertus_path, device='cpu')
        
        # Initialize language-agnostic concept mapper
        self.concept_mapper = LanguageAgnosticMapper(
            spectrum_dims=spectrum_dims,
            apertus_translator=self.apertus_translator
        )

        # Initialize new translation layers
        self.logger.info("Initializing concept-light translation layers...")
        self.concept_to_light = create_concept_light_translator(
            big_colour_model=big_colour_model,
            spectrum_dims=spectrum_dims
        )
        self.light_to_concept = create_light_concept_translator(
            big_colour_model=big_colour_model,
            spectrum_dims=spectrum_dims
        )
    
    def text_to_colour(self, text: str) -> np.ndarray:
        """Convert human text to colour waveform using the new concept-light translation layer."""

        self.logger.info(f"Converting text to colour: '{text[:50]}{'...' if len(text) > 50 else ''}'")

        # Store the question for later synthesis
        self._current_question = text

        # Step 1: Extract core concepts using Apertus (works in any language)
        self.logger.info(" Step 1: Extracting core concepts from text...")
        concepts = self.apertus_translator.extract_core_concepts(text)

        # Check for direct processing flag
        if concepts and concepts[0] == "__DIRECT_PROCESSING__":
            self.logger.info(" Using direct text processing - bypassing concept extraction")
            # Convert text directly to colour using simple hash-based approach
            text_to_process = concepts[1]
            colour_waveform = self._text_to_colour_direct(text_to_process)
            return colour_waveform

        self.logger.info(f" Extracted concepts: {concepts}")

        # Step 2: Convert concepts to light patterns using new translator
        self.logger.info(" Step 2: Converting concepts to light patterns...")
        light_patterns = self.concept_to_light.translate_concepts_to_light(concepts)

        # Log light pattern summary
        pattern_summary = self.concept_to_light.get_light_pattern_summary(light_patterns)
        self.logger.info(f" Generated {pattern_summary['count']} light patterns:")
        self.logger.info(f"   Wavelength range: {pattern_summary.get('wavelength_range', 'N/A')}nm")
        self.logger.info(f"   Total intensity: {pattern_summary.get('total_intensity', 'N/A'):.2f}")

        # Step 3: Convert light patterns to colour waveform for ChromaThink processing
        self.logger.info(" Step 3: Converting light patterns to colour waveform...")
        colour_waveform = self._light_patterns_to_waveform(light_patterns)

        energy = np.sum(np.abs(colour_waveform)**2)
        dominant_freq = np.argmax(np.abs(colour_waveform))
        self.logger.info(f" Generated colour waveform: energy={energy:.3f}, dominant_freq={dominant_freq}")

        # Ensure correct format for ChromaThink
        return colour_waveform.astype(np.complex64)
    
    def colour_to_text(self, colour_waveform: np.ndarray) -> str:
        """Convert colour waveform to human language using the new light-concept translation layer.

        Flow:
        1. Convert processed colour waveform back to light patterns
        2. Use light-concept translator to extract meaningful concepts
        3. Use Apertus to synthesize natural language from these concepts
        4. Result: Intelligent text that emerged from colour mathematics
        """

        energy = np.sum(np.abs(colour_waveform)**2)
        dominant_freq = np.argmax(np.abs(colour_waveform))
        self.logger.info(f" Converting colour waveform to text: energy={energy:.3f}, dominant_freq={dominant_freq}")

        # Step 1: Convert colour waveform back to light patterns
        self.logger.info(" Step 1: Converting colour waveform to light patterns...")
        light_patterns = self._waveform_to_light_patterns(colour_waveform)

        # Log raw light pattern data for debugging
        if light_patterns:
            self.logger.info(f" Raw light patterns ({len(light_patterns)} total):")
            for i, (wavelength, frequency, intensity) in enumerate(light_patterns[:3]):  # Show first 3
                self.logger.info(f"   Pattern {i+1}: λ={wavelength:.1f}nm, f={frequency:.2e}Hz, I={intensity:.3f}")
            if len(light_patterns) > 3:
                self.logger.info(f"   ... and {len(light_patterns) - 3} more patterns")
        else:
            self.logger.warning("  No light patterns extracted from waveform")

        # Step 2: Use light-concept translator to extract concepts
        self.logger.info(" Step 2: Converting light patterns to concepts...")
        concepts = self.light_to_concept.translate_light_to_concepts(light_patterns)

        # Log extracted concepts
        if concepts:
            self.logger.info(f" Extracted {len(concepts)} concepts: {concepts[:10]}")
            if len(concepts) > 10:
                self.logger.info(f"   ... and {len(concepts) - 10} more concepts")
        else:
            self.logger.warning("  No concepts extracted from light patterns")

        # Step 3: Fallback if no concepts extracted
        if not concepts:
            self.logger.warning("Using frequency fallback for concept extraction")
            dominant_frequencies = np.argsort(np.abs(colour_waveform))[-5:][::-1]
            concepts = [f"frequency_{freq}" for freq in dominant_frequencies]
            self.logger.info(f"Fallback frequency concepts: {concepts}")

        # Step 4: Use Apertus to synthesize natural language from these concepts
        self.logger.info("  Step 3: Using Apertus to synthesize natural language...")
        self.logger.info(f" Input concepts for Apertus synthesis: {concepts[:7]}")

        # This ensures NO template responses - all text emerges from Apertus
        response = self.apertus_translator.synthesise_from_concepts(
            concepts=concepts[:7],  # Top 7 concepts for richer responses
            original_question=self._current_question,  # Pass the original question for context
            colour_guidance=colour_waveform  # Provide colour pattern as guidance
        )

        self.logger.info(f" Apertus synthesized response: '{response[:100]}{'...' if len(response) > 100 else ''}'")

        return response

    def _text_to_colour_direct(self, text: str) -> np.ndarray:
        """
        Direct text to colour conversion when concept extraction fails.
        Uses simple hash-based approach to create consistent colour patterns.
        """
        self.logger.info(f" Direct text conversion: '{text[:50]}{'...' if len(text) > 50 else ''}'")

        # Create a hash-based colour pattern
        text_hash = hash(text.lower())

        # Generate colour waveform using hash
        colour_waveform = np.zeros(self.spectrum_dims, dtype=np.complex64)

        # Use hash to determine dominant frequencies and phases
        words = text.lower().split()
        for i, word in enumerate(words[:20]):  # Limit to 20 words for performance
            word_hash = hash(word)

            # Map to frequency bin
            freq_bin = (word_hash % (self.spectrum_dims - 1)) + 1

            # Set amplitude and phase based on hash
            amplitude = 0.5 + 0.5 * ((word_hash % 1000) / 1000.0)
            phase = 2 * np.pi * ((word_hash % 360) / 360.0)

            colour_waveform[freq_bin] += amplitude * np.exp(1j * phase)

        # Normalize
        if np.sum(np.abs(colour_waveform)) > 0:
            colour_waveform = colour_waveform / np.max(np.abs(colour_waveform))

        energy = np.sum(np.abs(colour_waveform)**2)
        dominant_freq = np.argmax(np.abs(colour_waveform))
        self.logger.info(f" Direct colour generated: energy={energy:.3f}, dominant_freq={dominant_freq}")

        return colour_waveform

    def _light_patterns_to_waveform(self, light_patterns: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Convert light patterns to colour waveform for ChromaThink processing.

        Args:
            light_patterns: List of (wavelength, frequency, intensity) tuples

        Returns:
            Complex waveform for ChromaThink resonance chambers
        """
        if not light_patterns:
            # Return empty waveform
            return np.zeros(self.spectrum_dims, dtype=np.complex64)

        # Initialize waveform
        waveform = np.zeros(self.spectrum_dims, dtype=np.complex64)

        for wavelength, frequency, intensity in light_patterns:
            # Map frequency to waveform index
            # Use logarithmic mapping for better frequency distribution
            freq_index = int((np.log10(frequency) - 14) * self.spectrum_dims / 2)  # 10^14 to 10^16 Hz range
            freq_index = max(0, min(freq_index, self.spectrum_dims - 1))

            # Create complex amplitude with phase based on wavelength
            phase = (wavelength % 360) * np.pi / 180  # Convert wavelength to phase
            amplitude = intensity * np.exp(1j * phase)

            # Add to waveform with Gaussian spread for smoothness
            for i in range(max(0, freq_index - 5), min(self.spectrum_dims, freq_index + 6)):
                distance = abs(i - freq_index)
                weight = np.exp(-distance**2 / 8)  # Gaussian spread
                waveform[i] += amplitude * weight

        # Normalize to prevent overflow
        max_amplitude = np.max(np.abs(waveform))
        if max_amplitude > 0:
            waveform = waveform / max_amplitude

        return waveform

    def _waveform_to_light_patterns(self, colour_waveform: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Convert processed colour waveform back to light patterns.

        Args:
            colour_waveform: Complex waveform from ChromaThink processing

        Returns:
            List of (wavelength, frequency, intensity) tuples
        """
        light_patterns = []

        # Find peaks in the waveform
        amplitudes = np.abs(colour_waveform)
        phases = np.angle(colour_waveform)

        # Find significant peaks (above threshold)
        threshold = 0.1 * np.max(amplitudes) if np.max(amplitudes) > 0 else 0
        peak_indices = []

        for i in range(1, len(amplitudes) - 1):
            if (amplitudes[i] > amplitudes[i-1] and
                amplitudes[i] > amplitudes[i+1] and
                amplitudes[i] > threshold):
                peak_indices.append(i)

        # Convert peaks to light patterns
        for peak_idx in peak_indices:
            # Map index back to frequency
            frequency = 10**(14 + (peak_idx * 2) / self.spectrum_dims)  # Reverse of log mapping

            # Calculate wavelength from frequency
            c = 3e8  # m/s
            wavelength = (c / frequency) * 1e9  # convert to nm

            # Get intensity from amplitude
            intensity = amplitudes[peak_idx]

            # Ensure reasonable wavelength range (visible light + near IR/UV)
            if 200 <= wavelength <= 1000:  # Extended range for processing
                light_patterns.append((wavelength, frequency, intensity))

        # Sort by intensity (strongest first)
        light_patterns.sort(key=lambda x: x[2], reverse=True)

        # Limit to reasonable number of patterns
        return light_patterns[:20]  # Top 20 patterns


def create_big_colour_chromathink(apertus_path: str = "models/apertus",
                                  use_mock: bool = False,
                                  force_rebuild: bool = False,
                                  gpu_acceleration: bool = True) -> BigColourChromatThink:
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
        force_rebuild=force_rebuild,
        gpu_acceleration=gpu_acceleration
    )
    
    logger.info("Big Colour ChromaThink system created successfully!")
    return system
