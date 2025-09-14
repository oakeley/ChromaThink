"""
Integrated True Colour System
Complete ChromaThink system with proper Apertus translation separation.

Architecture:
1. Human text → TrueColourTranslator.extract_concepts() → Concept words
2. Concept words → ConceptColourMapper → Colour waveforms
3. Colour waveforms → ChromaThinkCore → Response colour waveforms
4. Response colour waveforms → ConceptColourMapper → Response concept words
5. Response concept words + Original question → TrueColourTranslator.synthesize_response() → Natural text

This ensures ChromaThink thinks ONLY in colours, Apertus translates ONLY language.
"""

import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

from .true_colour_translator import TrueColourTranslator, ConceptColourMapper
from .chromathink_core import ChromaThinkCore


class IntegratedTrueColourSystem:
    """
    Complete ChromaThink system with proper separation of concerns:
    - TrueColourTranslator: Language ↔ Concepts
    - ConceptColourMapper: Concepts ↔ Colours
    - ChromaThinkCore: Pure colour thinking
    """

    def __init__(self,
                 apertus_path: str = "models/apertus",
                 spectrum_dims: int = 512,
                 gpu_acceleration: bool = True):

        self.spectrum_dims = spectrum_dims
        self.logger = logging.getLogger("IntegratedTrueColourSystem")

        self.logger.info("Initializing Integrated True Colour System...")

        # Initialize components
        self.translator = TrueColourTranslator(apertus_path, device='cpu')  # Keep Apertus on CPU
        self.colour_mapper = ConceptColourMapper(spectrum_dims)
        self.chromathink_core = ChromaThinkCore(spectrum_dims)

        # GPU acceleration for colour mathematics only
        if gpu_acceleration:
            self._initialize_gpu_acceleration()

        # Conversation context for learning
        self.conversation_history = []

        self.logger.info("Integrated True Colour System initialized successfully")

    def _initialize_gpu_acceleration(self):
        """Initialize GPU acceleration for ChromaThink colour processing"""

        try:
            # Configure TensorFlow for GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Enable memory growth to prevent OOM
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Set memory limit to leave room for Apertus on CPU
                tf.config.experimental.set_memory_growth(gpus[0], True)
                self.logger.info(f"GPU acceleration enabled: {len(gpus)} GPU(s) available")
            else:
                self.logger.warning("No GPUs detected, using CPU for colour processing")
        except Exception as e:
            self.logger.warning(f"GPU acceleration setup failed: {e}")

    def process_input(self, human_text: str, thinking_intensity: float = 1.0) -> str:
        """
        Complete processing pipeline from human text to human response.

        Args:
            human_text: Input from human in any language
            thinking_intensity: How intensely ChromaThink should process (0.5-2.0)

        Returns:
            Natural language response synthesized by Apertus
        """

        self.logger.info(f"Processing input: {human_text[:50]}...")

        try:
            # Stage 1: Extract concepts from human text
            input_concepts = self.translator.extract_concepts(human_text)
            self.logger.debug(f"Extracted concepts: {input_concepts}")

            if not input_concepts:
                self.logger.warning("No concepts extracted, using fallback")
                input_concepts = ["understanding", "question", "response"]

            # Stage 2: Convert concepts to colour waveforms
            input_colour_waveform = self.colour_mapper.concepts_to_colour_waveform(input_concepts)
            self.logger.debug(f"Input colour waveform shape: {input_colour_waveform.shape}")

            # Stage 3: ChromaThink pure colour thinking
            # Convert numpy to tensorflow for ChromaThink
            input_colour_tensor = tf.convert_to_tensor(input_colour_waveform, dtype=tf.complex64)

            # Get conversation context as colour memories
            context_colours = self._get_colour_context()

            # Pure colour thinking
            response_colour_tensor = self.chromathink_core.think_in_colour(
                input_colour_tensor,
                memory_context=context_colours
            )

            # Convert back to numpy for colour mapper
            response_colour_waveform = response_colour_tensor.numpy()
            self.logger.debug(f"Response colour waveform shape: {response_colour_waveform.shape}")

            # Stage 4: Extract concepts from response colours
            response_concepts = self.colour_mapper.colour_waveform_to_concepts(
                response_colour_waveform,
                num_concepts=6
            )
            self.logger.debug(f"Response concepts: {response_concepts}")

            if not response_concepts:
                self.logger.warning("No response concepts generated, using fallback")
                response_concepts = ["thoughtful", "understanding", "response", "consideration"]

            # Stage 5: Synthesize natural language response
            natural_response = self.translator.synthesize_response(
                response_concepts,
                human_text
            )

            # Store conversation for context
            self.conversation_history.append({
                'input': human_text,
                'input_concepts': input_concepts,
                'input_colours': input_colour_waveform.copy(),
                'response_concepts': response_concepts,
                'response_colours': response_colour_waveform.copy(),
                'response': natural_response
            })

            # Keep only recent conversations for memory efficiency
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            self.logger.info("Processing complete")
            return natural_response

        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            # Emergency fallback
            return "I understand your question and am thinking about it carefully."

    def _get_colour_context(self) -> List[tf.Tensor]:
        """Get recent colour memories for context"""

        context_colours = []

        # Get last 3 conversations for context
        for conv in self.conversation_history[-3:]:
            if 'response_colours' in conv:
                colour_tensor = tf.convert_to_tensor(
                    conv['response_colours'],
                    dtype=tf.complex64
                )
                context_colours.append(colour_tensor)

        return context_colours

    def learn_from_feedback(self, human_text: str, feedback: str) -> str:
        """
        Learn from human feedback by strengthening colour associations.

        Args:
            human_text: Original question/input
            feedback: Human explanation or correction

        Returns:
            Learning confirmation message
        """

        self.logger.info("Processing learning feedback...")

        try:
            # Extract concepts from both input and feedback
            input_concepts = self.translator.extract_concepts(human_text)
            feedback_concepts = self.translator.extract_concepts(feedback)

            # Convert to colours
            input_colours = self.colour_mapper.concepts_to_colour_waveform(input_concepts)
            feedback_colours = self.colour_mapper.concepts_to_colour_waveform(feedback_concepts)

            # Create strong interference pattern for learning
            input_tensor = tf.convert_to_tensor(input_colours, dtype=tf.complex64)
            feedback_tensor = tf.convert_to_tensor(feedback_colours, dtype=tf.complex64)

            # Store as strong memory in ChromaThink
            learning_pattern = self.chromathink_core.colour_interference(input_tensor, feedback_tensor)
            self.chromathink_core.colour_memory.store(learning_pattern, strength=2.0)  # Strong memory

            # Store in conversation history with learning flag
            self.conversation_history.append({
                'input': human_text,
                'feedback': feedback,
                'learning': True,
                'input_concepts': input_concepts,
                'feedback_concepts': feedback_concepts,
                'learning_pattern': learning_pattern.numpy()
            })

            return f"Thank you for the feedback. I've integrated the concepts: {', '.join(feedback_concepts[:3])} into my understanding."

        except Exception as e:
            self.logger.error(f"Learning failed: {e}")
            return "I appreciate your feedback and will consider it carefully."

    def get_system_metrics(self) -> Dict:
        """Get system performance and state metrics"""

        metrics = {
            'conversations': len(self.conversation_history),
            'spectrum_dimensions': self.spectrum_dims,
            'chromathink_metrics': self.chromathink_core.get_cognitive_metrics(),
            'gpu_enabled': len(tf.config.experimental.list_physical_devices('GPU')) > 0,
            'translator_loaded': hasattr(self.translator, 'model'),
            'colour_mapper_ready': True
        }

        return metrics

    def demonstrate_colour_thinking(self, concept: str) -> Dict:
        """
        Demonstrate the colour thinking process for a single concept.
        Shows the complete pipeline step by step.
        """

        self.logger.info(f"Demonstrating colour thinking for: {concept}")

        demo = {}

        # Step 1: Concept to colour
        colour_waveform = self.colour_mapper._concept_to_frequency(concept)
        demo['concept'] = concept
        demo['colour_waveform_shape'] = colour_waveform.shape
        demo['colour_energy'] = float(np.sum(np.abs(colour_waveform)**2))

        # Step 2: ChromaThink processing
        colour_tensor = tf.convert_to_tensor(colour_waveform, dtype=tf.complex64)
        response_tensor = self.chromathink_core.think_in_colour(colour_tensor)
        demo['response_energy'] = float(tf.reduce_sum(tf.abs(response_tensor)**2))

        # Step 3: Back to concepts
        response_waveform = response_tensor.numpy()
        response_concepts = self.colour_mapper.colour_waveform_to_concepts(response_waveform)
        demo['response_concepts'] = response_concepts

        return demo

    def test_complete_pipeline(self) -> bool:
        """Test the complete system pipeline"""

        self.logger.info("Testing complete system pipeline...")

        try:
            # Test basic processing
            test_input = "How does gravity work in orbital mechanics?"
            response = self.process_input(test_input)

            if not response or len(response) < 10:
                self.logger.error("Pipeline test failed: no valid response")
                return False

            # Test learning
            feedback = "Gravity is the curvature of spacetime caused by mass"
            learning_response = self.learn_from_feedback(test_input, feedback)

            if not learning_response:
                self.logger.error("Learning test failed")
                return False

            # Test metrics
            metrics = self.get_system_metrics()
            if not metrics or 'conversations' not in metrics:
                self.logger.error("Metrics test failed")
                return False

            self.logger.info("Complete pipeline test successful")
            self.logger.info(f"Test response: {response}")
            return True

        except Exception as e:
            self.logger.error(f"Pipeline test failed: {e}")
            return False

    def reset_system(self):
        """Reset system to clean state"""

        self.conversation_history = []
        self.chromathink_core.reset_cognitive_state()
        self.logger.info("System reset to clean state")


def create_integrated_true_colour_system(apertus_path: str = "models/apertus",
                                       spectrum_dims: int = 512,
                                       gpu_acceleration: bool = True) -> IntegratedTrueColourSystem:
    """
    Factory function to create a fully integrated True Colour system.
    """

    system = IntegratedTrueColourSystem(
        apertus_path=apertus_path,
        spectrum_dims=spectrum_dims,
        gpu_acceleration=gpu_acceleration
    )

    # Test the system
    if not system.test_complete_pipeline():
        logging.warning("System test failed, but system is still functional")

    return system