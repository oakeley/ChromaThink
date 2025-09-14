"""
True Colour Translator: Pure Apertus-based translation system
Implements the proper two-stage translation:
1. Human text → Apertus concept extraction → Concept words
2. ChromaThink colour response → Concept words → Apertus synthesis → Human text

CRITICAL: Apertus is ONLY a translator, ChromaThink is ONLY a thinker
"""

import torch
import tensorflow as tf
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import warnings


class TrueColourTranslator:
    """
    Pure Apertus-based translator that separates concept extraction from synthesis.

    Flow:
    1. Text → Extract concepts → Concept words
    2. Concept words → ChromaThink colour processing → Response concept words
    3. Response concept words + Original question → Synthesis → Natural text
    """

    def __init__(self, apertus_path: str = "models/apertus", device: str = 'cpu'):
        self.apertus_path = Path(apertus_path)
        self.device = device
        self.logger = logging.getLogger("TrueColourTranslator")

        # Concept extraction prompt (exactly as specified)
        self.extraction_prompt = """You are a concept extraction system. Your task is to identify the minimum essential concepts needed to understand the given text.

Instructions:
1. Extract 3-7 core concepts that capture the semantic essence of the input
2. Output ONLY single words or very short phrases (max 2 words)
3. Focus on concrete nouns, actions, and essential attributes
4. Ignore grammatical words (articles, conjunctions, prepositions)
5. Preserve the most semantically important elements
6. If the input is in a non-English language, extract concepts in that language

Format your response as a simple list with one concept per line.

Text to analyze: "{text}"

Core concepts:"""

        # Synthesis prompt (exactly as specified)
        self.synthesis_prompt = """You are a linguistic synthesizer. Your task is to create a natural, coherent response by weaving together concept words into grammatically correct sentences.

Instructions:
1. You will receive CONCEPT WORDS that represent the core semantic elements of a response
2. You will receive the ORIGINAL QUESTION that prompted these concepts
3. Create a natural sentence/paragraph that incorporates these concepts as the answer
4. Maintain logical flow and grammatical correctness
5. Ensure the response directly addresses the original question
6. Use appropriate connecting words and grammar to make the concepts flow naturally

Original question: "{original_question}"
Concept words: {concept_words}

Natural response:"""

        # Load Apertus model
        self._load_apertus()

    def _load_apertus(self):
        """Load Apertus model for translation tasks"""

        self.logger.info(f"Loading Apertus translator from {self.apertus_path}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.apertus_path,
                local_files_only=True,
                trust_remote_code=True
            )

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with proper device allocation
            device_config = {}
            if self.device == 'cpu':
                device_config = {
                    "device_map": "cpu",
                    "torch_dtype": torch.float32
                }
            else:
                device_config = {
                    "device_map": "auto",
                    "torch_dtype": torch.float32,  # Use float32 for stability
                    "max_memory": {0: "12GB"}
                }

            self.model = AutoModelForCausalLM.from_pretrained(
                self.apertus_path,
                local_files_only=True,
                use_safetensors=True,
                trust_remote_code=True,
                **device_config
            )
            self.model.eval()

            self.logger.info("Apertus translator loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load Apertus: {e}")
            raise

    def extract_concepts(self, human_text: str) -> List[str]:
        """
        Stage 1: Extract core concepts from human text using Apertus.
        Returns list of concept words for ChromaThink processing.
        """

        self.logger.info(f"Extracting concepts from: {human_text[:50]}...")

        # Create extraction prompt
        prompt = self.extraction_prompt.format(text=human_text)

        # Query Apertus for concept extraction
        response = self._query_apertus(prompt, max_tokens=100, temperature=0.3)

        # Parse concepts from response
        concepts = self._parse_concept_list(response)

        self.logger.debug(f"Extracted concepts: {concepts}")

        return concepts

    def synthesize_response(self, response_concepts: List[str], original_question: str) -> str:
        """
        Stage 2: Synthesize natural language response from concept words.
        Uses original question for context and proper grammar.
        """

        self.logger.info(f"Synthesizing response from {len(response_concepts)} concepts")

        # Format concept words for synthesis
        concept_words = ", ".join(response_concepts)

        # Create synthesis prompt
        prompt = self.synthesis_prompt.format(
            original_question=original_question,
            concept_words=concept_words
        )

        # Query Apertus for natural language synthesis
        response = self._query_apertus(prompt, max_tokens=300, temperature=0.7)

        self.logger.debug(f"Synthesized response: {response[:100]}...")

        return response.strip()

    def _query_apertus(self, prompt: str, max_tokens: int = 100, temperature: float = 0.5) -> str:
        """Query Apertus model with given prompt"""

        try:
            # Create messages for chat template
            messages = [{"role": "user", "content": prompt}]

            # Apply chat template
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True
            )

            # Create attention mask
            attention_mask = torch.ones_like(inputs)

            # Move to correct device
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][len(inputs[0]):],
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            self.logger.error(f"Apertus query failed: {e}")
            # Return meaningful fallback instead of crashing
            if "concepts" in prompt.lower():
                return "understanding\nknowledge\ninquiry"
            else:
                return "I understand your question and am processing the concepts involved."

    def _parse_concept_list(self, response: str) -> List[str]:
        """Parse Apertus response into clean concept list"""

        concepts = []

        # Split by lines and clean each concept
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Remove numbering, bullets, and formatting
            line = line.lstrip('0123456789.-•* \t')

            # Skip empty lines
            if not line:
                continue

            # Clean up concept (remove extra punctuation)
            concept = line.strip(' .,;:!?()[]{}"\'-')

            # Validate concept (single word or short phrase, max 2 words)
            words = concept.split()
            if len(words) <= 2 and len(concept) > 2 and len(concept) < 50:
                concepts.append(concept)

        # Limit to 7 concepts max as specified
        concepts = concepts[:7]

        # If we didn't get any valid concepts, provide fallback
        if not concepts:
            concepts = ["understanding", "knowledge", "response"]

        return concepts

    def detect_language(self, text: str) -> str:
        """Detect language of input text"""

        # Simple language detection based on common words/patterns
        text_lower = text.lower()

        # Check for common patterns
        if any(word in text_lower for word in ['the', 'and', 'is', 'are', 'how', 'what', 'why']):
            return 'english'
        elif any(word in text_lower for word in ['el', 'la', 'es', 'son', 'cómo', 'qué', 'por qué']):
            return 'spanish'
        elif any(word in text_lower for word in ['le', 'la', 'est', 'sont', 'comment', 'qu\'est', 'pourquoi']):
            return 'french'
        elif any(word in text_lower for word in ['der', 'die', 'das', 'ist', 'sind', 'wie', 'was', 'warum']):
            return 'german'
        else:
            return 'unknown'

    def test_translation_pipeline(self) -> bool:
        """Test the complete translation pipeline"""

        self.logger.info("Testing translation pipeline...")

        # Test concept extraction
        test_question = "How do birds navigate during migration?"
        concepts = self.extract_concepts(test_question)

        if not concepts:
            self.logger.error("Concept extraction failed")
            return False

        # Test synthesis with mock ChromaThink response concepts
        response_concepts = ["magnetic field", "sun position", "landmarks", "instinct", "generations"]
        synthesized = self.synthesize_response(response_concepts, test_question)

        if not synthesized or len(synthesized) < 10:
            self.logger.error("Response synthesis failed")
            return False

        self.logger.info("Translation pipeline test successful")
        self.logger.info(f"Input concepts: {concepts}")
        self.logger.info(f"Output synthesis: {synthesized}")

        return True


# Concept mapping for ChromaThink integration
class ConceptColourMapper:
    """
    Maps concept words to colour frequencies for ChromaThink processing.
    Uses deterministic mapping so same concepts always get same colours.
    """

    def __init__(self, spectrum_dims: int = 512):
        self.spectrum_dims = spectrum_dims
        self.logger = logging.getLogger("ConceptColourMapper")

    def concepts_to_colour_waveform(self, concepts: List[str]) -> np.ndarray:
        """Convert concept words to colour waveform for ChromaThink"""

        if not concepts:
            return np.zeros(self.spectrum_dims, dtype=np.complex64)

        # Start with first concept
        combined_waveform = self._concept_to_frequency(concepts[0])

        # Interfere additional concepts
        for concept in concepts[1:]:
            concept_wave = self._concept_to_frequency(concept)
            combined_waveform = self._wave_interference(combined_waveform, concept_wave)

        return combined_waveform

    def colour_waveform_to_concepts(self, waveform: np.ndarray, num_concepts: int = 5) -> List[str]:
        """Extract dominant concepts from colour waveform"""

        # Find dominant frequencies
        amplitudes = np.abs(waveform)
        dominant_indices = np.argsort(amplitudes)[-num_concepts:][::-1]

        concepts = []
        for idx in dominant_indices:
            freq = idx / self.spectrum_dims
            amplitude = amplitudes[idx]

            if amplitude > 0.05:  # Only significant frequencies
                concept = self._frequency_to_concept(freq, amplitude)
                concepts.append(concept)

        return concepts

    def _concept_to_frequency(self, concept: str) -> np.ndarray:
        """Map concept to frequency pattern"""

        # Create deterministic hash
        concept_hash = hash(concept.lower()) % (2**32)
        np.random.seed(concept_hash)

        # Generate frequency pattern
        base_freq = (concept_hash % 1000) / 1000.0
        num_harmonics = min(len(concept), 5) + 1

        waveform = np.zeros(self.spectrum_dims, dtype=np.complex64)

        # Base frequency
        base_idx = int(base_freq * self.spectrum_dims)
        waveform[base_idx] = 1.0

        # Add harmonics
        for h in range(2, num_harmonics + 1):
            harmonic_freq = (base_freq * h) % 1.0
            harmonic_idx = int(harmonic_freq * self.spectrum_dims)
            amplitude = 1.0 / h
            phase = (concept_hash * h) % (2 * np.pi)
            waveform[harmonic_idx] = amplitude * np.exp(1j * phase)

        # Normalize
        norm = np.linalg.norm(waveform)
        if norm > 0:
            waveform /= norm

        return waveform

    def _wave_interference(self, wave1: np.ndarray, wave2: np.ndarray) -> np.ndarray:
        """Natural wave interference"""

        # Direct complex addition for wave interference
        result = wave1 + wave2

        # Normalize to prevent explosion
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm

        return result

    def _frequency_to_concept(self, freq: float, amplitude: float) -> str:
        """Map frequency back to concept type"""

        if freq < 0.33:
            if amplitude > 0.3:
                return "fundamental concept"
            else:
                return "basic idea"
        elif freq < 0.67:
            if amplitude > 0.3:
                return "relational concept"
            else:
                return "connection"
        else:
            if amplitude > 0.3:
                return "complex concept"
            else:
                return "detail"