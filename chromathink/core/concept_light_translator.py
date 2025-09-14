"""
Concept-to-Light Translation Layer

This module translates extracted concept words into light patterns
(wavelength, frequency, intensity) that can be processed by the
ChromaThink resonance chambers.

Key Functions:
- Maps concept words to spectral properties using BCM embeddings
- Converts linguistic concepts to wavelength/frequency/intensity tuples
- Provides consistent concept-to-colour translation across languages
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path


class ConceptLightTranslator:
    """
    Translates concept words into light patterns for ChromaThink processing.

    This translator uses the Big Colour Model (BCM) token embeddings to create
    consistent mappings from concept words to spectral properties.
    """

    def __init__(self,
                 big_colour_model: Optional[Dict] = None,
                 spectrum_dims: int = 512,
                 wavelength_range: Tuple[float, float] = (380.0, 750.0),  # Visible light nm
                 intensity_range: Tuple[float, float] = (0.1, 1.0)):

        self.logger = logging.getLogger("ConceptLightTranslator")
        self.big_colour_model = big_colour_model
        self.spectrum_dims = spectrum_dims
        self.wavelength_range = wavelength_range  # nanometres
        self.intensity_range = intensity_range

        # Frequency range calculated from wavelength (c = λν)
        # c = 3e8 m/s, convert nm to m
        c = 3e8  # m/s
        self.frequency_range = (
            c / (wavelength_range[1] * 1e-9),  # Hz
            c / (wavelength_range[0] * 1e-9)   # Hz
        )

        self.logger.info(f"ConceptLightTranslator initialized:")
        self.logger.info(f"  Wavelength range: {wavelength_range[0]}-{wavelength_range[1]} nm")
        self.logger.info(f"  Frequency range: {self.frequency_range[0]:.2e}-{self.frequency_range[1]:.2e} Hz")
        self.logger.info(f"  Intensity range: {intensity_range[0]}-{intensity_range[1]}")

        # Concept cache for performance
        self._concept_cache = {}

    def translate_concepts_to_light(self, concepts: List[str]) -> List[Tuple[float, float, float]]:
        """
        Translate a list of concept words to light patterns.

        Args:
            concepts: List of concept words from Apertus extraction

        Returns:
            List of (wavelength_nm, frequency_hz, intensity) tuples
        """
        if not concepts:
            self.logger.warning("No concepts provided for translation")
            return []

        self.logger.info(f"Translating {len(concepts)} concepts to light patterns: {concepts}")

        light_patterns = []
        for concept in concepts:
            try:
                light_pattern = self._concept_to_light(concept)
                light_patterns.append(light_pattern)
                self.logger.debug(f"  '{concept}' → λ={light_pattern[0]:.1f}nm, f={light_pattern[1]:.2e}Hz, I={light_pattern[2]:.3f}")
            except Exception as e:
                self.logger.error(f"Failed to translate concept '{concept}': {e}")
                # Fallback to neutral light pattern
                fallback_pattern = self._generate_fallback_light_pattern(concept)
                light_patterns.append(fallback_pattern)

        self.logger.info(f"Generated {len(light_patterns)} light patterns from concepts")
        return light_patterns

    def _concept_to_light(self, concept: str) -> Tuple[float, float, float]:
        """
        Convert a single concept to light pattern using BCM embeddings.

        Args:
            concept: Single concept word

        Returns:
            (wavelength_nm, frequency_hz, intensity) tuple
        """
        # Check cache first
        if concept in self._concept_cache:
            return self._concept_cache[concept]

        if self.big_colour_model is None:
            # Fallback without BCM
            return self._generate_fallback_light_pattern(concept)

        try:
            # Get concept embedding from BCM
            concept_embedding = self._get_concept_embedding(concept)

            # Convert embedding to spectral properties
            wavelength, frequency, intensity = self._embedding_to_spectral(concept_embedding)

            # Cache result
            result = (wavelength, frequency, intensity)
            self._concept_cache[concept] = result

            return result

        except Exception as e:
            self.logger.warning(f"BCM lookup failed for '{concept}': {e}")
            return self._generate_fallback_light_pattern(concept)

    def _get_concept_embedding(self, concept: str) -> np.ndarray:
        """
        Extract concept embedding from Big Colour Model.

        Args:
            concept: Concept word to look up

        Returns:
            Embedding vector from BCM
        """
        # Handle different BCM formats
        if hasattr(self.big_colour_model, 'encoder') and hasattr(self.big_colour_model.encoder, 'token_colours'):
            # New BigColourModel object format
            token_patterns = self.big_colour_model.encoder.token_colours
            is_array_format = True
        elif 'token_patterns' in self.big_colour_model:
            # Legacy dict format
            token_patterns = self.big_colour_model['token_patterns']
            is_array_format = isinstance(token_patterns, np.ndarray)
        else:
            raise ValueError("BCM token_patterns not found")

        concept_lower = concept.lower()

        if is_array_format:
            # Array format - use hash-based fallback only
            self.logger.debug(f"Using array format fallback for concept '{concept}'")
            concept_hash = abs(hash(concept_lower))
            if len(token_patterns) > 0:
                pattern_index = concept_hash % len(token_patterns)
                return np.array(token_patterns[pattern_index])
            else:
                # Fallback to random pattern
                return np.random.random(self.spectrum_dims or 512).astype(np.float32)
        else:
            # Dict format - do normal lookup
            if concept_lower in token_patterns:
                return np.array(token_patterns[concept_lower])

            # Try partial matches for compound concepts
            for token, pattern in token_patterns.items():
                if concept_lower in token or token in concept_lower:
                    self.logger.debug(f"Using partial match '{token}' for concept '{concept}'")
                    return np.array(pattern)

            # No direct match - use semantic similarity
            return self._find_similar_concept_embedding(concept_lower, token_patterns)

    def _find_similar_concept_embedding(self, concept: str, token_patterns: Dict) -> np.ndarray:
        """
        Find most similar concept in BCM using simple heuristics.

        Args:
            concept: Target concept
            token_patterns: BCM token patterns dict

        Returns:
            Most similar embedding vector
        """
        # Simple heuristics for concept similarity
        best_match = None
        best_score = 0

        for token in token_patterns.keys():
            score = 0

            # Length similarity
            len_similarity = 1.0 - abs(len(token) - len(concept)) / max(len(token), len(concept))
            score += len_similarity * 0.2

            # Character overlap
            overlap = len(set(token) & set(concept))
            char_similarity = overlap / len(set(token) | set(concept))
            score += char_similarity * 0.8

            if score > best_score:
                best_score = score
                best_match = token

        if best_match:
            self.logger.debug(f"Best similarity match for '{concept}': '{best_match}' (score: {best_score:.3f})")
            return np.array(token_patterns[best_match])
        else:
            # Ultimate fallback - use mean of all embeddings
            all_embeddings = [np.array(pattern) for pattern in token_patterns.values()]
            mean_embedding = np.mean(all_embeddings, axis=0)
            self.logger.debug(f"Using mean embedding for concept '{concept}'")
            return mean_embedding

    def _embedding_to_spectral(self, embedding: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert BCM embedding vector to spectral properties.

        Args:
            embedding: BCM embedding vector

        Returns:
            (wavelength_nm, frequency_hz, intensity) tuple
        """
        # Normalize embedding to [0, 1] range
        embedding_norm = embedding - np.min(embedding)
        embedding_norm = embedding_norm / (np.max(embedding_norm) + 1e-8)

        # Calculate dominant frequency from embedding
        # Use weighted average of embedding dimensions
        weights = np.arange(len(embedding_norm)) + 1
        dominant_freq_ratio = np.average(embedding_norm, weights=weights)

        # Map to frequency range
        frequency = (self.frequency_range[0] +
                    dominant_freq_ratio * (self.frequency_range[1] - self.frequency_range[0]))

        # Calculate wavelength from frequency (c = λν)
        c = 3e8  # m/s
        wavelength = (c / frequency) * 1e9  # convert to nm

        # Calculate intensity from embedding energy
        intensity = np.linalg.norm(embedding_norm)
        intensity = self.intensity_range[0] + (intensity % 1) * (self.intensity_range[1] - self.intensity_range[0])

        return wavelength, frequency, intensity

    def _generate_fallback_light_pattern(self, concept: str) -> Tuple[float, float, float]:
        """
        Generate fallback light pattern when BCM lookup fails.

        Args:
            concept: Concept word

        Returns:
            (wavelength_nm, frequency_hz, intensity) tuple based on concept hash
        """
        # Use concept hash for consistent fallback
        concept_hash = hash(concept.lower())

        # Map hash to wavelength
        wavelength_ratio = (concept_hash % 1000) / 1000.0
        wavelength = (self.wavelength_range[0] +
                     wavelength_ratio * (self.wavelength_range[1] - self.wavelength_range[0]))

        # Calculate frequency from wavelength
        c = 3e8  # m/s
        frequency = c / (wavelength * 1e-9)

        # Map hash to intensity
        intensity_ratio = ((concept_hash // 1000) % 1000) / 1000.0
        intensity = (self.intensity_range[0] +
                    intensity_ratio * (self.intensity_range[1] - self.intensity_range[0]))

        self.logger.debug(f"Fallback pattern for '{concept}': λ={wavelength:.1f}nm, f={frequency:.2e}Hz, I={intensity:.3f}")
        return wavelength, frequency, intensity

    def get_light_pattern_summary(self, light_patterns: List[Tuple[float, float, float]]) -> Dict:
        """
        Summarise light patterns for logging/debugging.

        Args:
            light_patterns: List of (wavelength, frequency, intensity) tuples

        Returns:
            Summary statistics dict
        """
        if not light_patterns:
            return {"count": 0, "wavelength_range": None, "frequency_range": None, "intensity_range": None}

        wavelengths = [pattern[0] for pattern in light_patterns]
        frequencies = [pattern[1] for pattern in light_patterns]
        intensities = [pattern[2] for pattern in light_patterns]

        return {
            "count": len(light_patterns),
            "wavelength_range": (min(wavelengths), max(wavelengths)),
            "frequency_range": (min(frequencies), max(frequencies)),
            "intensity_range": (min(intensities), max(intensities)),
            "dominant_wavelength": np.mean(wavelengths),
            "total_intensity": sum(intensities)
        }


def create_concept_light_translator(big_colour_model: Optional[Dict] = None,
                                  spectrum_dims: int = 512) -> ConceptLightTranslator:
    """
    Factory function to create ConceptLightTranslator instance.

    Args:
        big_colour_model: BCM token patterns dict
        spectrum_dims: Spectrum dimensions

    Returns:
        Configured ConceptLightTranslator instance
    """
    translator = ConceptLightTranslator(
        big_colour_model=big_colour_model,
        spectrum_dims=spectrum_dims
    )
    return translator