"""
Light-to-Concept Translation Layer

This module translates processed light patterns from ChromaThink resonance
chambers back into concept words that can be used by Apertus for natural
language synthesis.

Key Functions:
- Maps spectral properties to concept words using BCM reverse lookup
- Converts wavelength/frequency/intensity patterns to linguistic concepts
- Provides intensity-based filtering and concept ranking
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path


class LightConceptTranslator:
    """
    Translates light patterns from ChromaThink processing back to concept words.

    This translator performs reverse lookup using the Big Colour Model (BCM)
    to find concept words that best match processed spectral patterns.
    """

    def __init__(self,
                 big_colour_model: Optional[Dict] = None,
                 spectrum_dims: int = 512,
                 intensity_threshold: float = 0.2,  # Minimum intensity to include concept
                 max_concepts: int = 20):           # Maximum concepts to return

        self.logger = logging.getLogger("LightConceptTranslator")
        self.big_colour_model = big_colour_model
        self.spectrum_dims = spectrum_dims
        self.intensity_threshold = intensity_threshold
        self.max_concepts = max_concepts

        self.logger.info(f"LightConceptTranslator initialized:")
        self.logger.info(f"  Intensity threshold: {intensity_threshold}")
        self.logger.info(f"  Max concepts: {max_concepts}")

        # Build reverse lookup index from BCM
        self._build_reverse_index()

        # Cache for performance
        self._pattern_cache = {}

    def translate_light_to_concepts(self,
                                  light_patterns: List[Tuple[float, float, float]],
                                  filter_intensity: bool = True) -> List[str]:
        """
        Translate processed light patterns back to concept words.

        Args:
            light_patterns: List of (wavelength_nm, frequency_hz, intensity) tuples
            filter_intensity: Whether to filter out low-intensity patterns

        Returns:
            List of concept words ranked by relevance
        """
        if not light_patterns:
            self.logger.warning("No light patterns provided for translation")
            return []

        self.logger.info(f"Translating {len(light_patterns)} light patterns to concepts")

        # Filter by intensity if requested
        if filter_intensity:
            filtered_patterns = [
                pattern for pattern in light_patterns
                if pattern[2] >= self.intensity_threshold
            ]
            self.logger.info(f"Filtered to {len(filtered_patterns)} patterns above intensity {self.intensity_threshold}")
            light_patterns = filtered_patterns

        if not light_patterns:
            self.logger.warning("No patterns above intensity threshold")
            return ["silence", "void", "empty"]  # Fallback concepts

        # Convert each light pattern to concepts
        all_concepts = []
        for wavelength, frequency, intensity in light_patterns:
            try:
                concepts = self._light_pattern_to_concepts(wavelength, frequency, intensity)
                all_concepts.extend(concepts)
                self.logger.debug(f"λ={wavelength:.1f}nm, I={intensity:.3f} → {concepts}")
            except Exception as e:
                self.logger.error(f"Failed to translate light pattern λ={wavelength:.1f}nm: {e}")

        if not all_concepts:
            self.logger.warning("No concepts extracted from light patterns")
            return ["unknown", "pattern", "energy"]  # Fallback concepts

        # Rank concepts by frequency and intensity
        ranked_concepts = self._rank_concepts(all_concepts, light_patterns)

        # Limit to max concepts
        final_concepts = ranked_concepts[:self.max_concepts]

        self.logger.info(f"Extracted {len(final_concepts)} ranked concepts: {final_concepts}")
        return final_concepts

    def _build_reverse_index(self):
        """
        Build reverse lookup index from BCM token patterns to spectral properties.
        """
        self.spectral_index = {}  # Maps (wavelength_bin, intensity_bin) to concepts

        if self.big_colour_model is None or 'token_patterns' not in self.big_colour_model:
            self.logger.warning("No BCM available - using fallback reverse index")
            self._build_fallback_index()
            return

        # Handle different BCM formats
        if hasattr(self.big_colour_model, 'encoder') and hasattr(self.big_colour_model.encoder, 'token_colours'):
            # New BigColourModel object format
            token_patterns = self.big_colour_model.encoder.token_colours
            self.logger.info(f"Building reverse index from {len(token_patterns)} BCM tokens (numpy array format)")
            is_array_format = True
        else:
            # Legacy dict format
            token_patterns = self.big_colour_model['token_patterns']
            self.logger.info(f"Building reverse index from {len(token_patterns)} BCM tokens (dict format)")
            is_array_format = False

        # Process tokens based on format
        if is_array_format:
            # Array format - enumerate with synthetic token names
            token_iterator = enumerate(token_patterns)
        else:
            # Dict format - use actual token names
            token_iterator = token_patterns.items()

        for item in token_iterator:
            if is_array_format:
                i, pattern = item
                token = f"token_{i}"
            else:
                token, pattern = item
            try:
                # Convert token pattern to spectral properties
                wavelength, frequency, intensity = self._pattern_to_spectral(np.array(pattern))

                # Create spectral bins for indexing
                wavelength_bin = int(wavelength // 10) * 10  # 10nm bins
                intensity_bin = round(intensity, 1)           # 0.1 intensity bins

                spectral_key = (wavelength_bin, intensity_bin)

                if spectral_key not in self.spectral_index:
                    self.spectral_index[spectral_key] = []

                self.spectral_index[spectral_key].append(token)

            except Exception as e:
                self.logger.debug(f"Skipping token '{token}': {e}")

        self.logger.info(f"Built reverse index with {len(self.spectral_index)} spectral bins")

    def _build_fallback_index(self):
        """
        Build fallback reverse index with common concepts.
        """
        # Map wavelength ranges to conceptual categories
        fallback_concepts = {
            (380, 450): ["blue", "cool", "calm", "water", "sky", "deep"],       # Violet/Blue
            (450, 495): ["blue", "ocean", "peace", "trust", "stability"],       # Blue
            (495, 570): ["green", "nature", "growth", "life", "harmony"],       # Green
            (570, 590): ["yellow", "sun", "energy", "bright", "warm"],          # Yellow
            (590, 620): ["orange", "fire", "creativity", "enthusiasm"],         # Orange
            (620, 750): ["red", "passion", "power", "strength", "heat"]         # Red
        }

        self.spectral_index = {}
        for (min_wave, max_wave), concepts in fallback_concepts.items():
            for wavelength in range(min_wave, max_wave, 10):  # 10nm steps
                for intensity in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    spectral_key = (wavelength, intensity)
                    self.spectral_index[spectral_key] = concepts.copy()

        self.logger.info(f"Built fallback reverse index with {len(self.spectral_index)} entries")

    def _pattern_to_spectral(self, pattern: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert BCM pattern to spectral properties (same as concept_light_translator).

        Args:
            pattern: BCM token pattern vector

        Returns:
            (wavelength_nm, frequency_hz, intensity) tuple
        """
        # Normalize pattern to [0, 1] range
        pattern_norm = pattern - np.min(pattern)
        pattern_norm = pattern_norm / (np.max(pattern_norm) + 1e-8)

        # Calculate dominant frequency from pattern
        weights = np.arange(len(pattern_norm)) + 1
        dominant_freq_ratio = np.average(pattern_norm, weights=weights)

        # Map to frequency range (same as concept_light_translator)
        c = 3e8  # m/s
        wavelength_range = (380.0, 750.0)  # nm
        frequency_range = (c / (wavelength_range[1] * 1e-9), c / (wavelength_range[0] * 1e-9))

        frequency = (frequency_range[0] +
                    dominant_freq_ratio * (frequency_range[1] - frequency_range[0]))

        # Calculate wavelength from frequency
        wavelength = (c / frequency) * 1e9  # convert to nm

        # Calculate intensity from pattern energy
        intensity = np.linalg.norm(pattern_norm)
        intensity = 0.1 + (intensity % 1) * 0.9  # Scale to [0.1, 1.0]

        return wavelength, frequency, intensity

    def _light_pattern_to_concepts(self, wavelength: float, frequency: float, intensity: float) -> List[str]:
        """
        Convert a single light pattern to concept words using reverse lookup.

        Args:
            wavelength: Wavelength in nm
            frequency: Frequency in Hz
            intensity: Intensity value

        Returns:
            List of concept words matching this pattern
        """
        # Create spectral bins for lookup
        wavelength_bin = int(wavelength // 10) * 10
        intensity_bin = round(intensity, 1)

        # Primary lookup
        spectral_key = (wavelength_bin, intensity_bin)
        if spectral_key in self.spectral_index:
            return self.spectral_index[spectral_key].copy()

        # Fallback: search nearby bins
        concepts = []
        for wave_offset in [-10, 0, 10]:
            for int_offset in [-0.1, 0.0, 0.1]:
                nearby_key = (wavelength_bin + wave_offset, round(intensity_bin + int_offset, 1))
                if nearby_key in self.spectral_index:
                    concepts.extend(self.spectral_index[nearby_key])

        if concepts:
            return list(set(concepts))  # Remove duplicates

        # Ultimate fallback: wavelength-based concepts
        return self._wavelength_to_concepts(wavelength, intensity)

    def _wavelength_to_concepts(self, wavelength: float, intensity: float) -> List[str]:
        """
        Fallback conversion from wavelength to concepts using color theory.

        Args:
            wavelength: Wavelength in nm
            intensity: Intensity value

        Returns:
            List of concept words based on wavelength
        """
        # Intensity modifiers
        intensity_concepts = []
        if intensity > 0.8:
            intensity_concepts.extend(["bright", "strong", "intense"])
        elif intensity > 0.5:
            intensity_concepts.extend(["clear", "distinct"])
        else:
            intensity_concepts.extend(["subtle", "soft", "gentle"])

        # Wavelength-based concepts
        if wavelength < 450:      # Violet/Blue
            base_concepts = ["blue", "deep", "calm", "cool"]
        elif wavelength < 495:    # Blue
            base_concepts = ["blue", "water", "peace", "trust"]
        elif wavelength < 570:    # Green
            base_concepts = ["green", "nature", "growth", "life"]
        elif wavelength < 590:    # Yellow
            base_concepts = ["yellow", "light", "energy", "warm"]
        elif wavelength < 620:    # Orange
            base_concepts = ["orange", "fire", "creative", "dynamic"]
        else:                     # Red
            base_concepts = ["red", "passion", "power", "heat"]

        return base_concepts + intensity_concepts

    def _rank_concepts(self, concepts: List[str], light_patterns: List[Tuple[float, float, float]]) -> List[str]:
        """
        Rank concepts by frequency and intensity weighting.

        Args:
            concepts: List of all extracted concepts
            light_patterns: Original light patterns for weighting

        Returns:
            List of concepts ranked by relevance
        """
        if not concepts:
            return []

        # Count concept frequencies
        concept_counts = {}
        concept_intensities = {}

        for i, concept in enumerate(concepts):
            if concept not in concept_counts:
                concept_counts[concept] = 0
                concept_intensities[concept] = 0.0

            concept_counts[concept] += 1

            # Add intensity weighting from corresponding light pattern
            if i < len(light_patterns):
                concept_intensities[concept] += light_patterns[i][2]  # intensity

        # Calculate ranking scores
        concept_scores = {}
        for concept in concept_counts:
            frequency_score = concept_counts[concept] / len(concepts)
            intensity_score = concept_intensities[concept] / max(concept_counts[concept], 1)

            # Combined score
            concept_scores[concept] = frequency_score * 0.6 + intensity_score * 0.4

        # Sort by score
        ranked_concepts = sorted(concept_scores.keys(), key=lambda c: concept_scores[c], reverse=True)

        self.logger.debug(f"Concept ranking: {[(c, round(concept_scores[c], 3)) for c in ranked_concepts[:5]]}")

        return ranked_concepts

    def get_translation_summary(self, light_patterns: List[Tuple[float, float, float]],
                               concepts: List[str]) -> Dict:
        """
        Generate translation summary for logging/debugging.

        Args:
            light_patterns: Input light patterns
            concepts: Extracted concepts

        Returns:
            Translation summary dict
        """
        if not light_patterns:
            return {"input_patterns": 0, "output_concepts": len(concepts)}

        wavelengths = [p[0] for p in light_patterns]
        intensities = [p[2] for p in light_patterns]

        return {
            "input_patterns": len(light_patterns),
            "output_concepts": len(concepts),
            "wavelength_range": (min(wavelengths), max(wavelengths)),
            "intensity_range": (min(intensities), max(intensities)),
            "average_intensity": sum(intensities) / len(intensities),
            "high_intensity_patterns": len([i for i in intensities if i > 0.7]),
            "primary_concepts": concepts[:5] if concepts else []
        }


def create_light_concept_translator(big_colour_model: Optional[Dict] = None,
                                   spectrum_dims: int = 512,
                                   intensity_threshold: float = 0.2) -> LightConceptTranslator:
    """
    Factory function to create LightConceptTranslator instance.

    Args:
        big_colour_model: BCM token patterns dict
        spectrum_dims: Spectrum dimensions
        intensity_threshold: Minimum intensity for concepts

    Returns:
        Configured LightConceptTranslator instance
    """
    translator = LightConceptTranslator(
        big_colour_model=big_colour_model,
        spectrum_dims=spectrum_dims,
        intensity_threshold=intensity_threshold
    )
    return translator