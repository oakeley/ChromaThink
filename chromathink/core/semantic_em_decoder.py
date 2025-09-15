#!/usr/bin/env python3
"""
Semantic EM Spectrum Decoder
Converts electromagnetic spectra back to language-independent concepts.
The reverse of semantic_em_encoder.py
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from .semantic_em_encoder import EMSpectrum, ConceptCluster, SemanticEMEncoder

logger = logging.getLogger(__name__)

class SemanticEMDecoder:
    """
    Decodes electromagnetic spectra back to semantic concepts and natural language.
    Like speech-to-text but for EM spectra → concepts → text.
    """

    def __init__(self, encoder: SemanticEMEncoder):
        self.encoder = encoder
        self.concept_clusters = encoder.concept_clusters

        # Build reverse lookup: spectrum → concepts
        self.spectrum_to_concept = self._build_spectrum_lookup()

    def _build_spectrum_lookup(self) -> Dict[Tuple[float, float], str]:
        """Build lookup table from spectrum characteristics to concept clusters."""
        spectrum_lookup = {}

        for cluster_id, cluster in self.concept_clusters.items():
            spectrum = cluster.assigned_spectrum
            if spectrum:
                # Use wavelength and bandwidth as key (rounded for lookup)
                key = (round(spectrum.wavelength, 1), round(spectrum.bandwidth, 1))
                spectrum_lookup[key] = cluster_id

        logger.info(f"Built spectrum lookup with {len(spectrum_lookup)} entries")
        return spectrum_lookup

    def decode_spectrum(self, em_spectrum: EMSpectrum, target_language: str = 'en') -> str:
        """
        Decode EM spectrum back to natural language text.

        Args:
            em_spectrum: The electromagnetic spectrum to decode
            target_language: Target language for output ('en', 'fr', 'de', etc.)

        Returns:
            Natural language text in target language
        """
        logger.info(f"Decoding spectrum: λ={em_spectrum.wavelength:.1f}nm, "
                   f"Δλ={em_spectrum.bandwidth:.1f}nm, I={em_spectrum.intensity:.3f}")

        # Step 1: Analyze spectrum to find concept matches
        concepts = self._spectrum_to_concepts(em_spectrum)

        if not concepts:
            return "[No concepts found in spectrum]"

        logger.info(f"Found concepts: {concepts}")

        # Step 2: Convert concepts to target language
        target_words = self._concepts_to_language(concepts, target_language)

        # Step 3: Synthesize natural language
        return self._synthesize_text(target_words)

    def _spectrum_to_concepts(self, em_spectrum: EMSpectrum) -> List[str]:
        """Find concept clusters that match the input spectrum."""
        concepts = []

        # Find exact matches first
        key = (round(em_spectrum.wavelength, 1), round(em_spectrum.bandwidth, 1))
        if key in self.spectrum_to_concept:
            cluster_id = self.spectrum_to_concept[key]
            concepts.append(cluster_id)
        else:
            # Find approximate matches
            best_matches = self._find_approximate_matches(em_spectrum)
            concepts.extend(best_matches)

        return concepts

    def _find_approximate_matches(self, em_spectrum: EMSpectrum, max_matches: int = 3) -> List[str]:
        """Find concept clusters with similar spectra."""
        matches = []

        for cluster_id, cluster in self.concept_clusters.items():
            spectrum = cluster.assigned_spectrum
            if spectrum:
                # Calculate spectral distance
                distance = self._spectral_distance(em_spectrum, spectrum)
                matches.append((cluster_id, distance))

        # Sort by distance and return best matches
        matches.sort(key=lambda x: x[1])
        return [cluster_id for cluster_id, distance in matches[:max_matches]]

    def _spectral_distance(self, spectrum1: EMSpectrum, spectrum2: EMSpectrum) -> float:
        """Calculate distance between two spectra."""
        wavelength_diff = abs(spectrum1.wavelength - spectrum2.wavelength)
        bandwidth_diff = abs(spectrum1.bandwidth - spectrum2.bandwidth)
        intensity_diff = abs(spectrum1.intensity - spectrum2.intensity)

        # Weighted distance (wavelength is most important)
        distance = (wavelength_diff * 2.0 + bandwidth_diff * 1.0 + intensity_diff * 0.5) / 3.5
        return distance

    def _concepts_to_language(self, concept_ids: List[str], target_language: str) -> List[str]:
        """Convert concept cluster IDs to words in target language."""
        target_words = []

        for concept_id in concept_ids:
            if concept_id in self.concept_clusters:
                cluster = self.concept_clusters[concept_id]

                # Find best token for target language
                best_token = self._select_token_for_language(cluster, target_language)
                if best_token:
                    target_words.append(best_token)

        return target_words

    def _select_token_for_language(self, cluster: ConceptCluster, target_language: str) -> Optional[str]:
        """Select the best token from cluster for target language."""
        # Simple heuristics for language selection
        language_patterns = {
            'en': ['love', 'gravity', 'water', 'happy'],  # English patterns
            'fr': ['amour', 'gravité', 'eau', 'bonheur'],  # French patterns
            'de': ['liebe', 'gravitation', 'wasser', 'glück'],  # German patterns
            'es': ['amor', 'gravedad', 'agua', 'feliz'],  # Spanish patterns
        }

        target_patterns = language_patterns.get(target_language, [])

        # First try to find exact language match
        for token in cluster.tokens:
            for pattern in target_patterns:
                if pattern.lower() in token.lower():
                    return token

        # Fallback to English-like tokens
        for token in cluster.tokens:
            if all(ord(char) < 128 for char in token):  # ASCII only (likely English)
                return token

        # Last resort: return first token
        return cluster.tokens[0] if cluster.tokens else None

    def _synthesize_text(self, words: List[str]) -> str:
        """Synthesize natural language from concept words."""
        if not words:
            return "[No words to synthesize]"

        # Simple synthesis: just join the words
        # In a real implementation, this would use proper grammar rules
        if len(words) == 1:
            return words[0]
        elif len(words) == 2:
            return f"{words[0]} {words[1]}"
        else:
            # Create simple sentence structure
            return " ".join(words)

    def decode_waveform(self, waveform: np.ndarray, spectrum_dims: int = 512) -> str:
        """
        Decode a complex waveform back to text by extracting spectral peaks.

        Args:
            waveform: Complex waveform from resonance chamber processing
            spectrum_dims: Dimension of the spectrum

        Returns:
            Decoded text
        """
        logger.info(f"Decoding waveform with shape {waveform.shape}")

        # Step 1: Extract spectral peaks from waveform
        spectra = self._waveform_to_spectra(waveform, spectrum_dims)

        # Step 2: Decode each spectrum to concepts
        all_concepts = []
        for spectrum in spectra:
            concepts = self._spectrum_to_concepts(spectrum)
            all_concepts.extend(concepts)

        # Step 3: Convert concepts to text
        target_words = self._concepts_to_language(all_concepts, 'en')

        # Step 4: Synthesize final text
        return self._synthesize_text(target_words)

    def _waveform_to_spectra(self, waveform: np.ndarray, spectrum_dims: int) -> List[EMSpectrum]:
        """Extract EM spectra from processed waveform."""
        spectra = []

        # Get magnitude spectrum
        if waveform.dtype == np.complex64 or waveform.dtype == np.complex128:
            magnitude = np.abs(waveform.flatten())
        else:
            magnitude = np.abs(waveform.flatten())

        # Find spectral peaks
        threshold = np.max(magnitude) * 0.1  # 10% threshold
        peak_indices = np.where(magnitude > threshold)[0]

        for peak_idx in peak_indices[:5]:  # Top 5 peaks
            # Convert frequency index to wavelength
            # Assuming frequency range: 10^14 to 10^16 Hz
            frequency_hz = 10**(14 + peak_idx * 2 / spectrum_dims)

            # Convert frequency to wavelength: λ = c/f
            wavelength_m = 3e8 / frequency_hz
            wavelength_nm = wavelength_m * 1e9

            # Clip to visible spectrum for now
            wavelength_nm = np.clip(wavelength_nm, 300, 800)

            # Estimate bandwidth and intensity
            intensity = float(magnitude[peak_idx] / np.max(magnitude))
            bandwidth = 20.0  # Default bandwidth

            spectrum = EMSpectrum(
                wavelength=wavelength_nm,
                bandwidth=bandwidth,
                intensity=intensity,
                fine_structure=None
            )

            spectra.append(spectrum)

        return spectra

    def analyze_spectrum_content(self, em_spectrum: EMSpectrum) -> Dict:
        """Analyze what semantic content is present in a spectrum."""
        analysis = {
            'wavelength': em_spectrum.wavelength,
            'bandwidth': em_spectrum.bandwidth,
            'intensity': em_spectrum.intensity,
            'semantic_category': self._classify_spectrum_category(em_spectrum),
            'matching_concepts': self._spectrum_to_concepts(em_spectrum),
            'language_options': {}
        }

        # Get language options for each concept
        for concept_id in analysis['matching_concepts']:
            if concept_id in self.concept_clusters:
                cluster = self.concept_clusters[concept_id]
                analysis['language_options'][concept_id] = cluster.tokens

        return analysis

    def _classify_spectrum_category(self, em_spectrum: EMSpectrum) -> str:
        """Classify semantic category based on spectrum characteristics."""
        wavelength = em_spectrum.wavelength

        if 400 <= wavelength <= 500:
            return 'physics'  # Blue-violet
        elif 600 <= wavelength <= 700:
            return 'emotion'  # Red-orange
        elif 500 <= wavelength <= 600:
            return 'action'   # Green
        elif 300 <= wavelength <= 400:
            return 'abstract' # UV
        elif 700 <= wavelength <= 800:
            return 'concrete' # Near-IR
        elif 550 <= wavelength <= 650:
            return 'social'   # Yellow-orange
        elif 450 <= wavelength <= 550:
            return 'temporal' # Blue-green
        else:
            return 'unknown'


def test_semantic_decoder():
    """Test the semantic EM decoder."""
    print("Testing Semantic EM Decoder...")
    print("=" * 50)

    # Create encoder and build dictionary
    encoder = SemanticEMEncoder()
    concept_dict = encoder.build_concept_dictionary()

    # Create decoder
    decoder = SemanticEMDecoder(encoder)

    print(f"Decoder initialized with {len(decoder.concept_clusters)} concept clusters")
    print()

    # Test encoding and decoding
    test_cases = [
        "love",
        "gravity",
        "explain gravity",
        "what is love"
    ]

    print("Testing encode → decode cycle:")
    print("-" * 40)

    for test_text in test_cases:
        # Encode to spectrum
        spectrum = encoder.encode_text(test_text)

        # Decode back to text
        decoded_text = decoder.decode_spectrum(spectrum, 'en')

        # Analyze spectrum content
        analysis = decoder.analyze_spectrum_content(spectrum)

        print(f"Original: '{test_text}'")
        print(f"Spectrum: λ={spectrum.wavelength:.1f}nm, Δλ={spectrum.bandwidth:.1f}nm")
        print(f"Category: {analysis['semantic_category']}")
        print(f"Decoded:  '{decoded_text}'")
        print(f"Concepts: {analysis['matching_concepts']}")
        print()

    # Test multilingual decoding
    print("Testing multilingual decoding:")
    print("-" * 40)

    love_spectrum = encoder.encode_text("love")

    for lang in ['en', 'fr', 'de']:
        decoded = decoder.decode_spectrum(love_spectrum, lang)
        print(f"Love in {lang}: '{decoded}'")


if __name__ == "__main__":
    test_semantic_decoder()