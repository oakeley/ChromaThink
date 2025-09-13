"""
Language-Agnostic Concept Mapper for ChromaThink

This module maps semantic concepts to frequency patterns regardless of language.
The same concept (e.g., "gravity"/"gravité"/"Schwerkraft") maps to the same frequency.
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import logging
import hashlib


class LanguageAgnosticMapper:
    """
    Maps semantic concepts to frequencies based on meaning rather than text.
    Concepts with the same meaning get the same frequency regardless of language.
    """
    
    def __init__(self, spectrum_dims: int = 512, apertus_translator=None):
        self.spectrum_dims = spectrum_dims
        self.apertus_translator = apertus_translator
        self.logger = logging.getLogger("LanguageAgnosticMapper")
        
        # Cache for concept translations and frequencies
        self.concept_cache = {}
        
    def concepts_to_waveform(self, concepts: List[str], original_text: str = "") -> np.ndarray:
        """
        Convert a list of concepts (in any language) to a unified waveform.
        Uses semantic meaning rather than literal text.
        """
        
        if not concepts:
            return np.zeros(self.spectrum_dims, dtype=complex)
        
        # Start with first concept
        combined_waveform = self._concept_to_frequency_pattern(concepts[0]).copy()
        
        # Add interference from other concepts
        for concept in concepts[1:]:
            concept_waveform = self._concept_to_frequency_pattern(concept)
            combined_waveform = self._interference_combine(combined_waveform, concept_waveform)
        
        # Apply context-dependent modulation if we have the original text
        if original_text:
            combined_waveform = self._apply_context_modulation(combined_waveform, original_text, concepts)
        
        return combined_waveform
    
    def _concept_to_frequency_pattern(self, concept: str) -> np.ndarray:
        """
        Map a concept to its frequency pattern based on semantic category.
        Same semantic concept gets same frequency regardless of language.
        """
        
        # Check cache first
        if concept in self.concept_cache:
            return self.concept_cache[concept]
        
        # Determine semantic category through pattern analysis
        concept_lower = concept.lower().strip()
        
        # Generate base frequency from semantic hash (not text hash)
        semantic_hash = self._get_semantic_hash(concept_lower)
        base_freq = (semantic_hash % 1000) / 1000.0
        
        # Determine number of harmonics and phase from concept characteristics
        num_harmonics = self._get_harmonic_count(concept_lower)
        phase_offset = self._get_phase_offset(concept_lower)
        
        # Generate frequency pattern
        waveform = self._generate_harmonic_pattern(base_freq, num_harmonics, phase_offset)
        
        # Cache the result
        self.concept_cache[concept] = waveform
        
        return waveform
    
    def _get_semantic_hash(self, concept: str) -> int:
        """
        Generate semantic hash based on concept characteristics rather than literal text.
        Similar concepts should have similar hashes regardless of language.
        """
        
        # Analyze concept characteristics
        length = len(concept)
        vowel_count = sum(1 for c in concept if c in 'aeiouäöüàéèíóúñ')  # Include accented vowels
        consonant_count = length - vowel_count
        
        # Check for common semantic indicators across languages
        semantic_features = 0
        
        # Science/physics indicators
        if any(pattern in concept for pattern in ['grav', 'forc', 'energ', 'phys', 'scien']):
            semantic_features += 1000
        
        # Action/verb indicators (common endings across languages)
        if any(pattern in concept for pattern in ['ing', 'er', 'ir', 'en', 'tion', 'sion', 'ment']):
            semantic_features += 500
        
        # Question/interrogative indicators
        if any(pattern in concept for pattern in ['qu', 'wh', 'pourqu', 'warum', 'como']):
            semantic_features += 750
        
        # Emotion indicators
        if any(pattern in concept for pattern in ['happ', 'joy', 'sad', 'ang', 'fear', 'love']):
            semantic_features += 300
        
        # Combine features for semantic hash
        semantic_value = (length * 37 + vowel_count * 73 + consonant_count * 23 + semantic_features) % 100000
        
        return semantic_value
    
    def _get_harmonic_count(self, concept: str) -> int:
        """Determine harmonic count based on concept complexity."""
        
        length = len(concept)
        
        # Simple concepts (short words) have fewer harmonics
        if length <= 3:
            return 2
        elif length <= 6:
            return 3 
        elif length <= 9:
            return 4
        else:
            return 5
    
    def _get_phase_offset(self, concept: str) -> float:
        """Determine phase offset based on concept characteristics."""
        
        # Use first and last characters to determine phase
        if len(concept) == 0:
            return 0.0
        
        first_char = ord(concept[0]) if concept else 0
        last_char = ord(concept[-1]) if len(concept) > 1 else first_char
        
        phase = ((first_char + last_char) % 100) / 100.0 * 2 * np.pi
        
        return phase
    
    def _generate_harmonic_pattern(self, base_freq: float, num_harmonics: int, phase_offset: float) -> np.ndarray:
        """Generate harmonic frequency pattern."""
        
        waveform = np.zeros(self.spectrum_dims, dtype=complex)
        
        # Base frequency
        base_idx = int(base_freq * self.spectrum_dims)
        waveform[base_idx] = 1.0 * np.exp(1j * phase_offset)
        
        # Add harmonics
        for h in range(2, num_harmonics + 1):
            harmonic_freq = (base_freq * h) % 1.0
            harmonic_idx = int(harmonic_freq * self.spectrum_dims)
            amplitude = 1.0 / h  # Harmonic rolloff
            phase = phase_offset + h * 0.2
            waveform[harmonic_idx] += amplitude * np.exp(1j * phase)
        
        # Normalize
        norm = np.linalg.norm(waveform)
        if norm > 0:
            waveform /= norm
        
        return waveform
    
    def _interference_combine(self, wave1: np.ndarray, wave2: np.ndarray) -> np.ndarray:
        """Combine waveforms through interference."""
        
        # Direct wave interference
        amplitude1 = np.abs(wave1)
        phase1 = np.angle(wave1)
        amplitude2 = np.abs(wave2)
        phase2 = np.angle(wave2)
        
        # Calculate interference
        phase_diff = phase1 - phase2
        result_amplitude = np.sqrt(
            amplitude1**2 + amplitude2**2 + 2*amplitude1*amplitude2*np.cos(phase_diff)
        )
        
        # Weighted phase combining
        total_amplitude = amplitude1 + amplitude2 + 1e-8
        result_phase = (amplitude1*phase1 + amplitude2*phase2) / total_amplitude
        
        result = result_amplitude * np.exp(1j * result_phase)
        
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
        
        return result
    
    def _apply_context_modulation(self, waveform: np.ndarray, original_text: str, concepts: List[str]) -> np.ndarray:
        """
        Apply context-dependent modulation based on the original text structure.
        This helps preserve meaning even when concepts are in different languages.
        """
        
        text_length = len(original_text.split())
        
        # Question modulation (works across languages)
        question_indicators = ['?', 'qu', 'wh', 'pourqu', 'warum', 'como', 'что', 'كيف']
        is_question = any(indicator in original_text.lower() for indicator in question_indicators)
        
        if is_question:
            # Questions get higher frequency components
            modulation = np.exp(2j * np.pi * np.arange(self.spectrum_dims) / self.spectrum_dims * 0.1)
            waveform *= (1.0 + 0.2 * modulation)
        
        # Length-based modulation (complexity)
        if text_length > 10:  # Complex statement
            # Add low-frequency components for complexity
            complexity_modulation = np.exp(-np.arange(self.spectrum_dims) / self.spectrum_dims * 2)
            waveform += 0.1 * complexity_modulation * np.exp(1j * np.pi/4)
        
        # Normalize
        norm = np.linalg.norm(waveform)
        if norm > 0:
            waveform /= norm
        
        return waveform
    
    def waveform_to_concepts(self, waveform: np.ndarray, target_language: str = "en") -> List[str]:
        """
        Convert waveform back to concepts in the target language.
        This is more complex and would ideally use Apertus for translation.
        """
        
        # For now, extract frequency patterns and map to generic concepts
        amplitudes = np.abs(waveform)
        dominant_indices = np.argsort(amplitudes)[-5:][::-1]
        
        concepts = []
        for idx in dominant_indices:
            freq = idx / self.spectrum_dims
            amplitude = amplitudes[idx]
            
            if amplitude > 0.1:
                # Generate concept based on frequency characteristics
                if freq < 0.3:
                    concepts.append("fundamental concept")
                elif freq < 0.6:
                    concepts.append("relational concept")
                else:
                    concepts.append("complex concept")
        
        return concepts[:3]  # Return top 3 concepts