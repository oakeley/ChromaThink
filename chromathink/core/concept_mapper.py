"""
Semantic Concept-to-Frequency Mapper for ChromaThink

This module provides meaningful bidirectional mapping between linguistic concepts 
and color frequency representations, replacing the random hash-based approach
with semantic understanding.
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path


class SemanticConceptMapper:
    """
    Maps semantic concepts to meaningful frequency patterns based on
    linguistic and cognitive principles rather than random hashing.
    """
    
    def __init__(self, spectrum_dims: int = 512):
        self.spectrum_dims = spectrum_dims
        self.logger = logging.getLogger("ConceptMapper")
        
        # Build semantic frequency maps
        self.concept_frequencies = self._build_semantic_frequency_map()
        self.emotion_frequencies = self._build_emotion_frequency_map()
        self.abstract_frequencies = self._build_abstract_frequency_map()
        
    def _build_semantic_frequency_map(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Map core concepts to (base_freq, harmonics, phase) patterns.
        Based on synesthetic research and cognitive linguistics.
        """
        
        # Low frequencies (0-0.2): Basic concepts, physical world
        basic_concepts = {
            # Physical objects - very low frequencies
            'object': (0.05, 2, 0.0),
            'thing': (0.05, 2, 0.1),
            'item': (0.05, 2, 0.2),
            
            # Body and senses - low frequencies
            'body': (0.08, 3, 0.0),
            'hand': (0.08, 3, 0.2),
            'eye': (0.09, 3, 0.4),
            'ear': (0.09, 3, 0.6),
            'mouth': (0.08, 3, 0.8),
            
            # Basic actions - low-mid frequencies  
            'move': (0.12, 2, 0.0),
            'go': (0.12, 2, 0.3),
            'come': (0.12, 2, 0.6),
            'take': (0.13, 2, 0.0),
            'give': (0.13, 2, 0.5),
            'make': (0.14, 2, 0.0),
            'do': (0.14, 2, 0.3),
        }
        
        # Mid-low frequencies (0.2-0.4): Social concepts, relationships
        social_concepts = {
            'person': (0.22, 3, 0.0),
            'people': (0.22, 4, 0.2),
            'human': (0.22, 3, 0.4),
            'friend': (0.25, 3, 0.0),
            'family': (0.25, 4, 0.3),
            'love': (0.28, 5, 0.0),
            'like': (0.28, 3, 0.5),
            'help': (0.30, 3, 0.0),
            'care': (0.30, 4, 0.3),
            'trust': (0.32, 3, 0.0),
            'respect': (0.32, 4, 0.4),
        }
        
        # Mid frequencies (0.4-0.6): Mental processes, cognition
        mental_concepts = {
            'think': (0.42, 4, 0.0),
            'know': (0.43, 3, 0.0),
            'understand': (0.44, 5, 0.0),
            'learn': (0.45, 4, 0.0),
            'remember': (0.46, 4, 0.3),
            'forget': (0.46, 4, 0.7),
            'believe': (0.48, 3, 0.0),
            'feel': (0.50, 5, 0.0),
            'emotion': (0.50, 6, 0.2),
            'mind': (0.52, 4, 0.0),
            'idea': (0.53, 3, 0.0),
            'thought': (0.54, 4, 0.0),
        }
        
        # Mid-high frequencies (0.6-0.8): Abstract concepts, language
        abstract_concepts = {
            'word': (0.62, 3, 0.0),
            'language': (0.63, 4, 0.0),
            'meaning': (0.64, 5, 0.0),
            'truth': (0.66, 3, 0.0),
            'reality': (0.67, 4, 0.0),
            'existence': (0.68, 5, 0.0),
            'time': (0.70, 4, 0.0),
            'space': (0.71, 4, 0.5),
            'infinity': (0.72, 6, 0.0),
            'concept': (0.74, 4, 0.0),
            'theory': (0.75, 3, 0.0),
            'system': (0.76, 4, 0.0),
        }
        
        # High frequencies (0.8-1.0): Meta-concepts, consciousness
        meta_concepts = {
            'consciousness': (0.82, 7, 0.0),
            'awareness': (0.83, 6, 0.0),
            'being': (0.84, 5, 0.0),
            'self': (0.85, 4, 0.0),
            'identity': (0.86, 5, 0.0),
            'soul': (0.87, 6, 0.0),
            'spirit': (0.87, 6, 0.3),
            'wisdom': (0.88, 5, 0.0),
            'enlightenment': (0.90, 8, 0.0),
            'transcendence': (0.92, 9, 0.0),
            'unity': (0.94, 7, 0.0),
            'oneness': (0.96, 10, 0.0),
        }
        
        # Combine all maps
        frequency_map = {}
        frequency_map.update(basic_concepts)
        frequency_map.update(social_concepts)  
        frequency_map.update(mental_concepts)
        frequency_map.update(abstract_concepts)
        frequency_map.update(meta_concepts)
        
        return frequency_map
    
    def _build_emotion_frequency_map(self) -> Dict[str, Tuple[float, float, float]]:
        """Map emotions to specific frequency patterns."""
        
        return {
            # Basic emotions - distinct frequency signatures
            'joy': (0.35, 7, 0.0),      # Bright, many harmonics
            'happy': (0.35, 6, 0.1),
            'glad': (0.35, 5, 0.2),
            
            'sad': (0.25, 2, np.pi),     # Low, few harmonics, phase inversion
            'sorrow': (0.24, 2, np.pi),
            'grief': (0.23, 1, np.pi),
            
            'angry': (0.18, 3, 0.7),     # Low, sharp harmonics
            'mad': (0.18, 3, 0.8),
            'furious': (0.17, 4, 0.9),
            
            'fear': (0.40, 8, 0.9),      # Mid-high, many chaotic harmonics  
            'afraid': (0.40, 7, 0.8),
            'scared': (0.39, 6, 0.7),
            
            'calm': (0.30, 1, 0.0),      # Clean, single tone
            'peace': (0.31, 1, 0.0),
            'serene': (0.32, 1, 0.0),
            
            'excited': (0.45, 9, 0.2),   # High energy, many harmonics
            'enthusiasm': (0.46, 8, 0.3),
            
            'confused': (0.50, 12, 0.5), # Many conflicting harmonics
            'uncertain': (0.51, 10, 0.6),
            
            'curious': (0.60, 5, 0.1),   # Higher frequency, moderate harmonics
            'wonder': (0.61, 6, 0.2),
        }
    
    def _build_abstract_frequency_map(self) -> Dict[str, Tuple[float, float, float]]:
        """Map abstract concepts to frequency patterns."""
        
        return {
            # Question words - specific high frequencies for inquiry
            'what': (0.78, 4, 0.0),
            'why': (0.79, 4, 0.2),
            'how': (0.80, 4, 0.4), 
            'when': (0.77, 4, 0.6),
            'where': (0.76, 4, 0.8),
            'who': (0.75, 4, 1.0),
            
            # Modal verbs - mid-high frequencies  
            'can': (0.65, 3, 0.0),
            'could': (0.65, 3, 0.2),
            'may': (0.66, 3, 0.0),
            'might': (0.66, 3, 0.2),
            'will': (0.67, 3, 0.0),
            'would': (0.67, 3, 0.2),
            'should': (0.68, 3, 0.0),
            'must': (0.69, 3, 0.0),
            
            # Logical connectors - specific patterns for reasoning
            'and': (0.33, 2, 0.0),       # Conjunction - combining
            'or': (0.34, 2, np.pi/2),    # Disjunction - alternatives  
            'but': (0.35, 2, np.pi),     # Contrast - opposition
            'because': (0.55, 3, 0.0),   # Causation - reasoning
            'therefore': (0.56, 3, 0.2),
            'however': (0.57, 3, np.pi/2),
            
            # Quantities and measures
            'all': (0.15, 1, 0.0),       # Totality - low, pure
            'some': (0.38, 3, 0.0),      # Partial - mid range
            'none': (0.05, 0, 0.0),      # Absence - very low
            'many': (0.40, 5, 0.0),      # Multiplicity - many harmonics
            'few': (0.36, 2, 0.0),       # Scarcity - few harmonics
        }
    
    def concept_to_waveform(self, concept: str) -> np.ndarray:
        """
        Convert a single concept to a meaningful color waveform.
        Uses semantic frequency mapping rather than random hashing.
        """
        
        concept = concept.lower().strip()
        
        # Check direct mappings first
        freq_params = None
        
        if concept in self.concept_frequencies:
            freq_params = self.concept_frequencies[concept]
        elif concept in self.emotion_frequencies:
            freq_params = self.emotion_frequencies[concept]  
        elif concept in self.abstract_frequencies:
            freq_params = self.abstract_frequencies[concept]
        
        if freq_params:
            base_freq, num_harmonics, phase_offset = freq_params
            return self._generate_harmonic_waveform(base_freq, num_harmonics, phase_offset)
        
        # Handle unknown words by semantic similarity
        return self._generate_semantic_waveform(concept)
    
    def _generate_harmonic_waveform(self, base_freq: float, num_harmonics: int, phase_offset: float) -> np.ndarray:
        """Generate a waveform with specified harmonic structure."""
        
        waveform = np.zeros(self.spectrum_dims, dtype=complex)
        
        # Base frequency
        base_idx = int(base_freq * self.spectrum_dims)
        waveform[base_idx] = 1.0 * np.exp(1j * phase_offset)
        
        # Add harmonics with decreasing amplitude
        for h in range(2, num_harmonics + 1):
            harmonic_idx = int((base_freq * h) % 1.0 * self.spectrum_dims)
            amplitude = 1.0 / h  # Harmonic rolloff
            phase = phase_offset + h * 0.1  # Phase evolution
            waveform[harmonic_idx] += amplitude * np.exp(1j * phase)
        
        # Normalize
        norm = np.linalg.norm(waveform)
        if norm > 0:
            waveform /= norm
            
        return waveform
    
    def _generate_semantic_waveform(self, concept: str) -> np.ndarray:
        """
        Generate semantically meaningful waveform for unknown concepts
        based on word structure and characteristics.
        """
        
        # Analyze word characteristics
        word_length = len(concept)
        vowel_count = sum(1 for c in concept if c in 'aeiou')
        consonant_count = word_length - vowel_count
        
        # Map characteristics to frequency parameters
        # Longer words -> lower frequencies (more complex concepts)
        base_freq = max(0.1, 0.8 - (word_length - 3) * 0.05)
        
        # More vowels -> more harmonics (flowing, musical)
        num_harmonics = min(8, max(1, vowel_count + 1))
        
        # Consonant pattern affects phase
        consonant_hash = sum(ord(c) for c in concept if c not in 'aeiou')
        phase_offset = (consonant_hash % 100) / 100.0 * 2 * np.pi
        
        # Check for semantic cues in the word
        if any(prefix in concept for prefix in ['un', 'dis', 'anti', 'non']):
            phase_offset += np.pi  # Negation -> phase inversion
            
        if concept.endswith('ly'):  # Adverbs -> higher frequencies
            base_freq += 0.1
            
        if concept.endswith('ing'):  # Present participles -> dynamic patterns
            num_harmonics += 2
            
        if concept.endswith('ed'):  # Past tense -> lower energy
            base_freq -= 0.05
            num_harmonics = max(1, num_harmonics - 1)
        
        return self._generate_harmonic_waveform(base_freq, num_harmonics, phase_offset)
    
    def text_to_waveform(self, text: str) -> np.ndarray:
        """
        Convert text to color waveform through semantic interference patterns.
        Each word contributes its frequency pattern through wave interference.
        """
        
        # Clean and tokenize text
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return np.zeros(self.spectrum_dims, dtype=complex)
        
        # Start with first word
        combined_waveform = self.concept_to_waveform(words[0]).copy()
        
        # Add interference from subsequent words
        for word in words[1:]:
            word_waveform = self.concept_to_waveform(word)
            combined_waveform = self._interference_combine(combined_waveform, word_waveform)
        
        return combined_waveform
    
    def _interference_combine(self, wave1: np.ndarray, wave2: np.ndarray) -> np.ndarray:
        """
        Combine two waveforms through constructive/destructive interference.
        This is how concepts interact in color space.
        """
        
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
        
        # Normalize to prevent amplitude explosion
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
            
        return result
    
    def waveform_to_concepts(self, waveform: np.ndarray, num_concepts: int = 5) -> List[str]:
        """
        Decode a color waveform back to the most likely concepts.
        Prioritizes concrete, meaningful terms over abstract function words.
        """
        
        concepts = []
        amplitudes = np.abs(waveform)
        phases = np.angle(waveform)
        
        # Find dominant frequency components
        dominant_indices = np.argsort(amplitudes)[-num_concepts*5:][::-1]
        
        # Convert frequencies back to concepts with prioritization
        concept_candidates = []
        
        for idx in dominant_indices:
            freq = idx / self.spectrum_dims
            amplitude = amplitudes[idx]
            phase = phases[idx]
            
            # Skip very low amplitude components
            if amplitude < 0.05:
                continue
            
            # Find closest matching concept
            best_concept = self._frequency_to_concept(freq, amplitude, phase)
            
            if best_concept and best_concept not in [c[0] for c in concept_candidates]:
                # Calculate concept priority (concrete concepts get higher priority)
                priority = self._calculate_concept_priority(best_concept, amplitude)
                concept_candidates.append((best_concept, priority, amplitude))
        
        # Sort by priority (concrete concepts first), then by amplitude
        concept_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Extract top concepts
        concepts = [concept for concept, _, _ in concept_candidates[:num_concepts]]
        
        return concepts
    
    def _frequency_to_concept(self, freq: float, amplitude: float, phase: float) -> Optional[str]:
        """Find the concept that best matches a given frequency signature."""
        
        best_match = None
        best_distance = float('inf')
        
        # Search all concept mappings
        all_concepts = {**self.concept_frequencies, **self.emotion_frequencies, **self.abstract_frequencies}
        
        for concept, (base_freq, num_harmonics, phase_offset) in all_concepts.items():
            
            # Calculate frequency distance
            freq_dist = abs(freq - base_freq)
            
            # Calculate phase distance (circular)
            phase_dist = min(abs(phase - phase_offset), 2*np.pi - abs(phase - phase_offset))
            
            # Weighted distance
            total_dist = freq_dist + 0.3 * phase_dist
            
            if total_dist < best_distance:
                best_distance = total_dist
                best_match = concept
        
        # Only return if reasonably close
        return best_match if best_distance < 0.15 else None
    
    def get_concept_similarity(self, concept1: str, concept2: str) -> float:
        """
        Calculate semantic similarity between two concepts based on their
        frequency patterns in color space.
        """
        
        wave1 = self.concept_to_waveform(concept1)
        wave2 = self.concept_to_waveform(concept2)
        
        # Calculate cross-correlation as similarity measure
        correlation = np.abs(np.vdot(wave1, wave2))
        
        return float(correlation)
    
    def _calculate_concept_priority(self, concept: str, amplitude: float) -> float:
        """
        Calculate priority for a concept, favoring concrete nouns and action verbs
        over abstract function words.
        """
        
        # High priority: concrete nouns and domain-specific terms
        high_priority_concepts = {
            # Science/Physics
            'gravity', 'force', 'energy', 'motion', 'speed', 'acceleration', 'mass', 'weight',
            'falling', 'jumping', 'moving', 'push', 'pull', 'pressure', 'temperature', 'light',
            'sound', 'wave', 'frequency', 'physics', 'science', 'experiment', 'theory',
            
            # Actions/Verbs
            'explain', 'describe', 'show', 'teach', 'learn', 'understand', 'analyze', 'solve',
            'create', 'build', 'design', 'calculate', 'measure', 'observe', 'discover',
            'jump', 'fall', 'move', 'run', 'walk', 'climb', 'fly', 'swim',
            
            # Body/Biology
            'body', 'brain', 'heart', 'hand', 'eye', 'muscle', 'bone', 'blood', 'cell',
            'life', 'living', 'organism', 'animal', 'human', 'person', 'health',
            
            # Technology/Computing
            'computer', 'program', 'software', 'data', 'algorithm', 'code', 'system',
            'network', 'internet', 'machine', 'robot', 'artificial', 'intelligence',
            
            # Emotions (concrete feelings)
            'happy', 'sad', 'angry', 'excited', 'calm', 'nervous', 'curious', 'surprised'
        }
        
        # Medium priority: abstract but meaningful concepts
        medium_priority_concepts = {
            'consciousness', 'awareness', 'thought', 'idea', 'concept', 'meaning', 'truth',
            'reality', 'existence', 'being', 'mind', 'soul', 'spirit', 'wisdom',
            'love', 'care', 'trust', 'respect', 'friendship', 'family', 'relationship'
        }
        
        # Low priority: function words and abstract connectors
        low_priority_concepts = {
            'who', 'what', 'when', 'where', 'how', 'why',  # Question words are abstract
            'and', 'or', 'but', 'because', 'therefore', 'however', 'could', 'would',
            'should', 'might', 'may', 'can', 'will', 'must', 'all', 'some', 'none',
            'many', 'few', 'thing', 'item', 'object'
        }
        
        concept_lower = concept.lower()
        
        if concept_lower in high_priority_concepts:
            return 3.0 + amplitude  # Highest priority
        elif concept_lower in medium_priority_concepts:
            return 2.0 + amplitude  # Medium priority  
        elif concept_lower in low_priority_concepts:
            return 0.5 + amplitude  # Lowest priority
        else:
            # Unknown concept - medium priority
            return 1.5 + amplitude