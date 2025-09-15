#!/usr/bin/env python3
"""
FrequencyMapper: Map concept clusters to N-dimensional frequency signatures.

This module implements Phase 1.2 of the Language-Independent Conceptual Frequency
Encoding System. It converts concept clusters into rich frequency signatures
containing primary frequency, harmonics, amplitude, phase, bandwidth, and modulation.

CRITICAL: No mock fallbacks - system must fail properly when errors occur.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Import ConceptCluster from Phase 1.1
from .concept_extractor import ConceptCluster

logger = logging.getLogger(__name__)

@dataclass
class FrequencySignature:
    """N-dimensional frequency signature representing a language-independent concept."""
    concept_id: str
    primary_frequency: float  # Hz - Core concept frequency
    harmonics: List[float]    # Hz - Related sub-concepts (up to 16 harmonics)
    amplitude: float          # Concept strength/importance
    phase: float             # radians - Conceptual relationships
    bandwidth: float         # Hz - Concept specificity (narrower = more specific)
    modulation: float        # Dynamic conceptual properties
    dimensions: int          # Number of frequency dimensions
    frequency_vector: np.ndarray  # N-dimensional frequency representation

class FrequencyMapper:
    """
    Map concept clusters to N-dimensional frequency signatures.

    This class transforms concept clusters from Phase 1.1 into rich frequency
    representations that capture semantic relationships in continuous space.
    """

    def __init__(self,
                 frequency_dimensions: int = 512,
                 frequency_range: Tuple[float, float] = (1.0, 1000.0),
                 max_harmonics: int = 16,
                 concept_spacing: float = 10.0):
        """
        Initialize FrequencyMapper.

        Args:
            frequency_dimensions: Number of dimensions in frequency space
            frequency_range: (min_freq, max_freq) in Hz for mapping
            max_harmonics: Maximum number of harmonics per concept
            concept_spacing: Minimum frequency separation between concepts (Hz)
        """
        self.frequency_dimensions = frequency_dimensions
        self.min_freq, self.max_freq = frequency_range
        self.max_harmonics = max_harmonics
        self.concept_spacing = concept_spacing
        self.logger = logging.getLogger("FrequencyMapper")

        # Validate parameters
        if frequency_dimensions <= 0:
            raise ValueError(f"Invalid frequency dimensions: {frequency_dimensions}")
        if self.min_freq >= self.max_freq:
            raise ValueError(f"Invalid frequency range: {frequency_range}")
        if max_harmonics <= 0:
            raise ValueError(f"Invalid max harmonics: {max_harmonics}")

        self.logger.info(f"FrequencyMapper initialized:")
        self.logger.info(f"  Frequency dimensions: {frequency_dimensions}")
        self.logger.info(f"  Frequency range: {self.min_freq:.1f} - {self.max_freq:.1f} Hz")
        self.logger.info(f"  Max harmonics: {max_harmonics}")
        self.logger.info(f"  Concept spacing: {concept_spacing:.1f} Hz")

    def map_concepts_to_frequencies(self, concept_clusters: List[ConceptCluster]) -> List[FrequencySignature]:
        """
        Map concept clusters to frequency signatures.

        Args:
            concept_clusters: List of concept clusters from Phase 1.1

        Returns:
            List of FrequencySignature objects representing concepts in frequency space

        Raises:
            RuntimeError: If frequency mapping fails
        """
        if not concept_clusters:
            raise ValueError("No concept clusters provided for frequency mapping")

        self.logger.info(f"Mapping {len(concept_clusters)} concept clusters to frequencies...")

        try:
            frequency_signatures = []
            allocated_frequencies = set()

            for i, cluster in enumerate(concept_clusters):
                # Generate primary frequency based on concept characteristics
                primary_freq = self._generate_primary_frequency(cluster, allocated_frequencies)
                allocated_frequencies.add(primary_freq)

                # Generate harmonics based on concept relationships
                harmonics = self._generate_harmonics(cluster, primary_freq)

                # Calculate amplitude from cluster quality
                amplitude = self._calculate_amplitude(cluster)

                # Generate phase from language diversity
                phase = self._calculate_phase(cluster)

                # Calculate bandwidth from concept specificity
                bandwidth = self._calculate_bandwidth(cluster)

                # Generate modulation from concept dynamics
                modulation = self._calculate_modulation(cluster)

                # Create N-dimensional frequency vector
                frequency_vector = self._create_frequency_vector(
                    primary_freq, harmonics, amplitude, phase, bandwidth, modulation
                )

                # Create frequency signature
                signature = FrequencySignature(
                    concept_id=cluster.concept_id,
                    primary_frequency=primary_freq,
                    harmonics=harmonics,
                    amplitude=amplitude,
                    phase=phase,
                    bandwidth=bandwidth,
                    modulation=modulation,
                    dimensions=self.frequency_dimensions,
                    frequency_vector=frequency_vector
                )

                frequency_signatures.append(signature)

            self.logger.info(f"Successfully mapped {len(frequency_signatures)} concepts to frequencies")
            self.logger.info(f"Frequency range used: {min(s.primary_frequency for s in frequency_signatures):.1f} - {max(s.primary_frequency for s in frequency_signatures):.1f} Hz")

            return frequency_signatures

        except Exception as e:
            self.logger.error(f"Frequency mapping failed: {e}")
            # NO MOCK FALLBACK - let it fail properly
            raise RuntimeError(f"Frequency mapping failed: {e}") from e

    def _generate_primary_frequency(self, cluster: ConceptCluster, allocated_frequencies: set) -> float:
        """
        Generate primary frequency for a concept cluster.

        Args:
            cluster: ConceptCluster to map
            allocated_frequencies: Set of already allocated frequencies

        Returns:
            Primary frequency in Hz
        """
        # Use cluster centroid for deterministic frequency assignment
        centroid_hash = hash(cluster.concept_id) % 1000000
        base_freq = self.min_freq + (centroid_hash / 1000000) * (self.max_freq - self.min_freq)

        # Adjust for concept quality (higher similarity = higher frequency)
        quality_factor = cluster.intra_cluster_similarity
        freq_adjustment = quality_factor * 50.0  # Up to 50 Hz boost for high quality

        primary_freq = base_freq + freq_adjustment

        # Ensure minimum spacing from other concepts
        while any(abs(primary_freq - freq) < self.concept_spacing for freq in allocated_frequencies):
            primary_freq += self.concept_spacing

        # Keep within valid range
        primary_freq = np.clip(primary_freq, self.min_freq, self.max_freq)

        return primary_freq

    def _generate_harmonics(self, cluster: ConceptCluster, primary_freq: float) -> List[float]:
        """
        Generate harmonic frequencies for concept relationships.

        Args:
            cluster: ConceptCluster to analyze
            primary_freq: Primary frequency to base harmonics on

        Returns:
            List of harmonic frequencies
        """
        harmonics = []

        # Number of harmonics based on cluster size and languages
        num_harmonics = min(self.max_harmonics, len(cluster.tokens) + len(cluster.languages))

        for h in range(1, num_harmonics + 1):
            # Integer harmonics with slight detuning based on language diversity
            language_factor = len(cluster.languages) / 10.0  # Small detuning
            harmonic_freq = primary_freq * (h + 1) * (1.0 + language_factor * 0.01)

            # Keep harmonics within valid range
            if harmonic_freq <= self.max_freq:
                harmonics.append(harmonic_freq)
            else:
                break

        return harmonics

    def _calculate_amplitude(self, cluster: ConceptCluster) -> float:
        """
        Calculate amplitude from cluster quality and importance.

        Args:
            cluster: ConceptCluster to analyze

        Returns:
            Amplitude value (0.0 to 1.0)
        """
        # Base amplitude from cluster similarity
        base_amplitude = cluster.intra_cluster_similarity

        # Boost for larger clusters (more widespread concept)
        size_factor = min(1.0, len(cluster.tokens) / 10.0)

        # Boost for multilingual concepts
        language_factor = min(1.0, len(cluster.languages) / 5.0)

        amplitude = base_amplitude * (1.0 + 0.2 * size_factor + 0.3 * language_factor)
        return np.clip(amplitude, 0.0, 1.0)

    def _calculate_phase(self, cluster: ConceptCluster) -> float:
        """
        Calculate phase from language relationships.

        Args:
            cluster: ConceptCluster to analyze

        Returns:
            Phase in radians (0 to 2π)
        """
        # Phase based on language diversity pattern
        language_hash = hash(tuple(sorted(cluster.languages))) % 1000
        phase = (language_hash / 1000.0) * 2 * np.pi

        return phase

    def _calculate_bandwidth(self, cluster: ConceptCluster) -> float:
        """
        Calculate bandwidth from concept specificity.

        Args:
            cluster: ConceptCluster to analyze

        Returns:
            Bandwidth in Hz
        """
        # Higher similarity = narrower bandwidth (more specific concept)
        specificity = cluster.intra_cluster_similarity

        # Base bandwidth: 1-20 Hz
        base_bandwidth = 20.0
        bandwidth = base_bandwidth * (1.0 - specificity + 0.1)  # 0.1 minimum factor

        return max(1.0, bandwidth)

    def _calculate_modulation(self, cluster: ConceptCluster) -> float:
        """
        Calculate modulation from concept dynamics.

        Args:
            cluster: ConceptCluster to analyze

        Returns:
            Modulation factor (0.0 to 1.0)
        """
        # Modulation based on token frequency variance
        token_freqs = list(cluster.token_frequencies.values())
        if len(token_freqs) > 1:
            freq_variance = np.var(token_freqs)
            modulation = min(1.0, freq_variance * 10.0)  # Scale variance
        else:
            modulation = 0.1  # Low modulation for single-token concepts

        return modulation

    def _create_frequency_vector(self, primary_freq: float, harmonics: List[float],
                                amplitude: float, phase: float, bandwidth: float,
                                modulation: float) -> np.ndarray:
        """
        Create N-dimensional frequency vector representation.

        Args:
            primary_freq: Primary frequency
            harmonics: List of harmonic frequencies
            amplitude: Amplitude factor
            phase: Phase in radians
            bandwidth: Bandwidth in Hz
            modulation: Modulation factor

        Returns:
            N-dimensional frequency vector
        """
        # Create frequency vector using spectral decomposition
        frequency_vector = np.zeros(self.frequency_dimensions, dtype=np.complex128)

        # Map primary frequency to vector space
        primary_index = int((primary_freq - self.min_freq) / (self.max_freq - self.min_freq) * self.frequency_dimensions)
        primary_index = np.clip(primary_index, 0, self.frequency_dimensions - 1)

        # Set primary frequency component
        frequency_vector[primary_index] = amplitude * np.exp(1j * phase)

        # Add harmonics
        for harmonic_freq in harmonics:
            harmonic_index = int((harmonic_freq - self.min_freq) / (self.max_freq - self.min_freq) * self.frequency_dimensions)
            harmonic_index = np.clip(harmonic_index, 0, self.frequency_dimensions - 1)

            # Harmonic amplitude decreases with order
            harmonic_amplitude = amplitude * 0.5 * (1.0 / (1 + len(harmonics)))
            frequency_vector[harmonic_index] += harmonic_amplitude * np.exp(1j * (phase + np.pi/4))

        # Apply bandwidth smoothing (Gaussian envelope)
        bandwidth_sigma = bandwidth / 10.0  # Convert to sigma
        for i in range(self.frequency_dimensions):
            if np.abs(frequency_vector[i]) > 0:
                # Apply Gaussian envelope around frequency peaks
                distance = abs(i - primary_index)
                envelope = np.exp(-0.5 * (distance / bandwidth_sigma) ** 2)
                frequency_vector[i] *= envelope

        # Apply modulation (frequency spreading)
        if modulation > 0.1:
            modulation_noise = np.random.normal(0, modulation * 0.1, self.frequency_dimensions)
            frequency_vector *= (1.0 + modulation_noise)

        return frequency_vector

    def validate_frequency_signatures(self, frequency_signatures: List[FrequencySignature]) -> Dict[str, float]:
        """
        Validate quality of frequency signatures.

        Args:
            frequency_signatures: List of frequency signatures to validate

        Returns:
            Dictionary of validation metrics
        """
        if not frequency_signatures:
            return {"error": "No frequency signatures to validate"}

        metrics = {}

        # Frequency distribution
        primary_freqs = [sig.primary_frequency for sig in frequency_signatures]
        metrics['freq_range_min'] = min(primary_freqs)
        metrics['freq_range_max'] = max(primary_freqs)
        metrics['freq_mean'] = np.mean(primary_freqs)
        metrics['freq_std'] = np.std(primary_freqs)

        # Amplitude statistics
        amplitudes = [sig.amplitude for sig in frequency_signatures]
        metrics['amplitude_mean'] = np.mean(amplitudes)
        metrics['amplitude_std'] = np.std(amplitudes)

        # Harmonic richness
        harmonic_counts = [len(sig.harmonics) for sig in frequency_signatures]
        metrics['avg_harmonics'] = np.mean(harmonic_counts)
        metrics['max_harmonics'] = max(harmonic_counts) if harmonic_counts else 0

        # Frequency spacing (concept separation)
        if len(primary_freqs) > 1:
            sorted_freqs = sorted(primary_freqs)
            spacings = [sorted_freqs[i+1] - sorted_freqs[i] for i in range(len(sorted_freqs)-1)]
            metrics['min_spacing'] = min(spacings)
            metrics['avg_spacing'] = np.mean(spacings)
        else:
            metrics['min_spacing'] = float('inf')
            metrics['avg_spacing'] = float('inf')

        # Vector norm distribution
        vector_norms = [np.linalg.norm(sig.frequency_vector) for sig in frequency_signatures]
        metrics['vector_norm_mean'] = np.mean(vector_norms)
        metrics['vector_norm_std'] = np.std(vector_norms)

        self.logger.info(f"Frequency signature validation metrics: {metrics}")
        return metrics


def create_frequency_mapper(frequency_dimensions: int = 512,
                          frequency_range: Tuple[float, float] = (1.0, 1000.0),
                          max_harmonics: int = 16,
                          concept_spacing: float = 10.0) -> FrequencyMapper:
    """
    Factory function to create FrequencyMapper with specified parameters.

    Args:
        frequency_dimensions: Number of dimensions in frequency space
        frequency_range: (min_freq, max_freq) in Hz for mapping
        max_harmonics: Maximum number of harmonics per concept
        concept_spacing: Minimum frequency separation between concepts (Hz)

    Returns:
        Configured FrequencyMapper instance
    """
    return FrequencyMapper(
        frequency_dimensions=frequency_dimensions,
        frequency_range=frequency_range,
        max_harmonics=max_harmonics,
        concept_spacing=concept_spacing
    )