#!/usr/bin/env python3
"""
ConceptExtractor: Extract language-independent concepts from LLM safetensor files.

This module implements Phase 1.1 of the Language-Independent Conceptual Frequency
Encoding System. It processes LLM safetensor files to extract semantic embeddings
and groups them into language-independent concept clusters.

CRITICAL: No mock fallbacks - system must fail properly when errors occur.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import hashlib

# Safetensors and transformers for loading model data
try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Clustering for concept grouping
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ConceptCluster:
    """A cluster of tokens representing the same language-independent concept."""
    concept_id: str
    tokens: List[str]
    languages: Set[str]
    embedding_centroid: np.ndarray
    intra_cluster_similarity: float
    token_frequencies: Dict[str, float]

class ConceptExtractor:
    """
    Extract language-independent concepts from LLM safetensor embeddings.

    This class processes safetensor files to identify semantic concepts that
    are represented consistently across different languages, forming the
    foundation for frequency-based concept encoding.
    """

    def __init__(self,
                 similarity_threshold: float = 0.85,
                 min_cluster_size: int = 2,
                 max_clusters: int = 10000):
        """
        Initialize ConceptExtractor.

        Args:
            similarity_threshold: Minimum cosine similarity for concept clustering
            min_cluster_size: Minimum tokens per concept cluster
            max_clusters: Maximum number of concept clusters to create
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.logger = logging.getLogger("ConceptExtractor")

        # Check dependencies - fail hard if not available
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors library required: pip install safetensors")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required: pip install transformers")
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn library required: pip install scikit-learn")

        self.logger.info(f"ConceptExtractor initialized:")
        self.logger.info(f"  Similarity threshold: {similarity_threshold}")
        self.logger.info(f"  Min cluster size: {min_cluster_size}")
        self.logger.info(f"  Max clusters: {max_clusters}")

    def extract_embeddings(self, safetensor_files: List[Path]) -> Dict[str, np.ndarray]:
        """
        Extract token embeddings from safetensor files.

        Args:
            safetensor_files: List of paths to safetensor files

        Returns:
            Dictionary mapping token_id -> embedding vector

        Raises:
            FileNotFoundError: If safetensor files don't exist
            RuntimeError: If embedding extraction fails
        """
        if not safetensor_files:
            raise ValueError("No safetensor files provided")

        # Verify all files exist
        for file_path in safetensor_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Safetensor file not found: {file_path}")

        self.logger.info(f"Extracting embeddings from {len(safetensor_files)} safetensor files...")

        try:
            # Load tokenizer to get vocabulary
            tokenizer_path = safetensor_files[0].parent
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            vocab = tokenizer.get_vocab()

            self.logger.info(f"Loaded tokenizer with {len(vocab)} tokens")

            # Extract embedding weights from safetensor files
            embeddings = {}
            embedding_tensor_found = False

            for file_path in safetensor_files:
                self.logger.info(f"Processing {file_path.name}...")

                with safe_open(file_path, framework="np") as f:
                    tensor_names = f.keys()

                    # Look for embedding tensor (common names)
                    embedding_tensor_name = None
                    for name in tensor_names:
                        if any(embed_key in name.lower() for embed_key in
                              ['embed_tokens.weight', 'embeddings.weight', 'word_embeddings.weight']):
                            embedding_tensor_name = name
                            break

                    if embedding_tensor_name:
                        embedding_matrix = f.get_tensor(embedding_tensor_name)
                        embedding_tensor_found = True

                        self.logger.info(f"Found embedding tensor '{embedding_tensor_name}' with shape {embedding_matrix.shape}")

                        # Map token indices to embeddings
                        for token, token_id in vocab.items():
                            if token_id < embedding_matrix.shape[0]:
                                embeddings[token] = embedding_matrix[token_id].copy()

                        break  # Only need embeddings from one file

            if not embedding_tensor_found:
                raise RuntimeError("No embedding tensor found in safetensor files")

            if not embeddings:
                raise RuntimeError("No embeddings extracted from safetensor files")

            self.logger.info(f"Successfully extracted {len(embeddings)} token embeddings")
            return embeddings

        except Exception as e:
            self.logger.error(f"Failed to extract embeddings: {e}")
            # NO MOCK FALLBACK - let it fail properly
            raise RuntimeError(f"Embedding extraction failed: {e}") from e

    def cluster_concepts(self, embeddings: Dict[str, np.ndarray]) -> List[ConceptCluster]:
        """
        Group embeddings into language-independent concept clusters.

        Args:
            embeddings: Dictionary of token -> embedding vector

        Returns:
            List of ConceptCluster objects representing language-independent concepts

        Raises:
            RuntimeError: If clustering fails
        """
        if not embeddings:
            raise ValueError("No embeddings provided for clustering")

        self.logger.info(f"Clustering {len(embeddings)} embeddings into concepts...")

        try:
            # Prepare data for clustering
            tokens = list(embeddings.keys())
            embedding_matrix = np.array([embeddings[token] for token in tokens])

            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            embedding_matrix = embedding_matrix / (norms + 1e-8)

            # Determine number of clusters
            num_clusters = min(self.max_clusters, len(tokens) // self.min_cluster_size)
            if num_clusters < 1:
                raise RuntimeError(f"Too few tokens ({len(tokens)}) for clustering")

            self.logger.info(f"Running KMeans clustering with {num_clusters} clusters...")

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embedding_matrix)

            # Build concept clusters
            concept_clusters = []
            cluster_groups = defaultdict(list)

            # Group tokens by cluster
            for token, cluster_id in zip(tokens, cluster_labels):
                cluster_groups[cluster_id].append(token)

            # Create ConceptCluster objects
            for cluster_id, cluster_tokens in cluster_groups.items():
                if len(cluster_tokens) < self.min_cluster_size:
                    continue  # Skip small clusters

                # Calculate cluster statistics
                cluster_embeddings = [embeddings[token] for token in cluster_tokens]
                centroid = np.mean(cluster_embeddings, axis=0)

                # Calculate intra-cluster similarity
                similarities = []
                for i, emb1 in enumerate(cluster_embeddings):
                    for j, emb2 in enumerate(cluster_embeddings[i+1:], i+1):
                        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        similarities.append(sim)

                avg_similarity = np.mean(similarities) if similarities else 0.0

                # Detect languages (simple heuristic)
                languages = self._detect_token_languages(cluster_tokens)

                # Calculate token frequencies (uniform for now)
                token_frequencies = {token: 1.0 / len(cluster_tokens) for token in cluster_tokens}

                # Create concept ID
                concept_id = self._generate_concept_id(cluster_tokens, cluster_id)

                concept_cluster = ConceptCluster(
                    concept_id=concept_id,
                    tokens=cluster_tokens,
                    languages=languages,
                    embedding_centroid=centroid,
                    intra_cluster_similarity=avg_similarity,
                    token_frequencies=token_frequencies
                )

                concept_clusters.append(concept_cluster)

            # Sort by cluster quality (similarity)
            concept_clusters.sort(key=lambda c: c.intra_cluster_similarity, reverse=True)

            self.logger.info(f"Created {len(concept_clusters)} concept clusters")
            self.logger.info(f"Average intra-cluster similarity: {np.mean([c.intra_cluster_similarity for c in concept_clusters]):.3f}")

            return concept_clusters

        except Exception as e:
            self.logger.error(f"Concept clustering failed: {e}")
            # NO MOCK FALLBACK - let it fail properly
            raise RuntimeError(f"Concept clustering failed: {e}") from e

    def _detect_token_languages(self, tokens: List[str]) -> Set[str]:
        """
        Simple heuristic to detect languages represented in token list.

        Args:
            tokens: List of tokens to analyze

        Returns:
            Set of detected language codes
        """
        languages = set()

        for token in tokens:
            # Check for non-Latin scripts
            if any(ord(char) > 127 for char in token):
                if any(0x0370 <= ord(char) <= 0x03FF for char in token):  # Greek
                    languages.add('el')
                elif any(0x0400 <= ord(char) <= 0x04FF for char in token):  # Cyrillic
                    languages.add('ru')
                elif any(0x4E00 <= ord(char) <= 0x9FFF for char in token):  # Chinese
                    languages.add('zh')
                elif any(0x0590 <= ord(char) <= 0x05FF for char in token):  # Hebrew
                    languages.add('he')
                elif any(0x0600 <= ord(char) <= 0x06FF for char in token):  # Arabic
                    languages.add('ar')
                else:
                    languages.add('unknown')
            else:
                # Simple heuristics for Latin-script languages
                if any(suffix in token.lower() for suffix in ['tion', 'sion', 'ing']):
                    languages.add('en')
                elif any(suffix in token.lower() for suffix in ['ment', 'ique', 'eur']):
                    languages.add('fr')
                elif any(char in token.lower() for char in ['ä', 'ö', 'ü', 'ß']):
                    languages.add('de')
                elif any(suffix in token.lower() for suffix in ['ción', 'miento', 'dad']):
                    languages.add('es')
                elif any(suffix in token.lower() for suffix in ['zione', 'mento', 'tà']):
                    languages.add('it')
                else:
                    languages.add('unknown')

        return languages if languages else {'unknown'}

    def _generate_concept_id(self, tokens: List[str], cluster_id: int) -> str:
        """
        Generate unique concept ID for a cluster.

        Args:
            tokens: Tokens in the cluster
            cluster_id: Cluster ID from clustering algorithm

        Returns:
            Unique concept identifier
        """
        # Use most frequent token as base, or first alphabetically
        base_token = sorted(tokens)[0] if tokens else "unknown"

        # Create hash of all tokens for uniqueness
        token_hash = hashlib.md5("|".join(sorted(tokens)).encode()).hexdigest()[:8]

        return f"concept_{cluster_id:04d}_{base_token[:10]}_{token_hash}"

    def validate_concept_clusters(self, concept_clusters: List[ConceptCluster]) -> Dict[str, float]:
        """
        Validate quality of concept clusters.

        Args:
            concept_clusters: List of concept clusters to validate

        Returns:
            Dictionary of validation metrics
        """
        if not concept_clusters:
            return {"error": "No concept clusters to validate"}

        metrics = {}

        # Average cluster size
        cluster_sizes = [len(c.tokens) for c in concept_clusters]
        metrics['avg_cluster_size'] = np.mean(cluster_sizes)
        metrics['min_cluster_size'] = min(cluster_sizes)
        metrics['max_cluster_size'] = max(cluster_sizes)

        # Average similarity
        similarities = [c.intra_cluster_similarity for c in concept_clusters]
        metrics['avg_similarity'] = np.mean(similarities)
        metrics['min_similarity'] = min(similarities)
        metrics['max_similarity'] = max(similarities)

        # Language coverage
        all_languages = set()
        for cluster in concept_clusters:
            all_languages.update(cluster.languages)
        metrics['total_languages'] = len(all_languages)

        # Multi-language clusters (good for language independence)
        multi_lang_clusters = sum(1 for c in concept_clusters if len(c.languages) > 1)
        metrics['multi_language_clusters'] = multi_lang_clusters
        metrics['multi_language_ratio'] = multi_lang_clusters / len(concept_clusters)

        self.logger.info(f"Validation metrics: {metrics}")
        return metrics


def create_concept_extractor(similarity_threshold: float = 0.85,
                           min_cluster_size: int = 2,
                           max_clusters: int = 10000) -> ConceptExtractor:
    """
    Factory function to create ConceptExtractor with specified parameters.

    Args:
        similarity_threshold: Minimum cosine similarity for concept clustering
        min_cluster_size: Minimum tokens per concept cluster
        max_clusters: Maximum number of concept clusters to create

    Returns:
        Configured ConceptExtractor instance
    """
    return ConceptExtractor(
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size,
        max_clusters=max_clusters
    )