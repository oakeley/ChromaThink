#!/usr/bin/env python3
"""
Semantic EM Spectrum Encoder
Converts language-independent concepts to electromagnetic spectra.
Like text-to-speech but using EM spectra as the intermediate representation.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
import logging

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EMSpectrum:
    """Electromagnetic spectrum representation of a concept."""
    wavelength: float      # Primary wavelength in nanometers
    bandwidth: float       # Spectral width (concept specificity)
    intensity: float       # Concept strength/importance
    fine_structure: np.ndarray  # Detailed spectral profile

    def __post_init__(self):
        """Initialize fine structure if not provided."""
        if self.fine_structure is None:
            # Create Gaussian profile
            wavelengths = np.linspace(self.wavelength - self.bandwidth/2,
                                    self.wavelength + self.bandwidth/2, 50)
            center = self.wavelength
            sigma = self.bandwidth / 6  # 99.7% within bandwidth
            profile = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)
            self.fine_structure = profile * self.intensity

@dataclass
class ConceptCluster:
    """A cluster of multilingual tokens representing the same concept."""
    cluster_id: str
    tokens: List[str]
    languages: List[str]
    embedding_center: np.ndarray
    semantic_category: str
    assigned_spectrum: EMSpectrum

class SemanticEMEncoder:
    """
    Encodes concepts to electromagnetic spectra based on semantic meaning.
    Language-independent: 'love'=amour=liebe=agápi → same EM signature
    """

    def __init__(self, safetensor_path: str = "models/apertus"):
        self.safetensor_path = Path(safetensor_path)
        self.concept_clusters = {}
        self.spectrum_assignments = {}

        # Spectral region assignments by semantic category
        self.spectral_regions = {
            'physics': (400, 500),      # Blue-violet: precise concepts
            'emotion': (600, 700),      # Red-orange: human feelings
            'action': (500, 600),       # Green: dynamic concepts
            'abstract': (300, 400),     # UV: philosophical concepts
            'concrete': (700, 800),     # Near-IR: tangible objects
            'social': (550, 650),       # Yellow-orange: relationships
            'temporal': (450, 550),     # Blue-green: time concepts
        }

    def build_concept_dictionary(self) -> Dict[str, ConceptCluster]:
        """
        Build language-independent concept dictionary from safetensor embeddings.
        Groups multilingual tokens that represent the same meaning.
        """
        logger.info("Building language-independent concept dictionary...")

        # Step 1: Load embeddings from safetensors
        embeddings, tokens = self._load_embeddings_from_safetensors()
        logger.info(f"Loaded {len(tokens)} tokens with embeddings")

        # Step 2: Cluster embeddings to find semantic groups
        concept_clusters = self._cluster_semantic_embeddings(embeddings, tokens)
        logger.info(f"Found {len(concept_clusters)} concept clusters")

        # Step 3: Assign EM spectra to clusters
        logger.info(f"Assigning EM spectra to {len(concept_clusters)} clusters...")
        for i, cluster in enumerate(concept_clusters):
            if i % 200 == 0:
                logger.info(f"Assigned spectra to {i}/{len(concept_clusters)} clusters ({i/len(concept_clusters)*100:.1f}%)")
            spectrum = self._assign_em_spectrum(cluster)
            cluster.assigned_spectrum = spectrum

        self.concept_clusters = {cluster.cluster_id: cluster for cluster in concept_clusters}

        return self.concept_clusters

    def _load_embeddings_from_safetensors(self) -> Tuple[np.ndarray, List[str]]:
        """Load token embeddings from Apertus safetensor files."""
        try:
            # Direct loading from safetensors files
            import safetensors
            from transformers import AutoTokenizer
            import torch

            logger.info("Loading embeddings directly from safetensors...")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(self.safetensor_path))
            logger.info(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")

            # Find embedding tensor in safetensors files
            safetensor_files = list(self.safetensor_path.glob("*.safetensors"))
            embeddings = None

            for file_path in safetensor_files:
                with safetensors.safe_open(file_path, framework="pt") as f:
                    for key in f.keys():
                        if 'embed' in key.lower() and 'token' in key.lower():
                            logger.info(f"Found embedding tensor: {key} in {file_path.name}")
                            tensor = f.get_tensor(key)
                            # Convert BFloat16 to float32 for GPU compatibility
                            if tensor.dtype == torch.bfloat16:
                                tensor = tensor.to(torch.float32)
                            # Move to GPU for processing
                            tensor = tensor.to(device)
                            embeddings = tensor.cpu().numpy()
                            break
                    if embeddings is not None:
                        break

            if embeddings is None:
                # Try common embedding weight names
                for file_path in safetensor_files:
                    with safetensors.safe_open(file_path, framework="pt") as f:
                        for key in ['model.embed_tokens.weight', 'embed_tokens.weight',
                                   'embeddings.word_embeddings.weight', 'lm_head.weight']:
                            if key in f.keys():
                                logger.info(f"Found embedding tensor: {key} in {file_path.name}")
                                tensor = f.get_tensor(key)
                                # Convert BFloat16 to float32 for GPU compatibility
                                if tensor.dtype == torch.bfloat16:
                                    tensor = tensor.to(torch.float32)
                                # Move to GPU for processing
                                tensor = tensor.to(device)
                                embeddings = tensor.cpu().numpy()
                                break
                    if embeddings is not None:
                        break

            if embeddings is None:
                raise ValueError("No embedding tensors found in safetensor files")

            # Create token list
            vocab_size = min(embeddings.shape[0], tokenizer.vocab_size)
            tokens = []

            for i in range(vocab_size):
                try:
                    token = tokenizer.decode([i], skip_special_tokens=False)
                    if token and token.strip():
                        # Clean token for better processing
                        clean_token = token.strip().replace('▁', '').replace('<0x', '').replace('>', '')
                        if clean_token and not clean_token.startswith('<') and len(clean_token) > 0:
                            tokens.append(clean_token)
                        else:
                            tokens.append(f"token_{i}")
                    else:
                        tokens.append(f"token_{i}")
                except:
                    tokens.append(f"token_{i}")

            # Trim embeddings to match tokens
            embeddings = embeddings[:len(tokens)]

            logger.info(f"Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
            logger.info(f"Sample tokens: {tokens[:10]}")

            return embeddings, tokens

        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            import traceback
            traceback.print_exc()
            # Critical error - cannot proceed without real embeddings
            raise RuntimeError(f"Failed to load embeddings: {e}. Cannot use mock fallback.")

    def _create_enhanced_mock_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Create enhanced mock embeddings with better multilingual coverage."""
        logger.warning("Using enhanced mock embeddings for testing")

        # Create comprehensive multilingual test data
        mock_tokens = [
            # Love in many languages
            'love', 'amour', 'liebe', 'amore', 'amor', 'agápi', 'rakkaus',
            'lyubóv', 'ài', 'gaol', 'cariad', 'kärlek', 'kjærlighet',

            # Gravity/Force in many languages
            'gravity', 'gravité', 'gravitation', 'gravità', 'gravedad',
            'βαρύτητα', 'painovoima', 'gravitátsiya', 'zhònglì', 'imearacht',

            # Water in many languages
            'water', 'eau', 'wasser', 'acqua', 'agua', 'νερό', 'vesi',
            'voda', 'shuǐ', 'uisce', 'vatten', 'vann',

            # Actions
            'run', 'courir', 'laufen', 'correre', 'correr', 'τρέχω',
            'juosta', 'bежать', 'pǎo', 'rith',

            # Emotions
            'happy', 'heureux', 'glücklich', 'felice', 'feliz', 'χαρούμενος',
            'iloinen', 'счастливый', 'kuàilè', 'sona',

            # Abstract concepts
            'think', 'penser', 'denken', 'pensare', 'pensar', 'σκέπτομαι',
            'ajatella', 'думать', 'xiǎng', 'ceapadh',

            # Time concepts
            'time', 'temps', 'zeit', 'tempo', 'tiempo', 'χρόνος',
            'aika', 'время', 'shíjiān', 'am',

            # Physical objects
            'house', 'maison', 'haus', 'casa', 'casa', 'σπίτι',
            'talo', 'дом', 'fángzi', 'teach'
        ]

        # Create semantically meaningful embeddings
        embeddings = []
        embedding_dim = 256  # Reasonable dimension

        # Define base vectors for semantic categories
        base_vectors = {
            'love': np.array([1.0, 0.8, 0.2, -0.3, 0.9, 0.1, -0.2, 0.7] + [0.0] * (embedding_dim - 8)),
            'gravity': np.array([0.2, -0.8, 1.0, 0.6, -0.1, 0.9, 0.3, -0.4] + [0.0] * (embedding_dim - 8)),
            'water': np.array([-0.3, 0.5, 0.1, 0.8, 0.4, -0.6, 1.0, 0.2] + [0.0] * (embedding_dim - 8)),
            'run': np.array([0.6, 0.3, -0.5, 0.2, 1.0, -0.1, 0.4, 0.8] + [0.0] * (embedding_dim - 8)),
            'happy': np.array([0.9, 0.7, 0.5, 0.2, 0.8, 0.6, 0.4, 0.3] + [0.0] * (embedding_dim - 8)),
            'think': np.array([0.1, -0.2, 0.8, 0.9, 0.3, -0.7, 0.5, 0.6] + [0.0] * (embedding_dim - 8)),
            'time': np.array([-0.5, 0.3, 0.7, -0.2, 0.6, 0.8, -0.4, 0.1] + [0.0] * (embedding_dim - 8)),
            'house': np.array([0.4, 0.6, -0.2, 0.8, 0.1, 0.5, 0.9, -0.3] + [0.0] * (embedding_dim - 8))
        }

        for token in mock_tokens:
            # Determine semantic category
            if any(love_word in token.lower() for love_word in ['love', 'amour', 'liebe', 'amore', 'amor', 'agápi']):
                base_vector = base_vectors['love']
                noise_scale = 0.1
            elif any(gravity_word in token.lower() for gravity_word in ['gravity', 'gravité', 'gravitation', 'gravità']):
                base_vector = base_vectors['gravity']
                noise_scale = 0.1
            elif any(water_word in token.lower() for water_word in ['water', 'eau', 'wasser', 'acqua', 'agua']):
                base_vector = base_vectors['water']
                noise_scale = 0.1
            elif any(run_word in token.lower() for run_word in ['run', 'courir', 'laufen', 'correre', 'correr']):
                base_vector = base_vectors['run']
                noise_scale = 0.1
            elif any(happy_word in token.lower() for happy_word in ['happy', 'heureux', 'glücklich', 'felice']):
                base_vector = base_vectors['happy']
                noise_scale = 0.1
            elif any(think_word in token.lower() for think_word in ['think', 'penser', 'denken', 'pensare']):
                base_vector = base_vectors['think']
                noise_scale = 0.1
            elif any(time_word in token.lower() for time_word in ['time', 'temps', 'zeit', 'tempo']):
                base_vector = base_vectors['time']
                noise_scale = 0.1
            elif any(house_word in token.lower() for house_word in ['house', 'maison', 'haus', 'casa']):
                base_vector = base_vectors['house']
                noise_scale = 0.1
            else:
                # Random embedding for unknown tokens
                base_vector = np.random.normal(0, 0.3, embedding_dim)
                noise_scale = 0.2

            # Add noise to create variation while maintaining similarity
            noise = np.random.normal(0, noise_scale, embedding_dim)
            embedding = base_vector + noise

            # Normalize to prevent numerical issues
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            embeddings.append(embedding)

        embeddings = np.array(embeddings)
        logger.info(f"Created {len(embeddings)} enhanced mock embeddings with dimension {embeddings.shape[1]}")

        return embeddings, mock_tokens

    def _create_mock_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Create mock embeddings for testing."""
        logger.warning("Using mock embeddings for testing")

        # Create some example multilingual tokens
        mock_tokens = [
            # Love in different languages
            'love', 'amour', 'liebe', 'agápi', 'rakkaus', 'lyubóv', 'ài', 'gaol',
            # Gravity in different languages
            'gravity', 'gravité', 'gravitation', 'βαρύτητα', 'painovoima', 'gravitátsiya',
            # Water in different languages
            'water', 'eau', 'wasser', 'νερό', 'vesi', 'voda', 'shuǐ', 'uisce',
            # Random other tokens
            'happiness', 'bonheur', 'glück', 'computer', 'ordinateur', 'rechner'
        ]

        # Create embeddings where similar concepts have similar vectors
        embeddings = []
        for token in mock_tokens:
            if any(love_word in token.lower() for love_word in ['love', 'amour', 'liebe', 'agápi']):
                # Love cluster - similar embeddings
                base_vector = np.array([1.0, 0.8, 0.2, -0.3, 0.9, 0.1, -0.2, 0.7])
                noise = np.random.normal(0, 0.1, 8)
                embedding = base_vector + noise
            elif any(gravity_word in token.lower() for gravity_word in ['gravity', 'gravité', 'gravitation']):
                # Gravity cluster
                base_vector = np.array([0.2, -0.8, 1.0, 0.6, -0.1, 0.9, 0.3, -0.4])
                noise = np.random.normal(0, 0.1, 8)
                embedding = base_vector + noise
            elif any(water_word in token.lower() for water_word in ['water', 'eau', 'wasser', 'νερό']):
                # Water cluster
                base_vector = np.array([-0.3, 0.5, 0.1, 0.8, 0.4, -0.6, 1.0, 0.2])
                noise = np.random.normal(0, 0.1, 8)
                embedding = base_vector + noise
            else:
                # Random embedding
                embedding = np.random.normal(0, 0.5, 8)

            embeddings.append(embedding)

        return np.array(embeddings), mock_tokens

    def _cluster_semantic_embeddings(self, embeddings: np.ndarray, tokens: List[str]) -> List[ConceptCluster]:
        """
        Cluster embeddings to find groups of tokens with similar semantic meaning.
        """
        logger.info("Clustering embeddings to find semantic concept groups...")

        # Clean embeddings: remove NaN and infinite values
        clean_embeddings = []
        clean_tokens = []

        for i, (embedding, token) in enumerate(zip(embeddings, tokens)):
            if np.isfinite(embedding).all():
                clean_embeddings.append(embedding)
                clean_tokens.append(token)
            else:
                logger.warning(f"Removing token '{token}' due to NaN/infinite embedding")

        if len(clean_embeddings) == 0:
            logger.error("No clean embeddings found!")
            return []

        clean_embeddings = np.array(clean_embeddings)
        logger.info(f"Using {len(clean_embeddings)} clean embeddings out of {len(embeddings)} total")

        # Skip cosine similarity matrix computation for large vocabularies (memory intensive)
        # For 131k tokens, cosine similarity would create 17 billion element matrix
        if len(clean_embeddings) > 10000:
            logger.info(f"Large vocabulary ({len(clean_embeddings)} tokens), skipping similarity matrix")
        else:
            # Use cosine similarity for small vocabularies only
            similarity_matrix = cosine_similarity(clean_embeddings)

            # Check for NaN in similarity matrix
            if np.isnan(similarity_matrix).any():
                logger.warning("NaN values found in similarity matrix, using fallback clustering")
                return self._fallback_clustering(clean_embeddings, clean_tokens)

        # Find clusters using KMeans (more robust than SpectralClustering for our case)
        from sklearn.cluster import KMeans

        # Determine number of clusters - aim for thousands as user requested
        # Use larger number for real vocabulary: roughly vocab_size / 100
        if len(clean_tokens) > 50000:  # Large vocabulary (real safetensors)
            n_clusters = max(1000, len(clean_tokens) // 100)  # 1k+ clusters for 131k tokens
            n_clusters = min(5000, n_clusters)  # Cap at 5k for memory
        else:  # Small vocabulary (mock data)
            n_clusters = min(20, max(3, int(np.sqrt(len(clean_tokens)))))

        logger.info(f"Creating {n_clusters} semantic clusters for {len(clean_tokens)} tokens")

        try:
            # Use GPU-accelerated clustering for large vocabularies
            if len(clean_embeddings) > 10000:
                logger.info(f"Using native GPU K-means for {len(clean_embeddings)} embeddings")
                logger.info("Moving embeddings to GPU...")

                # Convert to torch tensors on GPU
                embeddings_tensor = torch.tensor(clean_embeddings, dtype=torch.float32, device=device)
                if torch.cuda.is_available():
                    logger.info(f"GPU memory usage: {torch.cuda.memory_allocated()/1024**3:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

                logger.info("Starting native GPU K-means clustering (this may take 1-2 minutes)...")
                cluster_labels = self._gpu_kmeans_clustering(embeddings_tensor, n_clusters)
                logger.info("GPU clustering complete!")
            else:
                # Use CPU clustering for smaller vocabularies
                from sklearn.cluster import KMeans
                logger.info(f"Using CPU K-means for {len(clean_embeddings)} embeddings")
                clusterer = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10
                )
                cluster_labels = clusterer.fit_predict(clean_embeddings)
        except Exception as e:
            logger.warning(f"KMeans clustering failed: {e}, using fallback")
            return self._fallback_clustering(clean_embeddings, clean_tokens)

        # Group tokens by cluster (use clean tokens and embeddings)
        logger.info("Grouping tokens by cluster...")
        clusters = {}
        for i, (token, label) in enumerate(zip(clean_tokens, cluster_labels)):
            if i % 10000 == 0:
                logger.info(f"Processing token {i}/{len(clean_tokens)} ({i/len(clean_tokens)*100:.1f}%)")
            if label not in clusters:
                clusters[label] = {
                    'tokens': [],
                    'embeddings': [],
                    'indices': []
                }
            clusters[label]['tokens'].append(token)
            clusters[label]['embeddings'].append(clean_embeddings[i])
            clusters[label]['indices'].append(i)

        # Create ConceptCluster objects
        logger.info("Building concept clusters...")
        concept_clusters = []
        valid_clusters = 0
        for cluster_id, cluster_data in clusters.items():
            if len(cluster_data['tokens']) < 2:  # Skip singleton clusters
                continue

            valid_clusters += 1
            if valid_clusters % 200 == 0:
                logger.info(f"Built {valid_clusters} concept clusters...")

            # Calculate cluster center
            cluster_embeddings = np.array(cluster_data['embeddings'])
            embedding_center = np.mean(cluster_embeddings, axis=0)

            # Determine semantic category
            semantic_category = self._classify_semantic_category(cluster_data['tokens'])

            # Create cluster
            concept_cluster = ConceptCluster(
                cluster_id=f"concept_{cluster_id}",
                tokens=cluster_data['tokens'],
                languages=self._detect_languages(cluster_data['tokens']),
                embedding_center=embedding_center,
                semantic_category=semantic_category,
                assigned_spectrum=None  # Will be assigned later
            )

            concept_clusters.append(concept_cluster)

            # Log cluster info for first few clusters only
            if len(concept_clusters) <= 10:
                logger.info(f"Cluster {cluster_id} ({semantic_category}): {cluster_data['tokens'][:3]}...")

        return concept_clusters

    def _gpu_kmeans_clustering(self, embeddings_tensor: torch.Tensor, n_clusters: int, max_iters: int = 50) -> np.ndarray:
        """GPU-accelerated K-means clustering using PyTorch."""
        logger.info(f"Initializing GPU K-means with {n_clusters} clusters...")

        n_samples, n_features = embeddings_tensor.shape

        # Initialize centroids using K-means++ on GPU
        centroids = self._kmeans_plus_plus_init(embeddings_tensor, n_clusters)
        logger.info("K-means++ initialization complete")

        prev_labels = None
        for iteration in range(max_iters):
            # Compute distances to all centroids (GPU accelerated)
            distances = torch.cdist(embeddings_tensor, centroids, p=2)

            # Assign points to nearest centroid
            labels = torch.argmin(distances, dim=1)

            # Check for convergence
            if prev_labels is not None and torch.equal(labels, prev_labels):
                logger.info(f"K-means converged after {iteration + 1} iterations")
                break

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = embeddings_tensor[mask].mean(dim=0)
                else:
                    # Keep old centroid if no points assigned
                    new_centroids[k] = centroids[k]

            centroids = new_centroids
            prev_labels = labels.clone()

            if iteration % 10 == 0:
                logger.info(f"K-means iteration {iteration + 1}/{max_iters}")
                if torch.cuda.is_available():
                    logger.info(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

        logger.info(f"K-means completed after {iteration + 1} iterations")
        return labels.cpu().numpy()

    def _kmeans_plus_plus_init(self, embeddings_tensor: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """K-means++ initialization on GPU."""
        n_samples, n_features = embeddings_tensor.shape
        centroids = torch.zeros(n_clusters, n_features, device=embeddings_tensor.device, dtype=embeddings_tensor.dtype)

        # Choose first centroid randomly
        centroids[0] = embeddings_tensor[torch.randint(0, n_samples, (1,))]

        # Choose remaining centroids
        for c in range(1, n_clusters):
            # Compute distances to nearest centroid
            distances = torch.cdist(embeddings_tensor, centroids[:c], p=2)
            min_distances = torch.min(distances, dim=1)[0]

            # Choose next centroid with probability proportional to squared distance
            probs = min_distances ** 2
            probs = probs / probs.sum()

            # Sample from distribution
            cumulative_probs = torch.cumsum(probs, dim=0)
            rand_val = torch.rand(1, device=embeddings_tensor.device)
            chosen_idx = torch.searchsorted(cumulative_probs, rand_val)

            centroids[c] = embeddings_tensor[chosen_idx]

            if c % 100 == 0:
                logger.info(f"K-means++ init: {c}/{n_clusters} centroids")

        return centroids

    def _fallback_clustering(self, embeddings: np.ndarray, tokens: List[str]) -> List[ConceptCluster]:
        """Fallback clustering method using simple similarity grouping."""
        logger.info("Using fallback clustering based on token similarity")

        concept_clusters = []
        used_tokens = set()

        # Group similar tokens together
        for i, token in enumerate(tokens):
            if token in used_tokens:
                continue

            # Find similar tokens
            similar_tokens = [token]
            similar_embeddings = [embeddings[i]]
            used_tokens.add(token)

            for j, other_token in enumerate(tokens):
                if other_token in used_tokens or i == j:
                    continue

                # Simple string similarity for fallback
                if (self._token_similarity(token, other_token) > 0.6 or
                    self._embedding_similarity(embeddings[i], embeddings[j]) > 0.8):
                    similar_tokens.append(other_token)
                    similar_embeddings.append(embeddings[j])
                    used_tokens.add(other_token)

            if len(similar_tokens) > 0:
                # Create cluster
                embedding_center = np.mean(similar_embeddings, axis=0)
                semantic_category = self._classify_semantic_category(similar_tokens)

                concept_cluster = ConceptCluster(
                    cluster_id=f"fallback_concept_{len(concept_clusters)}",
                    tokens=similar_tokens,
                    languages=self._detect_languages(similar_tokens),
                    embedding_center=embedding_center,
                    semantic_category=semantic_category,
                    assigned_spectrum=None
                )

                concept_clusters.append(concept_cluster)

        logger.info(f"Fallback clustering created {len(concept_clusters)} clusters")
        return concept_clusters

    def _token_similarity(self, token1: str, token2: str) -> float:
        """Calculate simple string similarity between tokens."""
        # Simple character overlap similarity
        set1 = set(token1.lower())
        set2 = set(token2.lower())
        overlap = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return overlap / union if union > 0 else 0.0

    def _embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _classify_semantic_category(self, tokens: List[str]) -> str:
        """Classify the semantic category of a token cluster."""
        # Simple heuristic classification based on token analysis
        token_text = ' '.join(tokens).lower()

        if any(word in token_text for word in ['gravity', 'force', 'physics', 'energy', 'mass']):
            return 'physics'
        elif any(word in token_text for word in ['love', 'happy', 'sad', 'emotion', 'feel']):
            return 'emotion'
        elif any(word in token_text for word in ['run', 'jump', 'move', 'action', 'go', 'come']):
            return 'action'
        elif any(word in token_text for word in ['think', 'mind', 'philosophy', 'idea', 'concept']):
            return 'abstract'
        elif any(word in token_text for word in ['water', 'tree', 'rock', 'house', 'car']):
            return 'concrete'
        elif any(word in token_text for word in ['friend', 'family', 'social', 'people']):
            return 'social'
        elif any(word in token_text for word in ['time', 'when', 'before', 'after', 'now']):
            return 'temporal'
        else:
            return 'abstract'  # Default category

    def _detect_languages(self, tokens: List[str]) -> List[str]:
        """Detect languages present in token list (simple heuristic)."""
        # This is a simplified implementation
        # In practice, you'd use a proper language detection library
        languages = set()

        for token in tokens:
            # Check for non-Latin scripts
            if any(ord(char) > 127 for char in token):
                if any(0x0370 <= ord(char) <= 0x03FF for char in token):  # Greek
                    languages.add('greek')
                elif any(0x0400 <= ord(char) <= 0x04FF for char in token):  # Cyrillic
                    languages.add('russian')
                elif any(0x4E00 <= ord(char) <= 0x9FFF for char in token):  # Chinese
                    languages.add('chinese')
                else:
                    languages.add('unknown')
            else:
                # Simple heuristics for Latin-script languages
                if token.endswith('tion') or token.endswith('sion'):
                    languages.add('english')
                elif token.endswith('ment') or 'eau' in token:
                    languages.add('french')
                elif 'ä' in token or 'ö' in token or 'ü' in token:
                    languages.add('german')
                else:
                    languages.add('unknown')

        return list(languages) if languages else ['unknown']

    def _assign_em_spectrum(self, cluster: ConceptCluster) -> EMSpectrum:
        """Assign electromagnetic spectrum to a concept cluster."""
        category = cluster.semantic_category
        wavelength_range = self.spectral_regions.get(category, (400, 700))

        # Primary wavelength based on embedding characteristics
        embedding_hash = hash(tuple(cluster.embedding_center)) % 1000
        wavelength_ratio = embedding_hash / 1000.0

        wavelength = wavelength_range[0] + wavelength_ratio * (wavelength_range[1] - wavelength_range[0])

        # Bandwidth based on cluster coherence (tighter cluster = narrower spectrum)
        cluster_variance = np.var([np.linalg.norm(cluster.embedding_center - emb)
                                 for emb in [cluster.embedding_center]])  # Simplified
        bandwidth = max(5, min(50, 20 * (1 + cluster_variance)))

        # Intensity based on cluster size (more tokens = higher intensity)
        intensity = min(1.0, 0.3 + 0.1 * len(cluster.tokens))

        return EMSpectrum(
            wavelength=wavelength,
            bandwidth=bandwidth,
            intensity=intensity,
            fine_structure=None  # Will be computed in __post_init__
        )

    def encode_concept(self, concept: str) -> Optional[EMSpectrum]:
        """Encode a single concept to its EM spectrum."""
        # Find which cluster this concept belongs to
        for cluster in self.concept_clusters.values():
            if concept.lower() in [token.lower() for token in cluster.tokens]:
                return cluster.assigned_spectrum

        # If not found, return None or create default spectrum
        logger.warning(f"Concept '{concept}' not found in clusters")
        return None

    def encode_text(self, text: str) -> EMSpectrum:
        """
        Encode full text to combined EM spectrum.
        This is the main encoding function.
        """
        words = text.lower().split()
        spectra = []

        for word in words:
            spectrum = self.encode_concept(word)
            if spectrum:
                spectra.append(spectrum)

        if not spectra:
            # Return default spectrum if no concepts found
            return EMSpectrum(wavelength=550, bandwidth=50, intensity=0.1, fine_structure=None)

        # Combine spectra (like mixing light)
        return self._combine_spectra(spectra)

    def _combine_spectra(self, spectra: List[EMSpectrum]) -> EMSpectrum:
        """Combine multiple EM spectra (like mixing light)."""
        if len(spectra) == 1:
            return spectra[0]

        # Simple combination: weighted average
        total_intensity = sum(s.intensity for s in spectra)
        if total_intensity == 0:
            return EMSpectrum(wavelength=550, bandwidth=50, intensity=0, fine_structure=None)

        combined_wavelength = sum(s.wavelength * s.intensity for s in spectra) / total_intensity
        combined_bandwidth = sum(s.bandwidth * s.intensity for s in spectra) / total_intensity
        combined_intensity = min(1.0, total_intensity / len(spectra))

        return EMSpectrum(
            wavelength=combined_wavelength,
            bandwidth=combined_bandwidth,
            intensity=combined_intensity,
            fine_structure=None
        )

    def save_dictionary(self, filepath: str):
        """Save the concept dictionary to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.concept_clusters, f)
        logger.info(f"Saved concept dictionary to {filepath}")

    def load_dictionary(self, filepath: str):
        """Load concept dictionary from file."""
        with open(filepath, 'rb') as f:
            self.concept_clusters = pickle.load(f)
        logger.info(f"Loaded concept dictionary from {filepath}")


def test_semantic_encoder():
    """Test the semantic EM encoder."""
    print("Testing Semantic EM Encoder...")
    print("=" * 50)

    encoder = SemanticEMEncoder()

    # Build concept dictionary
    concept_dict = encoder.build_concept_dictionary()

    print(f"Built dictionary with {len(concept_dict)} concept clusters")
    print()

    # Test some multilingual concepts
    test_cases = [
        "love",
        "amour",  # French
        "liebe",  # German
        "gravity",
        "gravité",  # French
        "water",
        "eau",    # French
    ]

    print("Testing concept encoding:")
    print("-" * 30)

    for concept in test_cases:
        spectrum = encoder.encode_concept(concept)
        if spectrum:
            print(f"{concept:10} → λ={spectrum.wavelength:.1f}nm, "
                  f"Δλ={spectrum.bandwidth:.1f}nm, I={spectrum.intensity:.3f}")
        else:
            print(f"{concept:10} → NOT FOUND")

    print()
    print("Testing text encoding:")
    print("-" * 30)

    test_texts = [
        "explain gravity",
        "what is love",
        "water flows"
    ]

    for text in test_texts:
        spectrum = encoder.encode_text(text)
        print(f"'{text}' → λ={spectrum.wavelength:.1f}nm, "
              f"Δλ={spectrum.bandwidth:.1f}nm, I={spectrum.intensity:.3f}")


if __name__ == "__main__":
    test_semantic_encoder()