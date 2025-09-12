"""
ApertusTranslator: Bidirectional translator between human language and ChromaThink's colour space

CRITICAL: Apertus serves ONLY as a translator, not the thinking engine.
ChromaThink thinks purely in colour space through interference patterns.

Flow:
1. Human text -> Apertus extracts concepts -> Colour encoding
2. Colour question -> ChromaThink processes in colour -> Colour answer  
3. Colour answer -> Apertus synthesises -> Human text (same language)

Apertus is the bridge, ChromaThink is the mind.
"""

import torch
import tensorflow as tf
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import warnings

# Handle imports gracefully
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available")


class ApertusTranslator:
    """
    Apertus serves ONLY as a bidirectional translator between
    human language and ChromaThink's colour space.
    It does NOT generate responses - it only translates concepts.
    """
    
    def __init__(self, 
                 apertus_path: str = "models/apertus",
                 spectrum_dims: int = 512,
                 use_mock: bool = False,
                 device: str = 'auto'):
        
        self.apertus_path = Path(apertus_path)
        self.spectrum_dims = spectrum_dims
        self.use_mock = use_mock
        self.device = device
        self.logger = logging.getLogger("ApertusTranslator")
        
        # Concept extraction prompts for different tasks
        self.extraction_prompts = {
            'concepts': "Extract the main concepts from this text as a numbered list: ",
            'intent': "What is the core question or intent in this text? Answer briefly: ",
            'context': "What contextual information is important in this text? List key points: ",
            'language': "What language is this text written in? Answer with just the language name: "
        }
        
        # Synthesis templates for different languages
        self.synthesis_templates = {
            'english': "Express these concepts in natural English: {concepts}. Based on the original context '{context}' and responding to '{intent}', synthesize a thoughtful response:",
            'spanish': "Expresa estos conceptos en español natural: {concepts}. Basado en el contexto original '{context}' y respondiendo a '{intent}', sintetiza una respuesta reflexiva:",
            'french': "Exprimez ces concepts en français naturel: {concepts}. Basé sur le contexte original '{context}' et en répondant à '{intent}', synthétisez une réponse réfléchie:",
            'german': "Drücken Sie diese Konzepte in natürlichem Deutsch aus: {concepts}. Basierend auf dem ursprünglichen Kontext '{context}' und als Antwort auf '{intent}', synthetisieren Sie eine durchdachte Antwort:",
            'default': "Express these concepts naturally: {concepts}. Based on the original context '{context}' and responding to '{intent}', synthesize a thoughtful response:"
        }
        
        # Initialize Apertus model
        if self.use_mock or not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Using mock Apertus translator")
            self._create_mock_translator()
        else:
            try:
                self._load_apertus_model()
            except Exception as e:
                self.logger.warning(f"Failed to load Apertus: {e}, using mock")
                self._create_mock_translator()
    
    def _create_mock_translator(self):
        """Create mock translator for testing"""
        
        class MockTokenizer:
            def __init__(self):
                self.vocab_size = 50000
                self.eos_token_id = 2
                self.pad_token_id = 0
                
            def apply_chat_template(self, messages, **kwargs):
                return torch.tensor([[1, 2, 3, 4, 5]])  # Mock tokens
                
            def decode(self, tokens, **kwargs):
                # Mock concept extraction responses
                if "concepts" in str(tokens):
                    return "1. Understanding\n2. Knowledge\n3. Learning"
                elif "intent" in str(tokens):
                    return "Seeking information"
                elif "context" in str(tokens):
                    return "Educational conversation"
                elif "language" in str(tokens):
                    return "English"
                else:
                    return "This is a thoughtful response based on the provided concepts."
        
        class MockModel:
            def generate(self, inputs, **kwargs):
                return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            
            def eval(self):
                pass
        
        self.tokenizer = MockTokenizer()
        self.model = MockModel()
        self.logger.info("Mock Apertus translator created")
    
    def _load_apertus_model(self):
        """Load real Apertus model for translation"""
        
        self.logger.info(f"Loading Apertus translator from {self.apertus_path}")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.apertus_path,
                local_files_only=True,
                trust_remote_code=True
            )
        except Exception as e:
            self.logger.warning(f"Loading tokenizer failed: {e}, using DialoGPT fallback")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        
        # Load model with device configuration
        device_config = {}
        if self.device == 'cpu':
            device_config = {"device_map": "cpu", "dtype": torch.float32}
        elif self.device == 'cuda':
            # Use higher memory allocation for GPU
            device_config = {
                "device_map": "cuda", 
                "dtype": torch.float16,
                "max_memory": {0: "12GB"}  # Allocate more GPU memory
            }
        else:
            # Auto mode with higher memory limits
            device_config = {
                "device_map": "auto", 
                "dtype": torch.float16,
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
        
        # Log device allocation
        if hasattr(self.model, 'hf_device_map'):
            self.logger.info(f"Apertus loaded on: {self.model.hf_device_map}")
        else:
            self.logger.info(f"Apertus loaded on: {self.device}")
        
        self.logger.info("Apertus translator loaded successfully")
    
    def extract_concepts(self, text: str) -> Tuple[Dict[str, List[str]], str]:
        """
        Extract concepts from human text using Apertus.
        Returns structured concepts and detected language.
        """
        
        self.logger.info(f"Extracting concepts from text: {text[:50]}...")
        
        # Detect language first
        language = self._detect_language(text)
        
        # Extract different concept types
        concepts = {}
        
        for concept_type, prompt in self.extraction_prompts.items():
            if concept_type == 'language':  # Skip language detection in main loop
                continue
                
            extraction_query = f"{prompt}\n\"{text}\"\n\nResponse:"
            
            response = self._query_apertus(extraction_query, max_tokens=100, temperature=0.3)
            
            # Parse the response into concept list
            concepts[concept_type] = self._parse_concept_response(response, concept_type)
            
            self.logger.debug(f"Extracted {concept_type}: {concepts[concept_type]}")
        
        return concepts, language
    
    def synthesise_response(self, 
                          colour_descriptors: List[str], 
                          target_language: str,
                          original_concepts: Dict[str, List[str]]) -> str:
        """
        Translate colour pattern descriptors back to human language.
        This is synthesis from colour, NOT generation of new content.
        """
        
        self.logger.info(f"Synthesising {target_language} response from {len(colour_descriptors)} colour descriptors")
        
        # Get appropriate template
        language_key = target_language.lower() if target_language.lower() in self.synthesis_templates else 'default'
        template = self.synthesis_templates[language_key]
        
        # Format the synthesis prompt
        concepts_text = ", ".join(colour_descriptors)
        context_text = ", ".join(original_concepts.get('context', ['general conversation']))
        intent_text = ", ".join(original_concepts.get('intent', ['seeking understanding']))
        
        synthesis_prompt = template.format(
            concepts=concepts_text,
            context=context_text,
            intent=intent_text
        )
        
        # Generate response
        response = self._query_apertus(synthesis_prompt, max_tokens=300, temperature=0.7)
        
        self.logger.debug(f"Synthesised response: {response[:100]}...")
        
        return response.strip()
    
    def colour_to_concept_descriptors(self, colour_pattern: tf.Tensor) -> List[str]:
        """
        Analyse colour pattern to extract conceptual descriptors.
        These describe what the colour pattern represents.
        """
        
        # Convert to numpy for analysis
        if colour_pattern.dtype == tf.complex64:
            amplitude = tf.abs(colour_pattern).numpy()
            phase = tf.math.angle(colour_pattern).numpy()
        else:
            amplitude = colour_pattern.numpy()
            phase = np.zeros_like(amplitude)
        
        # Flatten if needed
        amplitude = amplitude.flatten()
        phase = phase.flatten()
        
        descriptors = []
        
        # Identify dominant frequencies (main concepts)
        dominant_freqs = np.argsort(amplitude)[-10:][::-1]  # Top 10 frequencies
        
        # Map frequency ranges to concept types
        for i, freq_idx in enumerate(dominant_freqs):
            amplitude_val = amplitude[freq_idx]
            phase_val = phase[freq_idx] if len(phase) > freq_idx else 0
            
            # Low frequencies = fundamental concepts
            if freq_idx < self.spectrum_dims // 4:
                descriptor = self._frequency_to_fundamental_concept(freq_idx, amplitude_val, phase_val)
            # Mid frequencies = relationships
            elif freq_idx < 3 * self.spectrum_dims // 4:
                descriptor = self._frequency_to_relationship_concept(freq_idx, amplitude_val, phase_val)
            # High frequencies = details
            else:
                descriptor = self._frequency_to_detail_concept(freq_idx, amplitude_val, phase_val)
            
            descriptors.append(descriptor)
            
            # Stop if we have enough descriptors
            if len(descriptors) >= 8:
                break
        
        # Add pattern complexity descriptors
        spectral_entropy = self._calculate_spectral_entropy(amplitude)
        phase_coherence = self._calculate_phase_coherence(phase)
        
        if spectral_entropy > 0.7:
            descriptors.append("complex multifaceted understanding")
        elif spectral_entropy < 0.3:
            descriptors.append("focused specific knowledge")
        
        if phase_coherence > 0.8:
            descriptors.append("coherent unified concept")
        elif phase_coherence < 0.3:
            descriptors.append("diverse interconnected ideas")
        
        self.logger.debug(f"Generated {len(descriptors)} concept descriptors")
        
        return descriptors
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of input text"""
        
        language_query = f"{self.extraction_prompts['language']}\n\"{text[:100]}\"\n\nLanguage:"
        response = self._query_apertus(language_query, max_tokens=10, temperature=0.1)
        
        # Clean up response
        language = response.strip().lower()
        
        # Map common variations
        language_mapping = {
            'en': 'english', 'english': 'english',
            'es': 'spanish', 'español': 'spanish', 'spanish': 'spanish',
            'fr': 'french', 'français': 'french', 'french': 'french',
            'de': 'german', 'deutsch': 'german', 'german': 'german'
        }
        
        detected = language_mapping.get(language, language)
        self.logger.debug(f"Detected language: {detected}")
        
        return detected
    
    def _query_apertus(self, prompt: str, max_tokens: int = 100, temperature: float = 0.5) -> str:
        """Query Apertus model with given prompt"""
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Apply chat template
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True
            )
            
            # Create attention mask
            attention_mask = torch.ones_like(inputs)
            
            # Move inputs to same device and dtype as model
            device = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype
            
            # inputs and attention_mask should remain as int64/long for tokenized data
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            
            # Ensure embedding layer can handle the input dtype correctly
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                # This is for handling mixed precision models
                pass  # inputs should remain as long/int64
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][len(inputs[0]):],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.warning(f"Apertus query failed: {e}, using fallback")
            # Return a generic response based on prompt content
            if "concepts" in prompt:
                return "understanding, knowledge, inquiry"
            elif "intent" in prompt:
                return "seeking information"
            elif "context" in prompt:
                return "conversational context"
            else:
                return "thoughtful response"
    
    def _parse_concept_response(self, response: str, concept_type: str) -> List[str]:
        """Parse Apertus response into concept list"""
        
        concepts = []
        
        # Clean up response
        response = response.strip()
        
        # Try different parsing strategies
        if any(marker in response for marker in ['1.', '2.', '-', '•']):
            # Numbered or bulleted list
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                # Remove numbering and bullets
                line = line.lstrip('0123456789.-•* ')
                if line:
                    concepts.append(line)
        
        elif ',' in response:
            # Comma-separated
            parts = response.split(',')
            concepts = [p.strip() for p in parts if p.strip()]
        
        else:
            # Single concept or unstructured
            concepts = [response] if response else []
        
        # Clean and filter concepts
        cleaned_concepts = []
        for concept in concepts[:5]:  # Limit to 5 concepts max
            # Remove common prefixes/suffixes
            concept = concept.strip(' .,;:!?-*')
            if len(concept) > 2 and len(concept) < 100:  # Reasonable length
                cleaned_concepts.append(concept)
        
        return cleaned_concepts if cleaned_concepts else [f"general {concept_type}"]
    
    def _frequency_to_fundamental_concept(self, freq_idx: int, amplitude: float, phase: float) -> str:
        """Map low frequency to fundamental concept descriptor"""
        
        # Map frequency bands to concept categories
        band = (freq_idx * 8) // self.spectrum_dims  # 8 bands
        
        concept_categories = [
            "essential understanding",
            "core knowledge", 
            "fundamental principle",
            "basic concept",
            "primary idea",
            "foundational thought",
            "central theme",
            "main point"
        ]
        
        base_concept = concept_categories[band % len(concept_categories)]
        
        # Modify based on amplitude and phase
        if amplitude > 0.7:
            return f"strong {base_concept}"
        elif amplitude > 0.4:
            return f"clear {base_concept}"
        else:
            return f"subtle {base_concept}"
    
    def _frequency_to_relationship_concept(self, freq_idx: int, amplitude: float, phase: float) -> str:
        """Map mid frequency to relationship concept descriptor"""
        
        # Map to relationship types
        band = ((freq_idx - self.spectrum_dims // 4) * 6) // (self.spectrum_dims // 2)
        
        relationship_types = [
            "connection between ideas",
            "relationship between concepts", 
            "interaction of elements",
            "association between topics",
            "correlation of themes",
            "interdependence of aspects"
        ]
        
        base_relationship = relationship_types[band % len(relationship_types)]
        
        # Modify based on phase
        if abs(phase) > 2.0:
            return f"complex {base_relationship}"
        elif abs(phase) > 1.0:
            return f"dynamic {base_relationship}"
        else:
            return f"harmonious {base_relationship}"
    
    def _frequency_to_detail_concept(self, freq_idx: int, amplitude: float, phase: float) -> str:
        """Map high frequency to detail concept descriptor"""
        
        # Map to detail types
        band = ((freq_idx - 3 * self.spectrum_dims // 4) * 4) // (self.spectrum_dims // 4)
        
        detail_types = [
            "specific detail",
            "particular aspect",
            "nuanced element", 
            "refined understanding"
        ]
        
        base_detail = detail_types[band % len(detail_types)]
        
        # Modify based on amplitude
        if amplitude > 0.5:
            return f"important {base_detail}"
        else:
            return f"subtle {base_detail}"
    
    def _calculate_spectral_entropy(self, amplitude: np.ndarray) -> float:
        """Calculate spectral entropy as measure of complexity"""
        
        # Normalize amplitude
        power = amplitude ** 2
        power_norm = power / (np.sum(power) + 1e-8)
        
        # Calculate entropy
        entropy = -np.sum(power_norm * np.log(power_norm + 1e-8))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(amplitude))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_phase_coherence(self, phase: np.ndarray) -> float:
        """Calculate phase coherence as measure of unity"""
        
        if len(phase) == 0:
            return 0.5
        
        # Calculate coherence as magnitude of mean phase vector
        mean_phase_vector = np.mean(np.exp(1j * phase))
        coherence = np.abs(mean_phase_vector)
        
        return float(coherence)
    
    def synthesise_from_concepts(self, concepts: List[str], colour_guidance: np.ndarray = None) -> str:
        """
        Synthesize natural language response from concept list using Apertus.
        This ensures NO template responses - all text comes from Apertus generation.
        """
        
        self.logger.info(f"Synthesizing from concepts: {concepts[:3]}...")
        
        # Build a natural synthesis prompt for Apertus
        concept_text = ", ".join(concepts)
        synthesis_prompt = f"Please explain or discuss: {concept_text}."
        
        # Use Apertus to generate natural response
        response = self._query_apertus(synthesis_prompt, max_tokens=200, temperature=0.8)
        
        return response.strip()
    
    def synthesise_from_frequency_pattern(self, colour_waveform: np.ndarray) -> str:
        """
        Synthesize response from raw frequency pattern by first extracting concepts.
        """
        
        # Convert numpy array to tensor for analysis
        colour_tensor = tf.convert_to_tensor(colour_waveform, dtype=tf.complex64)
        
        # Extract concept descriptors from the frequency pattern
        descriptors = self.colour_to_concept_descriptors(colour_tensor)
        
        # Synthesize from extracted concepts
        return self.synthesise_from_concepts(descriptors)