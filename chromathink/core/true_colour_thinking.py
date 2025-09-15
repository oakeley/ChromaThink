"""
CRITICAL: ChromaThink must NEVER have pre-programmed text responses.
Every response must emerge from colour interference patterns.

Flow:
1. Human text -> Apertus -> Concept extraction -> Colour encoding
2. Colour question -> ChromaThink interference/resonance -> Colour answer  
3. Colour answer -> Apertus -> Concept synthesis -> Human text (same language)

Apertus is the translator, ChromaThink is the thinker.
"""

import torch
import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import os
from pathlib import Path

# Handle imports gracefully
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .chromathink_core import ChromaThinkCore
from ..bootstrap.apertus_integration import ApertusWeightTranslator


class ApertusTranslator:
    """
    Apertus serves ONLY as a bidirectional translator between
    human language and ChromaThink's colour space.
    It does NOT generate responses - it only translates concepts.
    """
    
    def __init__(self, model_path: str = "models/apertus"):
        self.logger = logging.getLogger("ApertusTranslator")
        self.model_path = model_path
        
        # Check if Apertus is available
        if not os.path.exists(model_path) or not TRANSFORMERS_AVAILABLE:
            self.logger.warning(f"Apertus not available at {model_path}, using concept hashing")
            self.use_apertus = False
            self.tokenizer = None
            self.model = None
        else:
            try:
                # Load Apertus for concept extraction and synthesis
                self.logger.info(f"Loading Apertus from {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                # Load with CPU allocation to avoid conflicts with ChromaThink GPU usage
                device_config = {
                    "device_map": "cpu",  # Force CPU for classic system
                    "dtype": torch.float32,  # Use float32 for CPU
                    "local_files_only": True
                }
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **device_config
                )
                self.model.eval()
                self.use_apertus = True
                self.logger.info("Apertus loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load Apertus: {e}, using concept hashing")
                self.use_apertus = False
                self.tokenizer = None
                self.model = None
        
        # Concept extraction prompts for different languages
        self.extraction_prompts = {
            'concept': "Extract the core concepts from this text as a list: ",
            'intent': "What is the underlying question or intent: ",
            'context': "What contextual information is important: "
        }
        
    def extract_concepts(self, text: str, source_language: str = None) -> Tuple[Dict[str, List[str]], str]:
        """
        Extract concepts from human text using Apertus.
        Returns structured concepts, NOT a response.
        """
        
        if not self.use_apertus:
            return self._extract_concepts_fallback(text, source_language)
        
        # Detect language if not specified
        if source_language is None:
            source_language = self.detect_language(text)
        
        self.logger.info(f"Extracting concepts from {source_language} text")
        
        concepts = {}
        
        for concept_type, prompt in self.extraction_prompts.items():
            try:
                # Ask Apertus to extract specific concept types
                extraction_query = f"{prompt}\"{text}\"\nList the concepts:"
                
                messages = [{"role": "user", "content": extraction_query}]
                
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=True
                )
                
                # Create attention mask and move to same device as model
                attention_mask = torch.ones_like(inputs)
                device = next(self.model.parameters()).device
                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        max_new_tokens=100,
                        temperature=0.3,  # Low temperature for consistent extraction
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][len(inputs[0]):],
                    skip_special_tokens=True
                )
                
                # Parse concepts from response
                concepts[concept_type] = self.parse_concepts(response)
                
                self.logger.debug(f"Extracted {concept_type}: {concepts[concept_type]}")
                
            except Exception as e:
                self.logger.warning(f"Failed to extract {concept_type}: {e}")
                concepts[concept_type] = self._fallback_concept_extraction(text, concept_type)
        
        return concepts, source_language
    
    def synthesise_response(self, colour_pattern: tf.Tensor, 
                          target_language: str,
                          original_concepts: Dict) -> str:
        """
        Translate colour pattern back to human language.
        This is synthesis from colour, NOT generation of new content.
        """
        
        self.logger.info(f"Synthesising {target_language} response from colour pattern")
        
        if not self.use_apertus:
            return self._synthesise_response_fallback(colour_pattern, target_language, original_concepts)
        
        # Convert colour pattern to concept descriptors
        concept_descriptors = self.colour_to_concept_descriptors(colour_pattern)
        
        # Create synthesis prompt
        synthesis_prompt = self.create_synthesis_prompt(
            concept_descriptors,
            target_language,
            original_concepts
        )
        
        try:
            messages = [{"role": "user", "content": synthesis_prompt}]
            
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True
            )
            
            # Create attention mask and move to same device as model
            attention_mask = torch.ones_like(inputs)
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][len(inputs[0]):],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.warning(f"Failed to synthesize with Apertus: {e}")
            return self._synthesise_response_fallback(colour_pattern, target_language, original_concepts)
    
    def colour_to_concept_descriptors(self, colour_pattern: tf.Tensor) -> List[str]:
        """
        Analyse colour pattern to extract conceptual descriptors.
        These describe what the colour pattern represents.
        """
        
        # Convert to numpy for analysis
        try:
            if hasattr(colour_pattern, 'numpy'):
                colour_array = colour_pattern.numpy()
            elif tf.is_tensor(colour_pattern):
                colour_array = colour_pattern.numpy()
            else:
                colour_array = np.array(colour_pattern, dtype=np.complex64)
        except Exception as e:
            # Fallback for complex conversion issues
            if isinstance(colour_pattern, (list, tuple)):
                colour_array = np.array(colour_pattern, dtype=np.complex64)
            else:
                # Create a dummy pattern
                colour_array = np.random.random(512).astype(np.complex64)
        
        # Analyse colour properties
        amplitude = np.abs(colour_array)
        phase = np.angle(colour_array)
        
        # Identify dominant frequencies (main concepts)
        dominant_freqs = np.argsort(amplitude)[-10:][::-1]
        
        descriptors = []
        
        # Map frequency ranges to concept types
        for freq_idx in dominant_freqs:
            if freq_idx < 100:  # Low frequencies = fundamental concepts
                descriptors.append(f"fundamental_concept_{freq_idx}")
            elif freq_idx < 300:  # Mid frequencies = relationships
                descriptors.append(f"relationship_{freq_idx}")
            else:  # High frequencies = details
                descriptors.append(f"detail_{freq_idx}")
        
        # Add phase relationships (how concepts connect)
        phase_clusters = self.cluster_phases(phase[dominant_freqs])
        for cluster_id, cluster_phases in enumerate(phase_clusters):
            descriptors.append(f"connection_pattern_{cluster_id}")
        
        return descriptors
    
    def create_synthesis_prompt(self, descriptors: List[str],
                               language: str,
                               original_concepts: Dict) -> str:
        """
        Create a prompt for Apertus to synthesise the response.
        """
        
        prompt = f"""Synthesise a response in {language} that expresses these concepts:
        
        Original context: {original_concepts.get('context', [])}
        Responding to: {original_concepts.get('intent', [])}
        
        The response should integrate these elements:
        {', '.join(descriptors)}
        
        Synthesise a natural response that captures these concepts:"""
        
        return prompt
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        """
        
        if not self.use_apertus:
            return "english"  # Default fallback
        
        # Ask Apertus to identify the language
        detection_prompt = f"What language is this text written in? Reply with just the language name: \"{text[:100]}\""
        
        try:
            messages = [{"role": "user", "content": detection_prompt}]
            
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            language = self.tokenizer.decode(
                outputs[0][len(inputs[0]):],
                skip_special_tokens=True
            ).strip().lower()
            
            return language
        except:
            return "english"
    
    def parse_concepts(self, response: str) -> List[str]:
        """
        Parse concepts from Apertus response.
        """
        
        # Extract bullet points, comma-separated items, or lines
        concepts = []
        
        # Try different parsing strategies
        if '•' in response or '-' in response:
            # Bullet points
            lines = response.split('\n')
            for line in lines:
                line = line.strip().lstrip('•-*').strip()
                if line and len(line) > 2:
                    concepts.append(line)
        elif ',' in response:
            # Comma-separated
            parts = response.split(',')
            concepts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
        else:
            # Line-separated
            lines = response.split('\n')
            concepts = [l.strip() for l in lines if l.strip() and len(l.strip()) > 2]
        
        return concepts[:10]  # Limit to 10 concepts
    
    def cluster_phases(self, phases: np.ndarray) -> List[List[float]]:
        """
        Cluster phase values to identify connection patterns.
        """
        
        # Simple clustering based on phase similarity
        clusters = []
        used = set()
        
        for i, phase in enumerate(phases):
            if i in used:
                continue
            
            cluster = [phase]
            used.add(i)
            
            for j, other_phase in enumerate(phases):
                if j not in used:
                    # Check if phases are similar (within π/4)
                    if abs(phase - other_phase) < np.pi/4:
                        cluster.append(other_phase)
                        used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    # Fallback methods when Apertus is not available
    def _extract_concepts_fallback(self, text: str, source_language: str = None) -> Tuple[Dict[str, List[str]], str]:
        """Fallback concept extraction using simple heuristics."""
        
        if source_language is None:
            source_language = "english"
        
        # Simple word-based concept extraction
        words = text.lower().split()
        
        # Filter out common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        concepts = [word for word in words if word not in common_words and len(word) > 2]
        
        # Categorize concepts
        question_words = {"what", "why", "how", "when", "where", "who"}
        intent_concepts = [word for word in concepts if word in question_words]
        concept_concepts = [word for word in concepts if word not in question_words]
        
        return {
            'intent': intent_concepts[:5],
            'concepts': concept_concepts[:10],
            'context': concepts[:3]
        }, source_language
    
    def _fallback_concept_extraction(self, text: str, concept_type: str) -> List[str]:
        """Fallback for specific concept type extraction."""
        words = text.lower().split()
        return [w for w in words if len(w) > 3][:5]
    
    def _synthesise_response_fallback(self, colour_pattern: tf.Tensor, target_language: str, original_concepts: Dict) -> str:
        """Fallback response synthesis without Apertus."""
        
        # Analyze colour pattern properties
        if hasattr(colour_pattern, 'numpy'):
            colour_array = colour_pattern.numpy()
        else:
            colour_array = np.array(colour_pattern)
        
        amplitude = np.abs(colour_array)
        dominant_freq = np.argmax(amplitude)
        total_energy = np.sum(amplitude**2)
        
        # Extract concepts from original input
        all_concepts = []
        for concept_list in original_concepts.values():
            all_concepts.extend(concept_list)
        
        if not all_concepts:
            return f"The colour pattern shows frequency {dominant_freq} with energy {total_energy:.3f}, suggesting complex interference patterns."
        
        primary_concept = all_concepts[0] if all_concepts else "pattern"
        
        # Generate response based on colour properties
        if total_energy > 5.0:
            return f"{primary_concept.capitalize()} demonstrates rich interference patterns with dominant frequency {dominant_freq}. The high energy suggests deep resonance across multiple dimensions."
        elif total_energy > 1.0:
            return f"{primary_concept.capitalize()} shows moderate resonance patterns. The colour analysis reveals frequency {dominant_freq} as the central organizing principle."
        else:
            return f"{primary_concept.capitalize()} creates subtle interference patterns. The colour frequencies suggest emerging understanding at frequency {dominant_freq}."


class IntegratedChromaThink:
    """
    Complete system integrating Apertus translation with ChromaThink thinking.
    """
    
    def __init__(self, apertus_path: str = "models/apertus", spectrum_dims: int = 512, gpu_acceleration: bool = False):
        self.logger = logging.getLogger("IntegratedChromaThink")
        
        # Initialise components
        self.translator = ApertusTranslator(apertus_path)
        self.thinker = ChromaThinkCore(spectrum_dims)
        
        # Load Big Colour Model if available
        try:
            self.weight_translator = ApertusWeightTranslator(
                apertus_path=apertus_path,
                spectrum_dims=spectrum_dims,
                use_mock=(not os.path.exists(apertus_path)),
                extract_full_vocab=True
            )
            self.big_colour_model = self.weight_translator.build_big_colour_model()
        except Exception as e:
            self.logger.warning(f"Could not load Big Colour Model: {e}")
            self.weight_translator = None
            self.big_colour_model = None
        
        self.logger.info("Integrated ChromaThink ready")
    
    def process_input(self, human_text: str) -> str:
        """
        Complete processing pipeline:
        Text -> Concepts -> Colours -> Thinking -> Colours -> Text
        """
        
        self.logger.info(f"Processing input: {human_text[:50]}...")
        
        # Step 1: Extract concepts using Apertus
        concepts, source_language = self.translator.extract_concepts(human_text)
        self.logger.info(f"Extracted concepts in {source_language}")
        
        # Step 2: Convert concepts to colours
        concept_colours = {}
        for concept_type, concept_list in concepts.items():
            if self.big_colour_model and self.weight_translator:
                # Use Big Colour Model for encoding
                concept_colours[concept_type] = []
                for concept in concept_list:
                    try:
                        colour = self.weight_translator.encode_concept_to_waveform(concept)
                        if len(colour.shape) > 1:
                            colour = tf.squeeze(colour)
                        concept_colours[concept_type].append(colour)
                    except:
                        # Fallback to hash-based encoding
                        colour = self.thinker.encode_concept_to_colour(concept)
                        concept_colours[concept_type].append(colour)
            else:
                # Use hash-based encoding
                concept_colours[concept_type] = [
                    self.thinker.encode_concept_to_colour(c) for c in concept_list
                ]
        
        # Step 3: Think in pure colour space
        if concept_colours:
            response_colour = self.thinker.think_in_colour_complex(concept_colours)
        else:
            # Generate neutral curiosity response
            response_colour = tf.complex(
                tf.random.normal([self.thinker.spectrum_dims]),
                tf.random.normal([self.thinker.spectrum_dims])
            ) * 0.1
        
        self.logger.info("Colour thinking complete")
        
        # Step 4: Translate colour back to language
        response_text = self.translator.synthesise_response(
            response_colour,
            source_language,  # Respond in same language
            concepts  # Original concepts for context
        )
        
        self.logger.info(f"Generated {source_language} response")
        
        return response_text


def create_integrated_chromathink(apertus_path: str = "models/apertus",
                                spectrum_dims: int = 512,
                                gpu_acceleration: bool = False) -> IntegratedChromaThink:
    """
    Factory function to create integrated ChromaThink system.
    """
    return IntegratedChromaThink(apertus_path=apertus_path, spectrum_dims=spectrum_dims, gpu_acceleration=gpu_acceleration)