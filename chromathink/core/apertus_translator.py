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
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import warnings
import requests
import json

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
        
        # Ollama endpoint configuration
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "qwen3:4b"
        
        # Improved concept extraction prompt for Ollama
        self.concept_extraction_prompt = """You are a curt text simplification system. Your task is to identify the optimum sentence structure into which a query can be reformulated for clarity and ease of understanding of intent.

Instructions:
1. Extract the core concepts that capture the semantic essence of the input
2. Simplify the input structure for clarity and brevity
3. Focus on concrete nouns, actions, and essential attributes
4. Select a critical verb to include in the output phrase to describe the requested act
5. Ignore grammatical words (articles, conjunctions, prepositions)
6. Preserve the most semantically important elements so that meaning in the output phrase is retained
7. Render the output phrase in the same language as the input phrase, if more than one language is detected then render the output phrase in English
8. Do not provide explanations only return the simplified output phrase

Input text: "{text}"

Core concepts:"""
        
        # Keep extraction prompts for compatibility
        self.extraction_prompts = {
            'context': "Extract context keywords from:",
            'intent': "Identify the primary intent of:",
            'entities': "List key entities mentioned in:"
        }
        
        # Test Ollama connection
        try:
            self._test_ollama_connection()
            self.logger.info("Ollama connection established successfully")
        except Exception as e:
            self.logger.warning(f"Ollama connection test failed: {e}")
            if not use_mock:
                self.logger.warning("Ollama is required when not using mock mode")
    
    def _test_ollama_connection(self):
        """Test if Ollama is running and accessible."""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": "test",
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=5
            )
            response.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.ollama_url}: {e}")
    
    def _load_apertus_model(self):
        """Placeholder for compatibility - no longer loads Apertus model."""
        self.logger.info("Using Ollama for language processing")
        # Keep method for compatibility but don't load model
        self.tokenizer = None
        self.model = None
    
    def extract_concepts(self, text: str) -> Tuple[Dict[str, List[str]], str]:
        """
        Extract concepts from human text using Ollama.
        Returns structured concepts and detected language.
        """
        
        self.logger.info(f"Extracting concepts from text: {text[:50]}...")
        
        # Use simplified extraction - Ollama will handle language automatically
        concepts = {}
        
        for concept_type, prompt in self.extraction_prompts.items():
            extraction_query = f"{prompt}\n\"{text}\"\n\nResponse:"
            
            response = self._query_ollama(extraction_query, max_tokens=100, temperature=0.3)
            
            # Parse the response into concept list
            concepts[concept_type] = self._parse_concept_response(response, concept_type)
            
            self.logger.debug(f"Extracted {concept_type}: {concepts[concept_type]}")
        
        # Return concepts with default language (Ollama handles any language)
        return concepts, "auto"
    
    def synthesise_response(self, 
                          colour_descriptors: List[str], 
                          target_language: str,
                          original_concepts: Dict[str, List[str]]) -> str:
        """
        Translate colour pattern descriptors back to human language using Ollama.
        This is synthesis from colour, NOT generation of new content.
        """
        
        self.logger.info(f"Synthesising response from {len(colour_descriptors)} colour descriptors")
        
        # Create synthesis prompt for Ollama
        concepts_text = ", ".join(colour_descriptors)
        context_text = ", ".join(original_concepts.get('context', ['general conversation']))
        intent_text = ", ".join(original_concepts.get('intent', ['seeking understanding']))
        
        synthesis_prompt = f"""Express these concepts naturally: {concepts_text}. 
Based on the original context '{context_text}' and responding to '{intent_text}', 
synthesise a thoughtful response:"""
        
        # Generate response using Ollama
        response = self._query_ollama(synthesis_prompt, max_tokens=300, temperature=0.7)
        
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
    
    def _query_ollama(self, prompt: str, max_tokens: int = 100, temperature: float = 0.5) -> str:
        """Query Ollama model for text generation, handling thinking steps."""
        
        self.logger.debug(f"Querying Ollama with prompt: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
        
        try:
            # Use streaming to handle thinking process
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": True,  # Enable streaming
                    "options": {
                        "temperature": temperature,
                        "num_predict": 20480,  # Large enough for extensive thinking
                        # No stop tokens - let Ollama complete naturally
                    }
                },
                stream=True,
                timeout=3000  # 50 minutes for extensive thinking
            )
            
            response.raise_for_status()
            
            # Collect the streamed response
            full_response = ""
            thinking_logged = False
            think_tag_closed = False
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            chunk_text = chunk['response']
                            full_response += chunk_text
                            
                            # Log thinking message once when we detect thinking started
                            if not thinking_logged and '<think>' in full_response:
                                self.logger.info("Ollama is thinking...")
                                thinking_logged = True
                            
                            # Check if thinking has ended
                            if '</think>' in full_response:
                                think_tag_closed = True
                                self.logger.debug("Found </think> tag - thinking complete")
                        
                        # Only stop if Ollama says it's done AND we've either:
                        # 1. Seen </think> tag closed, or
                        # 2. Haven't started thinking (quick response)
                        if chunk.get('done', False):
                            if think_tag_closed:
                                self.logger.debug("Generation complete with </think> found")
                                break
                            elif not thinking_logged:
                                self.logger.debug("Generation complete without thinking phase")
                                break
                            # Otherwise, ignore the done flag and keep streaming
                            
                    except json.JSONDecodeError:
                        continue
            
            # Warn if thinking started but never completed properly
            if thinking_logged and not think_tag_closed:
                self.logger.warning("Thinking started but </think> not found - response may be incomplete")
            
            # Extract only the final answer
            generated_text = self._extract_final_answer(full_response, prompt)
            
            if not generated_text:
                self.logger.warning("Ollama returned empty response after processing")
                self.logger.debug(f"Full response was: {full_response[:500]}...")
                return ""
            
            self.logger.info(f"Ollama response: '{generated_text[:50]}{'...' if len(generated_text) > 50 else ''}'")
            return generated_text
            
        except requests.exceptions.Timeout:
            self.logger.error("Ollama request timed out after 50 minutes")
            return ""
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error querying Ollama: {e}")
            return ""
        except Exception as e:
            self.logger.error(f"Unexpected error querying Ollama: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return ""
    
    def _extract_final_answer(self, response: str, original_prompt: str) -> str:
        """Extract the final answer from Ollama response, handling thinking tags."""
        
        import re
        
        # First remove <think>...</think> blocks
        if '<think>' in response and '</think>' in response:
            # Extract everything after </think>
            parts = response.split('</think>')
            if len(parts) > 1:
                answer = parts[-1].strip()
                if answer:
                    # Remove quotes if present
                    if answer.startswith('"') and answer.endswith('"'):
                        return answer[1:-1]
                    return answer
        
        # Fallback to the previous extraction logic if no think tags
        # For concept extraction prompts, look for short simplified phrases
        if "Do not provide explanations only return the simplified output phrase" in original_prompt:
            # Split response into lines
            lines = response.strip().split('\n')
            
            # Look for the last non-empty line that looks like a simplified phrase
            thinking_indicators = [
                'thinking', 'we are given', 'input text:', 'let me', 'i need to',
                'first,', 'the task', 'instructions:', 'step', 'extract', 'simplify',
                'focus on', 'select', 'ignore', 'preserve', 'render', 'core concepts:'
            ]
            
            for line in reversed(lines):
                line = line.strip()
                if line and len(line) < 100:
                    if not any(indicator in line.lower() for indicator in thinking_indicators):
                        if line.startswith('"') and line.endswith('"'):
                            return line[1:-1]
                        elif not line.startswith('Input text:') and not line.startswith('Output:'):
                            return line
            
            # Look for patterns like "Output: X"
            output_match = re.search(r'(?:Output|Result|Answer|Response):\s*"?([^"\n]+)"?', response, re.IGNORECASE)
            if output_match:
                return output_match.group(1).strip()
        
        # General fallback cleaning
        cleaned = response.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "The answer is:", "Final answer:", "Response:", "Output:", 
            "Result:", "Core concepts:", "Simplified:"
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        return cleaned
    
    # Keep this method for backward compatibility
    def _query_apertus(self, prompt: str, max_tokens: int = 100, temperature: float = 0.5) -> str:
        """Redirects to Ollama query for backward compatibility."""
        return self._query_ollama(prompt, max_tokens, temperature)
    
    def _parse_concept_response(self, response: str, concept_type: str) -> List[str]:
        """Parse Ollama response into concept list."""
        
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
        """Map low frequency to fundamental concept descriptor."""
        
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
        """Map mid frequency to relationship concept descriptor."""
        
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
        """Map high frequency to detail concept descriptor."""
        
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
        """Calculate spectral entropy as measure of complexity."""
        
        # Normalise amplitude
        power = amplitude ** 2
        power_norm = power / (np.sum(power) + 1e-8)
        
        # Calculate entropy
        entropy = -np.sum(power_norm * np.log(power_norm + 1e-8))
        
        # Normalise by maximum possible entropy
        max_entropy = np.log(len(amplitude))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_phase_coherence(self, phase: np.ndarray) -> float:
        """Calculate phase coherence as measure of unity."""
        
        if len(phase) == 0:
            return 0.5
        
        # Calculate coherence as magnitude of mean phase vector
        mean_phase_vector = np.mean(np.exp(1j * phase))
        coherence = np.abs(mean_phase_vector)
        
        return float(coherence)
    
    def synthesise_from_concepts(self, concepts: List[str], original_question: str = None, colour_guidance: np.ndarray = None) -> str:
        """
        Synthesise natural language response from concept list using Ollama.
        This ensures NO template responses - all text comes from Ollama generation.
        """

        self.logger.info(f"  Ollama synthesis from concepts: {concepts[:3]}...")
        self.logger.info(f" Original question: '{original_question[:50] if original_question else 'None'}{'...' if original_question and len(original_question) > 50 else ''}'")

        if not concepts:
            self.logger.warning(" No concepts provided for synthesis")
            return "I cannot form a response from the given patterns."

        # Use the concepts as they are (should be coherent phrases now)
        concept_text = "; ".join(concepts[:3])  # Use semicolon to separate multiple concept phrases
        self.logger.info(f" Concept text for synthesis: '{concept_text}'")

        # Use the improved synthesis prompt structure
        synthesis_prompt = f"""You are a linguistic synthesiser. Your task is to create a natural, coherent response by weaving together concept words into grammatically correct sentences.

Instructions:
1. You will receive CONCEPT WORDS that represent the core semantic elements of a response
2. You will receive the ORIGINAL QUESTION that prompted these concepts
3. Create a natural sentence/paragraph that incorporates these concepts as the answer using the language of the question
4. Maintain logical flow and grammatical correctness
5. Ensure the response directly addresses the original question
6. Use appropriate connecting words and grammar to make the concepts flow naturally in the language used

Original question: {original_question or "general inquiry"}
Concept words: {concept_text}

Natural response:"""

        self.logger.debug(f" Synthesis prompt: {synthesis_prompt[:200]}...")

        # Use Ollama to generate natural response
        self.logger.info(" Querying Ollama for natural language synthesis...")
        response = self._query_ollama(synthesis_prompt, max_tokens=200, temperature=0.7)

        if not response or len(response.strip()) < 10:
            self.logger.error(" Ollama synthesis failed completely")
            self.logger.error(f" Failed synthesis prompt: {synthesis_prompt[:100]}...")
            raise RuntimeError("Ollama synthesis failed - no valid response generated. Check Ollama is running.")

        self.logger.info(f" Successfully synthesised response: '{response[:80]}{'...' if len(response) > 80 else ''}'")
        return response.strip()
    
    def synthesise_from_frequency_pattern(self, colour_waveform: np.ndarray) -> str:
        """
        Synthesise response from raw frequency pattern by first extracting concepts.
        """
        
        # Convert numpy array to tensor for analysis
        colour_tensor = tf.convert_to_tensor(colour_waveform, dtype=tf.complex64)
        
        # Extract concept descriptors from the frequency pattern
        descriptors = self.colour_to_concept_descriptors(colour_tensor)
        
        # Synthesise from extracted concepts
        return self.synthesise_from_concepts(descriptors)
    
    def extract_core_concepts(self, text: str) -> List[str]:
        """
        Extract core semantic concepts from text using Ollama.
        Language-agnostic - works with any language.
        """
        
        self.logger.info(f" Ollama concept extraction from: '{text[:50]}{'...' if len(text) > 50 else ''}'")

        # Use the multilingual concept extraction prompt
        extraction_prompt = self.concept_extraction_prompt.format(text=text)
        self.logger.debug(f"  Using extraction prompt: {extraction_prompt[:100]}...")

        # Query Ollama with lower temperature for more focused extraction
        self.logger.debug(" Querying Ollama for concept extraction...")
        response = self._query_ollama(extraction_prompt, max_tokens=100, temperature=0.3)
        self.logger.debug(f" Ollama raw response: '{response[:100]}{'...' if len(response) > 100 else ''}'")
        
        # The response should be a simplified phrase - return it as-is if valid
        if response and len(response.strip()) > 2:
            # Clean up the response
            cleaned_response = response.strip()
            
            # Remove quotes if present
            if cleaned_response.startswith('"') and cleaned_response.endswith('"'):
                cleaned_response = cleaned_response[1:-1]
            
            # Check if this looks like a valid simplified phrase
            # (contains at least one verb or action word)
            action_words = ['explain', 'describe', 'show', 'tell', 'find', 'calculate', 
                          'analyze', 'compare', 'define', 'understand', 'help', 'create',
                          'make', 'write', 'generate', 'solve', 'determine', 'identify']
            
            has_action = any(word in cleaned_response.lower() for word in action_words)
            
            if has_action and len(cleaned_response) < 100:
                # This is a valid simplified phrase - return it as a single concept
                self.logger.info(f" Extracted simplified phrase: '{cleaned_response}'")
                return [cleaned_response]
            
            # If it doesn't look like a simplified phrase, try parsing as concepts
            # This handles cases where Ollama might return a list instead
            concepts = self._parse_concept_response(response, "core")
            if concepts and concepts[0] != "general core":
                self.logger.info(f" Extracted concepts: {concepts}")
                return concepts
        
        # Enhanced fallback - if extraction fails, pass input directly
        self.logger.warning(f" Concept extraction failed for: {text[:50]}...")
        self.logger.info(" Using direct text fallback - bypassing concept extraction")
        # Return special flag to indicate direct processing
        return ["__DIRECT_PROCESSING__", text]
