"""
Analyse how ChromaThink formulates questions in its pre-trained state.
Track the colour patterns that emerge as curiosity.
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Dict, Any, Optional


class QuestionFormationDebugger:
    """
    Analyse how ChromaThink formulates questions in its pre-trained state.
    Track the colour patterns that emerge as curiosity.
    """
    
    def __init__(self, chromathink_model, apertus_translator=None):
        self.model = chromathink_model
        self.translator = apertus_translator
        self.logger = logging.getLogger("ChromaThink.Questions")
        
    def trace_question_formation(self, initial_state=None) -> Dict[str, Any]:
        """
        Debug the complete question formation pipeline.
        Track how colours become words.
        """
        
        self.logger.info("="*50)
        self.logger.info("TRACING QUESTION FORMATION")
        self.logger.info("="*50)
        
        # Start with initial colour state
        if initial_state is None:
            initial_state = self.model.curiosity.generate_wonder()
            self.logger.info("Generated random wonder state")
        
        # Log the colour state
        self.log_colour_state(initial_state, "Initial colour state")
        
        # Formulate question in colour space
        question_data = self.model.curiosity.formulate_question(initial_state)
        question_colour = question_data['question_colour']
        question_type = question_data.get('question_type', 'unknown')
        
        self.log_colour_state(question_colour, "Question colour pattern")
        
        # Identify question type from colour if not provided
        if question_type == 'unknown':
            question_type = self.identify_question_type(question_colour)
        self.logger.info(f"Detected question type: {question_type}")
        
        # Render to language (multiple attempts for debugging)
        rendered_texts = {}
        
        # Try different approaches to language rendering
        try:
            # Direct rendering using the model's method
            if hasattr(self.model, 'render_to_language'):
                text = self.model.render_to_language(question_colour, question_type=question_type)
                rendered_texts['direct'] = text
                self.logger.info(f"Direct rendering: {text}")
            else:
                # Fallback: create simple question from type
                fallback_questions = {
                    'what': 'What is this about?',
                    'why': 'Why does this happen?', 
                    'how': 'How does this work?',
                    'when': 'When does this occur?',
                    'where': 'Where is this happening?',
                    'who': 'Who is involved in this?'
                }
                text = fallback_questions.get(question_type, 'What is the meaning of this?')
                rendered_texts['fallback'] = text
                self.logger.info(f"Fallback rendering: {text}")
                
        except Exception as e:
            self.logger.warning(f"Could not render to language: {e}")
            rendered_texts['error'] = f"Rendering failed: {str(e)}"
        
        # Analyse colour-to-word mapping if possible
        try:
            self.analyse_colour_word_mapping(question_colour)
        except Exception as e:
            self.logger.warning(f"Could not analyse colour-word mapping: {e}")
        
        return {
            'initial_state': initial_state,
            'question_colour': question_colour,
            'question_type': question_type,
            'rendered_texts': rendered_texts,
            'question_data': question_data
        }
    
    def log_colour_state(self, colour_state, description: str):
        """
        Detailed logging of colour state properties.
        """
        
        # Convert to numpy for analysis
        if isinstance(colour_state, tf.Tensor):
            colour_np = colour_state.numpy()
        else:
            colour_np = np.array(colour_state)
        
        # Handle different tensor shapes
        if len(colour_np.shape) > 1:
            colour_np = colour_np.flatten()
        
        amplitude = np.abs(colour_np)
        phase = np.angle(colour_np)
        
        self.logger.info(f"\n{description}:")
        self.logger.info(f"  Shape: {colour_state.shape}")
        self.logger.info(f"  Dominant frequencies: {np.argsort(amplitude)[-5:][::-1]}")
        self.logger.info(f"  Amplitude range: [{amplitude.min():.4f}, {amplitude.max():.4f}]")
        self.logger.info(f"  Phase distribution: mean={phase.mean():.4f}, std={phase.std():.4f}")
        self.logger.info(f"  Spectral entropy: {self.calculate_entropy(amplitude):.4f}")
        self.logger.info(f"  Magnitude: {np.linalg.norm(amplitude):.4f}")
        
    def calculate_entropy(self, amplitude_spectrum):
        """Calculate spectral entropy of amplitude spectrum"""
        # Normalize to probability distribution
        normalized = amplitude_spectrum / (np.sum(amplitude_spectrum) + 1e-8)
        
        # Calculate entropy
        entropy = -np.sum(normalized * np.log(normalized + 1e-8))
        
        return entropy
        
    def identify_question_type(self, question_colour):
        """
        Reverse-engineer which question type was selected.
        """
        
        if not hasattr(self.model.curiosity, 'question_colours'):
            self.logger.warning("Model doesn't have question_colours attribute")
            return "unknown"
        
        best_match = None
        best_resonance = 0
        
        # Convert question_colour to numpy for compatibility
        if isinstance(question_colour, tf.Tensor):
            q_colour_np = question_colour.numpy()
        else:
            q_colour_np = np.array(question_colour)
        
        for q_type, q_colour in self.model.curiosity.question_colours.items():
            try:
                # Convert template colour to numpy
                if isinstance(q_colour, tf.Tensor):
                    template_np = q_colour.numpy()
                else:
                    template_np = np.array(q_colour)
                
                # Ensure shapes match
                if q_colour_np.shape != template_np.shape:
                    # Reshape to match
                    min_len = min(len(q_colour_np.flatten()), len(template_np.flatten()))
                    q_flat = q_colour_np.flatten()[:min_len]
                    t_flat = template_np.flatten()[:min_len]
                else:
                    q_flat = q_colour_np.flatten()
                    t_flat = template_np.flatten()
                
                # Calculate resonance (complex dot product)
                resonance = np.abs(np.sum(q_flat * np.conj(t_flat)))
                
                self.logger.debug(f"  Resonance with '{q_type}': {resonance:.4f}")
                
                if resonance > best_resonance:
                    best_resonance = resonance
                    best_match = q_type
                    
            except Exception as e:
                self.logger.debug(f"  Error calculating resonance for '{q_type}': {e}")
        
        return best_match or "unknown"
    
    def analyse_colour_word_mapping(self, colour):
        """
        Debug how colours map to words through the language renderer.
        """
        
        self.logger.info("\nAnalysing colour-to-word mapping:")
        
        # Check if model has language renderer
        if not hasattr(self.model, 'language_renderer'):
            self.logger.warning("Model doesn't have language_renderer")
            return
        
        try:
            # Get word probabilities
            word_probs = self.model.language_renderer.render(colour)
            
            if isinstance(word_probs, tf.Tensor):
                word_probs_np = word_probs.numpy()
            else:
                word_probs_np = np.array(word_probs)
            
            # Get top 10 words
            top_indices = np.argsort(word_probs_np)[-10:][::-1]
            
            for idx in top_indices:
                word = getattr(self.model.language_renderer, 'index_to_word', {}).get(
                    int(idx), f"<UNK_{idx}>"
                )
                prob = word_probs_np[idx]
                
                # Try to get the word's colour signature
                resonance = 0.0
                try:
                    if hasattr(self.model.language_renderer, 'word_colours'):
                        word_colour_tensor = self.model.language_renderer.word_colours[idx]
                        if isinstance(word_colour_tensor, tf.Tensor):
                            word_colour = word_colour_tensor.numpy()
                        else:
                            word_colour = np.array(word_colour_tensor)
                        
                        # Calculate resonance
                        colour_np = colour.numpy() if isinstance(colour, tf.Tensor) else np.array(colour)
                        
                        # Handle complex word colours (real, imag) format
                        if len(word_colour.shape) > 1 and word_colour.shape[-1] == 2:
                            word_colour_complex = word_colour[..., 0] + 1j * word_colour[..., 1]
                        else:
                            word_colour_complex = word_colour
                        
                        # Ensure shapes match
                        colour_flat = colour_np.flatten()
                        word_flat = word_colour_complex.flatten()
                        min_len = min(len(colour_flat), len(word_flat))
                        
                        resonance = np.abs(np.sum(
                            colour_flat[:min_len] * np.conj(word_flat[:min_len])
                        ))
                        
                except Exception as e:
                    self.logger.debug(f"Could not calculate resonance for word {word}: {e}")
                
                self.logger.info(f"  Word: '{word}', Prob: {prob:.4f}, Resonance: {resonance:.4f}")
                
        except Exception as e:
            self.logger.warning(f"Error in colour-word mapping analysis: {e}")
    
    def compare_question_formation(self, num_samples: int = 5) -> Dict[str, Any]:
        """
        Compare multiple question formations to understand patterns
        """
        
        self.logger.info(f"\nComparing {num_samples} question formations:")
        
        results = []
        
        for i in range(num_samples):
            self.logger.info(f"\n--- Sample {i+1} ---")
            
            result = self.trace_question_formation()
            results.append(result)
        
        # Analyse patterns across samples
        question_types = [r['question_type'] for r in results]
        type_counts = {}
        for qt in question_types:
            type_counts[qt] = type_counts.get(qt, 0) + 1
        
        self.logger.info(f"\nQuestion type distribution: {type_counts}")
        
        return {
            'samples': results,
            'type_distribution': type_counts,
            'num_samples': num_samples
        }