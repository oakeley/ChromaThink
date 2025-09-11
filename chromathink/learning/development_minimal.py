"""
DevelopmentalLearning: ChromaThink learns through dialogue (Minimal Version)

ChromaThink learns through dialogue, like a child asking endless 'why' questions.
This version works without external LLM dependencies for demonstration.
"""

import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional, Union

from .cognitive_spectrum import CognitiveSpectrum
from .curiosity_engine import CuriosityEngine
from .chromatic_memory import ChromaticMemory
from ..core.colour_utils import colour_distance


class DevelopmentalLearningMinimal:
    """
    Minimal version of developmental learning that works without external LLMs.
    Demonstrates the core colour-based learning concepts.
    """
    
    def __init__(self, spectrum_dims=512):
        
        # Initialise colour-based cognition
        self.spectrum_dims = spectrum_dims
        self.cognitive_spectrum = CognitiveSpectrum(spectrum_dims)
        
        # Curiosity engine - generates questions from colour states
        self.curiosity = CuriosityEngine(spectrum_dims)
        
        # Chromatic memory - stores learned associations
        self.colour_memory = ChromaticMemory(spectrum_dims)
        
        # Learning rate in colour space (not gradient descent)
        self.chromatic_plasticity = tf.Variable(0.1, trainable=False)
        
        # Developmental stage (child -> adolescent -> adult)
        self.developmental_stage = tf.Variable(0.0, trainable=False)
        
        # Simple mock responses for demonstration
        self.mock_responses = [
            "That's a wonderful question! Let me think about that carefully.",
            "Great curiosity! This is something many people wonder about.",
            "What an interesting way to approach this topic!",
            "I can see you're thinking deeply about this. That's excellent!",
            "This reminds me of other fascinating patterns in nature.",
            "Your question shows real insight into how things work.",
            "This is the kind of thinking that leads to great discoveries!",
            "I love how you're connecting different ideas together.",
            "That's a question that even experts find challenging.",
            "Your curiosity is really inspiring!"
        ]
    
    def learn_through_dialogue(self, 
                              initial_colour_state=None,
                              num_exchanges=10):
        """
        Engage in dialogue, learning from each exchange.
        Questions arise from colour curiosity, answers create new colour patterns.
        """
        
        # Start from random curiosity or provided state
        if initial_colour_state is None:
            colour_state = self.curiosity.generate_wonder()
        else:
            colour_state = initial_colour_state
        
        dialogue_history = []
        
        print(f"🎨 Starting learning dialogue with {num_exchanges} exchanges...")
        
        for exchange in range(num_exchanges):
            print(f"\\n--- Learning Exchange {exchange + 1} ---")
            
            # Generate question from current colour state
            question_data = self.curiosity.formulate_question(colour_state)
            question_colour = question_data['question_colour']
            
            # Translate to language (imperfect but necessary)
            question_text = self.render_to_language(
                question_colour, 
                question_type=question_data['question_type']
            )
            
            print(f"ChromaThink wonders: {self._describe_colour(question_colour)}")
            print(f"ChromaThink asks: {question_text}")
            
            # Get mock teacher response
            response_text = self.ask_mock_teacher(question_text)
            print(f"Teacher responds: {response_text}")
            
            # Translate response back to colour
            response_colour = self.encode_from_language(response_text)
            
            # Learn from the exchange
            colour_state = self.integrate_knowledge(
                question_colour, 
                response_colour,
                colour_state
            )
            
            # Store in chromatic memory
            resonance = self.calculate_resonance(question_colour, response_colour)
            self.colour_memory.store_association(
                question_colour,
                response_colour,
                strength=resonance
            )
            
            dialogue_history.append({
                'question_colour': question_colour,
                'question_text': question_text,
                'response_colour': response_colour,
                'response_text': response_text,
                'evolved_state': colour_state,
                'resonance': resonance
            })
            
            # Update developmental stage based on complexity of understanding
            self.update_developmental_stage(colour_state)
            
            print(f"Resonance: {resonance:.3f}, Development: {float(self.developmental_stage):.3f}")
        
        return dialogue_history
    
    def ask_mock_teacher(self, question: str):
        """
        Mock teacher that provides encouraging responses.
        """
        # Add some developmental stage awareness
        stage_value = float(self.developmental_stage.numpy())
        
        if stage_value < 0.3:  # Child stage
            prefix = "Let me explain simply: "
        elif stage_value < 0.7:  # Adolescent stage
            prefix = "That's a good question! "
        else:  # Adult stage
            prefix = "Excellent inquiry. "
        
        response = np.random.choice(self.mock_responses)
        return prefix + response
    
    def render_to_language(self, colour_state, question_type='what'):
        """
        Translate colour state to language question.
        """
        
        # Extract features from colour state
        if colour_state.dtype.is_complex:
            magnitude = tf.math.abs(colour_state)
            phase = tf.math.angle(colour_state)
        else:
            magnitude = colour_state
            phase = tf.zeros_like(magnitude)
        
        # Find dominant patterns
        avg_magnitude = tf.reduce_mean(magnitude)
        max_magnitude = tf.reduce_max(magnitude)
        dominant_freq = tf.argmax(magnitude, axis=-1)
        
        # Create rough language mapping based on colour properties
        templates = {
            'what': [
                "What is this pattern I'm sensing?",
                "What does this feeling represent?",
                "What am I experiencing right now?",
                "What is the meaning behind this?",
                "What connects these sensations?",
                "What should I understand about this?"
            ],
            'why': [
                "Why do I feel this resonance?",
                "Why is this pattern emerging?",
                "Why does this seem important?",
                "Why am I drawn to this idea?",
                "Why do these colours appear together?",
                "Why does this create such harmony?"
            ],
            'how': [
                "How does this pattern work?",
                "How can I understand this better?",
                "How do these frequencies connect?",
                "How should I interpret this?",
                "How do these elements relate?",
                "How can I learn more about this?"
            ]
        }
        
        question_templates = templates.get(question_type, templates['what'])
        base_question = np.random.choice(question_templates)
        
        # Add intensity based on colour magnitude
        if avg_magnitude > 0.7:
            intensity_words = ["strongly", "deeply", "intensely", "powerfully"]
            intensity = np.random.choice(intensity_words)
            base_question = base_question.replace("I", f"I {intensity}")
        
        return base_question
    
    def encode_from_language(self, text: str):
        """
        Encode language response to colour representation.
        """
        
        # Simple text encoding based on character and word features
        text_lower = text.lower()
        
        # Create features from text
        text_length = min(len(text), 500)  # Cap length
        word_count = min(len(text.split()), 50)  # Cap word count
        
        # Character frequency features
        char_features = []
        for char_set in ['aeiou', 'bcdfg', 'hjklm', 'npqrs', 'tvwxyz']:
            count = sum(1 for c in text_lower if c in char_set)
            char_features.append(count / max(1, len(text)))
        
        # Word pattern features
        word_features = [
            text.count('!') / max(1, len(text)),  # Excitement
            text.count('?') / max(1, len(text)),  # Questioning
            text.count('.') / max(1, len(text)),  # Statements
            len([w for w in text.split() if len(w) > 6]) / max(1, len(text.split())),  # Complex words
        ]
        
        # Combine features
        basic_features = [
            text_length / 500.0,  # Normalized length
            word_count / 50.0,    # Normalized word count
        ] + char_features + word_features
        
        # Pad or truncate to spectrum dimensions
        if len(basic_features) < self.spectrum_dims:
            # Use harmonic expansion to fill spectrum
            base_pattern = np.array(basic_features)
            expanded_features = []
            
            for i in range(self.spectrum_dims):
                harmonic = 1 + (i % len(base_pattern))
                feature_idx = i % len(base_pattern)
                value = base_pattern[feature_idx] * np.sin(harmonic * np.pi / 4)
                expanded_features.append(value)
            
            colour_features = np.array(expanded_features[:self.spectrum_dims])
        else:
            colour_features = np.array(basic_features[:self.spectrum_dims])
        
        # Convert to TensorFlow tensor
        colour_tensor = tf.constant(colour_features, dtype=tf.float32)
        colour_tensor = tf.expand_dims(colour_tensor, 0)  # Add batch dimension
        
        # Process through cognitive spectrum
        colour_representation = self.cognitive_spectrum(colour_tensor, training=False)
        
        return colour_representation
    
    def integrate_knowledge(self, question_colour, response_colour, current_state):
        """
        Update colour state based on new knowledge.
        """
        
        # Calculate how much the response resonates with the question
        resonance = self.calculate_resonance(question_colour, response_colour)
        
        # Stronger resonance means more integration
        integration_strength = self.chromatic_plasticity * resonance
        
        # Blend response into current state (cast integration strength to complex if needed)
        if current_state.dtype.is_complex:
            integration_factor = tf.cast(integration_strength, current_state.dtype)
            current_factor = tf.cast(1 - integration_strength, current_state.dtype)
        else:
            integration_factor = integration_strength
            current_factor = 1 - integration_strength
            
        new_state = current_factor * current_state + integration_factor * response_colour
        
        # Add harmonic overtones from the learning
        harmonics = self.generate_harmonics(question_colour, response_colour)
        new_state = new_state + 0.1 * harmonics
        
        return new_state
    
    def calculate_resonance(self, colour1, colour2):
        """
        Calculate how much two colours resonate with each other.
        """
        distance = colour_distance(colour1, colour2, metric='spectral')
        resonance = tf.exp(-distance)  # Convert distance to resonance
        return float(resonance)
    
    def generate_harmonics(self, question_colour, response_colour):
        """
        Generate harmonic patterns from question-response interaction.
        """
        if question_colour.dtype.is_complex and response_colour.dtype.is_complex:
            # Complex interference creates natural harmonics
            harmonic = question_colour * tf.math.conj(response_colour)
            harmonic_abs = tf.math.abs(harmonic) + 1e-8
            harmonic = harmonic / tf.cast(harmonic_abs, harmonic.dtype)  # Normalize
        else:
            # For real signals, create harmonic through multiplication
            harmonic = question_colour * response_colour
            harmonic_max = tf.reduce_max(tf.math.abs(harmonic)) + 1e-8
            harmonic = harmonic / tf.cast(harmonic_max, harmonic.dtype)
        
        return harmonic
    
    def update_developmental_stage(self, colour_state):
        """
        Update developmental stage based on complexity of colour state.
        """
        # Calculate complexity metrics
        if colour_state.dtype.is_complex:
            magnitude = tf.math.abs(colour_state)
            phase = tf.math.angle(colour_state)
            
            # Spectral entropy as complexity measure
            power_spectrum = magnitude ** 2
            normalized_spectrum = power_spectrum / tf.reduce_sum(power_spectrum)
            entropy = -tf.reduce_sum(
                normalized_spectrum * tf.math.log(normalized_spectrum + 1e-8)
            )
            
            # Phase coherence
            phase_coherence = tf.math.abs(tf.reduce_mean(tf.exp(tf.complex(0.0, phase))))
            
            complexity = float(entropy) / np.log(self.spectrum_dims)  # Normalize
            coherence = float(phase_coherence)
            
        else:
            # For real signals
            variance = tf.math.reduce_variance(colour_state)
            complexity = float(tf.tanh(variance))  # Bound between 0 and 1
            coherence = 0.5
        
        # Developmental stage increases with balanced complexity and coherence
        stage_update = 0.01 * (complexity + coherence) / 2.0
        
        new_stage = self.developmental_stage + stage_update
        new_stage = tf.clip_by_value(new_stage, 0.0, 1.0)
        
        self.developmental_stage.assign(new_stage)
    
    def _describe_colour(self, colour_state):
        """
        Provide a human-readable description of a colour state.
        """
        if colour_state.dtype.is_complex:
            magnitude = tf.math.abs(colour_state)
            phase = tf.math.angle(colour_state)
            
            avg_mag = float(tf.reduce_mean(magnitude))
            max_mag = float(tf.reduce_max(magnitude))
            dominant_freq = int(tf.argmax(magnitude, axis=-1)[0])
            
            return f"Mag: {avg_mag:.2f}, Peak: {max_mag:.2f}, Dom: {dominant_freq}"
        else:
            avg_val = float(tf.reduce_mean(colour_state))
            max_val = float(tf.reduce_max(colour_state))
            return f"Avg: {avg_val:.2f}, Max: {max_val:.2f}"
    
    def get_learning_metrics(self):
        """
        Return comprehensive learning metrics.
        """
        cognitive_metrics = self.cognitive_spectrum.get_cognitive_metrics()
        curiosity_metrics = self.curiosity.get_curiosity_metrics()
        memory_metrics = self.colour_memory.get_memory_metrics()
        
        return {
            'developmental_stage': float(self.developmental_stage),
            'chromatic_plasticity': float(self.chromatic_plasticity),
            'cognitive': cognitive_metrics,
            'curiosity': curiosity_metrics,
            'memory': memory_metrics
        }
    
    def reset_learning_state(self):
        """Reset the learning system to initial state."""
        self.developmental_stage.assign(0.0)
        self.chromatic_plasticity.assign(0.1)
        self.cognitive_spectrum.reset_memory()
        self.curiosity.reset_curiosity()
        self.colour_memory.reset_memory()
        
        print("🎨 Learning state reset - ChromaThink is ready for new experiences!")