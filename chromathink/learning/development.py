"""
DevelopmentalLearning: ChromaThink learns through dialogue

ChromaThink learns through dialogue, like a child asking endless 'why' questions.
Apertus serves as the patient teacher, responding in any language whilst
ChromaThink thinks only in colour.
"""

import tensorflow as tf
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Union

from .cognitive_spectrum import CognitiveSpectrum
from .curiosity_engine import CuriosityEngine
from .chromatic_memory import ChromaticMemory
from ..core.colour_utils import colour_distance


class DevelopmentalLearning:
    """
    ChromaThink learns through dialogue, like a child asking endless 'why' questions.
    Apertus serves as the patient teacher, responding in any language whilst
    ChromaThink thinks only in colour.
    """
    
    def __init__(self, 
                 spectrum_dims=512,
                 apertus_model="microsoft/DialoGPT-medium",  # Use a more accessible model
                 device="auto"):
        
        # Initialise colour-based cognition
        self.spectrum_dims = spectrum_dims
        self.cognitive_spectrum = CognitiveSpectrum(spectrum_dims)
        
        # Initialise Apertus as teacher (using DialoGPT for now)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(apertus_model)
            self.teacher = AutoModelForCausalLM.from_pretrained(
                apertus_model,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.teacher.eval()  # Teacher doesn't learn from student
            
            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            print(f"Warning: Could not load teacher model {apertus_model}: {e}")
            print("Falling back to mock teacher for demonstration")
            self.teacher = None
            self.tokenizer = None
        
        # Curiosity engine - generates questions from colour states
        self.curiosity = CuriosityEngine(spectrum_dims)
        
        # Chromatic memory - stores learned associations
        self.colour_memory = ChromaticMemory(spectrum_dims)
        
        # Learning rate in colour space (not gradient descent)
        self.chromatic_plasticity = tf.Variable(0.1, trainable=False)
        
        # Developmental stage (child -> adolescent -> adult)
        self.developmental_stage = tf.Variable(0.0, trainable=False)
        
        # Language-to-colour translation system
        self.language_encoder = self._build_language_encoder()
        self.colour_decoder = self._build_colour_decoder()
    
    def _build_language_encoder(self):
        """Build system to encode text to colour."""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.spectrum_dims * 2, activation='tanh'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.spectrum_dims, activation='linear')
        ])
    
    def _build_colour_decoder(self):
        """Build system to decode colour to language patterns."""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.spectrum_dims * 2, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(256, activation='tanh'),  # Language embedding size
            tf.keras.layers.Dense(100, activation='softmax')  # Concept probabilities
        ])
    
    def learn_through_dialogue(self, 
                              initial_colour_state=None,
                              num_exchanges=10,
                              languages=['english']):  # Simplified for now
        """
        Engage in dialogue with Apertus, learning from each exchange.
        Questions arise from colour curiosity, answers create new colour patterns.
        """
        
        # Start from random curiosity or provided state
        if initial_colour_state is None:
            colour_state = self.curiosity.generate_wonder()
        else:
            colour_state = initial_colour_state
        
        dialogue_history = []
        
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
            
            print(f"ChromaThink wonders (in colour): {self._describe_colour(question_colour)}")
            print(f"ChromaThink asks: {question_text}")
            
            # Ask Apertus teacher
            response_text = self.ask_teacher(question_text)
            print(f"Apertus responds: {response_text}")
            
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
            
            print(f"Resonance: {resonance:.3f}, Development stage: {float(self.developmental_stage):.3f}")
        
        return dialogue_history
    
    def ask_teacher(self, question: str, temperature=0.8, top_p=0.9):
        """
        Query Apertus with child-like questions.
        The teacher responds patiently, adjusting to developmental stage.
        """
        
        if self.teacher is None or self.tokenizer is None:
            # Mock teacher for demonstration
            return self._mock_teacher_response(question)
        
        try:
            # Adjust prompt based on developmental stage
            stage_value = float(self.developmental_stage.numpy())
            
            if stage_value < 0.3:  # Child stage
                prompt = f"Explain to a curious 5-year-old: {question}"
            elif stage_value < 0.7:  # Adolescent stage
                prompt = f"Explain with examples and connections: {question}"
            else:  # Adult stage
                prompt = f"Provide a detailed, nuanced explanation: {question}"
            
            # Encode and generate
            inputs = self.tokenizer.encode(prompt + self.tokenizer.eos_token, 
                                         return_tensors='pt')
            
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            
            with torch.no_grad():
                outputs = self.teacher.generate(
                    inputs,
                    max_new_tokens=150,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][len(inputs[0]):], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"Error with teacher model: {e}")
            return self._mock_teacher_response(question)
    
    def _mock_teacher_response(self, question: str):
        """Mock teacher responses for demonstration."""
        responses = [
            f"That's a wonderful question about {question.lower()}! Let me think...",
            f"Great curiosity! {question} is something many people wonder about.",
            f"I'm glad you asked about {question.lower()}. Here's what I think...",
            f"What an interesting way to think about {question.lower()}!",
            f"You're really thinking deeply about {question.lower()}. That's excellent!"
        ]
        return np.random.choice(responses)
    
    def render_to_language(self, colour_state, question_type='what', language='english'):
        """
        Translate colour state to language question.
        This is an approximation - colour is richer than language.
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
                "What is the meaning of this sensation?"
            ],
            'why': [
                "Why do I feel this way?",
                "Why is this pattern occurring?",
                "Why does this resonate with me?",
                "Why am I drawn to this idea?"
            ],
            'how': [
                "How does this work?",
                "How can I understand this better?",
                "How do these patterns connect?",
                "How should I think about this?"
            ]
        }
        
        question_templates = templates.get(question_type, templates['what'])
        base_question = np.random.choice(question_templates)
        
        # Add intensity based on colour magnitude
        if avg_magnitude > 0.7:
            intensity_words = ["strongly", "intensely", "deeply", "powerfully"]
            intensity = np.random.choice(intensity_words)
            base_question = base_question.replace("I", f"I {intensity}")
        
        return base_question
    
    def encode_from_language(self, text: str):
        """
        Encode language response to colour representation.
        This is approximate - we're translating from tokens to frequencies.
        """
        
        # Simple text encoding based on character and word features
        text_lower = text.lower()
        
        # Create features from text
        text_length = len(text)
        word_count = len(text.split())
        char_frequencies = np.zeros(256)  # ASCII characters
        
        for char in text:
            if ord(char) < 256:
                char_frequencies[ord(char)] += 1
        
        # Normalize character frequencies
        if np.sum(char_frequencies) > 0:
            char_frequencies = char_frequencies / np.sum(char_frequencies)
        
        # Create basic colour representation
        # This is a simplified encoding - in reality would use embeddings
        colour_features = np.concatenate([
            [text_length / 1000.0],  # Normalized length
            [word_count / 100.0],    # Normalized word count
            char_frequencies[:self.spectrum_dims-2]  # Character features
        ])
        
        # Ensure correct size
        if len(colour_features) < self.spectrum_dims:
            padding = np.zeros(self.spectrum_dims - len(colour_features))
            colour_features = np.concatenate([colour_features, padding])
        else:
            colour_features = colour_features[:self.spectrum_dims]
        
        # Convert to complex representation
        colour_tensor = tf.constant(colour_features, dtype=tf.float32)
        colour_tensor = tf.expand_dims(colour_tensor, 0)  # Add batch dimension
        colour_tensor = tf.cast(colour_tensor, tf.complex64)  # Convert to complex
        
        # Process through cognitive spectrum to get proper colour representation
        colour_representation = self.cognitive_spectrum(colour_tensor, training=False)
        
        return colour_representation
    
    def integrate_knowledge(self, question_colour, response_colour, current_state):
        """
        Update colour state based on new knowledge.
        Not replacement but integration—like mixing paints.
        """
        
        # Calculate how much the response resonates with the question
        resonance = self.calculate_resonance(question_colour, response_colour)
        
        # Stronger resonance means more integration  
        integration_strength = float(self.chromatic_plasticity) * resonance
        
        # Blend response into current state
        # Cast integration strength to complex if needed
        if current_state.dtype.is_complex:
            integration_strength = tf.cast(integration_strength, current_state.dtype)
            blend_factor = tf.cast(1 - integration_strength, current_state.dtype)
            harmonic_factor = tf.cast(0.1, current_state.dtype)
        else:
            blend_factor = 1 - integration_strength
            harmonic_factor = 0.1
            
        new_state = blend_factor * current_state + \
                    integration_strength * response_colour
        
        # Add harmonic overtones from the learning
        harmonics = self.generate_harmonics(question_colour, response_colour)
        
        # Ensure harmonics have the same dtype as new_state
        if new_state.dtype != harmonics.dtype:
            harmonics = tf.cast(harmonics, new_state.dtype)
            
        new_state = new_state + harmonic_factor * harmonics
        
        return new_state
    
    def calculate_resonance(self, colour1, colour2):
        """
        Calculate how much two colours resonate with each other.
        """
        # Use colour distance to measure resonance (inverse relationship)
        distance = colour_distance(colour1, colour2, metric='spectral')
        resonance = tf.exp(-distance)  # Convert distance to resonance
        
        return float(resonance)
    
    def generate_harmonics(self, question_colour, response_colour):
        """
        Generate harmonic patterns from question-response interaction.
        """
        # Simple harmonic generation by interference
        if question_colour.dtype.is_complex and response_colour.dtype.is_complex:
            # Complex interference creates natural harmonics
            harmonic = question_colour * tf.math.conj(response_colour)
            # Cast epsilon to match harmonic dtype
            epsilon = tf.cast(1e-8, harmonic.dtype)
            harmonic = harmonic / (tf.math.abs(harmonic) + epsilon)  # Normalize
        else:
            # For real signals, create harmonic through multiplication
            harmonic = question_colour * response_colour
            epsilon = tf.cast(1e-8, harmonic.dtype)
            harmonic = harmonic / (tf.reduce_max(tf.math.abs(harmonic)) + epsilon)
        
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
            
            return f"Mag: {avg_mag:.2f}, Max: {max_mag:.2f}, Dom: {dominant_freq}"
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
        
        print("Learning state reset - ChromaThink is ready for new experiences!")