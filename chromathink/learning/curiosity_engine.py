"""
CuriosityEngine: Generates questions and wonder from colour states

Like a child's endless "why" questions, but expressed as colour patterns.
"""

import tensorflow as tf
import numpy as np
from ..core.colour_utils import colour_distance
from ..layers.interference import InterferenceLayer


class CuriosityEngine(tf.keras.Model):
    """
    Generates curiosity and questions from colour states.
    Curiosity emerges from colour pattern gaps and unexpected resonances.
    """
    
    def __init__(self, 
                 spectrum_dims=512,
                 curiosity_threshold=0.3,
                 question_types=['what', 'why', 'how', 'when', 'where'],
                 name='curiosity_engine'):
        super().__init__(name=name)
        
        self.spectrum_dims = spectrum_dims
        self.curiosity_threshold = curiosity_threshold
        self.question_types = question_types
        
        # Pattern detection for gaps in understanding
        self.gap_detector = tf.keras.Sequential([
            tf.keras.layers.Dense(spectrum_dims // 2, activation='tanh'),
            tf.keras.layers.Dense(spectrum_dims // 4, activation='relu'),
            tf.keras.layers.Dense(spectrum_dims, activation='linear')
        ])
        
        # Question type classifier based on colour patterns
        self.question_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(len(question_types) * 2, activation='tanh'),
            tf.keras.layers.Dense(len(question_types), activation='softmax')
        ])
        
        # Interference layer for creating question complexity
        self.question_interference = InterferenceLayer(
            interference_type='full',
            amplitude_weighting=True,
            phase_coupling=True,
            nonlinear_mixing=True
        )
        
        # Memory of previous questions to avoid repetition
        self.question_memory = []
        self.max_memory_size = 50
    
    def generate_wonder(self, base_uncertainty=0.7):
        """
        Generate a pure state of wonder - the starting point for all curiosity.
        
        Args:
            base_uncertainty: Base level of uncertainty/exploration
        """
        
        # Create structured randomness representing wonder
        frequencies = tf.linspace(0.1, 20.0, self.spectrum_dims)
        
        # Multiple overlapping sine waves with golden ratio relationships
        golden_ratio = (1 + np.sqrt(5)) / 2
        wonder_components = []
        
        for i in range(5):
            freq_scale = golden_ratio ** i
            amplitude = base_uncertainty * (0.8 ** i)
            
            component = amplitude * tf.sin(
                2 * np.pi * frequencies * freq_scale / 10.0 + 
                tf.random.uniform([1]) * 2 * np.pi
            )
            wonder_components.append(component)
        
        # Combine components with interference
        wonder_state = tf.reduce_sum(tf.stack(wonder_components), axis=0)
        wonder_state = tf.expand_dims(wonder_state, 0)  # Add batch dimension
        
        # Add complex structure
        wonder_complex = tf.cast(wonder_state, tf.complex64)
        
        return wonder_complex
    
    def detect_knowledge_gaps(self, colour_state):
        """
        Detect gaps in knowledge by finding incomplete patterns.
        
        Args:
            colour_state: Current colour representation of knowledge
        """
        
        # Use gap detector to find missing patterns
        if colour_state.dtype.is_complex:
            # Extract magnitude for gap detection
            magnitude = tf.math.abs(colour_state)
        else:
            magnitude = colour_state
        
        # Ensure proper shape for gap detector (flatten extra dimensions)
        original_shape = magnitude.shape
        if len(magnitude.shape) > 2:
            magnitude = tf.reshape(magnitude, [magnitude.shape[0], -1])
        
        predicted_complete = self.gap_detector(magnitude, training=False)
        
        # Restore original shape if needed
        if len(original_shape) > 2:
            predicted_complete = tf.reshape(predicted_complete, original_shape)
        
        # Calculate gap as difference between predicted complete and current
        gap_magnitude = tf.reduce_mean(tf.square(predicted_complete - magnitude), axis=-1)
        
        # Find specific gap locations
        pointwise_gaps = tf.square(predicted_complete - magnitude)
        gap_threshold = tf.reduce_mean(pointwise_gaps) + tf.math.reduce_std(pointwise_gaps)
        
        gap_locations = tf.where(pointwise_gaps > gap_threshold)
        
        return {
            'gap_magnitude': gap_magnitude,
            'gap_locations': gap_locations,
            'completion_prediction': predicted_complete
        }
    
    def formulate_question(self, colour_state, curiosity_intensity=1.0):
        """
        Formulate a question based on current colour state and detected gaps.
        
        Args:
            colour_state: Current understanding as colour
            curiosity_intensity: How intensely curious (0.0 to 2.0)
        """
        
        # Detect knowledge gaps
        gaps = self.detect_knowledge_gaps(colour_state)
        
        # Only generate question if gap is significant enough
        gap_magnitude_scalar = float(tf.reduce_mean(gaps['gap_magnitude']))
        if gap_magnitude_scalar < self.curiosity_threshold:
            # Add some randomness to encourage exploration
            noise = tf.random.normal(tf.shape(colour_state)) * 0.1 * curiosity_intensity
            
            # Ensure dtype compatibility
            if colour_state.dtype.is_complex and not noise.dtype.is_complex:
                noise = tf.cast(noise, colour_state.dtype)
            elif not colour_state.dtype.is_complex and noise.dtype.is_complex:
                noise = tf.cast(noise, colour_state.dtype)
                
            colour_state = colour_state + noise
            gaps = self.detect_knowledge_gaps(colour_state)
        
        # Determine question type based on colour patterns
        if colour_state.dtype.is_complex:
            question_input = tf.math.abs(colour_state)
        else:
            question_input = colour_state
        
        # Ensure proper shape for question classifier
        if len(question_input.shape) > 2:
            question_input = tf.reshape(question_input, [question_input.shape[0], -1])
            
        question_probs = self.question_classifier(question_input, training=False)
        question_type_idx = tf.argmax(question_probs, axis=-1)[0]
        question_type = self.question_types[int(question_type_idx)]
        
        # Create question colour by interfering current state with gap pattern
        gap_colour = tf.cast(gaps['completion_prediction'], colour_state.dtype)
        gap_colour = tf.expand_dims(gap_colour, 0)  # Add batch dimension
        
        # Use interference to create question complexity
        question_colour = self.question_interference([colour_state, gap_colour])
        
        # Amplify based on curiosity intensity
        question_colour = question_colour * curiosity_intensity
        
        # Store in memory to avoid immediate repetition
        self._store_question_memory(question_colour)
        
        return {
            'question_colour': question_colour,
            'question_type': question_type,
            'gap_magnitude': gaps['gap_magnitude'],
            'curiosity_intensity': curiosity_intensity
        }
    
    def check_novelty(self, new_question_colour, novelty_threshold=0.5):
        """
        Check if a question is novel compared to recent questions.
        
        Args:
            new_question_colour: Colour representation of new question
            novelty_threshold: Minimum distance for novelty
        """
        
        if not self.question_memory:
            return True
        
        # Compare with recent questions
        distances = []
        for prev_question in self.question_memory[-10:]:  # Check last 10 questions
            distance = colour_distance(
                new_question_colour, 
                prev_question,
                metric='spectral'
            )
            distances.append(float(distance))
        
        min_distance = min(distances)
        return min_distance > novelty_threshold
    
    def _store_question_memory(self, question_colour):
        """Store question in memory for novelty checking."""
        self.question_memory.append(tf.identity(question_colour))
        
        # Maintain memory size limit
        if len(self.question_memory) > self.max_memory_size:
            self.question_memory.pop(0)
    
    def get_curiosity_metrics(self):
        """
        Return metrics about current curiosity state.
        """
        
        if not self.question_memory:
            return {
                'question_count': 0,
                'average_novelty': 0.0,
                'curiosity_diversity': 0.0
            }
        
        # Calculate diversity of recent questions
        recent_questions = self.question_memory[-5:]
        if len(recent_questions) < 2:
            diversity = 0.0
        else:
            distances = []
            for i, q1 in enumerate(recent_questions):
                for j, q2 in enumerate(recent_questions[i+1:], i+1):
                    dist = colour_distance(q1, q2, metric='spectral')
                    distances.append(float(dist))
            
            diversity = np.mean(distances) if distances else 0.0
        
        return {
            'question_count': len(self.question_memory),
            'average_novelty': diversity,
            'curiosity_diversity': diversity,
            'memory_utilization': len(self.question_memory) / self.max_memory_size
        }
    
    def reset_curiosity(self):
        """Reset curiosity state (clear question memory)."""
        self.question_memory = []