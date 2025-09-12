"""
TrueColourDevelopment: Developmental learning with pure colour-based cognition

This replaces DevelopmentalLearningMinimal with true colour thinking:
- NO pre-programmed text responses  
- All responses emerge from colour interference patterns
- Uses ApertusTranslator for language bridge
- ChromaThink thinks only in colour space

CRITICAL: ChromaThink never processes text directly.
"""

import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

from ..core.true_colour_thinking import IntegratedChromaThink


class TrueColourDevelopment:
    """
    Developmental learning system using pure colour-based cognition.
    
    This system learns through dialogue while maintaining ChromaThink's
    pure colour thinking - no text processing in the cognitive core.
    
    Learning Process:
    1. Human input -> ApertusTranslator extracts concepts -> Colours
    2. ChromaThink thinks in pure colour space -> Response colours
    3. ApertusTranslator synthesises -> Human text response
    4. Experience stored as colour associations
    """
    
    def __init__(self, 
                 spectrum_dims: int = 512,
                 apertus_path: str = "models/apertus",
                 use_mock_apertus: bool = False,
                 gpu_acceleration: bool = True):
        
        self.spectrum_dims = spectrum_dims
        self.logger = logging.getLogger("TrueColourDevelopment")
        
        # Initialize integrated ChromaThink system
        self.chromathink_system = IntegratedChromaThink(
            apertus_path=apertus_path,
            spectrum_dims=spectrum_dims,
            use_mock_apertus=use_mock_apertus,
            gpu_acceleration=gpu_acceleration
        )
        
        # Developmental stage tracking
        self.developmental_stage = tf.Variable(0.0, trainable=False, name='developmental_stage')
        
        # Learning rate in colour space
        self.chromatic_plasticity = tf.Variable(0.15, trainable=False, name='chromatic_plasticity')
        
        # Curiosity and engagement tracking
        self.curiosity_level = tf.Variable(0.8, trainable=False, name='curiosity_level')
        self.engagement_history = []
        
        # No mock responses - everything emerges from colour dynamics
        self.logger.info("TrueColourDevelopment initialized - pure colour cognition active")
    
    def learn_through_dialogue(self, 
                              teacher_inputs: List[str] = None,
                              num_exchanges: int = 10,
                              thinking_intensity: float = 1.0) -> Dict:
        """
        Learn through dialogue using pure colour thinking.
        
        Args:
            teacher_inputs: Optional list of teacher inputs (if None, generates curious questions)
            num_exchanges: Number of learning exchanges
            thinking_intensity: How intensely ChromaThink should think
        
        Returns:
            Dictionary with dialogue history and learning metrics
        """
        
        self.logger.info(f"Starting true colour learning dialogue - {num_exchanges} exchanges")
        
        dialogue_history = []
        
        for exchange in range(num_exchanges):
            self.logger.info(f"\n--- True Colour Learning Exchange {exchange + 1} ---")
            
            if teacher_inputs and exchange < len(teacher_inputs):
                # Use provided teacher input
                teacher_input = teacher_inputs[exchange]
            else:
                # Generate curious question through colour thinking
                teacher_input = self._generate_curious_question()
            
            self.logger.info(f"Input: {teacher_input}")
            
            # Process through ChromaThink's pure colour cognition
            response_text = self.chromathink_system.process_conversation(
                human_input=teacher_input,
                thinking_intensity=thinking_intensity,
                use_conversation_context=True
            )
            
            self.logger.info(f"ChromaThink Response: {response_text}")
            
            # Calculate engagement and learning metrics
            engagement_score = self._calculate_engagement(teacher_input, response_text)
            
            # Update developmental stage based on interaction complexity
            self._update_developmental_stage(teacher_input, response_text, engagement_score)
            
            # Store exchange
            exchange_data = {
                'exchange_number': exchange + 1,
                'teacher_input': teacher_input,
                'chromathink_response': response_text,
                'engagement_score': float(engagement_score),
                'developmental_stage': float(self.developmental_stage),
                'curiosity_level': float(self.curiosity_level),
                'thinking_intensity': thinking_intensity
            }
            
            dialogue_history.append(exchange_data)
            self.engagement_history.append(engagement_score)
            
            # Show progress
            self._display_learning_progress(exchange_data)
        
        # Final learning analysis
        learning_metrics = self._analyze_learning_session(dialogue_history)
        
        return {
            'dialogue_history': dialogue_history,
            'learning_metrics': learning_metrics,
            'final_developmental_stage': float(self.developmental_stage),
            'system_status': self.chromathink_system.get_system_status()
        }
    
    def _generate_curious_question(self) -> str:
        """
        Generate a curious question by thinking about wonder in colour space.
        This demonstrates ChromaThink's ability to generate novel curiosity.
        """
        
        # Think about the concept of curiosity in colour space
        curiosity_colour = self.chromathink_system.think_about(
            concept="curiosity and wonder",
            intensity=float(self.curiosity_level)
        )
        
        # Think about questioning in colour space
        question_colour = self.chromathink_system.think_about(
            concept="asking questions", 
            intensity=1.0
        )
        
        # Combine through interference to create curious question
        curious_question_colour = self.chromathink_system.thinker.interference_engine.interfere_colours(
            curiosity_colour, question_colour, 'constructive'
        )
        
        # Convert to concept descriptors
        question_descriptors = self.chromathink_system.translator.colour_to_concept_descriptors(
            curious_question_colour
        )
        
        # Synthesise into question
        mock_concepts = {
            'concepts': ['curiosity', 'learning', 'understanding'],
            'intent': ['seeking knowledge', 'wondering'],
            'context': ['educational dialogue', 'exploration']
        }
        
        question = self.chromathink_system.translator.synthesise_response(
            colour_descriptors=question_descriptors,
            target_language='english',
            original_concepts=mock_concepts
        )
        
        return question
    
    def _calculate_engagement(self, input_text: str, response_text: str) -> float:
        """
        Calculate engagement level based on colour complexity and response depth.
        """
        
        # Get system metrics
        system_status = self.chromathink_system.get_system_status()
        
        # Engagement based on colour complexity
        colour_complexity = system_status['colour_complexity']['recent_average']
        
        # Response length factor (longer responses show more engagement)
        response_length_factor = min(len(response_text) / 100.0, 2.0)  # Cap at 2.0
        
        # Combine factors
        engagement = (0.6 * colour_complexity + 0.4 * response_length_factor) * self.curiosity_level
        
        # Ensure reasonable bounds
        engagement = float(tf.clip_by_value(engagement, 0.1, 2.0))
        
        return engagement
    
    def _update_developmental_stage(self, input_text: str, response_text: str, engagement: float):
        """
        Update developmental stage based on interaction complexity and engagement.
        """
        
        # Calculate interaction complexity
        input_complexity = len(input_text.split()) / 20.0  # Normalized word count
        response_complexity = len(response_text.split()) / 20.0
        
        # Average recent engagement
        recent_engagement = np.mean(self.engagement_history[-5:]) if self.engagement_history else engagement
        
        # Developmental growth rate based on complexity and engagement
        growth_rate = self.chromatic_plasticity * (
            0.4 * input_complexity + 
            0.4 * response_complexity + 
            0.2 * recent_engagement
        ) * 0.01  # Scale factor
        
        # Update developmental stage (with cap at 1.0)
        new_stage = self.developmental_stage + growth_rate
        new_stage = tf.clip_by_value(new_stage, 0.0, 1.0)
        self.developmental_stage.assign(new_stage)
        
        # Update curiosity (decreases slightly with maturity, but stays active)
        curiosity_adjustment = -0.001 * growth_rate + 0.002 * engagement * 0.1
        new_curiosity = self.curiosity_level + curiosity_adjustment
        new_curiosity = tf.clip_by_value(new_curiosity, 0.3, 1.5)  # Keep curious!
        self.curiosity_level.assign(new_curiosity)
    
    def _display_learning_progress(self, exchange_data: Dict):
        """Display learning progress for this exchange."""
        
        stage = exchange_data['developmental_stage']
        engagement = exchange_data['engagement_score']
        curiosity = exchange_data['curiosity_level']
        
        # Determine developmental stage name
        if stage < 0.3:
            stage_name = "Child"
        elif stage < 0.6:
            stage_name = "Adolescent" 
        elif stage < 0.9:
            stage_name = "Young Adult"
        else:
            stage_name = "Mature Thinker"
        
        print(f"📊 Learning Progress:")
        print(f"   Development: {stage_name} ({stage:.3f})")
        print(f"   Engagement: {engagement:.3f}")
        print(f"   Curiosity: {curiosity:.3f}")
    
    def _analyze_learning_session(self, dialogue_history: List[Dict]) -> Dict:
        """Analyze the complete learning session."""
        
        if not dialogue_history:
            return {'error': 'No dialogue history to analyze'}
        
        # Extract metrics
        engagements = [exchange['engagement_score'] for exchange in dialogue_history]
        stages = [exchange['developmental_stage'] for exchange in dialogue_history]
        
        # Calculate progression
        initial_stage = stages[0]
        final_stage = stages[-1]
        development_growth = final_stage - initial_stage
        
        # Engagement analysis
        avg_engagement = np.mean(engagements)
        engagement_trend = np.polyfit(range(len(engagements)), engagements, 1)[0]  # Linear trend
        
        # Response complexity analysis
        response_lengths = [len(exchange['chromathink_response']) for exchange in dialogue_history]
        avg_response_length = np.mean(response_lengths)
        
        # Get final system status
        system_status = self.chromathink_system.get_system_status()
        
        learning_analysis = {
            'session_summary': {
                'total_exchanges': len(dialogue_history),
                'development_growth': float(development_growth),
                'average_engagement': float(avg_engagement),
                'engagement_trend': float(engagement_trend),
                'average_response_length': float(avg_response_length)
            },
            'developmental_progression': {
                'initial_stage': float(initial_stage),
                'final_stage': float(final_stage),
                'growth_rate': float(development_growth / len(dialogue_history))
            },
            'colour_cognition_metrics': {
                'memory_count': system_status['cognitive']['memory_count'],
                'cognitive_complexity': system_status['cognitive']['cognitive_complexity'], 
                'average_colour_complexity': system_status['colour_complexity']['recent_average'],
                'thinking_efficiency': system_status['conversation']['average_thinking_time']
            },
            'learning_quality_assessment': self._assess_learning_quality(dialogue_history)
        }
        
        return learning_analysis
    
    def _assess_learning_quality(self, dialogue_history: List[Dict]) -> Dict:
        """Assess the quality of learning based on various factors."""
        
        # Assess consistency of engagement
        engagements = [exchange['engagement_score'] for exchange in dialogue_history]
        engagement_stability = 1.0 - np.std(engagements) / max(np.mean(engagements), 0.1)
        
        # Assess response diversity (length variety indicates adaptability)
        response_lengths = [len(exchange['chromathink_response']) for exchange in dialogue_history]
        length_diversity = np.std(response_lengths) / max(np.mean(response_lengths), 1.0)
        
        # Assess developmental momentum (consistent growth)
        stages = [exchange['developmental_stage'] for exchange in dialogue_history]
        stage_momentum = np.mean(np.diff(stages)) if len(stages) > 1 else 0.0
        
        # Overall learning quality score
        quality_factors = [
            ('engagement_stability', engagement_stability, 0.3),
            ('response_diversity', min(length_diversity, 1.0), 0.3),
            ('developmental_momentum', max(stage_momentum * 10, 0.0), 0.4)
        ]
        
        quality_score = sum(factor * weight for _, factor, weight in quality_factors)
        quality_score = max(0.0, min(1.0, quality_score))  # Bound between 0-1
        
        return {
            'overall_quality_score': float(quality_score),
            'engagement_stability': float(engagement_stability),
            'response_diversity': float(length_diversity),
            'developmental_momentum': float(stage_momentum),
            'quality_rating': self._get_quality_rating(quality_score)
        }
    
    def _get_quality_rating(self, score: float) -> str:
        """Convert quality score to human-readable rating."""
        
        if score >= 0.8:
            return "Excellent Learning"
        elif score >= 0.6:
            return "Good Learning"
        elif score >= 0.4:
            return "Moderate Learning"
        elif score >= 0.2:
            return "Basic Learning"
        else:
            return "Limited Learning"
    
    def dream_consolidation(self, num_dream_cycles: int = 3) -> Dict:
        """
        Perform dream-like memory consolidation using colour dynamics.
        """
        
        self.logger.info(f"Starting dream consolidation with {num_dream_cycles} cycles")
        
        # Let ChromaThink dream
        dream_colours = self.chromathink_system.dream(
            dream_intensity=0.6,
            num_dream_cycles=num_dream_cycles
        )
        
        # Analyze dream patterns
        dream_analysis = {
            'num_dreams': len(dream_colours),
            'dream_complexities': [],
            'consolidation_effect': 0.0
        }
        
        if dream_colours:
            # Calculate complexity of each dream
            for dream_colour in dream_colours:
                complexity = self.chromathink_system.thinker._calculate_spectral_entropy(dream_colour)
                dream_analysis['dream_complexities'].append(float(complexity))
            
            # Consolidation effect on memory
            initial_memory_count = self.chromathink_system.thinker.colour_memory.memory_index
            final_memory_count = self.chromathink_system.thinker.colour_memory.memory_index
            dream_analysis['consolidation_effect'] = float(final_memory_count - initial_memory_count)
            
            # Slightly increase developmental stage from consolidation
            consolidation_boost = 0.005 * np.mean(dream_analysis['dream_complexities'])
            new_stage = tf.clip_by_value(
                self.developmental_stage + consolidation_boost, 0.0, 1.0
            )
            self.developmental_stage.assign(new_stage)
        
        self.logger.info("Dream consolidation complete")
        
        return dream_analysis
    
    def get_developmental_metrics(self) -> Dict:
        """Get comprehensive developmental and learning metrics."""
        
        # System metrics
        system_status = self.chromathink_system.get_system_status()
        
        # Learning history analysis
        recent_engagement = np.mean(self.engagement_history[-10:]) if self.engagement_history else 0.0
        
        return {
            'developmental_stage': float(self.developmental_stage),
            'stage_name': self._get_stage_name(float(self.developmental_stage)),
            'curiosity_level': float(self.curiosity_level),
            'chromatic_plasticity': float(self.chromatic_plasticity),
            'recent_engagement': float(recent_engagement),
            'total_interactions': len(self.engagement_history),
            'system_metrics': system_status,
            'learning_efficiency': self._calculate_learning_efficiency()
        }
    
    def _get_stage_name(self, stage_value: float) -> str:
        """Convert developmental stage value to name."""
        
        if stage_value < 0.25:
            return "Curious Child"
        elif stage_value < 0.50:
            return "Inquisitive Adolescent"
        elif stage_value < 0.75:
            return "Thoughtful Young Adult"
        else:
            return "Wise Mature Thinker"
    
    def _calculate_learning_efficiency(self) -> float:
        """Calculate learning efficiency based on development per interaction."""
        
        if len(self.engagement_history) < 2:
            return 0.0
        
        # Development rate per interaction
        development_per_interaction = float(self.developmental_stage) / max(len(self.engagement_history), 1)
        
        # Scale to reasonable range
        efficiency = development_per_interaction * 100  # Scale up for readability
        
        return float(efficiency)
    
    def reset_development(self):
        """Reset developmental state while preserving colour memories."""
        
        self.logger.info("Resetting developmental state")
        
        # Reset developmental variables
        self.developmental_stage.assign(0.0)
        self.curiosity_level.assign(0.8)
        self.chromatic_plasticity.assign(0.15)
        
        # Clear engagement history
        self.engagement_history.clear()
        
        # Reset ChromaThink system but preserve some memories
        self.chromathink_system.reset_system(preserve_language_patterns=True)
        
        self.logger.info("Developmental reset complete")