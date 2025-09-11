"""
ChromaticMemory: Stores learned associations as colour patterns

Memory that works like human memory - associative, fuzzy, and colour-based.
"""

import tensorflow as tf
import numpy as np
from ..core.colour_utils import colour_distance, colour_interpolation
from ..layers.resonance_chamber import ResonanceChamber


class ChromaticMemory(tf.keras.Model):
    """
    Associative memory system that stores and retrieves colour-based memories.
    Memories fade, strengthen, and interfere with each other like human memory.
    """
    
    def __init__(self, 
                 spectrum_dims=512,
                 memory_capacity=1000,
                 decay_rate=0.99,
                 consolidation_threshold=0.8,
                 name='chromatic_memory'):
        super().__init__(name=name)
        
        self.spectrum_dims = spectrum_dims
        self.memory_capacity = memory_capacity
        self.decay_rate = decay_rate
        self.consolidation_threshold = consolidation_threshold
        
        # Memory storage as resonance chambers
        self.memory_chamber = ResonanceChamber(
            dimensions=spectrum_dims,
            num_modes=64,
            boundary_conditions='mixed',
            quality_factor=0.9
        )
        
        # Memory banks
        self.episodic_memory = []  # Specific experiences
        self.semantic_memory = []  # General knowledge patterns
        self.emotional_memory = []  # Emotional associations
        
        # Memory strength tracking
        self.memory_strengths = []
        self.access_counts = []
        self.last_accessed = []
        
        # Association network
        self.association_weights = tf.Variable(
            tf.zeros([memory_capacity, memory_capacity]),
            trainable=False
        )
        
        # Current memory index
        self.memory_index = 0
    
    def store_association(self, 
                         question_colour, 
                         response_colour, 
                         strength=1.0,
                         memory_type='episodic'):
        """
        Store a question-response association in chromatic memory.
        
        Args:
            question_colour: Colour representation of question
            response_colour: Colour representation of response
            strength: Initial memory strength
            memory_type: Type of memory ('episodic', 'semantic', 'emotional')
        """
        
        # Ensure consistent shapes for interference
        if len(question_colour.shape) != len(response_colour.shape):
            # Reshape to match
            if len(question_colour.shape) > len(response_colour.shape):
                response_colour = tf.expand_dims(response_colour, 1)
            else:
                question_colour = tf.expand_dims(question_colour, 1)
        
        # Create memory trace by interfering question and response
        memory_trace = self.memory_chamber([question_colour, response_colour])
        
        # Choose memory bank
        if memory_type == 'episodic':
            memory_bank = self.episodic_memory
        elif memory_type == 'semantic':
            memory_bank = self.semantic_memory
        else:
            memory_bank = self.emotional_memory
        
        # Store memory
        memory_entry = {
            'question': tf.identity(question_colour),
            'response': tf.identity(response_colour),
            'trace': tf.identity(memory_trace),
            'strength': float(strength),
            'type': memory_type,
            'timestamp': self.memory_index
        }
        
        memory_bank.append(memory_entry)
        self.memory_strengths.append(strength)
        self.access_counts.append(0)
        self.last_accessed.append(self.memory_index)
        
        # Update association network
        self._update_associations(len(memory_bank) - 1, memory_trace)
        
        # Manage memory capacity
        if len(memory_bank) > self.memory_capacity // 3:  # Each bank gets 1/3 capacity
            self._consolidate_or_forget(memory_bank)
        
        self.memory_index += 1
        
        return memory_entry
    
    def retrieve_similar(self, 
                        query_colour, 
                        memory_type=None, 
                        top_k=5,
                        similarity_threshold=0.3):
        """
        Retrieve memories similar to query colour.
        
        Args:
            query_colour: Colour to search for
            memory_type: Type of memory to search (None for all)
            top_k: Number of memories to return
            similarity_threshold: Minimum similarity threshold
        """
        
        # Determine which memory banks to search
        if memory_type == 'episodic':
            banks = [('episodic', self.episodic_memory)]
        elif memory_type == 'semantic':
            banks = [('semantic', self.semantic_memory)]
        elif memory_type == 'emotional':
            banks = [('emotional', self.emotional_memory)]
        else:
            banks = [
                ('episodic', self.episodic_memory),
                ('semantic', self.semantic_memory),
                ('emotional', self.emotional_memory)
            ]
        
        # Calculate similarities
        candidates = []
        
        for bank_name, bank in banks:
            for i, memory in enumerate(bank):
                # Calculate similarity to both question and response
                q_similarity = 1.0 - colour_distance(
                    query_colour, memory['question'], metric='spectral'
                )
                r_similarity = 1.0 - colour_distance(
                    query_colour, memory['response'], metric='spectral'
                )
                trace_similarity = 1.0 - colour_distance(
                    query_colour, memory['trace'], metric='spectral'
                )
                
                # Combined similarity with memory strength weighting
                combined_similarity = (
                    0.4 * q_similarity + 
                    0.4 * r_similarity + 
                    0.2 * trace_similarity
                ) * memory['strength']
                
                if combined_similarity > similarity_threshold:
                    candidates.append({
                        'memory': memory,
                        'similarity': float(combined_similarity),
                        'bank': bank_name,
                        'index': i
                    })
        
        # Sort by similarity and return top k
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        retrieved = candidates[:top_k]
        
        # Update access counts and strengthen accessed memories
        for candidate in retrieved:
            self._access_memory(candidate['bank'], candidate['index'])
        
        return retrieved
    
    def associate_memories(self, memory1_colour, memory2_colour, strength=0.5):
        """
        Create association between two memory colours.
        
        Args:
            memory1_colour: First memory colour
            memory2_colour: Second memory colour  
            strength: Association strength
        """
        
        # Find memories matching these colours
        mem1_matches = self.retrieve_similar(memory1_colour, top_k=1)
        mem2_matches = self.retrieve_similar(memory2_colour, top_k=1)
        
        if mem1_matches and mem2_matches:
            idx1 = mem1_matches[0]['index']
            idx2 = mem2_matches[0]['index']
            
            # Update association weights
            current_weight = self.association_weights[idx1, idx2].numpy()
            new_weight = min(1.0, current_weight + strength)
            
            self.association_weights[idx1, idx2].assign(new_weight)
            self.association_weights[idx2, idx1].assign(new_weight)  # Symmetric
    
    def _update_associations(self, memory_idx, memory_trace):
        """Update associations with existing memories."""
        
        all_memories = (
            self.episodic_memory + 
            self.semantic_memory + 
            self.emotional_memory
        )
        
        for i, existing_memory in enumerate(all_memories):
            if i != memory_idx and i < tf.shape(self.association_weights)[0]:
                # Calculate natural association strength
                similarity = 1.0 - colour_distance(
                    memory_trace, 
                    existing_memory['trace'], 
                    metric='spectral'
                )
                
                if similarity > 0.5:  # Only create strong associations
                    association_strength = (similarity - 0.5) * 2.0  # Scale to 0-1
                    
                    current_weight = self.association_weights[memory_idx, i].numpy()
                    new_weight = min(1.0, current_weight + association_strength * 0.1)
                    
                    self.association_weights[memory_idx, i].assign(new_weight)
                    self.association_weights[i, memory_idx].assign(new_weight)
    
    def _access_memory(self, bank_name, memory_idx):
        """Update memory access statistics and strengthen memory."""
        
        if memory_idx < len(self.access_counts):
            self.access_counts[memory_idx] += 1
            self.last_accessed[memory_idx] = self.memory_index
            
            # Strengthen frequently accessed memories
            if self.access_counts[memory_idx] > 3:
                strength_boost = min(0.1, 1.0 / self.access_counts[memory_idx])
                self.memory_strengths[memory_idx] = min(
                    1.0, 
                    self.memory_strengths[memory_idx] + strength_boost
                )
    
    def _consolidate_or_forget(self, memory_bank):
        """
        Implement memory consolidation and forgetting.
        Strong memories get consolidated, weak ones are forgotten.
        """
        
        if len(memory_bank) <= self.memory_capacity // 6:  # Don't consolidate if small
            return
        
        # Calculate memory importance scores
        importance_scores = []
        current_time = self.memory_index
        
        for i, memory in enumerate(memory_bank):
            # Factors: strength, recency, access frequency
            strength = memory['strength']
            recency = 1.0 / (1.0 + current_time - memory['timestamp'])
            frequency = self.access_counts[i] if i < len(self.access_counts) else 0
            
            importance = 0.5 * strength + 0.3 * recency + 0.2 * frequency
            importance_scores.append((importance, i))
        
        # Sort by importance
        importance_scores.sort(reverse=True)
        
        # Keep top memories, forget bottom ones
        keep_count = self.memory_capacity // 6
        memories_to_keep = [memory_bank[i] for _, i in importance_scores[:keep_count]]
        
        # Consolidate similar memories
        consolidated_memories = self._consolidate_similar_memories(memories_to_keep)
        
        # Update memory bank
        memory_bank.clear()
        memory_bank.extend(consolidated_memories)
    
    def _consolidate_similar_memories(self, memories):
        """Consolidate similar memories into more general patterns."""
        
        if len(memories) < 2:
            return memories
        
        consolidated = []
        used_indices = set()
        
        for i, memory1 in enumerate(memories):
            if i in used_indices:
                continue
                
            similar_group = [memory1]
            used_indices.add(i)
            
            # Find similar memories
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                similarity = 1.0 - colour_distance(
                    memory1['trace'], 
                    memory2['trace'], 
                    metric='spectral'
                )
                
                if similarity > self.consolidation_threshold:
                    similar_group.append(memory2)
                    used_indices.add(j)
            
            # If group has multiple memories, consolidate them
            if len(similar_group) > 1:
                consolidated_memory = self._merge_memories(similar_group)
                consolidated.append(consolidated_memory)
            else:
                consolidated.append(memory1)
        
        return consolidated
    
    def _merge_memories(self, memory_group):
        """Merge a group of similar memories into one consolidated memory."""
        
        # Average the colour representations
        questions = tf.stack([m['question'] for m in memory_group])
        responses = tf.stack([m['response'] for m in memory_group])
        traces = tf.stack([m['trace'] for m in memory_group])
        
        avg_question = tf.reduce_mean(questions, axis=0)
        avg_response = tf.reduce_mean(responses, axis=0)
        avg_trace = tf.reduce_mean(traces, axis=0)
        
        # Combined strength
        total_strength = sum(m['strength'] for m in memory_group)
        avg_strength = min(1.0, total_strength / len(memory_group) * 1.2)  # Slight boost
        
        return {
            'question': avg_question,
            'response': avg_response,
            'trace': avg_trace,
            'strength': avg_strength,
            'type': 'consolidated',
            'timestamp': max(m['timestamp'] for m in memory_group)
        }
    
    def get_memory_metrics(self):
        """Return metrics about current memory state."""
        
        total_memories = (
            len(self.episodic_memory) + 
            len(self.semantic_memory) + 
            len(self.emotional_memory)
        )
        
        if total_memories == 0:
            return {
                'total_memories': 0,
                'average_strength': 0.0,
                'memory_utilization': 0.0,
                'consolidation_ratio': 0.0
            }
        
        # Calculate average strength
        avg_strength = np.mean(self.memory_strengths[-total_memories:]) if self.memory_strengths else 0.0
        
        # Count consolidated memories
        consolidated_count = sum(
            1 for m in (self.episodic_memory + self.semantic_memory + self.emotional_memory)
            if m['type'] == 'consolidated'
        )
        
        return {
            'total_memories': total_memories,
            'episodic_count': len(self.episodic_memory),
            'semantic_count': len(self.semantic_memory),
            'emotional_count': len(self.emotional_memory),
            'average_strength': float(avg_strength),
            'memory_utilization': total_memories / self.memory_capacity,
            'consolidation_ratio': consolidated_count / max(1, total_memories),
            'association_density': float(tf.reduce_mean(tf.math.abs(self.association_weights)))
        }
    
    def reset_memory(self):
        """Clear all memories (useful for starting fresh)."""
        self.episodic_memory.clear()
        self.semantic_memory.clear()
        self.emotional_memory.clear()
        self.memory_strengths.clear()
        self.access_counts.clear()
        self.last_accessed.clear()
        self.association_weights.assign(tf.zeros_like(self.association_weights))
        self.memory_index = 0