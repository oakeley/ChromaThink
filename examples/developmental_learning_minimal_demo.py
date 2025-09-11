"""
Demonstration of ChromaThink's Developmental Learning System (Minimal Version)

Shows how ChromaThink learns through dialogue, thinking only in colour.
This version works without external LLM dependencies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
from chromathink.learning.development_minimal import DevelopmentalLearningMinimal


def main():
    print("🎨 ChromaThink Developmental Learning Demo (Minimal)")
    print("=" * 55)
    
    # Initialize the learning system
    print("Initializing ChromaThink's colour-based cognition...")
    learner = DevelopmentalLearningMinimal(spectrum_dims=128)
    
    print(f"✓ ChromaThink initialized with {learner.spectrum_dims}-dimensional colour space")
    print(f"✓ Initial development stage: {float(learner.developmental_stage):.3f}")
    
    # Start learning dialogue
    print("\\n🤔 Beginning learning dialogue...")
    print("ChromaThink will think in colour and ask questions through curiosity...")
    
    try:
        # Conduct learning dialogue
        dialogue = learner.learn_through_dialogue(num_exchanges=5)
        
        print("\\n📊 Learning Session Complete!")
        print("=" * 50)
        
        # Show learning progression
        print("\\n🧠 Learning Progression:")
        for i, exchange in enumerate(dialogue):
            print(f"\\nExchange {i+1}:")
            print(f"  Question: {exchange['question_text']}")
            print(f"  Response: {exchange['response_text'][:80]}...")
            print(f"  Resonance: {exchange['resonance']:.3f}")
        
        # Show final metrics
        metrics = learner.get_learning_metrics()
        print("\\n📈 Final Learning Metrics:")
        print(f"  Development Stage: {metrics['developmental_stage']:.3f}")
        print(f"  Total Memories: {metrics['memory']['total_memories']}")
        print(f"  Questions Asked: {metrics['curiosity']['question_count']}")
        print(f"  Cognitive Complexity: {metrics['cognitive']['memory_complexity']:.3f}")
        
        # Demonstrate colour-based thinking
        print("\\n🌈 ChromaThink's Current Colour State:")
        current_state = learner.cognitive_spectrum.cognitive_memory
        if current_state is not None:
            print(f"  {learner._describe_colour(current_state)}")
        
        # Show memory retrieval
        print("\\n🧠 Testing Memory Retrieval:")
        if dialogue:
            test_question = dialogue[0]['question_colour']
            similar_memories = learner.colour_memory.retrieve_similar(
                test_question, 
                top_k=3
            )
            
            print(f"  Found {len(similar_memories)} similar memories")
            for i, memory in enumerate(similar_memories):
                print(f"    Memory {i+1}: Similarity {memory['similarity']:.3f}")
        
        print("\\n✨ Demo complete! ChromaThink has learned through colour-based dialogue.")
        
    except Exception as e:
        print(f"\\n❌ Error during learning: {e}")
        import traceback
        traceback.print_exc()
    
    # Demonstrate curiosity generation
    print("\\n🔍 Demonstrating Pure Curiosity Generation:")
    wonder_state = learner.curiosity.generate_wonder()
    print(f"  Wonder state: {learner._describe_colour(wonder_state)}")
    
    question_data = learner.curiosity.formulate_question(wonder_state)
    print(f"  Generated question type: {question_data['question_type']}")
    print(f"  Question intensity: {question_data['curiosity_intensity']:.3f}")


def demonstrate_colour_thinking():
    """
    Show how ChromaThink thinks purely in colour frequencies.
    """
    print("\\n🎨 Colour-Based Thinking Demo")
    print("-" * 30)
    
    # Create different "thoughts" as colour patterns
    thoughts = {
        "curiosity": np.array([0.8, 0.3, 0.1, 0.9] * 8),
        "understanding": np.array([0.2, 0.7, 0.8, 0.4] * 8),
        "confusion": np.array([0.9, 0.1, 0.9, 0.2] * 8),
        "joy": np.array([0.9, 0.8, 0.7, 0.9] * 8)
    }
    
    print("Thought patterns in colour space:")
    for emotion, pattern in thoughts.items():
        # Simple processing
        processed = np.tanh(pattern * 2.0)  # Nonlinear activation
        avg_activation = np.mean(processed)
        max_activation = np.max(processed)
        dominant_freq = np.argmax(processed)
        
        print(f"  {emotion:12}: avg={avg_activation:.3f}, max={max_activation:.3f}, dom={dominant_freq}")


def demonstrate_learning_progression():
    """
    Show how learning evolves through multiple sessions.
    """
    print("\\n📚 Multi-Session Learning Demo")
    print("-" * 35)
    
    learner = DevelopmentalLearningMinimal(spectrum_dims=64)
    
    print("Conducting multiple learning sessions...")
    for session in range(3):
        print(f"\\n--- Session {session + 1} ---")
        
        initial_stage = float(learner.developmental_stage)
        dialogue = learner.learn_through_dialogue(num_exchanges=3)
        final_stage = float(learner.developmental_stage)
        
        print(f"Development: {initial_stage:.3f} → {final_stage:.3f}")
        
        # Show memory growth
        metrics = learner.get_learning_metrics()
        print(f"Memories: {metrics['memory']['total_memories']}")
        print(f"Curiosity diversity: {metrics['curiosity']['curiosity_diversity']:.3f}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    main()
    demonstrate_colour_thinking()
    demonstrate_learning_progression()