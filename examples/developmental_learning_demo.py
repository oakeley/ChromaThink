"""
Demonstration of ChromaThink's Developmental Learning System

Shows how ChromaThink learns through dialogue, thinking only in colour
while interacting with language-based teachers.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
from chromathink.learning.development import DevelopmentalLearning


def main():
    print("🎨 ChromaThink Developmental Learning Demo")
    print("=" * 50)
    
    # Initialize the learning system
    print("Initializing ChromaThink's colour-based cognition...")
    learner = DevelopmentalLearning(
        spectrum_dims=128,  # Smaller for demo
        device="cpu"  # Use CPU for broader compatibility
    )
    
    print(f"✓ ChromaThink initialized with {learner.spectrum_dims}-dimensional colour space")
    print(f"✓ Initial development stage: {float(learner.developmental_stage):.3f}")
    
    # Start learning dialogue
    print("\\n🤔 Beginning learning dialogue...")
    print("ChromaThink will think in colour and ask questions through curiosity...")
    
    try:
        # Conduct learning dialogue
        dialogue = learner.learn_through_dialogue(
            num_exchanges=5,
            languages=['english']
        )
        
        print("\\n📊 Learning Session Complete!")
        print("=" * 50)
        
        # Show learning progression
        print("\\n🧠 Learning Progression:")
        for i, exchange in enumerate(dialogue):
            print(f"\\nExchange {i+1}:")
            print(f"  Question: {exchange['question_text']}")
            print(f"  Response: {exchange['response_text'][:100]}...")
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
        print("This may be due to missing transformer dependencies.")
        print("The system is designed to work with proper language models.")
    
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
    
    # Create a cognitive spectrum
    spectrum = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(32, activation='relu')  
    ])
    
    # Different "thoughts" as colour patterns
    thoughts = {
        "curiosity": tf.constant([[0.8, 0.3, 0.1, 0.9] * 8], dtype=tf.float32),
        "understanding": tf.constant([[0.2, 0.7, 0.8, 0.4] * 8], dtype=tf.float32),
        "confusion": tf.constant([[0.9, 0.1, 0.9, 0.2] * 8], dtype=tf.float32),
        "joy": tf.constant([[0.9, 0.8, 0.7, 0.9] * 8], dtype=tf.float32)
    }
    
    print("Thought patterns in colour space:")
    for emotion, pattern in thoughts.items():
        processed = spectrum(pattern)
        avg_activation = tf.reduce_mean(processed)
        max_activation = tf.reduce_max(processed)
        print(f"  {emotion:12}: avg={avg_activation:.3f}, max={max_activation:.3f}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    main()
    demonstrate_colour_thinking()