#!/usr/bin/env python3
"""
ChromaThink True Colour Chatbot Launcher
"""

import sys
import argparse
import logging
from pathlib import Path

# Set up enhanced logging for debugging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Suppress TensorFlow and other noisy logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('WARNING')

# Add project root to path for proper imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import components
from chromathink.cli.true_colour_chatbot import main as run_chatbot
from chromathink.bootstrap.apertus_integration import ApertusWeightTranslator
from chromathink.core.big_colour_integration import create_big_colour_chromathink


def build_big_colour_model(apertus_path: str = "models/apertus", force_rebuild: bool = False):
    """
    Build or load the Big Colour Model from Apertus safetensor files.
    Processes all 131k tokens for comprehensive concept encoding.
    """
    
    print("Building Big Colour Model from Apertus safetensors...")
    print(f"Apertus path: {apertus_path}")
    print("Extracting all 131,072 token embeddings...")
    
    # Check if model files exist
    apertus_dir = Path(apertus_path)
    if not apertus_dir.exists():
        raise FileNotFoundError(f"Apertus directory not found: {apertus_path}")
    
    safetensor_files = list(apertus_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensor files found in: {apertus_path}")
    
    print(f"Found {len(safetensor_files)} safetensor files")
    use_mock = False
    
    # Initialize weight translator with full vocabulary extraction
    translator = ApertusWeightTranslator(
        apertus_path=apertus_path,
        spectrum_dims=512,
        use_mock=use_mock,
        max_tokens=None,  # Auto-detect vocabulary size
        extract_full_vocab=True,
        force_rebuild=force_rebuild
    )
    
    # Build the big colour model
    print("Building comprehensive colour model...")
    big_colour_model = translator.build_big_colour_model()
    
    # Show statistics
    stats = big_colour_model.get_statistics()
    print("\nBig Colour Model Statistics:")
    print(f"   Vocabulary size: {stats['vocab_size']:,} tokens")
    print(f"   Attention patterns: {stats['attention_patterns']}")
    print(f"   Transformation patterns: {stats['transformation_patterns']}")
    print(f"   Spectrum dimensions: {stats['spectrum_dims']}")
    
    return translator, big_colour_model


def test_big_colour_model(translator, big_colour_model):
    """
    Test the big colour model with various concepts.
    """
    
    print("\nTesting Big Colour Model encoding/decoding...")
    
    test_concepts = [
        "consciousness and awareness",
        "creative thinking patterns",
        "wave interference dynamics",
        "colour frequency resonance",
        "neural pattern analysis"
    ]
    
    for concept in test_concepts:
        print(f"\nConcept: '{concept}'")
        
        # Encode to waveform
        waveform = translator.encode_concept_to_waveform(concept)
        
        # Analyze waveform properties
        amplitude = np.abs(waveform)
        dominant_freq = np.argmax(amplitude)
        total_energy = np.sum(amplitude**2)
        
        print(f"   Waveform: dominant freq {dominant_freq}, energy {total_energy:.3f}")
        
        # Decode back to concepts
        decoded_concepts = translator.decode_waveform_to_concept(waveform, num_concepts=3)
        
        print(f"   Decoded: {[concept[0] for concept in decoded_concepts]}")
        print(f"   Amplitudes: {[f'{concept[1]:.3f}' for concept in decoded_concepts]}")


def run_interactive_session(chromathink_system):
    """
    Run interactive session with the Big Colour ChromaThink system.
    """
    
    print("\nInteractive Big Colour ChromaThink Session")
    print("=" * 60)
    print("All responses generated from pure colour dynamics")
    print("Type 'help' for commands, 'quit' to exit")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("Human: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! Colour memories preserved.")
                break
            
            if user_input.lower() == 'help':
                print("\nBig Colour ChromaThink Commands:")
                print("  help        - Show this help")
                print("  spectrum    - Analyze colour spectrum of your input")
                print("  dream       - Perform memory consolidation")
                print("  stats       - Show system statistics")
                print("  intensity N - Set thinking intensity (0.1-3.0)")
                print("  quit        - Exit session")
                print()
                continue
            
            if user_input.lower().startswith('intensity '):
                try:
                    intensity_val = float(user_input.split()[1])
                    print(f"Thinking intensity set to {intensity_val}")
                    continue
                except (IndexError, ValueError):
                    print("Usage: intensity <number>")
                    continue
            
            if user_input.lower() == 'spectrum':
                print("Enter text to analyze:")
                text = input("   Text: ").strip()
                if text:
                    spectrum = chromathink_system.get_colour_spectrum(text)
                    print(f"\nColour Spectrum Analysis:")
                    print(f"   Total energy: {spectrum['total_energy']:.3f}")
                    print(f"   Spectral centroid: {spectrum['spectral_centroid']:.1f}")
                    print(f"   Dominant frequencies: {spectrum['dominant_frequencies'][:5]}")
                    print(f"   Descriptors: {[desc[0] for desc in spectrum['colour_descriptors']]}")
                print()
                continue
            
            if user_input.lower() == 'dream':
                dream_result = chromathink_system.dream_consolidation()
                print(f"\nDream: {dream_result}")
                print()
                continue
            
            if user_input.lower() == 'stats':
                stats = chromathink_system.get_system_statistics()
                print(f"\nSystem Statistics:")
                print(f"   Vocabulary size: {stats['big_colour_model']['vocab_size']:,}")
                print(f"   Colour memories: {stats['colour_memories']}")
                print(f"   Conversations: {stats['conversations']}")
                print(f"   Memory energy: {stats['memory_energy']:.3f}")
                print(f"   Architecture: {stats['thinking_architecture']}")
                print()
                continue
            
            # Detect learning patterns
            learning_keywords = ['learn from this', 'explanation:', 'let me teach', 'this is how', 'actually', 'here\'s the correct']
            is_teaching = any(keyword in user_input.lower() for keyword in learning_keywords)
            
            if is_teaching and len(chromathink_system.conversation_context) > 0:
                # Extract the previous question for context
                last_interaction = chromathink_system.conversation_context[-1]
                previous_question = last_interaction[0] if last_interaction else "previous question"
                
                print("\nDetected teaching pattern - integrating learning...")
                learning_response = chromathink_system.learn_from_example(previous_question, user_input)
                print(f"Learning: {learning_response}")
                print()
            
            # Process through colour thinking
            intensity = getattr(run_interactive_session, 'intensity', 1.0)
            
            print("\nThinking in colour space...")
            response = chromathink_system.think_about(user_input, intensity=intensity)
            
            print(f"\nChromaThink: {response}")
            print()
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Colour memories preserved.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Continuing with next input...\n")


def main():
    """
    Main launcher with Big Colour Model integration.
    """
    
    parser = argparse.ArgumentParser(description="ChromaThink True Colour Chatbot with Big Colour Model")
    parser.add_argument("--apertus-path", default="models/apertus", help="Path to Apertus model files")
    parser.add_argument("--build-model", action="store_true", help="Build Big Colour Model from safetensors")
    parser.add_argument("--test-model", action="store_true", help="Test Big Colour Model encoding/decoding")
    parser.add_argument("--use-mock", action="store_true", help="Use mock model for testing")
    parser.add_argument("--rebuild-with-full-vocab", action="store_true", help="Force rebuild of Big Colour Model with full 131k vocabulary")
    
    args = parser.parse_args()
    
    print("ChromaThink Big Colour System")
    print("=" * 50)
    
    if args.build_model or args.test_model or args.rebuild_with_full_vocab:
        # Import numpy for testing
        import numpy as np
        
        if args.build_model or args.test_model or args.rebuild_with_full_vocab:
            # Enable INFO logging for rebuild operations to show what's happening
            if args.rebuild_with_full_vocab:
                logging.getLogger("ChromaThink.Bootstrap").setLevel(logging.INFO)
            
            # Build the big colour model for testing
            translator, big_colour_model = build_big_colour_model(
                apertus_path=args.apertus_path,
                force_rebuild=args.rebuild_with_full_vocab
            )
            
            if args.test_model:
                test_big_colour_model(translator, big_colour_model)
            
            if args.rebuild_with_full_vocab:
                print("\nBig Colour Model rebuilt with full 131k vocabulary")
                # Exit after rebuild if no other action needed
                return
            
            if args.build_model and not args.test_model:
                # If only building, exit after build
                return
    
    # Default behavior: Launch interactive Big Colour ChromaThink system
    print("Creating Big Colour ChromaThink system...")
    
    # Enable detailed logging for debugging concept extraction and synthesis
    for logger_name in ['ChromaThink', 'ApertusTranslator', 'BigColourChromatThink', 'LanguageBridge', 'ChromaThink.Bootstrap']:
        logging.getLogger(logger_name).setLevel(logging.INFO)
    
    try:
        chromathink_system = create_big_colour_chromathink(
            apertus_path=args.apertus_path,
            use_mock=args.use_mock
        )
        
        # Show system info
        stats = chromathink_system.get_system_statistics()
        print(f"System ready with {stats['big_colour_model']['vocab_size']:,} token vocabulary")
        
        # Run interactive session
        run_interactive_session(chromathink_system)
    
    except Exception as e:
        print(f"Failed to initialize Big Colour system: {e}")
        print("System initialization failed - Apertus model is required")
        return


if __name__ == '__main__':
    main()
