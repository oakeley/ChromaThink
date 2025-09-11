"""
Main script to bootstrap ChromaThink from Apertus weights.
"""

import sys
import logging
from pathlib import Path
import warnings

# Add the current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from chromathink.learning.development_minimal import DevelopmentalLearningMinimal
    from chromathink.bootstrap import ApertusWeightTranslator, ChromaThinkBootstrap
    from chromathink.debug import QuestionFormationDebugger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure ChromaThink is properly installed")
    sys.exit(1)


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chromathink_bootstrap.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress some verbose warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)


def main():
    """
    Main script to bootstrap ChromaThink from Apertus weights.
    """
    
    setup_logging()
    logger = logging.getLogger("Main")
    
    # Path to Apertus model files
    apertus_path = Path("/home/edward/.cache/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509/snapshots/a76c9e6701dafe8dd9b173cee09bc93f2eba3bfc/")
    
    # Check if Apertus path exists
    if not apertus_path.exists():
        logger.warning(f"Apertus path {apertus_path} not found. Using mock model.")
        use_mock = True
    else:
        logger.info(f"Found Apertus path: {apertus_path}")
        # Check for actual safetensors files
        safetensor_files = list(apertus_path.glob("model-*-of-*.safetensors"))
        if len(safetensor_files) >= 4:
            logger.info(f"Found {len(safetensor_files)} safetensor files - using real Apertus model")
            use_mock = False
        else:
            logger.warning(f"Only found {len(safetensor_files)} safetensor files - using mock model")
            use_mock = True
    
    try:
        logger.info("Initialising ChromaThink...")
        chromathink = DevelopmentalLearningMinimal(
            spectrum_dims=128,  # Match the minimal demo
        )
        
        logger.info("Loading Apertus and analysing weights...")
        translator = ApertusWeightTranslator(
            apertus_path=apertus_path,
            spectrum_dims=128,
            use_mock=use_mock
        )
        
        # Debug pre-bootstrap state
        logger.info("\n" + "="*50)
        logger.info("PRE-BOOTSTRAP ANALYSIS")
        logger.info("="*50)
        
        debugger = QuestionFormationDebugger(chromathink, translator)
        pre_bootstrap_result = debugger.trace_question_formation()
        
        # Perform bootstrap
        logger.info("\n" + "="*50)
        logger.info("BOOTSTRAPPING FROM APERTUS")
        logger.info("="*50)
        
        bootstrap = ChromaThinkBootstrap(chromathink, translator)
        bootstrap_results = bootstrap.bootstrap_from_apertus()
        
        # Report bootstrap results
        logger.info("\nBootstrap Results:")
        for key, result in bootstrap_results.items():
            if isinstance(result, dict) and 'status' in result:
                logger.info(f"  {key}: {result['status']}")
                if result['status'] == 'error' and 'error' in result:
                    logger.error(f"    Error: {result['error']}")
            else:
                logger.info(f"  {key}: {result}")
        
        # Debug post-bootstrap state
        logger.info("\n" + "="*50)
        logger.info("POST-BOOTSTRAP ANALYSIS")
        logger.info("="*50)
        
        post_bootstrap_result = debugger.trace_question_formation()
        
        # Compare pre and post bootstrap
        logger.info("\n" + "="*50)
        logger.info("BOOTSTRAP COMPARISON")
        logger.info("="*50)
        
        logger.info("Pre-bootstrap question:")
        logger.info(f"  Type: {pre_bootstrap_result['question_type']}")
        for render_type, text in pre_bootstrap_result['rendered_texts'].items():
            logger.info(f"  {render_type}: {text[:100]}...")
        
        logger.info("\nPost-bootstrap question:")
        logger.info(f"  Type: {post_bootstrap_result['question_type']}")
        for render_type, text in post_bootstrap_result['rendered_texts'].items():
            logger.info(f"  {render_type}: {text[:100]}...")
        
        # Test learning capability
        logger.info("\n" + "="*50)
        logger.info("TESTING LEARNING CAPABILITY")
        logger.info("="*50)
        
        try:
            # Run a short learning session
            session_history = chromathink.learn_through_dialogue(num_exchanges=3)
            logger.info(f"Successfully completed learning session with {len(session_history)} exchanges")
            
            # Show final development metrics
            metrics = chromathink.get_development_metrics()
            logger.info(f"Final development metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Learning session failed: {e}")
        
        # Create checkpoints directory if it doesn't exist
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Try to save the bootstrapped model
        try:
            chromathink.save_development_state("checkpoints/chromathink_bootstrapped")
            logger.info("Saved bootstrapped model to checkpoints/")
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
        
        return chromathink
        
    except KeyboardInterrupt:
        logger.info("Bootstrap interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Bootstrap failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_mode():
    """
    Run a demonstration of the bootstrapping process
    """
    setup_logging()
    logger = logging.getLogger("Demo")
    
    logger.info("🎨 ChromaThink Bootstrap Demo")
    logger.info("=" * 40)
    
    # Run the main bootstrap
    bootstrapped_model = main()
    
    if bootstrapped_model:
        logger.info("\n✅ Bootstrap Demo Completed Successfully!")
        
        # Run some additional demonstrations
        logger.info("\n🧠 Testing Bootstrapped ChromaThink:")
        
        try:
            # Test curiosity generation
            wonder_state = bootstrapped_model.curiosity.generate_wonder()
            logger.info(f"Generated wonder state with magnitude: {abs(wonder_state.numpy()).max():.4f}")
            
            # Test question formulation
            question_data = bootstrapped_model.curiosity.formulate_question(wonder_state)
            logger.info(f"Formulated question of type: {question_data.get('question_type', 'unknown')}")
            
            # Test memory system
            memory_count = len(bootstrapped_model.colour_memory.episodic_memory)
            logger.info(f"Memory system contains: {memory_count} episodic memories")
            
        except Exception as e:
            logger.error(f"Demo tests failed: {e}")
        
    else:
        logger.error("❌ Bootstrap Demo Failed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_mode()
    else:
        bootstrapped_model = main()