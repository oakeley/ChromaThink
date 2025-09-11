"""
ChromaThink Learning Module

Developmental learning through colour-based cognition and multilingual dialogue.
"""

from .cognitive_spectrum import CognitiveSpectrum
from .curiosity_engine import CuriosityEngine
from .chromatic_memory import ChromaticMemory
from .development_minimal import DevelopmentalLearningMinimal

# Try to import the full version with external LLM support
try:
    from .development import DevelopmentalLearning
    HAS_FULL_LEARNING = True
except ImportError:
    HAS_FULL_LEARNING = False
    DevelopmentalLearning = None

__all__ = [
    'CognitiveSpectrum', 
    'CuriosityEngine',
    'ChromaticMemory',
    'DevelopmentalLearningMinimal'
]

if HAS_FULL_LEARNING:
    __all__.append('DevelopmentalLearning')