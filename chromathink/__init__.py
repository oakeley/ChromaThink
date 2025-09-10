"""
ChromaThink: A Colour-Wave Neural Architecture

This package implements an experimental neural network architecture that processes
thoughts as colours and waveforms rather than discrete tokens.
"""

__version__ = "0.1.0"

from .layers.cognitive_waveform import CognitiveWaveform
from .layers.interference import InterferenceLayer
from .layers.chromatic_resonance import ChromaticResonance
from .layers.resonance_chamber import ResonanceChamber

__all__ = [
    "CognitiveWaveform",
    "InterferenceLayer", 
    "ChromaticResonance",
    "ResonanceChamber"
]