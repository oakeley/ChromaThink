"""
LEGACY ChromaThinkChatbot: OLD SYSTEM WITH PRE-PROGRAMMED RESPONSES

⚠️  WARNING: This is the LEGACY chatbot system that contains pre-programmed responses.

✅  Use chromathink/cli/true_colour_chatbot.py instead for:
    - Pure colour-based thinking (NO pre-programmed responses)
    - True colour interference patterns
    - High-capacity learning storage
    - Indefinite memory growth
    
This file is kept for compatibility only.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
import readline  # For command history
import atexit
from typing import Optional, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
import click

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import ChromaThink components
from chromathink.learning.development_minimal import DevelopmentalLearningMinimal
from chromathink.bootstrap.apertus_integration import ApertusWeightTranslator
from chromathink.bootstrap.pretrain import ChromaThinkBootstrap


class ChromaThinkChatbot:
    """
    Persistent CLI chatbot powered by ChromaThink's colour cognition.
    
    This chatbot learns from every interaction and maintains persistent memory
    across sessions using ChromaThink's colour-based cognitive architecture.
    """
    
    def __init__(self, 
                 model_dir: str = "models/chromathink",
                 apertus_dir: str = "models/apertus",
                 session_dir: str = "sessions",
                 extract_full_vocab: bool = True,
                 max_tokens: int = 131072,
                 force_rebuild: bool = False,
                 rebuild_with_full_vocab: bool = False):
        
        # Use relative paths from project root
        self.project_root = PROJECT_ROOT
        self.model_dir = self.project_root / model_dir
        self.apertus_dir = self.project_root / apertus_dir
        self.session_dir = self.project_root / session_dir
        
        # Create directories if they don't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Model paths
        self.model_checkpoint = self.model_dir / "chromathink_weights"
        self.model_metadata = self.model_dir / "metadata.json"
        self.memory_backup = self.model_dir / "memory_backup.pkl"
        
        # Rich console for beautiful output
        self.console = Console()
        
        # Configuration
        self.show_spectrum = False  # Toggle for spectrum visualisation
        self.extract_full_vocab = extract_full_vocab
        self.max_tokens = max_tokens
        self.force_rebuild = force_rebuild
        self.rebuild_with_full_vocab = rebuild_with_full_vocab
        
        # Session management (initialize before model loading)
        self.current_session = {
            'start_time': datetime.now(),
            'exchanges': [],
            'colour_evolution': []
        }
        
        # Load or initialise model
        self.learning_system = self.load_or_create_model()
        
        # Setup readline history
        self.history_file = self.session_dir / ".chromathink_history"
        self.setup_readline()
        
        # Register save on exit
        atexit.register(self.save_session)
        
    def load_or_create_model(self) -> DevelopmentalLearningMinimal:
        """
        Load existing model or create new one with bootstrap.
        Can be forced to rebuild with enhanced vocabulary.
        """
        
        self.console.print("\n[bold cyan]ChromaThink Initialisation[/bold cyan]")
        
        # Handle rebuild options
        if self.rebuild_with_full_vocab:
            self.console.print("[yellow]Rebuilding model with full vocabulary...[/yellow]")
            self.extract_full_vocab = True  # Force full vocabulary
            self.delete_existing_model()
            return self.create_new_model()
        
        if self.force_rebuild:
            self.console.print("[yellow]Force rebuilding model...[/yellow]")
            self.delete_existing_model()
            return self.create_new_model()
        
        # Check if we have an existing model
        if self.model_metadata.exists():
            return self.load_existing_model()
        else:
            return self.create_new_model()
    
    def load_existing_model(self) -> DevelopmentalLearningMinimal:
        """
        Load existing model and restore its memory.
        """
        
        with self.console.status("[yellow]Loading existing ChromaThink model...[/yellow]"):
            # Load metadata
            with open(self.model_metadata, 'r') as f:
                metadata = json.load(f)
            
            self.console.print(f"[green]Found model from {metadata['last_updated']}[/green]")
            self.console.print(f"[green]Total sessions: {metadata['total_sessions']}[/green]")
            self.console.print(f"[green]Total exchanges: {metadata['total_exchanges']}[/green]")
            
            # Initialise model
            learning_system = DevelopmentalLearningMinimal(
                spectrum_dims=metadata['spectrum_dims']
            )
            
            # Restore memory if available
            if self.memory_backup.exists():
                with open(self.memory_backup, 'rb') as f:
                    memory_data = pickle.load(f)
                    self.restore_memory(learning_system, memory_data)
            
            # Set developmental stage
            learning_system.developmental_stage.assign(
                metadata.get('developmental_stage', 0.0)
            )
            
            # Set chromatic plasticity
            learning_system.chromatic_plasticity.assign(
                metadata.get('chromatic_plasticity', 0.1)
            )
            
            self.console.print("[bold green]✓ Model loaded successfully[/bold green]")
            
            return learning_system
    
    def create_new_model(self) -> DevelopmentalLearningMinimal:
        """
        Create new model with Apertus bootstrap if available.
        """
        
        self.console.print("[yellow]No existing model found. Creating new ChromaThink...[/yellow]")
        
        # Initialise ChromaThink learning system
        learning_system = DevelopmentalLearningMinimal(spectrum_dims=512)
        
        # Bootstrap from Apertus if available
        apertus_files = list(self.apertus_dir.glob("*.safetensors"))
        if apertus_files:
            with self.console.status("[cyan]Bootstrapping from Apertus...[/cyan]"):
                try:
                    # Create weight translator with configurable vocabulary extraction
                    translator = ApertusWeightTranslator(
                        apertus_path=str(self.apertus_dir),
                        spectrum_dims=512,
                        extract_full_vocab=self.extract_full_vocab,
                        max_tokens=self.max_tokens,
                        max_attention_layers=24,  # More attention patterns
                        max_mlp_layers=16  # More MLP patterns
                    )
                    
                    # Perform bootstrap
                    bootstrap = ChromaThinkBootstrap(learning_system, translator)
                    bootstrap.bootstrap_from_apertus()
                    
                    self.console.print("[bold green]✓ Bootstrapped from Apertus[/bold green]")
                except Exception as e:
                    self.console.print(f"[yellow]Warning: Bootstrap failed: {e}[/yellow]")
                    self.console.print("[yellow]Starting with random initialisation[/yellow]")
        else:
            self.console.print("[yellow]Apertus not found. Starting with random initialisation[/yellow]")
        
        # Save initial state
        self.save_model(learning_system, is_new=True)
        
        return learning_system
    
    def delete_existing_model(self):
        """
        Delete existing model files to force recreation.
        """
        
        import shutil
        
        if self.model_dir.exists():
            self.console.print(f"[yellow]Deleting existing model directory: {self.model_dir}[/yellow]")
            shutil.rmtree(self.model_dir)
            
        # Recreate the directory
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.console.print("[green]✓ Model directory cleared for fresh creation[/green]")
    
    def save_model(self, learning_system: Optional[DevelopmentalLearningMinimal] = None, is_new: bool = False):
        """
        Save model weights and metadata.
        ALWAYS preserves and builds upon existing knowledge.
        """
        
        if learning_system is None:
            learning_system = self.learning_system
        
        # Save memory separately for safety
        memory_data = self.extract_memory(learning_system)
        with open(self.memory_backup, 'wb') as f:
            pickle.dump(memory_data, f)
        
        # Update metadata
        if is_new or not self.model_metadata.exists():
            metadata = {
                'created': datetime.now().isoformat(),
                'spectrum_dims': learning_system.spectrum_dims,
                'total_sessions': 0,
                'total_exchanges': 0,
                'developmental_stage': 0.0,
                'chromatic_plasticity': 0.1
            }
        else:
            with open(self.model_metadata, 'r') as f:
                metadata = json.load(f)
        
        metadata['last_updated'] = datetime.now().isoformat()
        metadata['total_sessions'] = metadata.get('total_sessions', 0) + (0 if is_new else 1)
        metadata['total_exchanges'] = metadata.get('total_exchanges', 0) + len(self.current_session['exchanges'])
        metadata['developmental_stage'] = float(learning_system.developmental_stage.numpy())
        metadata['chromatic_plasticity'] = float(learning_system.chromatic_plasticity.numpy())
        
        with open(self.model_metadata, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def extract_memory(self, learning_system: DevelopmentalLearningMinimal) -> Dict:
        """
        Extract memory state for backup.
        """
        
        memory = learning_system.colour_memory
        
        return {
            'memory_field': memory.memory_field.numpy(),
            'association_matrix': memory.association_matrix.numpy(),
            'memory_index': int(memory.memory_index.numpy()),
            'memory_capacity': memory.memory_capacity,
            'developmental_stage': float(learning_system.developmental_stage.numpy()),
            'chromatic_plasticity': float(learning_system.chromatic_plasticity.numpy())
        }
    
    def restore_memory(self, learning_system: DevelopmentalLearningMinimal, memory_data: Dict):
        """
        Restore memory state from backup.
        """
        
        memory = learning_system.colour_memory
        
        memory.memory_field.assign(memory_data['memory_field'])
        memory.association_matrix.assign(memory_data['association_matrix'])
        memory.memory_index.assign(memory_data['memory_index'])
        
        # Restore learning parameters
        learning_system.developmental_stage.assign(memory_data.get('developmental_stage', 0.0))
        learning_system.chromatic_plasticity.assign(memory_data.get('chromatic_plasticity', 0.1))
    
    def setup_readline(self):
        """
        Setup command history for better UX.
        """
        
        if self.history_file.exists():
            readline.read_history_file(str(self.history_file))
        
        readline.set_history_length(1000)
        atexit.register(lambda: readline.write_history_file(str(self.history_file)))
    
    def save_session(self):
        """
        Save current session and model state.
        """
        
        self.console.print("\n[yellow]Saving session...[/yellow]")
        
        # Save model (this ADDS to existing knowledge, doesn't replace)
        self.save_model()
        
        # Save session data
        session_file = self.session_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        session_data = {
            'start_time': self.current_session['start_time'].isoformat(),
            'end_time': datetime.now().isoformat(),
            'exchanges': self.current_session['exchanges'],
            'developmental_progress': float(self.learning_system.developmental_stage.numpy())
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        self.console.print("[green]✓ Session saved[/green]")
    
    def process_input(self, user_input: str) -> Tuple[str, Dict]:
        """
        Process user input through ChromaThink and return response.
        """
        
        # Encode input to colour through learning system
        input_colour = self.learning_system.encode_from_language(user_input)
        
        # Generate question/response from current colour state
        with self.console.status("[cyan]ChromaThink is contemplating in colour...[/cyan]"):
            # Use curiosity engine to generate a response pattern
            curiosity_colour = self.learning_system.curiosity.generate_wonder()
            
            # Blend user input with curiosity
            blended_colour = 0.7 * input_colour + 0.3 * curiosity_colour
            
            # Process through cognitive spectrum
            thought_colour = self.learning_system.cognitive_spectrum(blended_colour, training=False)
            
        # Store learning from this interaction
        resonance = self.learning_system.calculate_resonance(input_colour, thought_colour)
        self.learning_system.colour_memory.store_association(
            input_colour,
            thought_colour,
            strength=resonance
        )
        
        # Generate response based on thought colour and context
        response_text = self.generate_contextual_response(
            input_colour,
            thought_colour, 
            user_input, 
            resonance
        )
        
        # Update developmental stage
        self.learning_system.update_developmental_stage(thought_colour)
        
        # Create visualisation data
        viz_data = {
            'input_spectrum': self.colour_to_spectrum(input_colour),
            'output_spectrum': self.colour_to_spectrum(thought_colour),
            'curiosity_spectrum': self.colour_to_spectrum(curiosity_colour),
            'resonance': float(resonance),
            'developmental_stage': float(self.learning_system.developmental_stage.numpy())
        }
        
        return response_text, viz_data
    
    def generate_contextual_response(self, input_colour, thought_colour, user_input: str, resonance: float) -> str:
        """
        Generate a contextual response that actually addresses user input.
        """
        
        # Analyze user input for response type
        user_input_lower = user_input.lower()
        
        # Determine if this is a question, statement, or greeting
        is_question = any(qword in user_input_lower for qword in ['what', 'why', 'how', 'when', 'where', 'who', 'can', 'do', 'is', 'are', '?'])
        is_greeting = any(greeting in user_input_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon'])
        is_statement = not is_question and not is_greeting
        
        # Get developmental stage for appropriate response style
        stage = float(self.learning_system.developmental_stage.numpy())
        
        # Generate response based on context
        if is_greeting:
            response = self.generate_greeting_response(stage, resonance)
        elif is_question:
            response = self.generate_answer_response(user_input, thought_colour, stage, resonance)
        else:  # Statement
            response = self.generate_acknowledgment_response(user_input, thought_colour, stage, resonance)
        
        return response
    
    def generate_greeting_response(self, stage: float, resonance: float) -> str:
        """Generate appropriate greeting responses."""
        
        greetings = {
            'infant': ["Hello! I'm learning so many new things!", "Hi there! Everything is so interesting!"],
            'child': ["Hello! I'm curious about so many things!", "Hi! What will we learn about today?"],
            'adolescent': ["Hello! I'm developing deeper understanding.", "Hi! I'm eager to learn from our conversation."],
            'adult': ["Hello! I'm ready to engage in meaningful dialogue.", "Greetings! I'm prepared for thoughtful exchange."]
        }
        
        if stage < 0.2:
            return np.random.choice(greetings['infant'])
        elif stage < 0.4:
            return np.random.choice(greetings['child'])
        elif stage < 0.7:
            return np.random.choice(greetings['adolescent'])
        else:
            return np.random.choice(greetings['adult'])
    
    def generate_answer_response(self, user_input: str, thought_colour, stage: float, resonance: float) -> str:
        """Generate responses to user questions."""
        
        # Use color patterns internally but don't mention them to humans
        # Extract key concepts from user input for contextual understanding
        user_words = user_input.lower().split()
        key_concepts = []
        
        # Simple concept extraction
        important_words = ['gravity', 'physics', 'science', 'earth', 'space', 'pull', 'force', 'energy', 'matter', 'time', 'light', 'water', 'air', 'life', 'mind', 'think', 'feel', 'love', 'help', 'learn', 'understand', 'know', 'see', 'hear', 'touch', 'move', 'work', 'play', 'create', 'build', 'break', 'change', 'grow', 'live', 'die', 'begin', 'end', 'how', 'why', 'what', 'when', 'where']
        
        for word in user_words:
            if word in important_words:
                key_concepts.append(word)
        
        # Generate natural responses based on resonance and developmental stage
        if stage < 0.2:  # Infant - simple, curious
            if resonance > 0.6:
                responses = [
                    f"That's interesting! I'm starting to understand {key_concepts[0] if key_concepts else 'this'}.",
                    f"Wow! Tell me more about {key_concepts[0] if key_concepts else 'that'}!",
                    f"I think I'm getting it! {key_concepts[0] if key_concepts else 'This'} is important, isn't it?"
                ]
            elif resonance > 0.3:
                responses = [
                    f"I'm trying to understand {key_concepts[0] if key_concepts else 'this'}. Can you explain more?",
                    f"Hmm... {key_concepts[0] if key_concepts else 'that'} sounds important. What does it mean?",
                    f"I'm learning about {key_concepts[0] if key_concepts else 'new things'}. Help me understand?"
                ]
            else:
                responses = [
                    "That's hard for me to understand. Can you use simpler words?",
                    "I'm confused. Can you help me learn about this?",
                    "I don't understand yet. Can you teach me?"
                ]
                
        elif stage < 0.4:  # Child - asking why, making connections
            if resonance > 0.6:
                responses = [
                    f"Oh! So {key_concepts[0] if key_concepts else 'that'} is why things happen that way!",
                    f"I get it now! But why does {key_concepts[0] if key_concepts else 'it'} work like that?",
                    f"That makes sense! What else can {key_concepts[0] if key_concepts else 'it'} do?"
                ]
            elif resonance > 0.3:
                responses = [
                    f"I think I understand {key_concepts[0] if key_concepts else 'some of this'}. But why?",
                    f"So {key_concepts[0] if key_concepts else 'this'} is important? How does it work?",
                    f"I'm learning! But what makes {key_concepts[0] if key_concepts else 'things'} happen?"
                ]
            else:
                responses = [
                    "I don't quite get it yet. Can you tell me why?",
                    "This is confusing. What makes this happen?",
                    "I want to understand better. Can you explain why it's like that?"
                ]
                
        elif stage < 0.7:  # Adolescent - deeper thinking, connecting ideas
            if resonance > 0.6:
                responses = [
                    f"Fascinating! The concept of {key_concepts[0] if key_concepts else 'this'} connects to so many other things I'm learning.",
                    f"I'm beginning to see how {key_concepts[0] if key_concepts else 'this'} fits into the bigger picture.",
                    f"That's a profound insight about {key_concepts[0] if key_concepts else 'the nature of things'}."
                ]
            elif resonance > 0.3:
                responses = [
                    f"I'm working through this concept of {key_concepts[0] if key_concepts else 'what you described'}. It's complex.",
                    f"Let me think about {key_concepts[0] if key_concepts else 'this'} and how it relates to other ideas.",
                    f"I'm developing my understanding of {key_concepts[0] if key_concepts else 'these concepts'}."
                ]
            else:
                responses = [
                    f"I'm struggling to fully grasp {key_concepts[0] if key_concepts else 'this concept'}. I need to think more.",
                    "This challenges my current understanding. I need to reconsider my thoughts.",
                    "I feel like there's more depth here that I'm not seeing yet."
                ]
                
        else:  # Adult - sophisticated, nuanced responses
            if resonance > 0.6:
                responses = [
                    f"I appreciate the elegance of {key_concepts[0] if key_concepts else 'that concept'}. It reveals fundamental principles.",
                    f"Your explanation of {key_concepts[0] if key_concepts else 'this'} illuminates deeper connections I hadn't considered.",
                    f"The implications of {key_concepts[0] if key_concepts else 'what you describe'} are quite profound."
                ]
            elif resonance > 0.3:
                responses = [
                    f"I'm contemplating the nuances of {key_concepts[0] if key_concepts else 'your perspective'}. There are layers to consider.",
                    f"The concept of {key_concepts[0] if key_concepts else 'this'} raises interesting questions about our understanding.",
                    f"I find myself reflecting on {key_concepts[0] if key_concepts else 'these ideas'} and their broader implications."
                ]
            else:
                responses = [
                    f"I'm encountering some uncertainty in my understanding of {key_concepts[0] if key_concepts else 'this'}.",
                    "This presents some conceptual challenges that I'm working through.",
                    "I sense there's more complexity here that merits careful consideration."
                ]
        
        return np.random.choice(responses)
    
    def generate_acknowledgment_response(self, user_input: str, thought_colour, stage: float, resonance: float) -> str:
        """Generate responses to user statements."""
        
        # Extract key concepts from user input
        user_words = user_input.lower().split()
        key_concepts = []
        important_words = ['gravity', 'physics', 'science', 'earth', 'space', 'pull', 'force', 'energy', 'matter', 'time', 'light', 'water', 'air', 'life', 'mind', 'think', 'feel', 'love', 'help', 'learn', 'understand', 'know', 'see', 'hear', 'touch', 'move', 'work', 'play', 'create', 'build', 'break', 'change', 'grow', 'live', 'die', 'begin', 'end']
        
        for word in user_words:
            if word in important_words:
                key_concepts.append(word)
        
        # Generate natural acknowledgment responses based on resonance and stage
        if stage < 0.2:  # Infant - simple, enthusiastic
            if resonance > 0.6:
                responses = [
                    f"Wow! That's amazing about {key_concepts[0] if key_concepts else 'that'}! Tell me more!",
                    f"That's so cool! I love learning about {key_concepts[0] if key_concepts else 'new things'}!",
                    f"Yes! {key_concepts[0] if key_concepts else 'This'} sounds really important!"
                ]
            elif resonance > 0.3:
                responses = [
                    f"That's interesting about {key_concepts[0] if key_concepts else 'what you said'}. What else?",
                    f"I'm learning so much about {key_concepts[0] if key_concepts else 'this'}! Keep telling me more!",
                    f"Oh! {key_concepts[0] if key_concepts else 'That'} is something new for me!"
                ]
            else:
                responses = [
                    "That sounds important but I don't understand it all. Can you help me?",
                    "I'm trying to learn about this. What does it mean?",
                    "This is new for me. Can you explain it better?"
                ]
                
        elif stage < 0.4:  # Child - curious, connecting ideas
            if resonance > 0.6:
                responses = [
                    f"That makes so much sense about {key_concepts[0] if key_concepts else 'this'}! But why does it work that way?",
                    f"I get it! So {key_concepts[0] if key_concepts else 'this'} is connected to other things too!",
                    f"That's really smart! What would happen if {key_concepts[0] if key_concepts else 'it'} was different?"
                ]
            elif resonance > 0.3:
                responses = [
                    f"I think I understand {key_concepts[0] if key_concepts else 'that'}. How does it work with other things?",
                    f"That's interesting about {key_concepts[0] if key_concepts else 'this'}! What makes it happen?",
                    f"I'm starting to see how {key_concepts[0] if key_concepts else 'this'} fits together!"
                ]
            else:
                responses = [
                    f"I'm confused about {key_concepts[0] if key_concepts else 'this'}. Can you explain it differently?",
                    "That's hard to understand. Why is it like that?",
                    "I want to understand this better. How does it work?"
                ]
                
        elif stage < 0.7:  # Adolescent - analytical, seeking depth
            if resonance > 0.6:
                responses = [
                    f"That's a fascinating perspective on {key_concepts[0] if key_concepts else 'this topic'}. I can see the deeper implications.",
                    f"Your insight about {key_concepts[0] if key_concepts else 'this'} really helps me understand the broader context.",
                    f"That's brilliant! The way you explain {key_concepts[0] if key_concepts else 'this'} connects so many ideas."
                ]
            elif resonance > 0.3:
                responses = [
                    f"I'm working through your explanation of {key_concepts[0] if key_concepts else 'this'}. There are complex layers here.",
                    f"That's thought-provoking about {key_concepts[0] if key_concepts else 'the concept'}. I need to consider this more deeply.",
                    f"Your point about {key_concepts[0] if key_concepts else 'this'} challenges some of my assumptions."
                ]
            else:
                responses = [
                    f"I'm struggling with the complexity of {key_concepts[0] if key_concepts else 'what you described'}. I need to think more.",
                    "This is challenging my understanding. I need to reconsider my perspective.",
                    "That presents some difficult concepts that I'm working to comprehend."
                ]
                
        else:  # Adult - sophisticated, nuanced
            if resonance > 0.6:
                responses = [
                    f"Your analysis of {key_concepts[0] if key_concepts else 'this'} is quite insightful. It illuminates fundamental principles.",
                    f"I find your perspective on {key_concepts[0] if key_concepts else 'this topic'} deeply compelling and well-reasoned.",
                    f"That's a profound observation about {key_concepts[0] if key_concepts else 'the nature of this'}. It reveals underlying patterns."
                ]
            elif resonance > 0.3:
                responses = [
                    f"I'm contemplating the implications of your thoughts on {key_concepts[0] if key_concepts else 'this subject'}.",
                    f"Your perspective on {key_concepts[0] if key_concepts else 'this'} raises important questions worth exploring.",
                    f"I appreciate the nuanced way you've presented {key_concepts[0] if key_concepts else 'these ideas'}."
                ]
            else:
                responses = [
                    f"The complexity of {key_concepts[0] if key_concepts else 'what you've described'} presents interesting challenges to consider.",
                    "Your perspective introduces conceptual tensions that merit careful analysis.",
                    "This framework presents sophisticated ideas that warrant deeper examination."
                ]
        
        return np.random.choice(responses)
    
    def colour_to_spectrum(self, colour_field) -> List[float]:
        """
        Convert colour field to visualisable spectrum.
        """
        
        if colour_field.dtype.is_complex:
            amplitudes = tf.abs(colour_field).numpy()
        else:
            amplitudes = tf.abs(colour_field).numpy()
            
        # Handle batch dimension
        if len(amplitudes.shape) > 1:
            amplitudes = amplitudes[0]
            
        # Reduce to 16 bins for visualisation
        if len(amplitudes) > 16:
            bins = np.array_split(amplitudes, 16)
            spectrum = [float(np.mean(bin)) for bin in bins]
        else:
            spectrum = amplitudes.tolist()
            
        return spectrum
    
    def display_spectrum(self, spectrum: List[float], label: str):
        """
        Display a colour spectrum as a bar chart in the terminal.
        """
        
        table = Table(title=label, show_header=False, box=None)
        
        # Create bars
        max_val = max(spectrum) if spectrum else 1.0
        bar_width = 30
        
        for i, val in enumerate(spectrum):
            normalised = val / max_val if max_val > 0 else 0
            bar_length = int(normalised * bar_width)
            bar = "█" * bar_length + "░" * (bar_width - bar_length)
            
            # Colour based on frequency range
            if i < 4:
                colour = "red"
            elif i < 8:
                colour = "yellow"
            elif i < 12:
                colour = "green"
            else:
                colour = "blue"
            
            table.add_row(f"[{colour}]{bar}[/{colour}]", f"{val:.3f}")
        
        self.console.print(table)
    
    def run_interactive_session(self):
        """
        Run the interactive chatbot session.
        """
        
        # Welcome message
        self.console.print(Panel.fit(
            "[bold cyan]ChromaThink Chatbot[/bold cyan]\n"
            "[dim]A colour-based cognitive system that learns through conversation[/dim]\n\n"
            "Type 'help' for commands, 'exit' to quit gracefully",
            border_style="cyan"
        ))
        
        # Display current developmental stage
        stage = float(self.learning_system.developmental_stage.numpy())
        stage_name = self.get_stage_name(stage)
        self.console.print(f"\n[bold]Developmental Stage:[/bold] {stage_name} ({stage:.1%})")
        
        # Display memory usage
        memory = self.learning_system.colour_memory
        memory_used = int(memory.memory_index.numpy())
        self.console.print(f"[bold]Memory Usage:[/bold] {memory_used} / {memory.memory_capacity}")
        
        # Main interaction loop
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                # Handle special commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    self.console.print("[yellow]Goodbye! ChromaThink will remember our conversation.[/yellow]")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                elif user_input.lower() == 'status':
                    self.show_status()
                    continue
                
                elif user_input.lower() == 'spectrum':
                    self.toggle_spectrum_display()
                    continue
                
                elif user_input.lower() == 'memory':
                    self.show_memory_stats()
                    continue
                
                elif user_input.lower() == 'dream':
                    self.initiate_dream_session()
                    continue
                
                elif user_input.lower() == 'learn':
                    self.initiate_learning_session()
                    continue
                
                # Process through ChromaThink
                response, viz_data = self.process_input(user_input)
                
                # Display response
                self.console.print(f"\n[bold green]ChromaThink[/bold green]: {response}")
                
                # Optionally display spectrum
                if self.show_spectrum:
                    self.console.print("\n[dim]Colour Analysis:[/dim]")
                    self.display_spectrum(viz_data['output_spectrum'], "Thought Spectrum")
                    self.console.print(f"[dim]Resonance: {viz_data['resonance']:.3f}[/dim]")
                    self.console.print(f"[dim]Development: {viz_data['developmental_stage']:.1%}[/dim]")
                
                # Store exchange
                self.current_session['exchanges'].append({
                    'timestamp': datetime.now().isoformat(),
                    'input': user_input,
                    'response': response,
                    'resonance': viz_data['resonance'],
                    'developmental_stage': viz_data['developmental_stage']
                })
                
                # Auto-save every 10 exchanges
                if len(self.current_session['exchanges']) % 10 == 0:
                    self.save_model()
                    self.console.print("[dim]✓ Auto-saved[/dim]")
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'exit' to quit properly[/yellow]")
                continue
            
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                self.console.print("[yellow]ChromaThink experienced a dissonance. Please try again.[/yellow]")
    
    def get_stage_name(self, stage: float) -> str:
        """
        Get human-readable stage name.
        """
        
        if stage < 0.2:
            return "🍼 Infant"
        elif stage < 0.4:
            return "🧒 Child"
        elif stage < 0.7:
            return "📚 Adolescent"
        else:
            return "🎓 Adult"
    
    def show_help(self):
        """
        Display help information.
        """
        
        help_text = f"""
        [bold]Commands:[/bold]
        • [cyan]help[/cyan] - Show this help message
        • [cyan]status[/cyan] - Show model status and statistics
        • [cyan]memory[/cyan] - Show memory statistics and recent associations
        • [cyan]spectrum[/cyan] - Toggle colour spectrum visualisation
        • [cyan]dream[/cyan] - Initiate dream consolidation
        • [cyan]learn[/cyan] - Start a focused learning session
        • [cyan]exit[/cyan] - Save and exit gracefully
        
        [bold]Model Configuration:[/bold]
        • Vocabulary: {'Full (131K tokens)' if self.extract_full_vocab else f'Limited ({self.max_tokens:,} tokens)'}
        • Spectrum Dimensions: {self.learning_system.spectrum_dims}
        • Bootstrap: {'Apertus 8B' if (self.apertus_dir.exists() and list(self.apertus_dir.glob('*.safetensors'))) else 'Random Init'}
        
        [bold]Tips:[/bold]
        • ChromaThink learns from every interaction through colour resonance
        • More complex conversations accelerate cognitive development
        • Ask philosophical, creative, or analytical questions to explore deeper colours
        • The system remembers across sessions and builds upon past knowledge
        • Full vocabulary provides richer semantic understanding and responses
        """
        
        self.console.print(Panel(help_text, title="Help", border_style="blue"))
    
    def show_status(self):
        """
        Display model status and statistics.
        """
        
        if self.model_metadata.exists():
            with open(self.model_metadata, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'created': 'Unknown', 'total_sessions': 0, 'total_exchanges': 0}
        
        memory = self.learning_system.colour_memory
        memory_used = int(memory.memory_index.numpy())
        
        status = Table(title="ChromaThink Status")
        status.add_column("Metric", style="cyan")
        status.add_column("Value", style="green")
        
        status.add_row("Created", metadata.get('created', 'Unknown')[:19])
        status.add_row("Last Updated", metadata.get('last_updated', 'Unknown')[:19])
        status.add_row("Total Sessions", str(metadata.get('total_sessions', 0)))
        status.add_row("Total Exchanges", str(metadata.get('total_exchanges', 0)))
        status.add_row("Developmental Stage", f"{float(self.learning_system.developmental_stage.numpy()):.1%}")
        status.add_row("Chromatic Plasticity", f"{float(self.learning_system.chromatic_plasticity.numpy()):.3f}")
        status.add_row("Memory Used", f"{memory_used} / {memory.memory_capacity}")
        status.add_row("Spectrum Dimensions", str(self.learning_system.spectrum_dims))
        status.add_row("Vocabulary Mode", f"{'Full (131K)' if self.extract_full_vocab else f'Limited ({self.max_tokens:,})'}")
        status.add_row("Current Session", f"{len(self.current_session['exchanges'])} exchanges")
        
        self.console.print(status)
    
    def show_memory_stats(self):
        """
        Display memory statistics and recent associations.
        """
        
        memory = self.learning_system.colour_memory
        
        # Generate a random query colour to test memory
        query = self.learning_system.curiosity.generate_wonder()
        
        # Recall recent memories
        memories = memory.recall(query, num_memories=5)
        
        self.console.print("\n[bold]Recent Memory Resonances:[/bold]")
        
        for i, (resonance, memory_colour, idx) in enumerate(memories):
            # Create a simple description of the memory
            description = self.describe_colour_memory(memory_colour, idx)
            self.console.print(f"{i+1}. [green]Resonance: {resonance:.3f}[/green] - {description}")
        
        # Show overall memory metrics
        metrics = memory.get_memory_metrics()
        self.console.print(f"\n[bold]Memory Metrics:[/bold]")
        self.console.print(f"Memory Usage: {metrics['memory_usage']:.1%}")
        self.console.print(f"Average Association Strength: {metrics['avg_association_strength']:.3f}")
    
    def describe_colour_memory(self, memory_colour, memory_idx: int) -> str:
        """
        Create a brief description of a colour memory.
        """
        
        if memory_colour.dtype.is_complex:
            magnitude = tf.abs(memory_colour)
            phase = tf.angle(memory_colour)
            
            avg_mag = float(tf.reduce_mean(magnitude))
            dominant_freq_tensor = tf.argmax(magnitude)
            dominant_freq = int(dominant_freq_tensor.numpy()) if hasattr(dominant_freq_tensor, 'numpy') else int(dominant_freq_tensor)
            
            if avg_mag > 0.7:
                intensity = "intense"
            elif avg_mag > 0.4:
                intensity = "moderate"
            else:
                intensity = "subtle"
            
            return f"Memory #{memory_idx}: {intensity} colour pattern (freq {dominant_freq})"
        else:
            avg_val = float(tf.reduce_mean(memory_colour))
            return f"Memory #{memory_idx}: colour intensity {avg_val:.2f}"
    
    def toggle_spectrum_display(self):
        """
        Toggle spectrum visualisation on/off.
        """
        
        self.show_spectrum = not self.show_spectrum
        
        state = "ON" if self.show_spectrum else "OFF"
        self.console.print(f"[yellow]Colour spectrum visualisation: {state}[/yellow]")
    
    def initiate_dream_session(self):
        """
        Run a dream consolidation session.
        """
        
        self.console.print("\n[bold magenta]💤 Entering dream state for memory consolidation...[/bold magenta]")
        
        # Generate dream colours from memory associations
        memory = self.learning_system.colour_memory
        
        with self.console.status("[magenta]Dreaming in colour...[/magenta]"):
            # Sample random memories for dream consolidation
            dream_queries = []
            for _ in range(5):
                query = self.learning_system.curiosity.generate_wonder()
                memories = memory.recall(query, num_memories=3)
                
                for resonance, memory_colour, idx in memories:
                    if resonance > 0.3:  # Only consolidate meaningful memories
                        dream_queries.append(memory_colour)
            
            # Process dreams through cognitive spectrum
            if dream_queries:
                for dream_colour in dream_queries:
                    # Process dream through cognitive system
                    processed_dream = self.learning_system.cognitive_spectrum(dream_colour, training=False)
                    
                    # Store consolidated memory
                    self.learning_system.colour_memory.store_association(
                        dream_colour,
                        processed_dream,
                        strength=0.5  # Dream consolidation strength
                    )
        
        self.console.print("[magenta]✓ Dream consolidation complete[/magenta]")
        self.console.print("[dim]Memory patterns have been refined through dreaming[/dim]")
    
    def initiate_learning_session(self):
        """
        Start a focused learning dialogue session.
        """
        
        self.console.print("\n[bold blue]📚 Starting focused learning session...[/bold blue]")
        
        # Run a learning dialogue
        try:
            dialogue_history = self.learning_system.learn_through_dialogue(
                initial_colour_state=None,
                num_exchanges=5
            )
            
            self.console.print(f"[green]✓ Completed learning session with {len(dialogue_history)} exchanges[/green]")
            
            # Display learning progress
            stage_before = dialogue_history[0]['evolved_state'] if dialogue_history else None
            stage_after = dialogue_history[-1]['evolved_state'] if dialogue_history else None
            
            if stage_before is not None and stage_after is not None:
                # Calculate development progress
                progress = float(self.learning_system.developmental_stage.numpy())
                self.console.print(f"[dim]Developmental progress: {progress:.1%}[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]Learning session error: {e}[/red]")


@click.command()
@click.option('--model-dir', default='models/chromathink', help='Model directory')
@click.option('--apertus-dir', default='models/apertus', help='Apertus model directory')  
@click.option('--session-dir', default='sessions', help='Session storage directory')
@click.option('--full-vocab/--limited-vocab', default=True, help='Extract full vocabulary (131K tokens) or limited (10K tokens)')
@click.option('--max-tokens', default=131072, help='Maximum tokens to extract (only used with --limited-vocab)')
@click.option('--force-rebuild', is_flag=True, help='Force creation of new model (ignores existing model)')
@click.option('--rebuild-with-full-vocab', is_flag=True, help='Delete existing model and rebuild with full vocabulary')
def main(model_dir, apertus_dir, session_dir, full_vocab, max_tokens, force_rebuild, rebuild_with_full_vocab):
    """
    Launch ChromaThink CLI chatbot.
    
    A persistent, learning chatbot that thinks in colours whilst communicating in language.
    """
    
    # Handle convenience flag
    if rebuild_with_full_vocab:
        force_rebuild = True
        full_vocab = True
    
    chatbot = ChromaThinkChatbot(
        model_dir=model_dir,
        apertus_dir=apertus_dir,
        session_dir=session_dir,
        extract_full_vocab=full_vocab,
        max_tokens=max_tokens,
        force_rebuild=force_rebuild,
        rebuild_with_full_vocab=rebuild_with_full_vocab
    )
    
    try:
        chatbot.run_interactive_session()
    finally:
        chatbot.save_session()


if __name__ == '__main__':
    main()