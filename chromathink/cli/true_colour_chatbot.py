"""
TrueColourChatbot: Pure colour-based CLI chatbot interface
"""

import os
import sys
import json
import h5py
import sqlite3
from pathlib import Path
from datetime import datetime
import readline
import atexit
from typing import Optional, Dict, List, Tuple, Any
import logging

import numpy as np
import tensorflow as tf
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.bar import Bar

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import true colour thinking system
from chromathink.core.true_colour_thinking import create_integrated_chromathink
from chromathink.learning.true_colour_development import TrueColourDevelopment
from chromathink.core.gpu_colour_ops import create_gpu_accelerator


class HighCapacityStorage:
    """
    High-capacity storage system for indefinite learning with 386GB RAM.
    Uses HDF5 for efficient colour pattern storage and SQLite for metadata.
    """
    
    def __init__(self, storage_dir: str = "learning_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # HDF5 for colour patterns (efficient binary storage)
        self.hdf5_path = self.storage_dir / "colour_memories.h5"
        self.db_path = self.storage_dir / "memory_metadata.db"
        
        self.logger = logging.getLogger("HighCapacityStorage")
        
        # Initialize storage
        self._init_hdf5_storage()
        self._init_sqlite_metadata()
        
        # Memory cache (utilize your 386GB RAM)
        self.ram_cache_size = 50000  # Cache 50K colour patterns in RAM
        self.colour_cache = {}
        self.metadata_cache = {}
        
        self.logger.info(f"High-capacity storage initialized at {self.storage_dir}")
    
    def _init_hdf5_storage(self):
        """Initialize HDF5 storage for colour patterns."""
        
        with h5py.File(self.hdf5_path, 'a') as f:
            # Create groups for different types of memories
            if 'episodic' not in f:
                f.create_group('episodic')
            if 'semantic' not in f:
                f.create_group('semantic')
            if 'emotional' not in f:
                f.create_group('emotional')
            if 'developmental' not in f:
                f.create_group('developmental')
    
    def _init_sqlite_metadata(self):
        """Initialize SQLite database for metadata."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS colour_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                complexity REAL NOT NULL,
                amplitude REAL NOT NULL,
                resonance_strength REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL NOT NULL,
                tags TEXT,
                context TEXT,
                hdf5_key TEXT NOT NULL,
                session_id TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_memory_type ON colour_memories(memory_type)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_complexity ON colour_memories(complexity)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON colour_memories(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def store_colour_memory(self, 
                           colour_pattern: tf.Tensor,
                           memory_type: str = 'episodic',
                           complexity: float = 0.0,
                           amplitude: float = 0.0,
                           resonance_strength: float = 1.0,
                           tags: List[str] = None,
                           context: str = '',
                           session_id: str = None) -> int:
        """
        Store a colour memory with high-capacity backend.
        
        Returns:
            Memory ID for future retrieval
        """
        
        timestamp = datetime.now().timestamp()
        memory_id = int(timestamp * 1000000)  # Microsecond precision ID
        
        # Convert to numpy for storage
        colour_array = colour_pattern.numpy() if hasattr(colour_pattern, 'numpy') else colour_pattern
        
        # Store in HDF5
        hdf5_key = f"{memory_type}/{memory_id}"
        
        with h5py.File(self.hdf5_path, 'a') as f:
            f[hdf5_key] = colour_array
        
        # Store metadata in SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO colour_memories 
            (memory_type, timestamp, complexity, amplitude, resonance_strength, 
             last_accessed, tags, context, hdf5_key, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory_type, timestamp, complexity, amplitude, resonance_strength,
            timestamp, json.dumps(tags or []), context, hdf5_key, session_id
        ))
        
        actual_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Add to RAM cache if space available
        if len(self.colour_cache) < self.ram_cache_size:
            self.colour_cache[actual_id] = colour_array
            self.metadata_cache[actual_id] = {
                'memory_type': memory_type,
                'timestamp': timestamp,
                'complexity': complexity,
                'amplitude': amplitude,
                'resonance_strength': resonance_strength,
                'tags': tags or []
            }
        
        return actual_id
    
    def recall_similar_colours(self, 
                              query_colour: tf.Tensor,
                              memory_type: str = None,
                              limit: int = 10,
                              min_complexity: float = 0.0,
                              similarity_threshold: float = 0.1) -> List[Tuple[int, tf.Tensor, float]]:
        """
        Recall similar colour memories using high-performance search.
        
        Returns:
            List of (memory_id, colour_pattern, similarity_score)
        """
        
        query_array = query_colour.numpy() if hasattr(query_colour, 'numpy') else query_colour
        
        # Build SQL query for metadata filtering
        sql = '''
            SELECT id, memory_type, complexity, amplitude, resonance_strength, hdf5_key
            FROM colour_memories 
            WHERE complexity >= ?
        '''
        params = [min_complexity]
        
        if memory_type:
            sql += ' AND memory_type = ?'
            params.append(memory_type)
        
        sql += ' ORDER BY last_accessed DESC LIMIT ?'
        params.append(limit * 3)  # Get more candidates for similarity filtering
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(sql, params)
        candidates = cursor.fetchall()
        conn.close()
        
        # Calculate similarities
        similarities = []
        
        for memory_id, mem_type, complexity, amplitude, resonance, hdf5_key in candidates:
            # Try RAM cache first
            if memory_id in self.colour_cache:
                colour_array = self.colour_cache[memory_id]
            else:
                # Load from HDF5
                with h5py.File(self.hdf5_path, 'r') as f:
                    if hdf5_key in f:
                        colour_array = f[hdf5_key][:]
                    else:
                        continue
            
            # Calculate similarity (complex dot product)
            if colour_array.dtype.kind == 'c':  # Complex array
                similarity = np.abs(np.vdot(query_array.flatten(), colour_array.flatten()))
            else:
                similarity = np.dot(query_array.flatten(), colour_array.flatten()) / (
                    np.linalg.norm(query_array) * np.linalg.norm(colour_array) + 1e-8
                )
            
            similarity = float(np.real(similarity))
            
            if similarity >= similarity_threshold:
                similarities.append((memory_id, tf.constant(colour_array), similarity))
        
        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:limit]
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total memories by type
        cursor.execute('''
            SELECT memory_type, COUNT(*), AVG(complexity), AVG(amplitude)
            FROM colour_memories 
            GROUP BY memory_type
        ''')
        type_stats = cursor.fetchall()
        
        # Overall stats
        cursor.execute('SELECT COUNT(*), AVG(complexity), MAX(complexity) FROM colour_memories')
        total_count, avg_complexity, max_complexity = cursor.fetchone()
        
        # Recent activity
        cursor.execute('''
            SELECT COUNT(*) FROM colour_memories 
            WHERE timestamp > ?
        ''', (datetime.now().timestamp() - 24*3600,))  # Last 24 hours
        recent_count = cursor.fetchone()[0]
        
        conn.close()
        
        # File sizes
        hdf5_size = self.hdf5_path.stat().st_size if self.hdf5_path.exists() else 0
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        
        return {
            'total_memories': total_count or 0,
            'recent_memories_24h': recent_count or 0,
            'average_complexity': avg_complexity or 0.0,
            'max_complexity': max_complexity or 0.0,
            'memories_by_type': dict((mem_type, count) for mem_type, count, _, _ in type_stats),
            'ram_cache_size': len(self.colour_cache),
            'hdf5_file_size_mb': hdf5_size / (1024*1024),
            'sqlite_file_size_mb': db_size / (1024*1024),
            'storage_directory': str(self.storage_dir)
        }


class TrueColourChatbot:
    """
    True colour-based chatbot with NO pre-programmed responses.
    All responses emerge from colour interference patterns.
    """
    
    def __init__(self, 
                 apertus_dir: str = "models/apertus",
                 storage_dir: str = "learning_storage",
                 spectrum_dims: int = 1024,  # Larger for high-capacity system
                 use_real_apertus: bool = True):
        
        self.spectrum_dims = spectrum_dims
        self.logger = logging.getLogger("TrueColourChatbot")
        
        # Rich console for beautiful output
        self.console = Console()
        
        # High-capacity storage system
        self.storage = HighCapacityStorage(storage_dir)
        
        # Initialize true colour thinking system
        self.chromathink = create_integrated_chromathink(
            apertus_path=apertus_dir,
            spectrum_dims=spectrum_dims,
            gpu_acceleration=True,
            use_mock_apertus=not use_real_apertus
        )
        
        # Developmental learning system
        self.development = TrueColourDevelopment(
            spectrum_dims=spectrum_dims,
            apertus_path=apertus_dir,
            use_mock_apertus=not use_real_apertus,
            gpu_acceleration=True
        )
        
        # GPU accelerator for high-performance computing
        self.gpu_accelerator = create_gpu_accelerator(
            spectrum_dims=spectrum_dims,
            mixed_precision=True
        )
        
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exchange_count = 0
        self.show_spectrum = False
        
        # Setup readline history
        self.history_file = Path(storage_dir) / ".chromathink_history"
        self._setup_readline()
        
        # Register cleanup
        atexit.register(self._cleanup)
        
        self.console.print(Panel(
            "[green] Chatbot Initialized[/green]\n\n"
            f" Colour-based thinking active\n"
            f" Spectral dimensions: {spectrum_dims}\n"  
            f" Storage ready\n"
            f" GPU acceleration enabled\n"
            f" Session ID: {self.session_id}",
            title="ChromaThink Ready"
        ))
    
    def _setup_readline(self):
        """Setup readline for command history."""
        
        if self.history_file.exists():
            readline.read_history_file(str(self.history_file))
        
        readline.set_history_length(1000)
    
    def _cleanup(self):
        """Cleanup on exit."""
        
        readline.write_history_file(str(self.history_file))
        
        # Save final storage stats
        stats = self.storage.get_storage_stats()
        self.console.print(f"\n Session complete - {stats['total_memories']} memories stored")
    
    def process_user_input(self, user_input: str, thinking_intensity: float = 1.0) -> str:
        # Process through ChromaThink's pure colour cognition
        response = self.chromathink.process_input(user_input)
        
        # Store this interaction in high-capacity storage
        # Get basic metrics from ChromaThink core
        try:
            cognitive_metrics = self.chromathink.thinker.get_cognitive_metrics()
            complexity = cognitive_metrics.get('cognitive_complexity', 0.5)
            amplitude = cognitive_metrics.get('cognitive_amplitude', 0.5)
        except:
            # Fallback values
            complexity = 0.5
            amplitude = 0.5
        
        # Store the experience (both input and response concepts as colours)
        self.storage.store_colour_memory(
            colour_pattern=self.chromathink.thinker.cognitive_state,
            memory_type='episodic',
            complexity=complexity,
            amplitude=amplitude,
            resonance_strength=1.0,
            tags=['conversation', 'user_interaction'],
            context=f"User: {user_input[:100]}... Response: {response[:100]}...",
            session_id=self.session_id
        )
        
        self.exchange_count += 1
        
        return response
    
    def show_colour_spectrum(self, colour_pattern: tf.Tensor = None):
        """Display colour spectrum visualization."""
        
        if colour_pattern is None:
            colour_pattern = self.chromathink.thinker.cognitive_state
        
        if tf.reduce_sum(tf.abs(colour_pattern)) == 0:
            self.console.print(" [dim]No active colour pattern[/dim]")
            return
        
        # Convert to amplitude spectrum
        if colour_pattern.dtype == tf.complex64:
            amplitudes = tf.abs(colour_pattern).numpy()
        else:
            amplitudes = colour_pattern.numpy()
        
        # Create spectrum bars
        spectrum_table = Table(title="Colour Spectrum", show_header=False, box=None)
        spectrum_table.add_column("Frequency", style="cyan", no_wrap=True)
        spectrum_table.add_column("Amplitude", style="green")
        
        # Show top 16 frequency components
        top_indices = np.argsort(amplitudes)[-16:][::-1]
        
        for i, freq_idx in enumerate(top_indices):
            amplitude = amplitudes[freq_idx]
            freq_label = f"F{freq_idx:03d}"
            
            # Create bar visualization
            bar_length = int(amplitude * 30)  # Scale to 30 chars max
            bar = "█" * bar_length + "░" * (30 - bar_length)
            bar_text = f"[bright_red]{bar}[/bright_red] {amplitude:.3f}"
            
            spectrum_table.add_row(freq_label, bar_text)
        
        self.console.print(spectrum_table)
    
    def show_storage_stats(self):
        """Display comprehensive storage statistics."""
        
        stats = self.storage.get_storage_stats()
        
        # Main stats table
        stats_table = Table(title="📊 High-Capacity Storage Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Memories", f"{stats['total_memories']:,}")
        stats_table.add_row("Recent (24h)", f"{stats['recent_memories_24h']:,}")
        stats_table.add_row("RAM Cache", f"{stats['ram_cache_size']:,}")
        stats_table.add_row("Average Complexity", f"{stats['average_complexity']:.4f}")
        stats_table.add_row("Max Complexity", f"{stats['max_complexity']:.4f}")
        stats_table.add_row("HDF5 Size", f"{stats['hdf5_file_size_mb']:.1f} MB")
        stats_table.add_row("SQLite Size", f"{stats['sqlite_file_size_mb']:.1f} MB")
        
        self.console.print(stats_table)
        
        # Memory types breakdown
        if stats['memories_by_type']:
            type_table = Table(title="Memory Types")
            type_table.add_column("Type", style="cyan")
            type_table.add_column("Count", style="green")
            
            for mem_type, count in stats['memories_by_type'].items():
                type_table.add_row(mem_type.title(), f"{count:,}")
            
            self.console.print(type_table)
    
    def dream_consolidation(self):
        """Perform dream consolidation with high-capacity storage."""
        
        self.console.print(" [yellow]Initiating dream consolidation...[/yellow]")
        
        # ChromaThink dreams
        dream_colours = self.chromathink.dream(dream_intensity=0.7, num_dream_cycles=5)
        
        # Store dreams in high-capacity storage
        for i, dream_colour in enumerate(dream_colours):
            complexity = self.chromathink.thinker._calculate_spectral_entropy(dream_colour)
            amplitude = float(tf.reduce_mean(tf.abs(dream_colour)))
            
            self.storage.store_colour_memory(
                colour_pattern=dream_colour,
                memory_type='emotional',  # Dreams are emotional memories
                complexity=complexity,
                amplitude=amplitude,
                resonance_strength=0.8,
                tags=['dream', 'consolidation', f'cycle_{i+1}'],
                context=f"Dream consolidation cycle {i+1}",
                session_id=self.session_id
            )
        
        self.console.print(f" [green]Dream complete - {len(dream_colours)} dreams stored[/green]")
    
    def developmental_learning_session(self):
        """Run a developmental learning session."""
        
        self.console.print(" [yellow]Starting developmental learning session...[/yellow]")
        
        learning_topics = [
            "What is the nature of understanding?",
            "How do concepts form and evolve?",
            "What is the relationship between learning and memory?",
            "Can artificial minds experience genuine curiosity?", 
            "What does it mean to truly comprehend something?"
        ]
        
        results = self.development.learn_through_dialogue(
            teacher_inputs=learning_topics,
            num_exchanges=5,
            thinking_intensity=1.3
        )
        
        # Store learning session results
        final_complexity = results['learning_metrics']['colour_cognition_metrics']['average_colour_complexity']
        development_growth = results['learning_metrics']['session_summary']['development_growth']
        
        self.storage.store_colour_memory(
            colour_pattern=self.development.chromathink_system.thinker.cognitive_state,
            memory_type='developmental',
            complexity=final_complexity,
            amplitude=1.0,
            resonance_strength=1.0,
            tags=['learning_session', 'development', 'growth'],
            context=f"Developmental growth: {development_growth:.4f}",
            session_id=self.session_id
        )
        
        # Show results
        metrics = results['learning_metrics']
        self.console.print(Panel(
            f" Development Growth: {metrics['session_summary']['development_growth']:.4f}\n"
            f" Learning Quality: {metrics['learning_quality_assessment']['quality_rating']}\n"
            f" Final Complexity: {final_complexity:.4f}",
            title="Learning Session Complete"
        ))
    
    def interactive_chat(self):
        """Main interactive chat loop with colour thinking."""
        
        self.console.print(Panel(
            "[bold green]Welcome to ChromaThink Chat![/bold green]\n\n"
            "Every response emerges from pure colour interference patterns.\n"
            "Commands:\n"
            "• [cyan]spectrum[/cyan] - Show colour spectrum\n"
            "• [cyan]dream[/cyan] - Memory consolidation\n" 
            "• [cyan]learn[/cyan] - Developmental learning session\n"
            "• [cyan]stats[/cyan] - Storage statistics\n"
            "• [cyan]intensity X[/cyan] - Set thinking intensity (0.5-2.0)\n"
            "• [cyan]quit[/cyan] - Exit\n",
            title="Colour Thinking Active"
        ))
        
        current_intensity = 1.0
        
        while True:
            try:
                # User input
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.console.print("\n [green]Goodbye! Your memories are preserved.[/green]")
                    break
                
                elif user_input.lower() == 'spectrum':
                    self.show_colour_spectrum()
                    continue
                
                elif user_input.lower() == 'dream':
                    self.dream_consolidation()
                    continue
                
                elif user_input.lower() == 'learn':
                    self.developmental_learning_session()
                    continue
                
                elif user_input.lower() == 'stats':
                    self.show_storage_stats()
                    continue
                
                elif user_input.lower().startswith('intensity '):
                    try:
                        intensity = float(user_input.split()[1])
                        if 0.1 <= intensity <= 3.0:
                            current_intensity = intensity
                            self.console.print(f" [yellow]Thinking intensity set to {intensity}[/yellow]")
                        else:
                            self.console.print(" [red]Intensity must be between 0.1 and 3.0[/red]")
                    except (IndexError, ValueError):
                        self.console.print(" [red]Usage: intensity <float>[/red]")
                    continue
                
                # Process through pure colour thinking
                with self.console.status("[yellow] Thinking in colour space...[/yellow]"):
                    response = self.process_user_input(user_input, current_intensity)
                
                # Display response with metrics
                try:
                    cognitive_metrics = self.chromathink.thinker.get_cognitive_metrics()
                    complexity = cognitive_metrics.get('cognitive_complexity', 0.5)
                    thinking_time = 0.5  # Fixed for now
                except:
                    complexity = 0.5
                    thinking_time = 0.5
                
                self.console.print(f"\n[bold magenta]ChromaThink[/bold magenta]: {response}")
                
                # Show metrics
                metrics_text = (
                    f" {thinking_time:.3f}s | "
                    f" {complexity:.3f} | "
                    f" {current_intensity} | "
                    f" {self.exchange_count} exchanges"
                )
                self.console.print(f"[dim]{metrics_text}[/dim]")
                
                # Show spectrum if enabled
                if self.show_spectrum:
                    self.show_colour_spectrum()
                
                # Periodic storage stats
                if self.exchange_count % 10 == 0:
                    stats = self.storage.get_storage_stats()
                    self.console.print(f"[dim] {stats['total_memories']:,} total memories stored[/dim]")
                
            except KeyboardInterrupt:
                self.console.print("\n\n [green]Goodbye! Your memories are preserved.[/green]")
                break
            
            except Exception as e:
                self.console.print(f"[red] Error: {e}[/red]")
                self.logger.exception("Chat error")


def main():
    """Main entry point for chatbot."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check for Apertus model
    apertus_path = Path("models/apertus")
    use_real_apertus = apertus_path.exists()
    
    console = Console()
    
    if use_real_apertus:
        console.print(" [green]Found Apertus model - using real 8B parameter translation[/green]")
    else:
        console.print(" [yellow]Using mock Apertus translator for demonstration[/yellow]")
    
    try:
        # Initialize chatbot
        chatbot = TrueColourChatbot(
            apertus_dir="models/apertus",
            storage_dir="learning_storage",
            spectrum_dims=1024,
            use_real_apertus=use_real_apertus
        )
        
        # Start interactive chat
        chatbot.interactive_chat()
        
    except Exception as e:
        console.print(f"[red] Failed to initialize chatbot: {e}[/red]")
        logging.exception("Chatbot initialization failed")


if __name__ == "__main__":
    main()
