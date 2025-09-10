"""Visualization utilities for colour spaces and frequency spectra"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from typing import Optional, Tuple, List
import warnings


class ColourSpaceVisualizer:
    """
    Visualizer for colour-wave representations and frequency spectra
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        
    def plot_colour_field(self, colour_field: tf.Tensor, 
                         sample_index: int = 0,
                         title: str = "Colour Field",
                         save_path: Optional[str] = None):
        """
        Plot a colour field as both magnitude/phase and RGB approximation
        
        Args:
            colour_field: Complex tensor of shape [..., dimensions]
            sample_index: Which sample to plot if batched
            title: Plot title
            save_path: Optional path to save plot
        """
        if len(colour_field.shape) > 1:
            field = colour_field[sample_index]
        else:
            field = colour_field
            
        magnitude = tf.math.abs(field).numpy()
        phase = tf.math.angle(field).numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=14)
        
        # Magnitude plot
        axes[0, 0].plot(magnitude)
        axes[0, 0].set_title('Magnitude')
        axes[0, 0].set_xlabel('Frequency Bin')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Phase plot
        axes[0, 1].plot(phase)
        axes[0, 1].set_title('Phase')
        axes[0, 1].set_xlabel('Frequency Bin')
        axes[0, 1].set_ylabel('Phase (radians)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Power spectrum
        power = magnitude ** 2
        axes[1, 0].semilogy(power)
        axes[1, 0].set_title('Power Spectrum (Log Scale)')
        axes[1, 0].set_xlabel('Frequency Bin')
        axes[1, 0].set_ylabel('Power')
        axes[1, 0].grid(True, alpha=0.3)
        
        # RGB approximation
        rgb_approx = self._complex_to_rgb(field)
        axes[1, 1].imshow(rgb_approx.reshape(1, -1, 3), aspect='auto')
        axes[1, 1].set_title('RGB Approximation')
        axes[1, 1].set_xlabel('Frequency Bin')
        axes[1, 1].set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_interference_pattern(self, wave1: tf.Tensor, wave2: tf.Tensor, 
                                 result: tf.Tensor,
                                 sample_index: int = 0,
                                 title: str = "Wave Interference",
                                 save_path: Optional[str] = None):
        """
        Plot interference between two waves and the result
        
        Args:
            wave1, wave2: Input waves
            result: Interference result
            sample_index: Which sample to plot
            title: Plot title
            save_path: Optional path to save plot
        """
        if len(wave1.shape) > 1:
            w1 = wave1[sample_index].numpy()
            w2 = wave2[sample_index].numpy()
            res = result[sample_index].numpy()
        else:
            w1 = wave1.numpy()
            w2 = wave2.numpy()
            res = result.numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=14)
        
        x = np.arange(len(w1))
        
        # Magnitude comparison
        axes[0, 0].plot(x, np.abs(w1), label='Wave 1', alpha=0.7)
        axes[0, 0].plot(x, np.abs(w2), label='Wave 2', alpha=0.7)
        axes[0, 0].plot(x, np.abs(res), label='Result', linewidth=2)
        axes[0, 0].set_title('Magnitude')
        axes[0, 0].set_xlabel('Frequency Bin')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Phase comparison
        axes[0, 1].plot(x, np.angle(w1), label='Wave 1', alpha=0.7)
        axes[0, 1].plot(x, np.angle(w2), label='Wave 2', alpha=0.7)
        axes[0, 1].plot(x, np.angle(res), label='Result', linewidth=2)
        axes[0, 1].set_title('Phase')
        axes[0, 1].set_xlabel('Frequency Bin')
        axes[0, 1].set_ylabel('Phase (radians)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Power comparison
        axes[1, 0].semilogy(x, np.abs(w1)**2, label='Wave 1', alpha=0.7)
        axes[1, 0].semilogy(x, np.abs(w2)**2, label='Wave 2', alpha=0.7)
        axes[1, 0].semilogy(x, np.abs(res)**2, label='Result', linewidth=2)
        axes[1, 0].set_title('Power Spectrum')
        axes[1, 0].set_xlabel('Frequency Bin')
        axes[1, 0].set_ylabel('Power')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # RGB visualization of all three
        rgb1 = self._complex_to_rgb(tf.constant(w1))
        rgb2 = self._complex_to_rgb(tf.constant(w2))
        rgb_res = self._complex_to_rgb(tf.constant(res))
        
        rgb_stack = np.stack([rgb1, rgb2, rgb_res], axis=0)
        axes[1, 1].imshow(rgb_stack, aspect='auto')
        axes[1, 1].set_title('RGB Visualization')
        axes[1, 1].set_xlabel('Frequency Bin')
        axes[1, 1].set_ylabel('Wave 1 | Wave 2 | Result')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_resonance_evolution(self, evolution_history: List[tf.Tensor],
                                title: str = "Resonance Evolution",
                                save_path: Optional[str] = None):
        """
        Plot evolution of resonance over time
        
        Args:
            evolution_history: List of tensors showing evolution
            title: Plot title
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=14)
        
        # Extract data from first sample of each timestep
        magnitudes = [tf.math.abs(step[0]).numpy() for step in evolution_history]
        phases = [tf.math.angle(step[0]).numpy() for step in evolution_history]
        
        # Magnitude evolution heatmap
        mag_matrix = np.stack(magnitudes, axis=0)
        im1 = axes[0, 0].imshow(mag_matrix, aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Magnitude Evolution')
        axes[0, 0].set_xlabel('Frequency Bin')
        axes[0, 0].set_ylabel('Time Step')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Phase evolution heatmap
        phase_matrix = np.stack(phases, axis=0)
        im2 = axes[0, 1].imshow(phase_matrix, aspect='auto', cmap='hsv')
        axes[0, 1].set_title('Phase Evolution')
        axes[0, 1].set_xlabel('Frequency Bin')
        axes[0, 1].set_ylabel('Time Step')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Total power over time
        powers = [tf.reduce_sum(tf.math.abs(step[0])**2).numpy() for step in evolution_history]
        axes[1, 0].plot(powers, marker='o')
        axes[1, 0].set_title('Total Power Over Time')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Total Power')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Spectral centroid over time
        centroids = []
        for step in evolution_history:
            power = tf.math.abs(step[0])**2
            freqs = tf.range(len(power), dtype=tf.float32)
            centroid = tf.reduce_sum(power * freqs) / (tf.reduce_sum(power) + 1e-8)
            centroids.append(centroid.numpy())
        
        axes[1, 1].plot(centroids, marker='o', color='orange')
        axes[1, 1].set_title('Spectral Centroid Over Time')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Centroid (Frequency Bin)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_standing_wave_analysis(self, analysis_dict: dict,
                                   title: str = "Standing Wave Analysis",
                                   save_path: Optional[str] = None):
        """
        Plot standing wave analysis results
        
        Args:
            analysis_dict: Dictionary from get_standing_wave_analysis()
            title: Plot title
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=14)
        
        # Mode amplitudes
        mode_amps = analysis_dict['mode_amplitudes'][0].numpy()  # First sample
        axes[0, 0].bar(range(len(mode_amps)), mode_amps)
        axes[0, 0].set_title('Modal Amplitudes')
        axes[0, 0].set_xlabel('Mode Number')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Resonant frequencies
        if 'resonant_frequencies' in analysis_dict:
            freqs = analysis_dict['resonant_frequencies'].numpy()
            axes[0, 1].plot(freqs, mode_amps, 'o-')
            axes[0, 1].set_title('Mode Amplitudes vs Frequency')
            axes[0, 1].set_xlabel('Resonant Frequency')
            axes[0, 1].set_ylabel('Amplitude')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Standing wave ratio
        if 'standing_wave_ratio' in analysis_dict:
            swr = analysis_dict['standing_wave_ratio'].numpy()
            axes[1, 0].hist(swr, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Standing Wave Ratio Distribution')
            axes[1, 0].set_xlabel('SWR')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Chamber memory if available
        if 'chamber_memory' in analysis_dict:
            memory = analysis_dict['chamber_memory'].numpy()
            axes[1, 1].plot(memory, marker='s', markersize=4)
            axes[1, 1].set_title('Chamber Memory State')
            axes[1, 1].set_xlabel('Mode Number')
            axes[1, 1].set_ylabel('Memory Value')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _complex_to_rgb(self, complex_tensor: tf.Tensor) -> np.ndarray:
        """
        Convert complex tensor to RGB approximation
        
        Args:
            complex_tensor: Complex tensor
            
        Returns:
            RGB array of shape [..., 3]
        """
        magnitude = tf.math.abs(complex_tensor)
        phase = tf.math.angle(complex_tensor)
        
        # Normalize magnitude to [0, 1]
        mag_normalized = magnitude / (tf.reduce_max(magnitude) + 1e-8)
        
        # Convert phase to hue [0, 1]
        hue = (phase + np.pi) / (2 * np.pi)
        
        # Use magnitude as saturation and value
        saturation = tf.ones_like(hue)
        value = mag_normalized
        
        # Stack HSV
        hsv = tf.stack([hue, saturation, value], axis=-1)
        
        # Convert to RGB
        rgb = hsv_to_rgb(hsv.numpy())
        
        return rgb


def plot_frequency_spectrum(waveform: tf.Tensor, 
                           frequencies: Optional[tf.Tensor] = None,
                           sample_index: int = 0,
                           log_scale: bool = True,
                           title: str = "Frequency Spectrum",
                           save_path: Optional[str] = None):
    """
    Plot frequency spectrum of a waveform
    
    Args:
        waveform: Complex waveform tensor
        frequencies: Optional frequency values for x-axis
        sample_index: Which sample to plot if batched
        log_scale: Use log scale for y-axis
        title: Plot title
        save_path: Optional path to save plot
    """
    if len(waveform.shape) > 1:
        wave = waveform[sample_index]
    else:
        wave = waveform
    
    magnitude = tf.math.abs(wave).numpy()
    power = magnitude ** 2
    
    if frequencies is not None:
        x_values = frequencies.numpy()
        xlabel = 'Frequency'
    else:
        x_values = np.arange(len(wave))
        xlabel = 'Frequency Bin'
    
    plt.figure(figsize=(12, 6))
    
    # Magnitude plot
    plt.subplot(1, 2, 1)
    if log_scale:
        plt.semilogy(x_values, magnitude + 1e-12)  # Add small value to avoid log(0)
        plt.ylabel('Magnitude (log scale)')
    else:
        plt.plot(x_values, magnitude)
        plt.ylabel('Magnitude')
    
    plt.xlabel(xlabel)
    plt.title(f'{title} - Magnitude')
    plt.grid(True, alpha=0.3)
    
    # Power plot
    plt.subplot(1, 2, 2)
    if log_scale:
        plt.semilogy(x_values, power + 1e-12)
        plt.ylabel('Power (log scale)')
    else:
        plt.plot(x_values, power)
        plt.ylabel('Power')
    
    plt.xlabel(xlabel)
    plt.title(f'{title} - Power')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_interference_patterns(wave1: tf.Tensor, wave2: tf.Tensor,
                              interference_result: tf.Tensor,
                              sample_index: int = 0,
                              title: str = "Interference Patterns",
                              save_path: Optional[str] = None):
    """
    Plot interference patterns between waves
    
    Args:
        wave1, wave2: Input waves
        interference_result: Result of interference
        sample_index: Which sample to plot
        title: Plot title
        save_path: Optional path to save plot
    """
    visualizer = ColourSpaceVisualizer()
    visualizer.plot_interference_pattern(
        wave1, wave2, interference_result,
        sample_index=sample_index,
        title=title,
        save_path=save_path
    )