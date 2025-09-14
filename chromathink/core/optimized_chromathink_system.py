"""
Optimized ChromaThink System with proper GPU utilization and CPU parallelization.
Includes full pipeline visibility and debugging.
"""

import numpy as np
import tensorflow as tf
import torch
from typing import List, Dict, Optional, Tuple
import logging
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os

# Configure for maximum CPU utilization
os.environ['TF_NUM_INTEROP_THREADS'] = str(mp.cpu_count())
os.environ['TF_NUM_INTRAOP_THREADS'] = str(mp.cpu_count())
os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())

# Configure TensorFlow for aggressive GPU utilization
tf.config.threading.set_inter_op_parallelism_threads(mp.cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(mp.cpu_count())


class OptimizedGPUColourProcessor:
    """
    GPU-optimized colour processing with aggressive resource utilization.
    """

    def __init__(self, spectrum_dims: int = 512):
        self.spectrum_dims = spectrum_dims
        self.logger = logging.getLogger("OptimizedGPUProcessor")

        # Force GPU memory growth and utilization
        self._configure_gpu_aggressive()

        # Pre-compile GPU operations
        self._precompile_gpu_ops()

    def _configure_gpu_aggressive(self):
        """Configure GPU for maximum utilization"""

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Allow memory growth but set high limit
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    # Set large memory limit to force GPU usage
                    tf.config.experimental.set_memory_growth(gpu, False)  # Disable growth
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12288)]  # 12GB
                    )

                self.logger.info(f"Configured {len(gpus)} GPUs for aggressive utilization")

            except Exception as e:
                self.logger.warning(f"GPU configuration failed: {e}")

    def _precompile_gpu_ops(self):
        """Pre-compile GPU operations to avoid JIT overhead"""

        with tf.device('/GPU:0'):
            try:
                # Create dummy tensors to trigger compilation
                dummy_complex = tf.complex(
                    tf.random.normal([1000, self.spectrum_dims]),
                    tf.random.normal([1000, self.spectrum_dims])
                )

                # Pre-compile common operations
                _ = tf.abs(dummy_complex)
                _ = tf.angle(dummy_complex)
                _ = tf.reduce_sum(dummy_complex * tf.conj(dummy_complex))
                _ = tf.matmul(dummy_complex, tf.transpose(tf.conj(dummy_complex)))

                self.logger.info("GPU operations pre-compiled successfully")

            except Exception as e:
                self.logger.warning(f"GPU pre-compilation failed: {e}")

    @tf.function(experimental_relax_shapes=True)
    def batch_resonance_processing(self, colour_batch: tf.Tensor) -> tf.Tensor:
        """
        Batch process multiple colour patterns simultaneously on GPU.
        """

        batch_size = tf.shape(colour_batch)[0]

        # Parallel resonance computation
        with tf.device('/GPU:0'):
            # Phase modulation
            phase_shifts = tf.linspace(0.0, 2*np.pi, batch_size)
            phase_matrix = tf.exp(tf.complex(0.0, tf.expand_dims(phase_shifts, 1)))

            # Apply phase modulation
            modulated = colour_batch * phase_matrix

            # Interference patterns
            interference_matrix = tf.matmul(modulated, tf.transpose(tf.conj(modulated)))

            # Extract dominant modes
            eigenvalues, eigenvectors = tf.linalg.eigh(tf.real(interference_matrix))

            # Transform back to colour space
            dominant_indices = tf.argsort(eigenvalues, direction='DESCENDING')[:batch_size//2]
            dominant_modes = tf.gather(eigenvectors, dominant_indices, axis=1)

            # Project colours onto dominant modes
            projected = tf.matmul(tf.cast(dominant_modes, tf.complex64), modulated)

            # Apply nonlinear transformation
            amplitude = tf.abs(projected)
            phase = tf.angle(projected)

            # Nonlinear amplitude shaping
            shaped_amplitude = tf.tanh(amplitude * 2.0) * tf.sqrt(amplitude)

            # Reconstruct complex signal
            result = shaped_amplitude * tf.exp(tf.complex(0.0, phase))

        return result

    def process_colour_thinking(self, input_colours: tf.Tensor,
                              batch_size: int = 32) -> tf.Tensor:
        """
        Process colour thinking with aggressive GPU utilization.
        """

        self.logger.info("Starting GPU-accelerated colour processing")
        start_time = time.time()

        with tf.device('/GPU:0'):
            # Expand to batch for parallel processing
            if len(input_colours.shape) == 1:
                # Create multiple variations for parallel processing
                variations = []
                for i in range(batch_size):
                    variation = input_colours * (0.8 + 0.4 * i / batch_size)
                    phase_shift = 2 * np.pi * i / batch_size
                    variation = variation * tf.exp(tf.complex(0.0, phase_shift))
                    variations.append(variation)

                colour_batch = tf.stack(variations)
            else:
                colour_batch = input_colours

            # Process in parallel on GPU
            processed_batch = self.batch_resonance_processing(colour_batch)

            # Reduce to single output through interference
            result = tf.reduce_mean(processed_batch, axis=0)

            # Normalize
            norm = tf.abs(tf.reduce_sum(result * tf.conj(result)))
            result = result / tf.sqrt(norm + 1e-8)

        processing_time = time.time() - start_time
        self.logger.info(f"GPU processing completed in {processing_time:.3f}s")

        return result


class DebugApertusTranslator:
    """
    Apertus translator with full debugging and pipeline visibility.
    """

    def __init__(self, model_path: str = "models/apertus"):
        self.logger = logging.getLogger("DebugApertusTranslator")
        self.model_path = model_path

        # Load model with CPU optimization
        self._load_optimized_model()

    def _load_optimized_model(self):
        """Load Apertus with CPU optimization"""

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.logger.info("Loading Apertus with CPU optimization...")

            # Configure for CPU parallelization
            torch.set_num_threads(mp.cpu_count())

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load on CPU with maximum threads
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                device_map="cpu",
                torch_dtype=torch.float32,
                use_safetensors=True,
                trust_remote_code=True,
                low_cpu_mem_usage=False  # Use more RAM for speed
            )

            # Set to eval mode with threading
            self.model.eval()

            # Warm up model
            self._warmup_model()

            self.logger.info(f"Apertus loaded successfully on CPU with {mp.cpu_count()} threads")

        except Exception as e:
            self.logger.error(f"Failed to load Apertus: {e}")
            raise

    def _warmup_model(self):
        """Warm up model to reduce first-query latency"""

        warmup_text = "Test warmup query for model initialization"
        try:
            self.extract_concepts_debug(warmup_text)
            self.logger.info("Model warmed up successfully")
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")

    def extract_concepts_debug(self, text: str, show_debug: bool = True) -> Tuple[List[str], Dict]:
        """
        Extract concepts with full debugging information.
        """

        debug_info = {
            'input_text': text,
            'input_length': len(text),
            'extraction_prompt': None,
            'raw_response': None,
            'parsed_concepts': None,
            'processing_time': 0
        }

        start_time = time.time()

        # Stage 1: Create extraction prompt
        extraction_prompt = f"""You are a concept extraction system. Your task is to identify the minimum essential concepts needed to understand the given text.

Instructions:
1. Extract 3-7 core concepts that capture the semantic essence of the input
2. Output ONLY single words or very short phrases (max 2 words)
3. Focus on concrete nouns, actions, and essential attributes
4. Ignore grammatical words (articles, conjunctions, prepositions)
5. Preserve the most semantically important elements
6. If the input is in a non-English language, extract concepts in that language

Format your response as a simple list with one concept per line.

Text to analyze: "{text}"

Core concepts:"""

        debug_info['extraction_prompt'] = extraction_prompt

        if show_debug:
            print(f"\n🔍 CONCEPT EXTRACTION DEBUG")
            print(f"   Input: {text}")
            print(f"   Prompt length: {len(extraction_prompt)} chars")

        # Stage 2: Query model with optimized settings
        try:
            messages = [{"role": "user", "content": extraction_prompt}]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True
            )

            attention_mask = torch.ones_like(inputs)

            # Generate with optimized settings for CPU
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=1,  # Disable beam search for speed
                    use_cache=True
                )

            raw_response = self.tokenizer.decode(
                outputs[0][len(inputs[0]):],
                skip_special_tokens=True
            ).strip()

            debug_info['raw_response'] = raw_response

            # Stage 3: Parse concepts
            concepts = self._parse_concepts(raw_response)
            debug_info['parsed_concepts'] = concepts

            processing_time = time.time() - start_time
            debug_info['processing_time'] = processing_time

            if show_debug:
                print(f"   Raw response: {raw_response}")
                print(f"   Parsed concepts: {concepts}")
                print(f"   Processing time: {processing_time:.3f}s")

            return concepts, debug_info

        except Exception as e:
            self.logger.error(f"Concept extraction failed: {e}")
            fallback_concepts = self._fallback_extract(text)
            debug_info['parsed_concepts'] = fallback_concepts
            debug_info['processing_time'] = time.time() - start_time
            return fallback_concepts, debug_info

    def synthesize_response_debug(self, concepts: List[str], original_question: str,
                                show_debug: bool = True) -> Tuple[str, Dict]:
        """
        Synthesize response with full debugging.
        """

        debug_info = {
            'input_concepts': concepts,
            'original_question': original_question,
            'synthesis_prompt': None,
            'raw_response': None,
            'final_response': None,
            'processing_time': 0
        }

        start_time = time.time()

        # Stage 1: Create synthesis prompt
        concept_words = ", ".join(concepts)
        synthesis_prompt = f"""You are a linguistic synthesizer. Your task is to create a natural, coherent response by weaving together concept words into grammatically correct sentences.

Instructions:
1. You will receive CONCEPT WORDS that represent the core semantic elements of a response
2. You will receive the ORIGINAL QUESTION that prompted these concepts
3. Create a natural sentence/paragraph that incorporates these concepts as the answer
4. Maintain logical flow and grammatical correctness
5. Ensure the response directly addresses the original question
6. Use appropriate connecting words and grammar to make the concepts flow naturally

Original question: "{original_question}"
Concept words: {concept_words}

Natural response:"""

        debug_info['synthesis_prompt'] = synthesis_prompt

        if show_debug:
            print(f"\n📝 RESPONSE SYNTHESIS DEBUG")
            print(f"   Concepts: {concepts}")
            print(f"   Original question: {original_question}")

        # Stage 2: Generate response
        try:
            messages = [{"role": "user", "content": synthesis_prompt}]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True
            )

            attention_mask = torch.ones_like(inputs)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=1,
                    use_cache=True
                )

            raw_response = self.tokenizer.decode(
                outputs[0][len(inputs[0]):],
                skip_special_tokens=True
            ).strip()

            # Clean up response
            final_response = self._clean_response(raw_response)

            debug_info['raw_response'] = raw_response
            debug_info['final_response'] = final_response

            processing_time = time.time() - start_time
            debug_info['processing_time'] = processing_time

            if show_debug:
                print(f"   Raw response: {raw_response[:100]}...")
                print(f"   Final response: {final_response}")
                print(f"   Processing time: {processing_time:.3f}s")

            return final_response, debug_info

        except Exception as e:
            self.logger.error(f"Response synthesis failed: {e}")
            fallback_response = f"Based on the concepts {concept_words}, this relates to fundamental principles and interactions."
            debug_info['final_response'] = fallback_response
            debug_info['processing_time'] = time.time() - start_time
            return fallback_response, debug_info

    def _parse_concepts(self, raw_response: str) -> List[str]:
        """Parse raw response into clean concept list"""

        concepts = []
        lines = raw_response.strip().split('\n')

        for line in lines:
            line = line.strip()
            # Remove numbering and bullets
            line = line.lstrip('0123456789.-•* \t')

            if line and len(line) > 1 and len(line) < 50:
                # Clean concept
                concept = line.strip(' .,;:!?()[]{}"\'-')
                if len(concept.split()) <= 2:  # Max 2 words
                    concepts.append(concept)

        return concepts[:7]  # Max 7 concepts

    def _clean_response(self, raw_response: str) -> str:
        """Clean up synthesized response"""

        # Remove common artifacts
        response = raw_response.strip()

        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'

        return response

    def _fallback_extract(self, text: str) -> List[str]:
        """Fallback concept extraction when model fails"""

        words = text.lower().replace('?', '').replace('!', '').split()
        # Filter for meaningful words
        concepts = [w for w in words if len(w) > 3 and w not in [
            'what', 'how', 'why', 'when', 'where', 'does', 'can', 'will',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'
        ]]
        return concepts[:5]


class OptimizedChromatThinkSystem:
    """
    Complete optimized ChromaThink system with visibility and performance.
    """

    def __init__(self, apertus_path: str = "models/apertus"):
        self.logger = logging.getLogger("OptimizedChromatThink")

        print("🚀 Initializing Optimized ChromaThink System")
        print("=" * 60)

        # Initialize components
        self.gpu_processor = OptimizedGPUColourProcessor()
        self.apertus = DebugApertusTranslator(apertus_path)

        print("✅ All components loaded - system ready for high-performance processing")

    def process_with_full_debug(self, user_input: str) -> str:
        """
        Process input with complete pipeline visibility.
        """

        print(f"\n" + "="*80)
        print(f"🧠 PROCESSING: {user_input}")
        print("="*80)

        total_start = time.time()

        # Stage 1: Concept Extraction
        print(f"\n📝 STAGE 1: CONCEPT EXTRACTION")
        print("-" * 40)
        concepts, extract_debug = self.apertus.extract_concepts_debug(user_input)

        # Stage 2: Concept to Colour Conversion
        print(f"\n🌈 STAGE 2: CONCEPT → COLOUR CONVERSION")
        print("-" * 40)
        colour_start = time.time()

        # Convert concepts to colours (simplified for debugging)
        input_colours = self._concepts_to_colours_debug(concepts)
        colour_time = time.time() - colour_start

        print(f"   Concepts: {concepts}")
        print(f"   Colour tensor shape: {input_colours.shape}")
        print(f"   Colour energy: {tf.reduce_sum(tf.abs(input_colours)**2):.4f}")
        print(f"   Conversion time: {colour_time:.3f}s")

        # Stage 3: GPU Colour Processing
        print(f"\n🧠 STAGE 3: GPU COLOUR THINKING")
        print("-" * 40)
        gpu_start = time.time()

        response_colours = self.gpu_processor.process_colour_thinking(input_colours)
        gpu_time = time.time() - gpu_start

        print(f"   Input energy: {tf.reduce_sum(tf.abs(input_colours)**2):.4f}")
        print(f"   Output energy: {tf.reduce_sum(tf.abs(response_colours)**2):.4f}")
        print(f"   Energy ratio: {tf.reduce_sum(tf.abs(response_colours)**2) / tf.reduce_sum(tf.abs(input_colours)**2):.3f}")
        print(f"   GPU processing time: {gpu_time:.3f}s")

        # Stage 4: Colour to Concepts
        print(f"\n💭 STAGE 4: COLOUR → CONCEPT CONVERSION")
        print("-" * 40)
        concept_start = time.time()

        response_concepts = self._colours_to_concepts_debug(response_colours)
        concept_time = time.time() - concept_start

        print(f"   Response concepts: {response_concepts}")
        print(f"   Conversion time: {concept_time:.3f}s")

        # Stage 5: Response Synthesis
        print(f"\n📜 STAGE 5: RESPONSE SYNTHESIS")
        print("-" * 40)

        final_response, synth_debug = self.apertus.synthesize_response_debug(
            response_concepts, user_input
        )

        total_time = time.time() - total_start

        # Summary
        print(f"\n📊 PROCESSING SUMMARY")
        print("-" * 40)
        print(f"   Concept extraction: {extract_debug['processing_time']:.3f}s")
        print(f"   Colour conversion: {colour_time:.3f}s")
        print(f"   GPU thinking: {gpu_time:.3f}s")
        print(f"   Response synthesis: {synth_debug['processing_time']:.3f}s")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   GPU utilization: {gpu_time/total_time*100:.1f}% of total time")

        return final_response

    def _concepts_to_colours_debug(self, concepts: List[str]) -> tf.Tensor:
        """Convert concepts to colours with debugging"""

        spectrum_dims = 512
        combined = tf.zeros(spectrum_dims, dtype=tf.complex64)

        for i, concept in enumerate(concepts[:5]):
            # Deterministic hash-based colour generation
            concept_hash = hash(concept.lower()) % (2**16)
            base_freq = (concept_hash % 100) / 100.0

            freq_idx = int(base_freq * spectrum_dims)
            amplitude = 1.0 / (i + 1)
            phase = (concept_hash % 628) / 100.0

            # Add to tensor with proper GPU placement
            with tf.device('/GPU:0'):
                combined = combined.numpy()
                combined[freq_idx] += amplitude * np.exp(1j * phase)
                combined = tf.constant(combined, dtype=tf.complex64)

        # Normalize
        norm = tf.sqrt(tf.reduce_sum(tf.abs(combined)**2))
        if norm > 0:
            combined = combined / norm

        return combined

    def _colours_to_concepts_debug(self, colours: tf.Tensor) -> List[str]:
        """Convert colours back to concepts with debugging"""

        amplitudes = tf.abs(colours)
        dominant_indices = tf.argsort(amplitudes, direction='DESCENDING')[:5]

        concepts = []
        for idx in dominant_indices:
            freq = idx / colours.shape[0]
            amplitude = amplitudes[idx]

            if amplitude > 0.05:
                if freq < 0.25:
                    concepts.append("fundamental")
                elif freq < 0.5:
                    concepts.append("relational")
                elif freq < 0.75:
                    concepts.append("dynamic")
                else:
                    concepts.append("complex")

        return concepts[:4]


def create_optimized_system(apertus_path: str = "models/apertus") -> OptimizedChromatThinkSystem:
    """Create optimized ChromaThink system"""
    return OptimizedChromatThinkSystem(apertus_path)