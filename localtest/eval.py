#!/usr/bin/env python3
"""
Qwen3 Sudoku Benchmarking Script
===============================

This script benchmarks Qwen3-1.7B performance on Sudoku puzzles
before and after RL training. Provides comprehensive analysis.
"""

import json
import torch
import numpy as np
import pandas as pd
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# Core imports
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )
    print("✅ Transformers imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Install: pip install transformers torch accelerate bitsandbytes")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    
    # Model settings
    model_name: str = "Qwen/Qwen3-1.7B"
    use_8bit: bool = True
    enable_thinking: bool = False
    
    # Generation settings
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_new_tokens: int = 600
    
    # Benchmarking settings
    timeout_seconds: int = 30
    max_retries: int = 2

class Qwen3SudokuBenchmarker:
    """Comprehensive benchmarking for Qwen3 on Sudoku tasks"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🔧 Initializing Qwen3 Benchmarker")
        logger.info(f"   Model: {config.model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   8-bit: {config.use_8bit}")
        logger.info(f"   Thinking mode: {config.enable_thinking}")
        
        # Load model and tokenizer
        self.load_model()
    
    def load_model(self, model_path: Optional[str] = None):
        """Load Qwen3 model and tokenizer"""
        
        model_to_load = model_path if model_path else self.config.model_name
        logger.info(f"📥 Loading model: {model_to_load}")
        
        # Setup quantization
        if self.config.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
        else:
            quantization_config = None
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name if not model_path else model_path
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_to_load,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if not self.config.use_8bit else None,
                device_map="auto",
                trust_remote_code=True
            )
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def format_sudoku_grid(self, quiz_string: str) -> str:
        """Format Sudoku for visual display"""
        if len(quiz_string) != 81:
            raise ValueError(f"Invalid puzzle length: {len(quiz_string)}")
        
        grid_lines = []
        for row in range(9):
            line_chars = []
            for col in range(9):
                char = quiz_string[row * 9 + col]
                line_chars.append('_' if char == '0' else char)
            
            line = f"{' '.join(line_chars[0:3])} │ {' '.join(line_chars[3:6])} │ {' '.join(line_chars[6:9])}"
            grid_lines.append(line)
            
            if row == 2 or row == 5:
                grid_lines.append("──────┼───────┼──────")
        
        return '\n'.join(grid_lines)
    
    def find_empty_positions(self, quiz_string: str) -> List[str]:
        """Find empty cell positions"""
        empty_positions = []
        for i in range(81):
            if quiz_string[i] == '0':
                row, col = (i // 9) + 1, (i % 9) + 1
                empty_positions.append(f"R{row}C{col}")
        return empty_positions
    
    def create_benchmark_messages(self, puzzle: str, clue_count: int) -> List[Dict]:
        """Create messages for benchmarking"""
        
        formatted_grid = self.format_sudoku_grid(puzzle)
        empty_positions = self.find_empty_positions(puzzle)
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert Sudoku solver. Use logical reasoning to solve puzzles step by step. Always provide your final answer in the requested format."
            },
            {
                "role": "user",
                "content": f"""Solve this Sudoku puzzle with {clue_count} given clues.

PUZZLE:
{formatted_grid}

EMPTY CELLS TO FILL: {', '.join(empty_positions)}

Rules: Each row, column, and 3×3 box must contain digits 1-9 exactly once.

IMPORTANT: Provide your solution in this EXACT format:
SOLUTION:
R1C1: digit
R2C3: digit
...

Think through the constraints systematically and provide ALL missing digits."""
            }
        ]
        
        return messages
    
    def generate_solution(self, messages: List[Dict]) -> Tuple[str, float, Dict]:
        """Generate solution with timing and metadata"""
        
        start_time = time.time()
        
        try:
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.enable_thinking
            )
            
            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            # Generate with timeout protection
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Extract response
            input_length = model_inputs.input_ids.shape[1]
            output_ids = generated_ids[0][input_length:]
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            
            # Extract metadata
            metadata = {
                'input_tokens': model_inputs.input_ids.shape[1],
                'output_tokens': len(output_ids),
                'total_tokens': model_inputs.input_ids.shape[1] + len(output_ids),
                'generation_time': generation_time,
                'tokens_per_second': len(output_ids) / generation_time if generation_time > 0 else 0,
                'has_thinking': '<think>' in response and '</think>' in response
            }
            
            return response, generation_time, metadata
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return "", time.time() - start_time, {'error': str(e)}
    
    def parse_solution_response(self, response: str) -> Dict[str, int]:
        """Parse model response to extract solutions"""
        
        solutions = {}
        
        # Handle thinking mode responses
        if '<think>' in response and '</think>' in response:
            think_end = response.rfind('</think>')
            final_response = response[think_end + 8:] if think_end != -1 else response
        else:
            final_response = response
        
        # Look for SOLUTION: section
        if 'SOLUTION:' in final_response.upper():
            solution_start = final_response.upper().find('SOLUTION:') + len('SOLUTION:')
            solution_text = final_response[solution_start:].strip()
        else:
            solution_text = final_response
        
        # Extract R1C1: digit patterns
        pattern = r'R(\d+)C(\d+):\s*(\d)'
        matches = re.findall(pattern, solution_text, re.IGNORECASE)
        
        for match in matches:
            row, col, digit = int(match[0]), int(match[1]), int(match[2])
            if 1 <= row <= 9 and 1 <= col <= 9 and 1 <= digit <= 9:
                cell_key = f"R{row}C{col}"
                solutions[cell_key] = digit
        
        return solutions
    
    def evaluate_solution(self, puzzle: str, solution: str, response: str) -> Dict:
        """Comprehensive evaluation of model solution"""
        
        # Get expected solutions
        expected_solutions = {}
        for i in range(81):
            if puzzle[i] == '0':
                row, col = (i // 9) + 1, (i % 9) + 1
                cell_key = f"R{row}C{col}"
                expected_solutions[cell_key] = int(solution[i])
        
        # Parse model predictions
        predicted_solutions = self.parse_solution_response(response)
        
        # Calculate metrics
        total_empty_cells = len(expected_solutions)
        attempted_cells = len(predicted_solutions)
        correct_cells = 0
        incorrect_cells = 0
        
        # Check correctness
        for cell_key, expected_digit in expected_solutions.items():
            if cell_key in predicted_solutions:
                if predicted_solutions[cell_key] == expected_digit:
                    correct_cells += 1
                else:
                    incorrect_cells += 1
        
        # Calculate derived metrics
        accuracy = correct_cells / total_empty_cells if total_empty_cells > 0 else 0
        coverage = attempted_cells / total_empty_cells if total_empty_cells > 0 else 0
        precision = correct_cells / attempted_cells if attempted_cells > 0 else 0
        perfect_solution = (correct_cells == total_empty_cells and attempted_cells == total_empty_cells)
        
        # Analyze thinking quality if present
        thinking_quality = self.analyze_thinking_quality(response)
        
        return {
            'total_empty_cells': total_empty_cells,
            'attempted_cells': attempted_cells,
            'correct_cells': correct_cells,
            'incorrect_cells': incorrect_cells,
            'accuracy': accuracy,
            'coverage': coverage,
            'precision': precision,
            'perfect_solution': perfect_solution,
            'thinking_quality': thinking_quality,
            'expected_solutions': expected_solutions,
            'predicted_solutions': predicted_solutions
        }
    
    def analyze_thinking_quality(self, response: str) -> Dict:
        """Analyze quality of thinking content"""
        
        if '<think>' not in response or '</think>' not in response:
            return {'has_thinking': False, 'score': 0.0, 'reasoning_indicators': 0}
        
        # Extract thinking content
        think_start = response.find('<think>') + 7
        think_end = response.find('</think>')
        thinking_content = response[think_start:think_end].lower()
        
        # Count reasoning indicators
        reasoning_patterns = [
            'constraint', 'eliminate', 'possible', 'cannot', 'must be',
            'row', 'column', 'box', 'block', 'only option',
            'rule out', 'deduce', 'logic', 'because', 'therefore'
        ]
        
        reasoning_count = sum(1 for pattern in reasoning_patterns if pattern in thinking_content)
        thinking_length = len(thinking_content.split())
        
        return {
            'has_thinking': True,
            'score': min(reasoning_count / 8.0, 1.0),  # Normalize to 0-1
            'reasoning_indicators': reasoning_count,
            'thinking_length': thinking_length
        }
    
    def benchmark_single_puzzle(self, sample: Dict) -> Dict:
        """Benchmark a single puzzle"""
        
        logger.debug(f"🧩 Benchmarking puzzle: {sample['id']}")
        
        # Create messages
        messages = self.create_benchmark_messages(sample['puzzle'], sample['clue_count'])
        
        # Generate solution
        response, generation_time, metadata = self.generate_solution(messages)
        
        # Evaluate solution
        evaluation = self.evaluate_solution(sample['puzzle'], sample['solution'], response)
        
        # Compile results
        result = {
            'puzzle_id': sample['id'],
            'difficulty': sample['difficulty'],
            'clue_count': sample['clue_count'],
            'empty_count': sample['empty_count'],
            'response': response,
            'generation_time': generation_time,
            'metadata': metadata,
            'evaluation': evaluation,
            'success': evaluation['perfect_solution']
        }
        
        return result
    
    def benchmark_dataset(self, data_path: str, max_samples: Optional[int] = None) -> Dict:
        """Benchmark entire dataset"""
        
        logger.info(f"📊 Starting benchmark on dataset: {data_path}")
        
        # Load dataset
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            raise
        
        # Limit samples if specified
        if max_samples and max_samples < len(dataset):
            dataset = dataset[:max_samples]
            logger.info(f"🔢 Limited to first {max_samples} samples")
        
        logger.info(f"📋 Benchmarking {len(dataset)} puzzles...")
        
        # Benchmark each puzzle
        results = []
        successful_puzzles = 0
        
        for i, sample in enumerate(dataset):
            try:
                result = self.benchmark_single_puzzle(sample)
                results.append(result)
                
                if result['success']:
                    successful_puzzles += 1
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    current_success_rate = successful_puzzles / (i + 1)
                    logger.info(f"Progress: {i+1}/{len(dataset)} | Success rate: {current_success_rate:.2%}")
                
            except Exception as e:
                logger.error(f"❌ Error on puzzle {i}: {e}")
                continue
        
        # Calculate aggregate statistics
        aggregate_stats = self.calculate_aggregate_stats(results)
        
        logger.info(f"✅ Benchmark completed!")
        logger.info(f"   Total puzzles: {len(results)}")
        logger.info(f"   Perfect solutions: {successful_puzzles}")
        logger.info(f"   Success rate: {successful_puzzles/len(results):.2%}")
        
        return {
            'benchmark_config': {
                'model_name': self.config.model_name,
                'dataset_path': data_path,
                'total_samples': len(results),
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'results': results,
            'aggregate_stats': aggregate_stats
        }
    
    def calculate_aggregate_stats(self, results: List[Dict]) -> Dict:
        """Calculate comprehensive aggregate statistics"""
        
        if not results:
            return {}
        
        # Extract metrics
        perfect_solutions = sum(1 for r in results if r['evaluation']['perfect_solution'])
        total_puzzles = len(results)
        
        # Difficulty-based analysis
        difficulty_stats = {}
        for difficulty in ['expert', 'hard', 'medium', 'easy', 'beginner']:
            diff_results = [r for r in results if r.get('difficulty') == difficulty]
            if diff_results:
                difficulty_stats[difficulty] = {
                    'count': len(diff_results),
                    'perfect_rate': sum(1 for r in diff_results if r['evaluation']['perfect_solution']) / len(diff_results),
                    'avg_accuracy': np.mean([r['evaluation']['accuracy'] for r in diff_results]),
                    'avg_coverage': np.mean([r['evaluation']['coverage'] for r in diff_results]),
                    'avg_time': np.mean([r['generation_time'] for r in diff_results])
                }
        
        # Overall metrics
        accuracies = [r['evaluation']['accuracy'] for r in results]
        coverages = [r['evaluation']['coverage'] for r in results]
        times = [r['generation_time'] for r in results]
        thinking_scores = [r['evaluation']['thinking_quality']['score'] for r in results]
        
        return {
            'overall': {
                'total_puzzles': total_puzzles,
                'perfect_solutions': perfect_solutions,
                'success_rate': perfect_solutions / total_puzzles,
                'avg_accuracy': np.mean(accuracies),
                'avg_coverage': np.mean(coverages),
                'avg_generation_time': np.mean(times),
                'avg_thinking_score': np.mean(thinking_scores),
                'accuracy_std': np.std(accuracies),
                'time_std': np.std(times)
            },
            'by_difficulty': difficulty_stats,
            'distribution': {
                'accuracy_quartiles': np.percentile(accuracies, [25, 50, 75]).tolist(),
                'time_quartiles': np.percentile(times, [25, 50, 75]).tolist(),
                'perfect_by_clue_count': self.analyze_by_clue_count(results)
            }
        }
    
    def analyze_by_clue_count(self, results: List[Dict]) -> Dict:
        """Analyze performance by clue count"""
        
        clue_performance = {}
        
        for result in results:
            clue_count = result['clue_count']
            if clue_count not in clue_performance:
                clue_performance[clue_count] = {'total': 0, 'perfect': 0}
            
            clue_performance[clue_count]['total'] += 1
            if result['evaluation']['perfect_solution']:
                clue_performance[clue_count]['perfect'] += 1
        
        # Calculate success rates
        for clue_count in clue_performance:
            stats = clue_performance[clue_count]
            stats['success_rate'] = stats['perfect'] / stats['total']
        
        return clue_performance
    
    def save_benchmark_results(self, benchmark_data: Dict, output_path: str):
        """Save comprehensive benchmark results"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save complete results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        
        # Save summary CSV
        csv_path = output_path.with_suffix('.csv')
        summary_data = []
        
        for result in benchmark_data['results']:
            summary_data.append({
                'puzzle_id': result['puzzle_id'],
                'difficulty': result['difficulty'],
                'clue_count': result['clue_count'],
                'empty_count': result['empty_count'],
                'perfect_solution': result['evaluation']['perfect_solution'],
                'accuracy': result['evaluation']['accuracy'],
                'coverage': result['evaluation']['coverage'],
                'generation_time': result['generation_time'],
                'thinking_score': result['evaluation']['thinking_quality']['score'],
                'tokens_generated': result['metadata'].get('output_tokens', 0)
            })
        
        pd.DataFrame(summary_data).to_csv(csv_path, index=False)
        
        # Print summary
        stats = benchmark_data['aggregate_stats']['overall']
        print(f"\n📊 BENCHMARK SUMMARY")
        print(f"=" * 50)
        print(f"Model: {benchmark_data['benchmark_config']['model_name']}")
        print(f"Total puzzles: {stats['total_puzzles']}")
        print(f"Perfect solutions: {stats['perfect_solutions']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Average accuracy: {stats['avg_accuracy']:.3f}")
        print(f"Average coverage: {stats['avg_coverage']:.3f}")
        print(f"Average time: {stats['avg_generation_time']:.2f}s")
        print(f"Average thinking score: {stats['avg_thinking_score']:.3f}")
        
        logger.info(f"💾 Results saved to: {output_path}")
        logger.info(f"📊 Summary CSV: {csv_path}")

def main():
    """Main benchmarking function"""
    
    print("🎯 Qwen3-1.7B Sudoku Benchmarking")
    print("=" * 50)
    
    # Configuration
    config = BenchmarkConfig(
        model_name="Qwen/Qwen3-1.7B",
        use_8bit=True,
        enable_thinking=False,
        max_new_tokens=600
    )
    
    # Check for test data
    test_data_path = "test_data/sudoku_rl_test.json"
    if not Path(test_data_path).exists():
        print(f"❌ Test data not found: {test_data_path}")
        print("💡 Run the data preparation script first!")
        return
    
    try:
        # Initialize benchmarker
        benchmarker = Qwen3SudokuBenchmarker(config)
        
        # Ask which model to benchmark
        print(f"\n🤔 Which model do you want to benchmark?")
        print(f"1. Base Qwen3-1.7B (pre-training)")
        print(f"2. RL-trained model (post-training)")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            # Benchmark base model
            model_name = "Base Qwen3-1.7B"
            output_file = "benchmarks/qwen3_base_benchmark.json"
        elif choice == "2":
            # Look for RL-trained model
            rl_model_path = "models/qwen3_rl_final"
            if Path(rl_model_path).exists():
                benchmarker.load_model(rl_model_path)
                model_name = "RL-trained Qwen3-1.7B"
                output_file = "benchmarks/qwen3_rl_benchmark.json"
            else:
                print(f"❌ RL model not found: {rl_model_path}")
                print("💡 Train the RL model first!")
                return
        else:
            print("❌ Invalid choice")
            return
        
        print(f"\n🚀 Starting benchmark of {model_name}...")
        
        # Run benchmark
        benchmark_data = benchmarker.benchmark_dataset(test_data_path, max_samples=None)
        
        # Save results
        benchmarker.save_benchmark_results(benchmark_data, output_file)
        
        print(f"\n🎉 Benchmark completed!")
        print(f"📁 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()