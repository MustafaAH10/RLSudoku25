#!/usr/bin/env python3
"""
Complete Solution RL Benchmark
Uses ALL 384 test samples for comprehensive evaluation
"""

import sys
sys.path.append('../shared_utils')

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_pipeline import DataPipelineManager, PuzzleData

# Hardcoded model paths
BASE_MODEL_PATH = "../models/base_model"
TRAINED_MODEL_PATH = "../models/experiment_2_final_model"

def load_model(model_type: str = "base"):
    """Load model by type: 'base' or 'trained'"""
    if model_type == "base":
        model_path = BASE_MODEL_PATH
        print(f"Loading base model from {model_path}")
    elif model_type == "trained":
        model_path = TRAINED_MODEL_PATH
        print(f"Loading trained model from {model_path}")
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Use 'base' or 'trained'")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    
    return model, tokenizer

def create_complete_solution_prompt(puzzle: List[List[int]]) -> str:
    """Create prompt for complete solution"""
    grid_str = ""
    for i, row in enumerate(puzzle):
        if i in [3, 6]:
            grid_str += "─────── ┼ ─────── ┼ ───────\n"
        
        row_str = ""
        for j, cell in enumerate(row):
            if j in [3, 6]:
                row_str += " │ "
            if cell == 0:
                row_str += "  _  "
            else:
                row_str += f"  {cell}  "
        grid_str += row_str + "\n"
    
    return f"""You are an expert Sudoku solver. Solve this puzzle completely in one attempt.

{grid_str}

Rules:
- Numbers 1-9 must appear exactly once in each row, column, and 3x3 box
- Fill ALL empty cells (_) with correct digits
- Provide complete solution for all empty cells

Analyze the puzzle systematically and provide your complete solution in this exact format:
<answer>R1C1: 5, R2C3: 8, R3C7: 2, R4C2: 1, R5C9: 6, ...</answer>

Think step by step and provide the complete solution."""

def solve_puzzle_completely(model, tokenizer, puzzle: List[List[int]], solution: Dict[str, int]) -> Dict:
    """Solve puzzle completely in one attempt"""
    
    # Generate prompt
    prompt = create_complete_solution_prompt(puzzle)
    
    # Generate complete solution
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    # Decode and parse response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_response = full_response[len(prompt):].strip()
    predicted_solution = parse_complete_solution(generated_response)
    
    # Evaluate solution
    correct_predictions = 0
    total_predictions = len(predicted_solution)
    coverage = 0
    
    # Check correctness
    for cell_key, predicted_digit in predicted_solution.items():
        if cell_key in solution and solution[cell_key] == predicted_digit:
            correct_predictions += 1
    
    # Calculate coverage (how many empty cells were attempted)
    coverage = total_predictions / len(solution) if solution else 0
    
    # Calculate accuracy
    accuracy = correct_predictions / max(total_predictions, 1)
    
    # Check if puzzle is completely and correctly solved
    is_perfect = (correct_predictions == len(solution) and 
                  total_predictions == len(solution))
    
    # Create final puzzle state
    final_puzzle = [row[:] for row in puzzle]  # Deep copy
    final_accuracy = 0
    correct_cells = 0
    
    for cell_key, digit in predicted_solution.items():
        if cell_key.startswith('R') and 'C' in cell_key:
            try:
                row_col = cell_key[1:].split('C')
                row, col = int(row_col[0]) - 1, int(row_col[1]) - 1
                if 0 <= row < 9 and 0 <= col < 9:
                    final_puzzle[row][col] = digit
            except:
                pass
    
    # Count correct cells in final state
    for i in range(9):
        for j in range(9):
            cell_key = f"R{i+1}C{j+1}"
            if final_puzzle[i][j] != 0:
                if cell_key in solution and solution[cell_key] == final_puzzle[i][j]:
                    correct_cells += 1
                elif puzzle[i][j] == final_puzzle[i][j]:  # Original given cell
                    correct_cells += 1
    
    final_accuracy = correct_cells / 81
    
    return {
        'predicted_solution': predicted_solution,
        'final_puzzle': final_puzzle,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        'accuracy': accuracy,
        'coverage': coverage,
        'final_accuracy': final_accuracy,
        'is_perfect': is_perfect,
        'full_interaction': {  # Complete prompt/response logging
            'prompt': prompt,
            'full_response': full_response,
            'generated_response': generated_response,
            'original_puzzle': puzzle
        }
    }

def parse_complete_solution(response: str) -> Dict[str, int]:
    """Parse complete solution from model response"""
    import re
    
    solution = {}
    
    # Look for answer block
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1)
        
        # Find all R#C#: # patterns
        matches = re.findall(r'R(\d)C(\d):\s*(\d)', answer_content)
        for match in matches:
            row, col, digit = int(match[0]), int(match[1]), int(match[2])
            if 1 <= row <= 9 and 1 <= col <= 9 and 1 <= digit <= 9:
                cell_key = f"R{row}C{col}"
                solution[cell_key] = digit
    
    # Fallback: look for any R#C#: # pattern in the response
    if not solution:
        matches = re.findall(r'R(\d)C(\d):\s*(\d)', response)
        for match in matches:
            row, col, digit = int(match[0]), int(match[1]), int(match[2])
            if 1 <= row <= 9 and 1 <= col <= 9 and 1 <= digit <= 9:
                cell_key = f"R{row}C{col}"
                solution[cell_key] = digit
    
    return solution

def run_benchmark(model_type: str = "base"):
    """Run benchmark on all test samples"""
    print("🎯 Complete Solution RL Benchmark")
    print("=" * 50)
    
    # Load model
    model, tokenizer = load_model(model_type)
    
    # Load test data
    data_manager = DataPipelineManager()
    test_data = data_manager.load_split("test")
    print(f"📊 Loaded {len(test_data)} test samples")
    
    results = []
    total_accuracy = 0
    total_coverage = 0
    total_final_accuracy = 0
    perfect_solutions = 0
    
    print("\n🔄 Running benchmark...")
    for i, puzzle_dict in enumerate(tqdm(test_data, desc="Benchmarking")):
        puzzle_data = PuzzleData.from_dict(puzzle_dict)
        
        # Run complete solution
        result = solve_puzzle_completely(
            model, tokenizer, 
            puzzle_data.puzzle, 
            puzzle_data.solution
        )
        
        # Store results
        result['puzzle_id'] = i
        result['difficulty'] = puzzle_data.difficulty
        result['clue_count'] = puzzle_data.clue_count
        result['empty_cells'] = len(puzzle_data.empty_cells)
        results.append(result)
        
        # Accumulate metrics
        total_accuracy += result['accuracy']
        total_coverage += result['coverage']
        total_final_accuracy += result['final_accuracy']
        if result['is_perfect']:
            perfect_solutions += 1
    
    # Calculate final metrics
    n_puzzles = len(results)
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'model_type': model_type,
        'total_puzzles': n_puzzles,
        'avg_accuracy': total_accuracy / n_puzzles,
        'avg_coverage': total_coverage / n_puzzles,
        'avg_final_accuracy': total_final_accuracy / n_puzzles,
        'perfect_solve_rate': perfect_solutions / n_puzzles,
        'perfect_solutions': perfect_solutions,
        'detailed_results': results
    }
    
    # Save results
    results_path = Path("benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n📊 BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Total puzzles: {n_puzzles}")
    print(f"Average accuracy: {metrics['avg_accuracy']:.3f}")
    print(f"Average coverage: {metrics['avg_coverage']:.3f}")
    print(f"Average final accuracy: {metrics['avg_final_accuracy']:.3f}")
    print(f"Perfect solve rate: {metrics['perfect_solve_rate']:.3f}")
    print(f"Perfect solutions: {perfect_solutions}/{n_puzzles}")
    print(f"\n💾 Results saved to {results_path}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Complete Solution RL Benchmark")
    parser.add_argument("--model_type", type=str, default="base", choices=["base", "trained"],
                        help="Model type to use: 'base' for base model, 'trained' for RL-trained model (default: base)")
    
    args = parser.parse_args()
    
    run_benchmark(args.model_type)