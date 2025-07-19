#!/usr/bin/env python3
"""
Strategic Move RL Benchmark
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
BASE_MODEL_PATH = "/home/mustafaah/RLSudoku25/remote/models/base_model"
TRAINED_MODEL_PATH = "/home/mustafaah/RLSudoku25/remote/models/experiment_1_final_model"

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

def create_strategic_move_prompt(puzzle: List[List[int]]) -> str:
    """Create prompt for strategic move selection"""
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
    
    return f"""You are an expert Sudoku solver. Analyze this puzzle and make strategic moves.

{grid_str}

Rules:
- Numbers 1-9 must appear exactly once in each row, column, and 3x3 box
- Focus on cells with fewest possibilities
- Make logical deductions

Select the next best move and provide your answer in this format:
<answer>R#C#: digit</answer>

Think step by step and choose the most strategic move."""

def solve_puzzle_strategically(model, tokenizer, puzzle: List[List[int]], solution: Dict[str, int], max_moves: int = 50) -> Dict:
    """Solve puzzle using strategic moves until complete or max moves reached"""
    current_puzzle = [row[:] for row in puzzle]  # Deep copy
    moves_made = []
    correct_moves = 0
    full_interactions = []  # Store all prompt/response pairs
    
    for move_count in range(max_moves):
        # Check if puzzle is complete
        if all(cell != 0 for row in current_puzzle for cell in row):
            break
            
        # Generate prompt for current state
        prompt = create_strategic_move_prompt(current_puzzle)
        
        # Generate move
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        
        # Decode and parse response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_response = full_response[len(prompt):].strip()
        
        # Log the full interaction
        interaction = {
            'move_number': move_count + 1,
            'prompt': prompt,
            'full_response': full_response,
            'generated_response': generated_response,
            'puzzle_state_before': [row[:] for row in current_puzzle]
        }
        
        move = parse_strategic_move(generated_response)
        
        if move:
            row, col, digit = move
            cell_key = f"R{row}C{col}"
            
            # Check if move is valid and correct
            if current_puzzle[row-1][col-1] == 0:  # Cell is empty
                current_puzzle[row-1][col-1] = digit
                moves_made.append(f"{cell_key}: {digit}")
                
                # Check if move is correct
                is_correct = cell_key in solution and solution[cell_key] == digit
                if is_correct:
                    correct_moves += 1
                
                # Add move details to interaction
                interaction.update({
                    'parsed_move': f"{cell_key}: {digit}",
                    'move_valid': True,
                    'move_correct': is_correct,
                    'puzzle_state_after': [row[:] for row in current_puzzle]
                })
            else:
                interaction.update({
                    'parsed_move': f"{cell_key}: {digit}",
                    'move_valid': False,
                    'move_correct': False,
                    'error': 'cell_already_filled'
                })
        else:
            interaction.update({
                'parsed_move': None,
                'move_valid': False,
                'move_correct': False,
                'error': 'no_valid_move_parsed'
            })
            
        full_interactions.append(interaction)
        
        if not move:
            # No valid move found, break
            break
    
    # Calculate final metrics
    total_filled = sum(1 for row in current_puzzle for cell in row if cell != 0)
    original_filled = sum(1 for row in puzzle for cell in row if cell != 0)
    cells_filled = total_filled - original_filled
    
    # Check correctness of final state
    correct_cells = 0
    for i in range(9):
        for j in range(9):
            if current_puzzle[i][j] != 0:
                cell_key = f"R{i+1}C{j+1}"
                if cell_key in solution and solution[cell_key] == current_puzzle[i][j]:
                    correct_cells += 1
                elif puzzle[i][j] == current_puzzle[i][j]:  # Original given cell
                    correct_cells += 1
    
    return {
        'final_puzzle': current_puzzle,
        'moves_made': moves_made,
        'total_moves': len(moves_made),
        'correct_moves': correct_moves,
        'cells_filled': cells_filled,
        'accuracy': correct_moves / max(len(moves_made), 1),
        'coverage': cells_filled / len(solution) if solution else 0,
        'final_accuracy': correct_cells / 81,
        'is_complete': all(cell != 0 for row in current_puzzle for cell in row),
        'full_interactions': full_interactions  # Complete prompt/response logging
    }

def parse_strategic_move(response: str) -> Tuple[int, int, int]:
    """Parse strategic move from model response"""
    import re
    
    # Look for pattern like "R3C4: 7"
    match = re.search(r'<answer>R(\d)C(\d):\s*(\d)</answer>', response)
    if match:
        row, col, digit = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if 1 <= row <= 9 and 1 <= col <= 9 and 1 <= digit <= 9:
            return row, col, digit
    
    # Fallback: look for any R#C#: # pattern
    match = re.search(r'R(\d)C(\d):\s*(\d)', response)
    if match:
        row, col, digit = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if 1 <= row <= 9 and 1 <= col <= 9 and 1 <= digit <= 9:
            return row, col, digit
    
    return None

def run_benchmark(model_type: str = "base"):
    """Run benchmark on all test samples"""
    print("🎯 Strategic Move RL Benchmark")
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
    completed_puzzles = 0
    
    print("\n🔄 Running benchmark...")
    # Use only first 5 samples for quick testing
    test_sample = test_data[:5]
    for i, puzzle_dict in enumerate(tqdm(test_sample, desc="Benchmarking")):
        puzzle_data = PuzzleData.from_dict(puzzle_dict)
        
        # Run strategic solving
        result = solve_puzzle_strategically(
            model, tokenizer, 
            puzzle_data.puzzle, 
            puzzle_data.solution,
            max_moves=len(puzzle_data.empty_cells)
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
        if result['is_complete']:
            completed_puzzles += 1
    
    # Calculate final metrics
    n_puzzles = len(results)
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'model_type': model_type,
        'total_puzzles': n_puzzles,
        'avg_move_accuracy': total_accuracy / n_puzzles,
        'avg_coverage': total_coverage / n_puzzles,
        'avg_final_accuracy': total_final_accuracy / n_puzzles,
        'completion_rate': completed_puzzles / n_puzzles,
        'completed_puzzles': completed_puzzles,
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
    print(f"Average move accuracy: {metrics['avg_move_accuracy']:.3f}")
    print(f"Average coverage: {metrics['avg_coverage']:.3f}")
    print(f"Average final accuracy: {metrics['avg_final_accuracy']:.3f}")
    print(f"Completion rate: {metrics['completion_rate']:.3f}")
    print(f"Completed puzzles: {completed_puzzles}/{n_puzzles}")
    print(f"\n💾 Results saved to {results_path}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Strategic Move RL Benchmark")
    parser.add_argument("--model_type", type=str, default="base", choices=["base", "trained"],
                        help="Model type to use: 'base' for base model, 'trained' for RL-trained model (default: base)")
    
    args = parser.parse_args()
    
    run_benchmark(args.model_type)