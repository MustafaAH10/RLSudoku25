"""
Sudoku Evaluation Script

Before running this script, make sure you have:
1. Installed required packages: pip install torch transformers wandb huggingface_hub
2. Authenticated with Hugging Face: huggingface-cli login
3. Set up wandb account and login

If you get authentication errors, run:
huggingface-cli login

And enter your Hugging Face token from: https://huggingface.co/settings/tokens
"""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Tuple
import json
import wandb
from datetime import datetime
import os

class SudokuEvaluator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-8B-Instruct"):
        """Initialize the evaluator with Qwen model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print("Loading tokenizer and model...")
        
        # Try different model names if the main one fails
        model_options = [
            "Qwen/Qwen2.5-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct", 
            "Qwen/Qwen2-7B-Instruct",
            "microsoft/DialoGPT-medium",  # Fallback option
            "gpt2-medium"  # Another fallback
        ]
        
        model_loaded = False
        for model_option in model_options:
            try:
                print(f"Trying to load model: {model_option}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_option)
                
                # Use CPU if CUDA is not available, and appropriate dtype
                if self.device.type == "cpu":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_option,
                        torch_dtype=torch.float32,  # Use float32 for CPU
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_option,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                
                print(f"Successfully loaded model: {model_option}")
                model_loaded = True
                break
                
            except Exception as e:
                print(f"Failed to load {model_option}: {str(e)}")
                continue
        
        if not model_loaded:
            raise RuntimeError("Could not load any model. Please check your internet connection and Hugging Face authentication.")
        
        # Move model to device if using CPU
        if self.device.type == "cpu":
            self.model = self.model.to(self.device)
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def string_to_grid(self, puzzle_string: str) -> List[List[int]]:
        """Convert 81-character string to 9x9 grid"""
        grid = []
        for i in range(9):
            row = []
            for j in range(9):
                digit = int(puzzle_string[i * 9 + j])
                row.append(digit)
            grid.append(row)
        return grid
    
    def grid_to_string(self, grid: List[List[int]]) -> str:
        """Convert 9x9 grid back to 81-character string"""
        return ''.join(str(grid[i][j]) for i in range(9) for j in range(9))
    
    def format_grid_display(self, grid: List[List[int]]) -> str:
        """Format grid for display with proper Sudoku formatting"""
        display = ""
        for i, row in enumerate(grid):
            if i % 3 == 0 and i != 0:
                display += "------+-------+------\n"
            row_str = ""
            for j, cell in enumerate(row):
                if j % 3 == 0 and j != 0:
                    row_str += "| "
                row_str += f"{cell if cell != 0 else '.'} "
            display += row_str + "\n"
        return display
    
    def get_empty_cells(self, puzzle_string: str) -> List[str]:
        """Get positions of empty cells (0s) in the puzzle in rXcY format"""
        empty_cells = []
        for i in range(81):
            if puzzle_string[i] == '0':
                row = i // 9 + 1  # 1-indexed
                col = i % 9 + 1   # 1-indexed
                empty_cells.append(f"r{row}c{col}")
        return empty_cells
    
    def create_prompt(self, puzzle_string: str) -> str:
        """Create a structured prompt for Sudoku solving"""
        grid = self.string_to_grid(puzzle_string)
        grid_display = self.format_grid_display(grid)
        empty_cells = self.get_empty_cells(puzzle_string)
        
        # Format empty cells for display
        empty_cells_str = ", ".join(empty_cells)
        
        prompt = f"""Solve this Sudoku puzzle. Analyze the puzzle and provide values for all empty cells.

Initial puzzle:
{grid_display}

Empty cells to fill: {empty_cells_str}
Total empty cells: {len(empty_cells)}

Instructions:
1. Analyze the puzzle using standard Sudoku techniques (naked singles, hidden singles, etc.)
2. Identify key constraints and logical deductions
3. Provide your reasoning for the overall solving strategy
4. Fill in all empty cells with the correct numbers

After your analysis, provide the solution for each empty cell in the following format:
SOLUTION:
r1c3: 5
r2c7: 8
r3c1: 4
...
(continue for all empty cells)

Begin solving:"""
        
        return prompt
    
    def is_valid_sudoku(self, grid: List[List[int]]) -> bool:
        """Check if a Sudoku grid is valid (no conflicts)"""
        # Check rows
        for row in grid:
            seen = set()
            for num in row:
                if num != 0:
                    if num in seen:
                        return False
                    seen.add(num)
        
        # Check columns
        for col in range(9):
            seen = set()
            for row in range(9):
                num = grid[row][col]
                if num != 0:
                    if num in seen:
                        return False
                    seen.add(num)
        
        # Check 3x3 boxes
        for box_row in range(3):
            for box_col in range(3):
                seen = set()
                for i in range(3):
                    for j in range(3):
                        num = grid[box_row * 3 + i][box_col * 3 + j]
                        if num != 0:
                            if num in seen:
                                return False
                            seen.add(num)
        
        return True
    
    def is_complete(self, grid: List[List[int]]) -> bool:
        """Check if puzzle is completely filled"""
        for row in grid:
            for cell in row:
                if cell == 0:
                    return False
        return True
    
    def extract_final_grid(self, response: str, original_puzzle: str) -> str:
        """Extract the final solved grid from model response using rXcY: digit format"""
        # Start with the original puzzle
        result_grid = list(original_puzzle)
        
        # Look for the SOLUTION: tag
        solution_marker = "SOLUTION:"
        if solution_marker in response:
            solution_part = response.split(solution_marker)[1].strip()
        else:
            # Fallback: use the entire response
            solution_part = response
        
        # Parse rXcY: digit format
        import re
        pattern = r'r(\d+)c(\d+):\s*(\d+)'
        matches = re.findall(pattern, solution_part)
        
        filled_count = 0
        for match in matches:
            try:
                row = int(match[0]) - 1  # Convert to 0-indexed
                col = int(match[1]) - 1  # Convert to 0-indexed
                digit = match[2]
                
                # Validate bounds
                if 0 <= row < 9 and 0 <= col < 9 and digit.isdigit() and '1' <= digit <= '9':
                    position = row * 9 + col
                    # Only fill if it was originally empty
                    if original_puzzle[position] == '0':
                        result_grid[position] = digit
                        filled_count += 1
            except (ValueError, IndexError):
                continue
        
        print(f"Debug: Filled {filled_count} cells from rXcY format")
        
        # If we didn't get enough matches, try fallback methods
        if filled_count < original_puzzle.count('0') // 2:  # If we filled less than half
            print("Debug: Trying fallback extraction methods...")
            
            # Method 2: Look for any complete 9x9 grid in the response
            lines = solution_part.split('\n')
            grid_digits = []
            consecutive_digit_lines = 0
            
            for line in lines:
                digits_in_line = [c for c in line if c.isdigit()]
                
                if len(digits_in_line) == 9:
                    grid_digits.extend(digits_in_line)
                    consecutive_digit_lines += 1
                    
                    if consecutive_digit_lines == 9:
                        break
                else:
                    if consecutive_digit_lines > 0 and consecutive_digit_lines < 9:
                        grid_digits = grid_digits[:-consecutive_digit_lines*9]
                    consecutive_digit_lines = 0
            
            # If we found a complete grid, use it
            if len(grid_digits) == 81:
                try:
                    grid_ints = [int(d) for d in grid_digits]
                    if all(1 <= d <= 9 for d in grid_ints):
                        print("Debug: Using complete grid fallback")
                        return ''.join(grid_digits)
                except ValueError:
                    pass
        
        result_string = ''.join(result_grid)
        return result_string
    
    def evaluate_single_puzzle(self, puzzle: str, solution: str, clue_count: int) -> dict:
        """Evaluate model on a single puzzle"""
        prompt = self.create_prompt(puzzle)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,  # Reduced from 1024
                temperature=0.3,     # Lower temperature for more focused responses
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):]  # Remove the prompt from response
        
        # Extract the final grid from response
        predicted_solution = self.extract_final_grid(response, puzzle)
        
        # Evaluate correctness
        predicted_grid = self.string_to_grid(predicted_solution)
        solution_grid = self.string_to_grid(solution)
        
        is_valid = self.is_valid_sudoku(predicted_grid)
        is_complete = self.is_complete(predicted_grid)
        is_correct = (predicted_solution == solution)
        
        # Calculate partial correctness
        correct_cells = sum(1 for i in range(81) if predicted_solution[i] == solution[i])
        accuracy = correct_cells / 81
        
        return {
            'puzzle': puzzle,
            'solution': solution,
            'predicted_solution': predicted_solution,
            'clue_count': clue_count,
            'is_valid': is_valid,
            'is_complete': is_complete,
            'is_correct': is_correct,
            'accuracy': accuracy,
            'empty_cells_count': puzzle.count('0'),
            'filled_correctly': sum(1 for i in range(81) if puzzle[i] == '0' and predicted_solution[i] == solution[i]),
            'response': response[:300]  # Reduced truncation for storage
        }
    
    def evaluate_dataset(self, test_file: str, max_samples: int = 100) -> dict:
        """Evaluate model on a test dataset"""
        print(f"Loading test data from {test_file}")
        df = pd.read_csv(test_file)
        
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
        
        results = []
        
        for idx, row in df.iterrows():
            print(f"Evaluating puzzle {idx + 1}/{len(df)}")
            
            result = self.evaluate_single_puzzle(
                row['puzzle'], 
                row['solution'], 
                row['clue_count']
            )
            results.append(result)
            
            # Log progress
            if (idx + 1) % 10 == 0:
                correct_so_far = sum(1 for r in results if r['is_correct'])
                print(f"Progress: {idx + 1}/{len(df)}, Correct: {correct_so_far}/{len(results)}")
        
        # Calculate overall metrics
        total_puzzles = len(results)
        correct_puzzles = sum(1 for r in results if r['is_correct'])
        valid_puzzles = sum(1 for r in results if r['is_valid'])
        complete_puzzles = sum(1 for r in results if r['is_complete'])
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        
        # Group by difficulty
        difficulty_metrics = {}
        for result in results:
            clue_count = result['clue_count']
            if clue_count not in difficulty_metrics:
                difficulty_metrics[clue_count] = []
            difficulty_metrics[clue_count].append(result)
        
        difficulty_summary = {}
        for clue_count, group_results in difficulty_metrics.items():
            difficulty_summary[clue_count] = {
                'count': len(group_results),
                'correct': sum(1 for r in group_results if r['is_correct']),
                'valid': sum(1 for r in group_results if r['is_valid']),
                'complete': sum(1 for r in group_results if r['is_complete']),
                'avg_accuracy': np.mean([r['accuracy'] for r in group_results])
            }
        
        summary = {
            'total_puzzles': total_puzzles,
            'correct_puzzles': correct_puzzles,
            'valid_puzzles': valid_puzzles,
            'complete_puzzles': complete_puzzles,
            'overall_accuracy': correct_puzzles / total_puzzles,
            'valid_rate': valid_puzzles / total_puzzles,
            'completion_rate': complete_puzzles / total_puzzles,
            'avg_cell_accuracy': avg_accuracy,
            'difficulty_breakdown': difficulty_summary
        }
        
        return {
            'summary': summary,
            'detailed_results': results
        }

def main():
    print("=== Sudoku Baseline Evaluation ===")
    print("This script evaluates a language model on Sudoku solving")
    print()
    
    # Check if test data exists
    test_file = "data/test_set.csv"
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found.")
        print("Please run data_preparation.py first to create the datasets.")
        print()
        print("Run: python data_preparation.py")
        return
    
    # Initialize wandb
    wandb.init(
        project="sudoku-rl-baseline",
        name=f"qwen-baseline-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model": "Language Model",
            "task": "sudoku_solving",
            "evaluation_type": "baseline"
        }
    )
    
    # Initialize evaluator
    try:
        evaluator = SudokuEvaluator()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print()
        print("Please try:")
        print("1. huggingface-cli login")
        print("2. Check your internet connection")
        print("3. Make sure you have the required packages installed")
        wandb.finish()
        return
    
    print("Starting baseline evaluation...")
    results = evaluator.evaluate_dataset(test_file, max_samples=20)  # Start with 20 samples for testing
    
    # Log results to wandb
    wandb.log(results['summary'])
    
    # Also log some detailed metrics
    wandb.log({
        "avg_empty_cells": np.mean([r['empty_cells_count'] for r in results['detailed_results']]),
        "avg_filled_correctly": np.mean([r['filled_correctly'] for r in results['detailed_results']]),
        "empty_cell_fill_rate": np.mean([r['filled_correctly']/r['empty_cells_count'] if r['empty_cells_count'] > 0 else 0 
                                        for r in results['detailed_results']])
    })
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"results/baseline_eval_{timestamp}.json"
    os.makedirs("results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation Results:")
    print(f"Overall Accuracy: {results['summary']['overall_accuracy']:.3f}")
    print(f"Valid Solutions: {results['summary']['valid_rate']:.3f}")
    print(f"Completion Rate: {results['summary']['completion_rate']:.3f}")
    print(f"Average Cell Accuracy: {results['summary']['avg_cell_accuracy']:.3f}")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    wandb.finish()

if __name__ == "__main__":
    main()