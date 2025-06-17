import json
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re
from datetime import datetime

class SudokuBenchmark:
    def __init__(self, model_name, max_tokens=2500, temperature=0.1):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded on device: {self.model.device}")
        print(f"Using max_tokens: {max_tokens}, temperature: {temperature}")
    
    def format_sudoku_grid(self, grid):
        """Format 9x9 grid for display with box separators"""
        formatted = []
        for i, row in enumerate(grid):
            if i in [3, 6]:
                formatted.append("──────┼───────┼──────")
            
            row_str = ""
            for j, cell in enumerate(row):
                if j in [3, 6]:
                    row_str += "│ "
                if cell == 0:
                    row_str += "_ "
                else:
                    row_str += f"{cell} "
            formatted.append(row_str.strip())
        
        return "\n".join(formatted)
    
    def create_sudoku_prompt(self, puzzle_grid):
        """Create the prompt for solving Sudoku"""
        formatted_grid = self.format_sudoku_grid(puzzle_grid)
        
        prompt = f"""Solve this Sudoku puzzle completely. Provide ONLY the solution in the exact format shown below.

Puzzle:
```
{formatted_grid}
```

Instructions:
- Fill ALL empty cells (marked with _)
- Each row, column, and 3x3 box must contain digits 1-9 exactly once
- Provide your complete solution using ONLY this format: RnCm: digit
- List every empty cell's solution

SOLUTION:
"""
        return prompt
    
    def parse_solution(self, response_text):
        """Parse the model's response to extract cell solutions"""
        solutions = {}
        
        # Look for patterns like "R1C3: 5" or "R1C3 = 5"
        patterns = [
            r'R(\d)C(\d):\s*(\d)',
            r'R(\d)C(\d)\s*=\s*(\d)',
            r'R(\d)C(\d)\s+(\d)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            for match in matches:
                row, col, digit = int(match[0]), int(match[1]), int(match[2])
                if 1 <= row <= 9 and 1 <= col <= 9 and 1 <= digit <= 9:
                    solutions[f"R{row}C{col}"] = digit
        
        return solutions
    
    def evaluate_solution(self, puzzle_grid, expected_solutions, predicted_solutions):
        """Evaluate the quality of the solution"""
        total_empty_cells = sum(row.count(0) for row in puzzle_grid)
        attempted_cells = len(predicted_solutions)
        correct_cells = 0
        incorrect_cells = 0
        
        for cell, predicted_digit in predicted_solutions.items():
            if cell in expected_solutions:
                if expected_solutions[cell] == predicted_digit:
                    correct_cells += 1
                else:
                    incorrect_cells += 1
        
        accuracy = correct_cells / total_empty_cells if total_empty_cells > 0 else 0
        coverage = attempted_cells / total_empty_cells if total_empty_cells > 0 else 0
        precision = correct_cells / attempted_cells if attempted_cells > 0 else 0
        perfect_solution = (correct_cells == total_empty_cells and incorrect_cells == 0)
        
        return {
            "total_empty_cells": total_empty_cells,
            "attempted_cells": attempted_cells,
            "correct_cells": correct_cells,
            "incorrect_cells": incorrect_cells,
            "accuracy": accuracy,
            "coverage": coverage,
            "precision": precision,
            "perfect_solution": perfect_solution,
            "thinking_quality": {
                "has_thinking": "<thinking>" in str(predicted_solutions),
                "score": 0.0,
                "reasoning_indicators": 0
            }
        }
    
    def solve_puzzle(self, puzzle_data):
        """Solve a single puzzle"""
        puzzle_grid = puzzle_data["puzzle"]
        expected_solutions = puzzle_data["solution"]
        
        # Create prompt
        prompt = self.create_sudoku_prompt(puzzle_grid)
        
        # Tokenize and generate
        start_time = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                top_p=0.95,
                repetition_penalty=1.1
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        # Parse solution
        predicted_solutions = self.parse_solution(response)
        
        # Evaluate
        evaluation = self.evaluate_solution(puzzle_grid, expected_solutions, predicted_solutions)
        
        # Calculate token stats
        input_tokens = len(inputs.input_ids[0])
        output_tokens = len(outputs[0]) - input_tokens
        total_tokens = len(outputs[0])
        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
        
        return {
            "response": response,
            "generation_time": generation_time,
            "metadata": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second,
                "has_thinking": "<thinking>" in response.lower()
            },
            "evaluation": evaluation,
            "expected_solutions": expected_solutions,
            "predicted_solutions": predicted_solutions
        }
    
    def run_benchmark(self, test_data_path, total_samples=None):
        """Run benchmark on test dataset"""
        print(f"Loading test data from: {test_data_path}")
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        if total_samples:
            test_data = test_data[:total_samples]
        
        print(f"Running benchmark on {len(test_data)} puzzles...")
        
        results = []
        for i, puzzle_data in enumerate(tqdm(test_data, desc="Solving puzzles")):
            try:
                result = self.solve_puzzle(puzzle_data)
                results.append({
                    "puzzle_id": puzzle_data.get("id", f"puzzle_{i}"),
                    "difficulty": puzzle_data.get("difficulty", "unknown"),
                    "clue_count": puzzle_data.get("clue_count", 0),
                    "empty_count": sum(row.count(0) for row in puzzle_data["puzzle"]),
                    "success": result["evaluation"]["perfect_solution"],
                    **result
                })
                
                # Print progress for perfect solutions
                if result["evaluation"]["perfect_solution"]:
                    print(f"✓ Perfect solution found for puzzle {i+1}")
                    
            except Exception as e:
                print(f"Error solving puzzle {i+1}: {e}")
                results.append({
                    "puzzle_id": puzzle_data.get("id", f"puzzle_{i}"),
                    "difficulty": puzzle_data.get("difficulty", "unknown"),
                    "error": str(e),
                    "success": False
                })
        
        return {
            "benchmark_config": {
                "model_name": self.model_name,
                "dataset_path": test_data_path,
                "total_samples": len(test_data),
                "timestamp": datetime.now().isoformat()
            },
            "results": results
        }

def main():
    parser = argparse.ArgumentParser(description="Benchmark Sudoku solving performance")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--test_data", type=str, default="test_data/sudoku_rl_test.json", help="Test data path")
    parser.add_argument("--output_file", type=str, required=True, help="Output file for results")
    parser.add_argument("--max_tokens", type=int, default=2500, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--total_samples", type=int, help="Limit number of test samples")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = SudokuBenchmark(
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    results = benchmark.run_benchmark(args.test_data, args.total_samples)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    total_puzzles = len(results["results"])
    perfect_solutions = sum(1 for r in results["results"] if r.get("success", False))
    avg_accuracy = sum(r.get("evaluation", {}).get("accuracy", 0) for r in results["results"]) / total_puzzles
    
    print(f"\n{'='*50}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Model: {args.model_name}")
    print(f"Total puzzles: {total_puzzles}")
    print(f"Perfect solutions: {perfect_solutions} ({perfect_solutions/total_puzzles*100:.1f}%)")
    print(f"Average accuracy: {avg_accuracy:.3f}")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()