"""
Efficient Sudoku CoT Generator for 4 Million Sudoku Dataset - OPTIMIZED VERSION
===============================================================================

This approach generates efficient CoT for puzzles with minimal token usage
by providing empty cell positions and asking for direct solutions
"""

import kagglehub
import pandas as pd
import numpy as np
import json
import time
import asyncio
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
import random
from pathlib import Path
import re
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class OptimizedSudokuCoTGenerator:
    """Generate efficient CoT for 9×9 Sudoku with minimal token usage"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = "deepseek-reasoner"
    
    def format_sudoku_grid(self, quiz_string: str) -> str:
        """Convert flattened string to readable 9x9 grid"""
        
        if len(quiz_string) != 81:
            raise ValueError(f"Quiz string must be 81 characters, got {len(quiz_string)}")
        
        lines = []
        for i in range(9):
            row = []
            for j in range(9):
                char = quiz_string[i * 9 + j]
                row.append('.' if char == '0' else char)
            
            # Group by 3s with separators
            row_str = f"{' '.join(row[0:3])} | {' '.join(row[3:6])} | {' '.join(row[6:9])}"
            lines.append(row_str)
            
            # Add horizontal separator after rows 2 and 5
            if i == 2 or i == 5:
                lines.append("------+-------+------")
        
        return '\n'.join(lines)
    
    def find_empty_cells(self, quiz_string: str) -> List[Tuple[int, int]]:
        """Find positions of empty cells (0s) in the puzzle"""
        empty_cells = []
        for i in range(81):
            if quiz_string[i] == '0':
                row = i // 9 + 1  # 1-indexed
                col = i % 9 + 1   # 1-indexed
                empty_cells.append((row, col))
        return empty_cells
    
    def create_efficient_prompt(self, quiz_string: str, empty_cells: List[Tuple[int, int]], clue_count: int) -> str:
        """Create efficient prompt that asks for specific cell solutions"""
        
        formatted_grid = self.format_sudoku_grid(quiz_string)
        
        # Format empty cells for the prompt
        empty_cells_str = ", ".join([f"({r},{c})" for r, c in empty_cells])
        
        prompt = f"""Solve this Sudoku puzzle. You need to fill {len(empty_cells)} empty cell(s).

PUZZLE:
{formatted_grid}

EMPTY CELLS TO FILL: {empty_cells_str}
(Format: (row, column) where row 1-9 from top, column 1-9 from left)

INSTRUCTIONS:
1. Quickly analyze the constraints for each empty cell using ANY Sudoku constraints (row, column, 3x3 box) - YOU DO NOT HAVE TO CHECK ALL CONSTRAINTS. 
2. For each empty cell, determine the missing digit that fits the grid.
3. Return your solution in this EXACT format:

SOLUTION:
(row,col):digit
(row,col):digit
...

Example: If cell (3,1) needs digit 7 and cell (5,9) needs digit 2:
SOLUTION:
(3,1):7
(5,9):2

Rules: Each row, column, and 3×3 box must contain digits 1-9 exactly once."""

        return prompt
    
    def extract_cell_solutions(self, response_content: str, empty_cells: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
        """Extract cell solutions from the response"""
        
        solutions = {}
        
        # Look for SOLUTION section
        if 'SOLUTION:' in response_content:
            solution_start = response_content.find('SOLUTION:') + len('SOLUTION:')
            solution_section = response_content[solution_start:].strip()
        else:
            solution_section = response_content
        
        # Pattern to match (row,col):digit format
        pattern = r'\((\d+),(\d+)\):(\d)'
        matches = re.findall(pattern, solution_section)
        
        for match in matches:
            row, col, digit = int(match[0]), int(match[1]), int(match[2])
            if (row, col) in empty_cells and 1 <= digit <= 9:
                solutions[(row, col)] = digit
        
        return solutions
    
    def apply_solutions_to_grid(self, quiz_string: str, solutions: Dict[Tuple[int, int], int]) -> str:
        """Apply the solutions to create the complete grid string"""
        
        grid = list(quiz_string)
        
        for (row, col), digit in solutions.items():
            # Convert to 0-indexed position
            pos = (row - 1) * 9 + (col - 1)
            if 0 <= pos < 81 and grid[pos] == '0':
                grid[pos] = str(digit)
        
        return ''.join(grid)
    
    def validate_solution(self, solved_grid: str, expected_solution: str) -> bool:
        """Check if the solved grid matches the expected solution"""
        return solved_grid == expected_solution
    
    async def solve_puzzle_with_cot(self, quiz: str, solution: str, clue_count: int) -> Dict:
        """Solve puzzle and generate CoT with validation"""
        
        empty_cells = self.find_empty_cells(quiz)
        prompt = self.create_efficient_prompt(quiz, empty_cells, clue_count)
        
        try:
            start_time = time.time()
            
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert Sudoku solver. Analyze constraints efficiently and provide exact solutions in the requested format."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,  # Reduced since we need less formatting
                    temperature=0.1,
                    timeout=90
                ),
                timeout=120
            )
            
            elapsed_time = time.time() - start_time
            
            reasoning_content = response.choices[0].message.reasoning_content or ""
            final_answer = response.choices[0].message.content or ""
            
            # Extract solutions from response
            extracted_solutions = self.extract_cell_solutions(final_answer, empty_cells)
            
            # Apply solutions to create complete grid
            solved_grid = self.apply_solutions_to_grid(quiz, extracted_solutions)
            
            # Check if all empty cells were filled
            all_filled = len(extracted_solutions) == len(empty_cells)
            
            # Check correctness
            is_correct = all_filled and self.validate_solution(solved_grid, solution)
            
            result = {
                'success': True,
                'quiz': quiz,
                'solution': solution,
                'clue_count': clue_count,
                'empty_cells': empty_cells,
                'extracted_solutions': extracted_solutions,
                'solved_grid': solved_grid,
                'all_cells_filled': all_filled,
                'is_correct': is_correct,
                'reasoning_content': reasoning_content,
                'final_answer': final_answer,
                'response_time_seconds': elapsed_time,
                'token_usage': response.usage.total_tokens if response.usage else 0,
                'prompt': prompt
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'quiz': quiz,
                'solution': solution,
                'clue_count': clue_count,
                'error': str(e),
                'error_type': type(e).__name__
            }

class SudokuDatasetProcessor:
    """Process the 4 Million Sudoku dataset"""
    
    def __init__(self):
        self.dataset_path = None
        self.df = None
    
    def download_and_load_dataset(self) -> pd.DataFrame:
        """Download and load the dataset"""
        
        print("Downloading 4 Million Sudoku dataset...")
        try:
            path = kagglehub.dataset_download("informoney/4-million-sudoku-puzzles-easytohard")
            print(f"Dataset downloaded to: {path}")
            
            # Find CSV file in the path
            csv_files = list(Path(path).glob("*.csv"))
            if csv_files:
                csv_path = csv_files[0]
                print(f"Loading CSV: {csv_path}")
                
                self.df = pd.read_csv(csv_path)
                print(f"Loaded dataset with {len(self.df)} puzzles")
                print(f"Columns: {list(self.df.columns)}")
                
                # Show clue distribution
                if 'clue_numbers' in self.df.columns:
                    clue_dist = self.df['clue_numbers'].value_counts().sort_index()
                    print(f"Clue distribution:\n{clue_dist.head(10)}")
                
                return self.df
            else:
                print("No CSV files found in dataset path")
                return None
                
        except Exception as e:
            print(f"Error downloading/loading dataset: {e}")
            return None
    
    def get_puzzles_by_clues(self, clue_count: int, num_puzzles: int = 50) -> List[Tuple[str, str, int]]:
        """Get puzzles with specific number of clues"""
        
        if self.df is None:
            self.download_and_load_dataset()
        
        if self.df is None:
            return []
        
        # Filter by clue count
        filtered_df = self.df[self.df['clue_numbers'] == clue_count]
        
        if len(filtered_df) == 0:
            print(f"No puzzles found with {clue_count} clues")
            return []
        
        print(f"Found {len(filtered_df)} puzzles with {clue_count} clues")
        
        # Sample the requested number
        sample_size = min(num_puzzles, len(filtered_df))
        sample_df = filtered_df.sample(n=sample_size, random_state=42)
        
        puzzles = []
        for _, row in sample_df.iterrows():
            quiz = row['quizzes']
            solution = row['solutions'] 
            puzzles.append((quiz, solution, clue_count))
        
        return puzzles

class OptimizedSudokuCoTTrainingGenerator:
    """Generate training data with efficient CoT for Sudoku"""
    
    def __init__(self, api_key: str):
        self.cot_generator = OptimizedSudokuCoTGenerator(api_key)
        self.dataset_processor = SudokuDatasetProcessor()
    
    async def generate_cot_dataset(self, clue_count: int = 80, num_examples: int = 20) -> List[Dict]:
        """Generate CoT dataset for puzzles with specific clue count"""
        
        print(f"Generating efficient CoT dataset for {clue_count}-clue puzzles...")
        
        # Get puzzles from dataset
        puzzles = self.dataset_processor.get_puzzles_by_clues(clue_count, num_examples)
        
        if not puzzles:
            print("No puzzles found!")
            return []
        
        results = []
        total_cost = 0
        successful = 0
        
        for i, (quiz, solution, clues) in enumerate(puzzles):
            print(f"\nProcessing puzzle {i+1}/{len(puzzles)} (clues: {clues})...")
            
            try:
                result = await self.cot_generator.solve_puzzle_with_cot(quiz, solution, clues)
                
                if result['success']:
                    results.append(result)
                    
                    # Calculate cost (DeepSeek pricing: $0.55 per million tokens)
                    tokens = result.get('token_usage', 0)
                    cost = tokens * 0.00000055
                    total_cost += cost
                    
                    if result.get('is_correct', False):
                        successful += 1
                        print(f"  ✅ Correct! ({successful}/{i+1}) | Tokens: {tokens} | Cost: ${cost:.6f}")
                        # Show the extracted solutions
                        solutions = result.get('extracted_solutions', {})
                        if solutions:
                            solution_str = ", ".join([f"({r},{c}):{d}" for (r,c), d in solutions.items()])
                            print(f"     Solutions: {solution_str}")
                    else:
                        print(f"  ❌ Incorrect | Tokens: {tokens} | Cost: ${cost:.6f}")
                        # Show what went wrong
                        expected_cells = len(result.get('empty_cells', []))
                        filled_cells = len(result.get('extracted_solutions', {}))
                        print(f"     Expected {expected_cells} cells, filled {filled_cells}")
                        
                        if not result.get('all_cells_filled', False):
                            print(f"     Missing cells not filled!")
                        else:
                            print(f"     All cells filled but solution incorrect")
                else:
                    print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"  ❌ Exception: {e}")
        
        print(f"\n📊 GENERATION COMPLETE:")
        print(f"  💰 Total cost: ${total_cost:.6f}")
        print(f"  ✅ Successful: {successful}/{len(results)}")
        print(f"  📈 Success rate: {successful/len(results)*100:.1f}%" if results else "  📈 Success rate: 0%")
        
        return results
    
    def save_to_txt_file(self, results: List[Dict], filename: str = "sudoku_cot_optimized.txt"):
        """Save all results to a single text file with clear formatting"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("OPTIMIZED SUDOKU CHAIN OF THOUGHT TRAINING DATA\n")
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(results):
                f.write(f"EXAMPLE {i+1}\n")
                f.write("-" * 40 + "\n")
                
                # Metadata
                f.write(f"Clues: {result.get('clue_count', 'Unknown')}\n")
                f.write(f"Empty Cells: {len(result.get('empty_cells', []))}\n")
                f.write(f"Correct: {result.get('is_correct', False)}\n")
                f.write(f"Tokens: {result.get('token_usage', 0)}\n")
                f.write(f"Time: {result.get('response_time_seconds', 0):.1f}s\n\n")
                
                # Original puzzle
                f.write("ORIGINAL PUZZLE:\n")
                puzzle_formatted = self.cot_generator.format_sudoku_grid(result.get('quiz', ''))
                f.write(puzzle_formatted)
                f.write("\n\n")
                
                # Empty cells to fill
                empty_cells = result.get('empty_cells', [])
                f.write(f"EMPTY CELLS TO FILL: {empty_cells}\n\n")
                
                # Efficient prompt
                f.write("EFFICIENT PROMPT:\n")
                prompt = result.get('prompt', 'No prompt available')
                f.write(prompt)
                f.write("\n\n")
                
                # Chain of thought (reasoning)
                f.write("CHAIN OF THOUGHT (Model's Internal Reasoning):\n")
                reasoning = result.get('reasoning_content', 'No reasoning available')
                f.write(reasoning)
                f.write("\n\n")
                
                # Model's response
                f.write("MODEL'S RESPONSE:\n")
                answer = result.get('final_answer', 'No answer available')
                f.write(answer)
                f.write("\n\n")
                
                # Extracted solutions
                f.write("EXTRACTED SOLUTIONS:\n")
                solutions = result.get('extracted_solutions', {})
                if solutions:
                    for (row, col), digit in solutions.items():
                        f.write(f"  ({row},{col}): {digit}\n")
                else:
                    f.write("  None extracted\n")
                f.write("\n")
                
                # Correct solution
                f.write("CORRECT SOLUTION GRID:\n")
                correct_solution = self.cot_generator.format_sudoku_grid(result.get('solution', ''))
                f.write(correct_solution)
                f.write("\n\n")
                
                # Validation details
                f.write("VALIDATION:\n")
                f.write(f"Expected:     {result.get('solution', 'Unknown')}\n")
                f.write(f"Model Output: {result.get('solved_grid', 'None')}\n")
                f.write(f"Match: {result.get('is_correct', False)}\n")
                f.write(f"All Cells Filled: {result.get('all_cells_filled', False)}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"Optimized training data saved to {filename}")
    
    def save_json_dataset(self, results: List[Dict], filename: str = "sudoku_cot_optimized.json"):
        """Save as clean JSON for training"""
        
        clean_data = []
        for result in results:
            clean_entry = {
                'puzzle': result['quiz'],
                'solution': result['solution'], 
                'clue_count': result.get('clue_count', 0),
                'empty_cells': result.get('empty_cells', []),
                'extracted_solutions': result.get('extracted_solutions', {}),
                'is_correct': result.get('is_correct', False),
                'all_cells_filled': result.get('all_cells_filled', False),
                'reasoning': result.get('reasoning_content', ''),
                'model_response': result.get('final_answer', ''),
                'tokens': result.get('token_usage', 0),
                'response_time': result.get('response_time_seconds', 0)
            }
            clean_data.append(clean_entry)
        
        with open(filename, 'w') as f:
            json.dump(clean_data, f, indent=2, default=str)  # default=str handles tuple keys
        
        correct_count = sum(1 for entry in clean_data if entry['is_correct'])
        print(f"Optimized dataset with {len(clean_data)} examples ({correct_count} correct) saved to {filename}")

# Main execution
async def main():
    """Generate optimized Sudoku CoT training data"""
    
    # Get API key from environment variable
    API_KEY = os.getenv('DEEPSEEK_API_KEY')
    if not API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables. Please set it in .env file")
    
    CLUE_COUNT = 80  # Start with easiest puzzles (only 1 empty cell)
    NUM_EXAMPLES = 5  # Test with more examples
    
    # Initialize generator
    generator = OptimizedSudokuCoTTrainingGenerator(API_KEY)
    
    # Generate dataset
    print(f"Generating {NUM_EXAMPLES} optimized examples with {CLUE_COUNT} clues...")
    results = await generator.generate_cot_dataset(CLUE_COUNT, NUM_EXAMPLES)
    
    if results:
        # Save to text file (complete format)
        generator.save_to_txt_file(results, f"sudoku_cot_{CLUE_COUNT}clues_optimized.txt")
        
        # Save as JSON (clean format for training)
        generator.save_json_dataset(results, f"sudoku_cot_{CLUE_COUNT}clues_optimized.json")
        
        print(f"\n🎯 Generated {len(results)} optimized examples!")
        print(f"📄 Text file: sudoku_cot_{CLUE_COUNT}clues_optimized.txt")
        print(f"📊 JSON file: sudoku_cot_{CLUE_COUNT}clues_optimized.json")
        
        # Show stats
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        avg_tokens = sum(r.get('token_usage', 0) for r in results) / len(results) if results else 0
        total_cost = sum(r.get('token_usage', 0) * 0.00000055 for r in results if r.get('success', False))
        
        print(f"✅ Correct solutions: {correct_count}/{len(results)}")
        print(f"📊 Average tokens per puzzle: {avg_tokens:.0f}")
        print(f"💰 Total cost: ${total_cost:.6f}")
        
    else:
        print("❌ No results generated. Check API key and try again.")

if __name__ == "__main__":
    asyncio.run(main())