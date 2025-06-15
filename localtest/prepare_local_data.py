#!/usr/bin/env python3
"""
Standalone Script: Prepare 100 Sudoku Puzzles for RL Testing
============================================================

This script downloads the Kaggle dataset and prepares 100 puzzles
across different difficulties for local RL testing.
"""

import pandas as pd
import numpy as np
import json
import random
from pathlib import Path
import hashlib
from typing import List, Dict, Tuple
import kagglehub

class SudokuTestDataPreparator:
    """Prepare 100 test puzzles for RL validation"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.df = None
        
    def download_kaggle_dataset(self) -> pd.DataFrame:
        """Download and load the Kaggle Sudoku dataset"""
        print("🔄 Downloading Kaggle Sudoku dataset...")
        
        try:
            # Download dataset
            path = kagglehub.dataset_download("informoney/4-million-sudoku-puzzles-easytohard")
            print(f"✅ Dataset downloaded to: {path}")
            
            # Find and load CSV
            csv_files = list(Path(path).glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in dataset")
                
            csv_path = csv_files[0]
            print(f"📁 Loading CSV: {csv_path}")
            
            # Load dataset
            self.df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(self.df):,} puzzles")
            
            # Add unique identifiers
            self.df['puzzle_hash'] = self.df['quizzes'].apply(
                lambda x: hashlib.md5(x.encode()).hexdigest()[:12]
            )
            
            print(f"📊 Clue count distribution:")
            clue_dist = self.df['clue_numbers'].value_counts().sort_index()
            print(f"   Min clues: {clue_dist.index.min()}")
            print(f"   Max clues: {clue_dist.index.max()}")
            print(f"   Most common: {clue_dist.index[clue_dist.argmax()]} clues ({clue_dist.max():,} puzzles)")
            
            return self.df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def format_sudoku_for_llm(self, quiz_string: str) -> str:
        """Format Sudoku puzzle in LLM-friendly structure"""
        if len(quiz_string) != 81:
            raise ValueError(f"Invalid puzzle length: {len(quiz_string)}")
        
        # Convert to 2D grid representation
        grid_lines = []
        for row in range(9):
            line_chars = []
            for col in range(9):
                char = quiz_string[row * 9 + col]
                line_chars.append('_' if char == '0' else char)
            
            # Group into 3x3 blocks with separators
            line = f"{' '.join(line_chars[0:3])} | {' '.join(line_chars[3:6])} | {' '.join(line_chars[6:9])}"
            grid_lines.append(line)
            
            # Add horizontal separators
            if row == 2 or row == 5:
                grid_lines.append("------+-------+------")
        
        return '\n'.join(grid_lines)
    
    def find_empty_positions(self, quiz_string: str) -> List[str]:
        """Find empty cell positions in (R,C) format"""
        empty_positions = []
        for i in range(81):
            if quiz_string[i] == '0':
                row = (i // 9) + 1  # 1-indexed
                col = (i % 9) + 1   # 1-indexed
                empty_positions.append(f"R{row}C{col}")
        return empty_positions
    
    def create_llm_prompt(self, puzzle: str, clue_count: int) -> str:
        """Create the complete prompt for LLM training"""
        
        formatted_grid = self.format_sudoku_for_llm(puzzle)
        empty_positions = self.find_empty_positions(puzzle)
        
        prompt = f"""<|im_start|>system
You are an expert Sudoku solver. Analyze the puzzle and provide the solution for each empty cell marked with '_'.
<|im_end|>
<|im_start|>user
Solve this Sudoku puzzle. Fill in the missing numbers (marked with '_') so that each row, column, and 3x3 box contains all digits 1-9.

PUZZLE ({clue_count} clues given):
{formatted_grid}

EMPTY CELLS TO FILL: {', '.join(empty_positions)}

Provide your solution in this exact format:
SOLUTION:
R1C1: digit
R2C3: digit
...

Think step by step and solve systematically.
<|im_end|>
<|im_start|>assistant
I'll solve this Sudoku step by step.

Looking at the puzzle systematically:

SOLUTION:"""

        return prompt
    
    def create_target_response(self, puzzle: str, solution: str) -> str:
        """Create the target response showing correct solutions"""
        
        empty_positions = []
        solution_lines = []
        
        for i in range(81):
            if puzzle[i] == '0':
                row = (i // 9) + 1
                col = (i % 9) + 1
                digit = solution[i]
                empty_positions.append(f"R{row}C{col}")
                solution_lines.append(f"R{row}C{col}: {digit}")
        
        return '\n'.join(solution_lines)
    
    def get_stratified_sample(self, total_samples: int = 100) -> List[Dict]:
        """Get stratified sample across difficulty levels"""
        
        if self.df is None:
            print("❌ Dataset not loaded. Loading now...")
            self.download_kaggle_dataset()
            
        if self.df is None:
            raise ValueError("Could not load dataset")
        
        # Define difficulty distribution
        difficulty_targets = {
            'expert': (17, 25, 10),    # 17-25 clues, 10 samples
            'hard': (26, 35, 20),      # 26-35 clues, 20 samples  
            'medium': (36, 50, 30),    # 36-50 clues, 30 samples
            'easy': (51, 65, 25),      # 51-65 clues, 25 samples
            'beginner': (66, 80, 15)   # 66-80 clues, 15 samples
        }
        
        all_samples = []
        used_hashes = set()
        
        for difficulty, (min_clues, max_clues, target_count) in difficulty_targets.items():
            print(f"🎯 Sampling {target_count} {difficulty} puzzles ({min_clues}-{max_clues} clues)...")
            
            # Filter by clue range
            mask = (self.df['clue_numbers'] >= min_clues) & (self.df['clue_numbers'] <= max_clues)
            difficulty_df = self.df[mask]
            
            # Remove already used puzzles
            difficulty_df = difficulty_df[~difficulty_df['puzzle_hash'].isin(used_hashes)]
            
            if len(difficulty_df) < target_count:
                print(f"⚠️  Only {len(difficulty_df)} available, need {target_count}")
                target_count = len(difficulty_df)
            
            # Sample without replacement
            sampled = difficulty_df.sample(n=target_count, random_state=self.random_seed)
            
            # Process samples
            for idx, row in sampled.iterrows():
                sample = {
                    'id': f"{difficulty}_{len(all_samples)+1:03d}",
                    'puzzle_hash': row['puzzle_hash'],
                    'difficulty': difficulty,
                    'puzzle': row['quizzes'],
                    'solution': row['solutions'],
                    'clue_count': int(row['clue_numbers']),
                    'empty_count': 81 - int(row['clue_numbers']),
                    'prompt': self.create_llm_prompt(row['quizzes'], int(row['clue_numbers'])),
                    'target_response': self.create_target_response(row['quizzes'], row['solutions']),
                    'formatted_grid': self.format_sudoku_for_llm(row['quizzes'])
                }
                
                all_samples.append(sample)
                used_hashes.add(row['puzzle_hash'])
            
            print(f"✅ Added {len(sampled)} {difficulty} samples")
        
        # Shuffle final samples
        random.shuffle(all_samples)
        print(f"\n🎉 Created {len(all_samples)} total samples")
        
        return all_samples
    
    def save_test_dataset(self, samples: List[Dict], output_dir: str = "test_data"):
        """Save the test dataset in multiple formats"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save complete dataset as JSON
        json_path = output_path / "sudoku_test_100.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        # Save simplified version for RL training
        rl_data = []
        for sample in samples:
            rl_sample = {
                'id': sample['id'],
                'puzzle': sample['puzzle'],
                'solution': sample['solution'],
                'clue_count': sample['clue_count'],
                'difficulty': sample['difficulty'],
                'prompt': sample['prompt'],
                'target_response': sample['target_response']
            }
            rl_data.append(rl_sample)
        
        rl_path = output_path / "sudoku_rl_test.json" 
        with open(rl_path, 'w', encoding='utf-8') as f:
            json.dump(rl_data, f, indent=2, ensure_ascii=False)
        
        # Save CSV for inspection
        csv_data = []
        for sample in samples:
            csv_row = {
                'id': sample['id'],
                'difficulty': sample['difficulty'],
                'clue_count': sample['clue_count'],
                'empty_count': sample['empty_count'],
                'puzzle_hash': sample['puzzle_hash'],
                'puzzle': sample['puzzle'],
                'solution': sample['solution']
            }
            csv_data.append(csv_row)
        
        csv_path = output_path / "sudoku_test_100.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        
        # Save statistics
        stats = self.generate_statistics(samples)
        stats_path = output_path / "dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n💾 Dataset saved:")
        print(f"   📄 Complete data: {json_path}")
        print(f"   🤖 RL training: {rl_path}")
        print(f"   📊 CSV format: {csv_path}")
        print(f"   📈 Statistics: {stats_path}")
        
        return rl_path
    
    def generate_statistics(self, samples: List[Dict]) -> Dict:
        """Generate dataset statistics"""
        
        stats = {
            'total_samples': len(samples),
            'difficulty_distribution': {},
            'clue_distribution': {},
            'sample_prompts': []
        }
        
        # Count by difficulty
        for sample in samples:
            difficulty = sample['difficulty']
            stats['difficulty_distribution'][difficulty] = stats['difficulty_distribution'].get(difficulty, 0) + 1
        
        # Count by clue count
        for sample in samples:
            clues = sample['clue_count']
            stats['clue_distribution'][str(clues)] = stats['clue_distribution'].get(str(clues), 0) + 1
        
        # Sample prompts for inspection
        stats['sample_prompts'] = [
            {
                'id': sample['id'],
                'difficulty': sample['difficulty'],
                'clue_count': sample['clue_count'],
                'prompt_preview': sample['prompt'][:200] + "..." if len(sample['prompt']) > 200 else sample['prompt']
            }
            for sample in samples[:3]  # First 3 samples
        ]
        
        return stats
    
    def validate_dataset(self, samples: List[Dict]) -> bool:
        """Validate the prepared dataset"""
        
        print("\n🔍 Validating dataset...")
        
        # Check uniqueness
        hashes = [s['puzzle_hash'] for s in samples]
        if len(set(hashes)) != len(hashes):
            print("❌ Duplicate puzzles found!")
            return False
        
        # Check format consistency
        for i, sample in enumerate(samples):
            try:
                # Validate puzzle format
                if len(sample['puzzle']) != 81:
                    print(f"❌ Sample {i}: Invalid puzzle length")
                    return False
                
                # Validate solution format  
                if len(sample['solution']) != 81:
                    print(f"❌ Sample {i}: Invalid solution length")
                    return False
                
                # Check clue count consistency
                actual_clues = 81 - sample['puzzle'].count('0')
                if actual_clues != sample['clue_count']:
                    print(f"❌ Sample {i}: Clue count mismatch")
                    return False
                    
            except Exception as e:
                print(f"❌ Sample {i}: Validation error - {e}")
                return False
        
        print("✅ All validations passed!")
        return True

def main():
    """Main execution function"""
    
    print("🎲 Sudoku Test Dataset Preparation")
    print("=" * 50)
    
    # Initialize preparator
    preparator = SudokuTestDataPreparator(random_seed=42)
    
    # Download and prepare dataset
    try:
        preparator.download_kaggle_dataset()
        
        # Create 100 test samples
        print("\n🎯 Creating stratified test dataset...")
        samples = preparator.get_stratified_sample(total_samples=100)
        
        # Validate dataset
        if not preparator.validate_dataset(samples):
            print("❌ Dataset validation failed!")
            return
        
        # Save dataset
        rl_data_path = preparator.save_test_dataset(samples)
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Created 100 test puzzles across all difficulties")
        print(f"📁 RL training data ready at: {rl_data_path}")
        print(f"\n🚀 Next step: Run the RL training script!")
        
        # Show sample
        print(f"\n📋 Sample puzzle preview:")
        sample = samples[0]
        print(f"ID: {sample['id']}")
        print(f"Difficulty: {sample['difficulty']} ({sample['clue_count']} clues)")
        print(f"Grid:\n{sample['formatted_grid']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have internet connection and kagglehub installed:")
        print("   pip install kagglehub pandas numpy")

if __name__ == "__main__":
    main()