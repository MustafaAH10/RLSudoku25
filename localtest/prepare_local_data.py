#!/usr/bin/env python3
"""
Enhanced Dataset Preparation: 100 Training + 100 Test Samples
============================================================

This script creates TWO separate datasets:
1. 100 samples for RL training (training_data/)
2. 100 samples for benchmarking (test_data/) 
All 200 samples are guaranteed to be unique.
"""

import pandas as pd
import numpy as np
import json
import random
from pathlib import Path
import hashlib
from typing import List, Dict, Tuple
import kagglehub

class EnhancedSudokuDataPreparator:
    """Prepare training and test datasets with guaranteed uniqueness"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.df = None
        self.all_used_hashes = set()  # Track ALL used puzzles across both sets
        
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
    
    def format_sudoku_for_qwen3(self, quiz_string: str) -> str:
        """Format Sudoku puzzle optimized for Qwen3"""
        if len(quiz_string) != 81:
            raise ValueError(f"Invalid puzzle length: {len(quiz_string)}")
        
        # Convert to visually clear grid with Unicode box drawing
        grid_lines = []
        for row in range(9):
            line_chars = []
            for col in range(9):
                char = quiz_string[row * 9 + col]
                line_chars.append('_' if char == '0' else char)
            
            # Create visually appealing 3x3 blocks
            line = f"{' '.join(line_chars[0:3])} │ {' '.join(line_chars[3:6])} │ {' '.join(line_chars[6:9])}"
            grid_lines.append(line)
            
            # Add horizontal separators after rows 2 and 5
            if row == 2 or row == 5:
                grid_lines.append("──────┼───────┼──────")
        
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
    
    def create_qwen3_messages(self, puzzle: str, clue_count: int) -> List[Dict]:
        """Create Qwen3-optimized message format"""
        
        formatted_grid = self.format_sudoku_for_qwen3(puzzle)
        empty_positions = self.find_empty_positions(puzzle)
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert Sudoku solver. Use step-by-step logical reasoning to solve puzzles systematically. Think through constraints carefully."
            },
            {
                "role": "user",
                "content": f"""Solve this Sudoku puzzle with {clue_count} given clues.

PUZZLE:
{formatted_grid}

EMPTY CELLS TO FILL: {', '.join(empty_positions)}

Rules: Each row, column, and 3×3 box must contain digits 1-9 exactly once.

Provide your solution in this EXACT format:
SOLUTION:
R1C1: digit
R2C3: digit
...

Think step by step about constraints and logical deductions."""
            }
        ]
        
        return messages
    
    def create_target_response(self, puzzle: str, solution: str) -> str:
        """Create the target response showing correct solutions"""
        
        solution_lines = []
        
        for i in range(81):
            if puzzle[i] == '0':
                row = (i // 9) + 1
                col = (i % 9) + 1
                digit = solution[i]
                solution_lines.append(f"R{row}C{col}: {digit}")
        
        return '\n'.join(solution_lines)
    
    def get_stratified_sample(self, total_samples: int, dataset_name: str) -> List[Dict]:
        """Get stratified sample for either training or test set"""
        
        if self.df is None:
            print("❌ Dataset not loaded. Loading now...")
            self.download_kaggle_dataset()
            
        if self.df is None:
            raise ValueError("Could not load dataset")
        
        # Define difficulty distribution (same for both sets)
        difficulty_targets = {
            'expert': (17, 25, int(total_samples * 0.10)),    # 10% expert
            'hard': (26, 35, int(total_samples * 0.20)),      # 20% hard
            'medium': (36, 50, int(total_samples * 0.30)),    # 30% medium
            'easy': (51, 65, int(total_samples * 0.25)),      # 25% easy
            'beginner': (66, 80, int(total_samples * 0.15))   # 15% beginner
        }
        
        # Adjust for rounding
        total_allocated = sum(count for _, _, count in difficulty_targets.values())
        if total_allocated < total_samples:
            difficulty_targets['medium'] = (
                difficulty_targets['medium'][0],
                difficulty_targets['medium'][1], 
                difficulty_targets['medium'][2] + (total_samples - total_allocated)
            )
        
        all_samples = []
        
        for difficulty, (min_clues, max_clues, target_count) in difficulty_targets.items():
            print(f"🎯 Sampling {target_count} {difficulty} puzzles for {dataset_name} ({min_clues}-{max_clues} clues)...")
            
            # Filter by clue range
            mask = (self.df['clue_numbers'] >= min_clues) & (self.df['clue_numbers'] <= max_clues)
            difficulty_df = self.df[mask]
            
            # Remove already used puzzles across ALL datasets
            difficulty_df = difficulty_df[~difficulty_df['puzzle_hash'].isin(self.all_used_hashes)]
            
            if len(difficulty_df) < target_count:
                print(f"⚠️  Only {len(difficulty_df)} available, need {target_count}")
                target_count = len(difficulty_df)
            
            # Sample without replacement
            sampled = difficulty_df.sample(n=target_count, random_state=self.random_seed + len(all_samples))
            
            # Process samples
            for idx, row in sampled.iterrows():
                sample = {
                    'id': f"{dataset_name}_{difficulty}_{len(all_samples)+1:03d}",
                    'puzzle_hash': row['puzzle_hash'],
                    'difficulty': difficulty,
                    'puzzle': row['quizzes'],
                    'solution': row['solutions'],
                    'clue_count': int(row['clue_numbers']),
                    'empty_count': 81 - int(row['clue_numbers']),
                    'messages': self.create_qwen3_messages(row['quizzes'], int(row['clue_numbers'])),
                    'target_response': self.create_target_response(row['quizzes'], row['solutions']),
                    'formatted_grid': self.format_sudoku_for_qwen3(row['quizzes'])
                }
                
                all_samples.append(sample)
                self.all_used_hashes.add(row['puzzle_hash'])  # Track globally
            
            print(f"✅ Added {len(sampled)} {difficulty} samples to {dataset_name}")
        
        # Shuffle final samples
        random.shuffle(all_samples)
        print(f"✅ Created {len(all_samples)} total samples for {dataset_name}")
        
        return all_samples
    
    def create_both_datasets(self) -> Tuple[List[Dict], List[Dict]]:
        """Create both training and test datasets ensuring uniqueness"""
        
        print("🎯 Creating training and test datasets...")
        print("=" * 60)
        
        # Create training dataset first
        print("\n📚 CREATING TRAINING DATASET (100 samples)")
        train_samples = self.get_stratified_sample(100, "train")
        
        print(f"\n📊 CREATING TEST DATASET (100 samples)")
        test_samples = self.get_stratified_sample(100, "test")
        
        # Verify no overlap
        train_hashes = {s['puzzle_hash'] for s in train_samples}
        test_hashes = {s['puzzle_hash'] for s in test_samples}
        overlap = train_hashes & test_hashes
        
        if overlap:
            raise ValueError(f"❌ Found {len(overlap)} overlapping puzzles!")
        
        print(f"\n✅ UNIQUENESS VERIFIED:")
        print(f"   Training puzzles: {len(train_samples)}")
        print(f"   Test puzzles: {len(test_samples)}")
        print(f"   Total unique puzzles: {len(train_hashes | test_hashes)}")
        print(f"   No overlaps: ✅")
        
        return train_samples, test_samples
    
    def save_datasets(self, train_samples: List[Dict], test_samples: List[Dict]):
        """Save both datasets to separate directories"""
        
        # Create directories
        train_dir = Path("training_data")
        test_dir = Path("test_data")
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # Save training dataset
        print(f"\n💾 Saving training dataset...")
        train_path = train_dir / "sudoku_rl_train.json"
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, indent=2, ensure_ascii=False)
        
        train_csv_path = train_dir / "sudoku_rl_train.csv"
        train_df_data = [{
            'id': s['id'], 'difficulty': s['difficulty'], 'clue_count': s['clue_count'],
            'puzzle': s['puzzle'], 'solution': s['solution']
        } for s in train_samples]
        pd.DataFrame(train_df_data).to_csv(train_csv_path, index=False)
        
        # Save test dataset
        print(f"💾 Saving test dataset...")
        test_path = test_dir / "sudoku_rl_test.json"
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_samples, f, indent=2, ensure_ascii=False)
        
        test_csv_path = test_dir / "sudoku_rl_test.csv"
        test_df_data = [{
            'id': s['id'], 'difficulty': s['difficulty'], 'clue_count': s['clue_count'],
            'puzzle': s['puzzle'], 'solution': s['solution']
        } for s in test_samples]
        pd.DataFrame(test_df_data).to_csv(test_csv_path, index=False)
        
        # Save statistics
        self.save_dataset_stats(train_samples, test_samples)
        
        print(f"\n✅ DATASETS SAVED:")
        print(f"   Training: {train_path}")
        print(f"   Training CSV: {train_csv_path}")
        print(f"   Test: {test_path}")
        print(f"   Test CSV: {test_csv_path}")
    
    def save_dataset_stats(self, train_samples: List[Dict], test_samples: List[Dict]):
        """Save comprehensive statistics for both datasets"""
        
        def get_stats(samples, name):
            return {
                'name': name,
                'total_samples': len(samples),
                'difficulty_distribution': {
                    d: len([s for s in samples if s['difficulty'] == d])
                    for d in ['expert', 'hard', 'medium', 'easy', 'beginner']
                },
                'clue_stats': {
                    'min': min(s['clue_count'] for s in samples),
                    'max': max(s['clue_count'] for s in samples),
                    'mean': sum(s['clue_count'] for s in samples) / len(samples),
                },
                'sample_puzzles': [
                    {
                        'id': s['id'],
                        'difficulty': s['difficulty'], 
                        'clue_count': s['clue_count'],
                        'grid_preview': s['formatted_grid'][:100] + "..."
                    }
                    for s in samples[:2]  # First 2 samples as examples
                ]
            }
        
        stats = {
            'creation_date': pd.Timestamp.now().isoformat(),
            'total_unique_puzzles': len(train_samples) + len(test_samples),
            'training_set': get_stats(train_samples, 'training'),
            'test_set': get_stats(test_samples, 'test'),
            'uniqueness_verified': True
        }
        
        # Save to both directories
        for directory in [Path("training_data"), Path("test_data")]:
            stats_path = directory / "dataset_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
        
        print(f"📈 Statistics saved to both directories")

def main():
    """Main execution function"""
    
    print("🎲 Enhanced Sudoku Dataset Preparation")
    print("=" * 60)
    print("Creating 200 unique puzzles:")
    print("• 100 for RL training (training_data/)")
    print("• 100 for benchmarking (test_data/)")
    print("=" * 60)
    
    # Initialize preparator
    preparator = EnhancedSudokuDataPreparator(random_seed=42)
    
    try:
        # Download dataset
        preparator.download_kaggle_dataset()
        
        # Create both datasets
        train_samples, test_samples = preparator.create_both_datasets()
        
        # Save datasets
        preparator.save_datasets(train_samples, test_samples)
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Created 200 unique Sudoku puzzles")
        print(f"📁 Training data: training_data/sudoku_rl_train.json")
        print(f"📁 Test data: test_data/sudoku_rl_test.json")
        print(f"\n🚀 Next steps:")
        print(f"1. Run benchmarking: python benchmark_qwen3.py")
        print(f"2. Run RL training: python qwen3_rl_trainer.py")
        print(f"3. Run post-RL benchmarking to compare results")
        
        # Show samples from both sets
        print(f"\n📋 Sample from training set:")
        train_sample = train_samples[0]
        print(f"   ID: {train_sample['id']}")
        print(f"   Difficulty: {train_sample['difficulty']} ({train_sample['clue_count']} clues)")
        
        print(f"\n📋 Sample from test set:")
        test_sample = test_samples[0]
        print(f"   ID: {test_sample['id']}")
        print(f"   Difficulty: {test_sample['difficulty']} ({test_sample['clue_count']} clues)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have internet connection and required packages:")
        print("   pip install kagglehub pandas numpy")

if __name__ == "__main__":
    main()