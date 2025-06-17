#!/usr/bin/env python3
"""
Fixed Data Preparation using Kagglehub
=====================================
Downloads 4M Sudoku dataset and creates train/val/test splits
"""

import pandas as pd
import numpy as np
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import kagglehub

class SudokuDataPreparator:
    """Prepare Sudoku datasets from Kaggle with proper splits"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.df = None
        self.used_hashes = set()
        
    def download_kaggle_dataset(self) -> pd.DataFrame:
        """Download Kaggle Sudoku dataset"""
        print("🔄 Downloading Kaggle Sudoku dataset...")
        
        try:
            # Download dataset
            path = kagglehub.dataset_download("informoney/4-million-sudoku-puzzles-easytohard")
            print(f"✅ Dataset downloaded to: {path}")
            
            # Find CSV file
            csv_files = list(Path(path).glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found")
                
            csv_path = csv_files[0]
            print(f"📁 Loading: {csv_path}")
            
            # Load dataset
            self.df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(self.df):,} puzzles")
            
            # Add hash for uniqueness
            self.df['puzzle_hash'] = self.df['quizzes'].apply(
                lambda x: hashlib.md5(x.encode()).hexdigest()[:12]
            )
            
            # Show distribution
            print(f"📊 Clue distribution: {self.df['clue_numbers'].min()}-{self.df['clue_numbers'].max()} clues")
            
            return self.df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def quiz_to_grid(self, quiz_string: str) -> List[List[int]]:
        """Convert quiz string to 9x9 grid"""
        grid = []
        for i in range(9):
            row = []
            for j in range(9):
                digit = int(quiz_string[i * 9 + j])
                row.append(digit)
            grid.append(row)
        return grid
    
    def solution_to_dict(self, puzzle_string: str, solution_string: str) -> Dict[str, int]:
        """Convert solution to dictionary of empty cells"""
        solution_dict = {}
        
        for i in range(81):
            if puzzle_string[i] == '0':  # Empty cell
                row = (i // 9) + 1  # 1-indexed
                col = (i % 9) + 1   # 1-indexed
                digit = int(solution_string[i])
                solution_dict[f"R{row}C{col}"] = digit
        
        return solution_dict
    
    def assign_difficulty(self, clue_count: int) -> str:
        """Assign difficulty based on clue count"""
        if clue_count >= 60:
            return "beginner"
        elif clue_count >= 50:
            return "easy"
        elif clue_count >= 35:
            return "medium"
        elif clue_count >= 25:
            return "hard"
        else:
            return "expert"
    
    def create_dataset_splits(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create train/val/test splits with stratified sampling"""
        
        if self.df is None:
            self.download_kaggle_dataset()
        
        print("\n🎯 Creating stratified dataset splits...")
        
        # Target sizes
        TRAIN_SIZE = 5000  # More training data for RL
        VAL_SIZE = 1000    # Validation during training
        TEST_SIZE = 1000   # Final evaluation
        
        print(f"📊 Target sizes:")
        print(f"   Training: {TRAIN_SIZE}")
        print(f"   Validation: {VAL_SIZE}")
        print(f"   Test: {TEST_SIZE}")
        
        # Get samples for each split
        train_samples = self.get_stratified_sample(TRAIN_SIZE, "train")
        val_samples = self.get_stratified_sample(VAL_SIZE, "val")
        test_samples = self.get_stratified_sample(TEST_SIZE, "test")
        
        return train_samples, val_samples, test_samples
    
    def get_stratified_sample(self, target_size: int, split_name: str) -> List[Dict]:
        """Get stratified sample by difficulty"""
        
        # Difficulty distribution (balanced)
        difficulty_ratios = {
            'expert': 0.20,    
            'hard': 0.20,      
            'medium': 0.20,    
            'easy': 0.20,      
            'beginner': 0.20   
        }
        
        all_samples = []
        
        for difficulty, ratio in difficulty_ratios.items():
            target_count = int(target_size * ratio)
            
            # Define clue ranges for each difficulty
            clue_ranges = {
                'expert': (17, 30),
                'hard': (31, 44),
                'medium': (45, 59),
                'easy': (60, 69),
                'beginner': (70, 80)
            }
            
            min_clues, max_clues = clue_ranges[difficulty]
            
            print(f"🎯 Sampling {target_count} {difficulty} puzzles for {split_name} ({min_clues}-{max_clues} clues)")
            
            # Filter by clue range and exclude used puzzles
            mask = (
                (self.df['clue_numbers'] >= min_clues) & 
                (self.df['clue_numbers'] <= max_clues) &
                (~self.df['puzzle_hash'].isin(self.used_hashes))
            )
            available_df = self.df[mask]
            
            if len(available_df) < target_count:
                print(f"⚠️  Only {len(available_df)} available, adjusting target")
                target_count = len(available_df)
            
            # Sample
            sampled = available_df.sample(n=target_count, random_state=self.random_seed + len(all_samples))
            
            # Convert to our format
            for idx, row in sampled.iterrows():
                puzzle_grid = self.quiz_to_grid(row['quizzes'])
                solution_dict = self.solution_to_dict(row['quizzes'], row['solutions'])
                
                sample = {
                    'id': f"{split_name}_{difficulty}_{len(all_samples)+1:03d}",
                    'difficulty': difficulty,
                    'puzzle': puzzle_grid,
                    'solution': solution_dict,
                    'clue_count': int(row['clue_numbers']),
                    'empty_count': 81 - int(row['clue_numbers']),
                    'raw_puzzle': row['quizzes'],
                    'raw_solution': row['solutions']
                }
                
                all_samples.append(sample)
                self.used_hashes.add(row['puzzle_hash'])
            
            print(f"✅ Added {len(sampled)} {difficulty} samples")
        
        # Shuffle and adjust if needed
        random.shuffle(all_samples)
        
        # Adjust to exact target size if needed
        if len(all_samples) != target_size:
            print(f"🔧 Adjusting from {len(all_samples)} to {target_size} samples")
            all_samples = all_samples[:target_size]
        
        print(f"✅ Created {len(all_samples)} samples for {split_name}")
        return all_samples
    
    def save_datasets(self, train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]):
        """Save all datasets"""
        
        # Create directories
        for dir_name in ['training_data', 'test_data']:
            Path(dir_name).mkdir(exist_ok=True)
        
        # Save files
        datasets = [
            (train_samples, 'training_data/sudoku_rl_train.json'),
            (val_samples, 'training_data/sudoku_rl_val.json'),  # Val goes with training
            (test_samples, 'test_data/sudoku_rl_test.json')
        ]
        
        for samples, filepath in datasets:
            print(f"💾 Saving {len(samples)} samples to {filepath}")
            with open(filepath, 'w') as f:
                json.dump(samples, f, indent=2)
        
        # Create legacy format for compatibility
        with open('train_data.json', 'w') as f:
            json.dump(train_samples, f, indent=2)
        with open('val_data.json', 'w') as f:
            json.dump(val_samples, f, indent=2)
        
        # Save statistics
        self.save_statistics(train_samples, val_samples, test_samples)
        
        print(f"\n✅ All datasets saved!")
        print(f"📁 Training: training_data/sudoku_rl_train.json ({len(train_samples)} samples)")
        print(f"📁 Validation: training_data/sudoku_rl_val.json ({len(val_samples)} samples)")
        print(f"📁 Test: test_data/sudoku_rl_test.json ({len(test_samples)} samples)")
    
    def save_statistics(self, train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]):
        """Save dataset statistics"""
        
        def get_difficulty_stats(samples):
            stats = {}
            for difficulty in ['expert', 'hard', 'medium', 'easy', 'beginner']:
                count = len([s for s in samples if s['difficulty'] == difficulty])
                stats[difficulty] = count
            return stats
        
        statistics = {
            'total_samples': len(train_samples) + len(val_samples) + len(test_samples),
            'splits': {
                'train': {
                    'count': len(train_samples),
                    'difficulty_distribution': get_difficulty_stats(train_samples)
                },
                'val': {
                    'count': len(val_samples),
                    'difficulty_distribution': get_difficulty_stats(val_samples)
                },
                'test': {
                    'count': len(test_samples),
                    'difficulty_distribution': get_difficulty_stats(test_samples)
                }
            },
            'uniqueness_verified': True
        }
        
        # Save to both directories
        for directory in ['training_data', 'test_data']:
            stats_path = Path(directory) / 'dataset_statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(statistics, f, indent=2)
        
        # Print summary
        print(f"\n📊 Dataset Statistics:")
        for split_name, split_stats in statistics['splits'].items():
            print(f"   {split_name.capitalize()}: {split_stats['count']} samples")
            for diff, count in split_stats['difficulty_distribution'].items():
                print(f"     {diff}: {count}")

def main():
    """Main function"""
    print("🎲 Sudoku Dataset Preparation with Kagglehub")
    print("=" * 50)
    
    try:
        preparator = SudokuDataPreparator(random_seed=42)
        
        # Download and prepare
        preparator.download_kaggle_dataset()
        
        # Create splits
        train_samples, val_samples, test_samples = preparator.create_dataset_splits()
        
        # Save datasets
        preparator.save_datasets(train_samples, val_samples, test_samples)
        
        print(f"\n🎉 SUCCESS! Ready for RL training")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have: pip install kagglehub pandas numpy")

if __name__ == "__main__":
    main()