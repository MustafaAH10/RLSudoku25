#!/usr/bin/env python3
"""
Create balanced 2000 sample dataset with 70/15/15 split and equal difficulty distribution.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

def string_to_grid(s):
    """Convert sudoku string to 9x9 grid"""
    return [[int(c) for c in s[i*9:(i+1)*9]] for i in range(9)]

def create_balanced_splits():
    """Create balanced 2000 sample dataset with 70/15/15 split and equal difficulty distribution."""
    print("Creating balanced splits with 2000 samples (70/15/15 split)...")
    
    # Read the CSV data
    df = pd.read_csv('/home/mustafaah/RLSudoku25/remote/shared_data/raw/sudoku_puzzles.csv')
    print(f"Loaded {len(df)} puzzles from CSV")
    
    # We have 64 difficulty levels (17-80 clues), each with 62,500 samples
    # For 2000 total samples: ~31 samples per difficulty level
    difficulty_levels = sorted(df['clue_numbers'].unique())
    samples_per_level = 2000 // len(difficulty_levels)  # 31 samples per level
    
    print(f"Found {len(difficulty_levels)} difficulty levels (clues {min(difficulty_levels)}-{max(difficulty_levels)})")
    print(f"Sampling {samples_per_level} puzzles per difficulty level")
    
    # Calculate split sizes per difficulty level
    train_per_level = int(samples_per_level * 0.7)  # ~22 samples  
    val_per_level = int(samples_per_level * 0.15)   # ~5 samples
    test_per_level = samples_per_level - train_per_level - val_per_level  # ~4 samples
    
    print(f"Per difficulty level: {train_per_level} train, {val_per_level} val, {test_per_level} test")
    
    train_data = []
    test_data = []
    val_data = []
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for difficulty in difficulty_levels:
        # Get samples for this difficulty level
        difficulty_samples = df[df['clue_numbers'] == difficulty]
        
        # Randomly sample the required number
        sampled = difficulty_samples.sample(n=samples_per_level, random_state=42)
        
        # Convert to the required format
        converted_samples = []
        for _, row in sampled.iterrows():
            puzzle_grid = string_to_grid(row['quizzes'])
            solution_grid = string_to_grid(row['solutions'])
            
            # Convert solution to dictionary format expected by experiments
            solution_dict = {}
            empty_cells = []
            for i in range(9):
                for j in range(9):
                    if puzzle_grid[i][j] == 0:
                        cell_key = f"R{i+1}C{j+1}"
                        solution_dict[cell_key] = solution_grid[i][j]
                        empty_cells.append(cell_key)
            
            # Calculate difficulty based on clue count
            clue_count = int(row['clue_numbers'])
            if clue_count >= 65:
                difficulty = "easy"
            elif clue_count >= 50:
                difficulty = "medium"
            elif clue_count >= 35:
                difficulty = "hard"
            else:
                difficulty = "expert"
            
            difficulty_score = 1.0 - (clue_count / 81.0)
            
            converted_samples.append({
                'puzzle': puzzle_grid,
                'solution': solution_dict,
                'difficulty': difficulty,
                'clue_count': clue_count,
                'empty_cells': empty_cells,
                'difficulty_score': difficulty_score
            })
        
        # Split into train/val/test
        train_data.extend(converted_samples[:train_per_level])
        val_data.extend(converted_samples[train_per_level:train_per_level + val_per_level])
        test_data.extend(converted_samples[train_per_level + val_per_level:])
    
    # Shuffle the final datasets
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    np.random.shuffle(val_data)
    
    # Save the splits
    splits_dir = Path('/home/mustafaah/RLSudoku25/remote/shared_data/splits')
    splits_dir.mkdir(exist_ok=True)
    
    with open(splits_dir / 'train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(splits_dir / 'test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    with open(splits_dir / 'val.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Created balanced splits:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Total: {len(train_data) + len(test_data) + len(val_data)} samples")
    
    # Verify difficulty distribution
    print("\nDifficulty distribution verification:")
    for split_name, split_data in [('Train', train_data), ('Test', test_data), ('Val', val_data)]:
        difficulties = [sample['clue_count'] for sample in split_data]
        unique_difficulties = sorted(set(difficulties))
        min_clues = min(difficulties)
        max_clues = max(difficulties)
        avg_clues = sum(difficulties) / len(difficulties)
        print(f"  {split_name}: {len(unique_difficulties)} difficulty levels, clues {min_clues}-{max_clues}, avg {avg_clues:.1f}")
    
    print(f"\nSaved splits to {splits_dir}/")

if __name__ == "__main__":
    create_balanced_splits()