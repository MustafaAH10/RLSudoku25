import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import os
from collections import Counter
import random
import json
import kagglehub

class SudokuDataPreparator:
    def __init__(self, output_dir: str = "data", auto_download: bool = True):
        """
        Initialize the data preparator
        
        Args:
            output_dir: Directory to save processed datasets
            auto_download: Whether to automatically download dataset from Kaggle
        """
        self.output_dir = output_dir
        self.data = None
        self.input_file = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        if auto_download:
            self.download_dataset()
        
    def download_dataset(self):
        """Download the Sudoku dataset from Kaggle"""
        print("Downloading dataset from Kaggle...")
        try:
            # Download latest version
            path = kagglehub.dataset_download("informoney/4-million-sudoku-puzzles-easytohard")
            print(f"Dataset downloaded to: {path}")
            
            # Find the CSV file in the downloaded directory
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            
            if not csv_files:
                raise FileNotFoundError("No CSV files found in downloaded dataset")
            
            # Use the first CSV file found (should be the main dataset)
            self.input_file = os.path.join(path, csv_files[0])
            print(f"Using dataset file: {self.input_file}")
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please manually download the dataset and update the INPUT_FILE variable")
            raise
        
    def load_and_process_data(self):
        """Load the original dataset and add unique IDs"""
        print("Loading original dataset...")
        
        # Load the data - assuming column names are 'quizzes', 'solutions', 'clue_count'
        # Adjust column names based on actual CSV structure
        self.data = pd.read_csv(self.input_file)
        
        # Check column names and rename if necessary
        print(f"Original columns: {list(self.data.columns)}")
        
        # Rename columns to standard format if needed
        column_mapping = {}
        for col in self.data.columns:
            if 'quiz' in col.lower() or 'puzzle' in col.lower():
                column_mapping[col] = 'puzzle'
            elif 'solution' in col.lower():
                column_mapping[col] = 'solution'
            elif 'clue' in col.lower() or 'hint' in col.lower():
                column_mapping[col] = 'clue_count'
        
        if column_mapping:
            self.data = self.data.rename(columns=column_mapping)
            print(f"Renamed columns: {column_mapping}")
        
        # Add unique IDs
        self.data['id'] = range(len(self.data))
        
        # Remove any duplicates based on puzzle
        print(f"Original dataset size: {len(self.data)}")
        self.data = self.data.drop_duplicates(subset=['puzzle'], keep='first')
        print(f"After removing duplicates: {len(self.data)}")
        
        # Validate data format
        self.validate_data()
        
        print(f"Data loaded successfully. Shape: {self.data.shape}")
        print(f"Clue count distribution:")
        print(self.data['clue_count'].value_counts().sort_index())
        
    def validate_data(self):
        """Validate that the data is in correct format"""
        print("Validating data format...")
        
        # Check that puzzles and solutions are 81 characters
        invalid_puzzles = self.data[self.data['puzzle'].str.len() != 81]
        invalid_solutions = self.data[self.data['solution'].str.len() != 81]
        
        if len(invalid_puzzles) > 0:
            print(f"Warning: {len(invalid_puzzles)} puzzles are not 81 characters long")
            self.data = self.data[self.data['puzzle'].str.len() == 81]
        
        if len(invalid_solutions) > 0:
            print(f"Warning: {len(invalid_solutions)} solutions are not 81 characters long")
            self.data = self.data[self.data['solution'].str.len() == 81]
        
        # Check that puzzles contain only digits 0-9
        invalid_chars_puzzle = self.data[~self.data['puzzle'].str.match(r'^[0-9]{81}$')]
        invalid_chars_solution = self.data[~self.data['solution'].str.match(r'^[1-9]{81}$')]
        
        if len(invalid_chars_puzzle) > 0:
            print(f"Warning: {len(invalid_chars_puzzle)} puzzles contain invalid characters")
            self.data = self.data[self.data['puzzle'].str.match(r'^[0-9]{81}$')]
        
        if len(invalid_chars_solution) > 0:
            print(f"Warning: {len(invalid_chars_solution)} solutions contain invalid characters")
            self.data = self.data[self.data['solution'].str.match(r'^[1-9]{81}$')]
        
        print(f"Data validation complete. Final dataset size: {len(self.data)}")
    
    def get_difficulty_distribution(self, total_samples: int, min_clues: int = 17, max_clues: int = 80) -> Dict[int, int]:
        """
        Calculate how many samples to take from each difficulty level
        
        Args:
            total_samples: Total number of samples needed
            min_clues: Minimum number of clues (hardest)
            max_clues: Maximum number of clues (easiest)
        
        Returns:
            Dictionary mapping clue_count to number of samples
        """
        # Get available clue counts in our data
        available_clues = sorted(self.data['clue_count'].unique())
        available_clues = [c for c in available_clues if min_clues <= c <= max_clues]
        
        if not available_clues:
            raise ValueError(f"No puzzles found with clue counts between {min_clues} and {max_clues}")
        
        print(f"Available clue counts: {available_clues}")
        
        # Create a weighted distribution that gives more samples to middle difficulties
        # and ensures harder puzzles (fewer clues) are well represented
        distribution = {}
        
        # Inverse weighting: fewer clues = higher weight
        weights = {}
        for clue_count in available_clues:
            # Give higher weight to harder puzzles (fewer clues)
            weights[clue_count] = 1.0 / (clue_count - min_clues + 1)
        
        # Normalize weights
        total_weight = sum(weights.values())
        for clue_count in available_clues:
            weights[clue_count] /= total_weight
        
        # Calculate samples per difficulty
        remaining_samples = total_samples
        for i, clue_count in enumerate(available_clues):
            if i == len(available_clues) - 1:  # Last one gets remaining samples
                distribution[clue_count] = remaining_samples
            else:
                samples = max(1, int(total_samples * weights[clue_count]))
                distribution[clue_count] = min(samples, remaining_samples)
                remaining_samples -= samples
        
        # Ensure we don't exceed available puzzles for each difficulty
        for clue_count in distribution:
            available_count = len(self.data[self.data['clue_count'] == clue_count])
            if distribution[clue_count] > available_count:
                print(f"Warning: Requested {distribution[clue_count]} puzzles with {clue_count} clues, "
                      f"but only {available_count} available")
                distribution[clue_count] = available_count
        
        return distribution
    
    def sample_by_difficulty(self, distribution: Dict[int, int], exclude_ids: set = None) -> pd.DataFrame:
        """
        Sample puzzles according to difficulty distribution
        
        Args:
            distribution: Dictionary mapping clue_count to number of samples
            exclude_ids: Set of IDs to exclude from sampling
        
        Returns:
            DataFrame with sampled puzzles
        """
        if exclude_ids is None:
            exclude_ids = set()
        
        sampled_data = []
        
        for clue_count, num_samples in distribution.items():
            if num_samples <= 0:
                continue
                
            # Filter data for this difficulty level, excluding already used IDs
            difficulty_data = self.data[
                (self.data['clue_count'] == clue_count) & 
                (~self.data['id'].isin(exclude_ids))
            ]
            
            if len(difficulty_data) < num_samples:
                print(f"Warning: Only {len(difficulty_data)} puzzles available for "
                      f"{clue_count} clues, requested {num_samples}")
                num_samples = len(difficulty_data)
            
            # Sample without replacement
            sampled = difficulty_data.sample(n=num_samples, random_state=42)
            sampled_data.append(sampled)
            
            # Add sampled IDs to exclude set
            exclude_ids.update(sampled['id'].tolist())
        
        if not sampled_data:
            return pd.DataFrame()
        
        result = pd.concat(sampled_data, ignore_index=True)
        return result.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    def create_datasets(self):
        """Create all required datasets"""
        print("Creating datasets...")
        
        used_ids = set()
        datasets = {}
        
        # 1. CoT Training Set (1000 samples with varying difficulty)
        print("Creating CoT training set (1000 samples)...")
        cot_distribution = self.get_difficulty_distribution(1000)
        print(f"CoT training distribution: {cot_distribution}")
        
        cot_train = self.sample_by_difficulty(cot_distribution, used_ids)
        used_ids.update(cot_train['id'].tolist())
        datasets['cot_train'] = cot_train
        
        # 2. RL Training Set (5000 samples)
        print("Creating RL training set (5000 samples)...")
        rl_distribution = self.get_difficulty_distribution(5000)
        print(f"RL training distribution: {rl_distribution}")
        
        rl_train = self.sample_by_difficulty(rl_distribution, used_ids)
        used_ids.update(rl_train['id'].tolist())
        datasets['rl_train'] = rl_train
        
        # 3. Validation Set (1000 samples)
        print("Creating validation set (1000 samples)...")
        val_distribution = self.get_difficulty_distribution(1000)
        print(f"Validation distribution: {val_distribution}")
        
        val_set = self.sample_by_difficulty(val_distribution, used_ids)
        used_ids.update(val_set['id'].tolist())
        datasets['val_set'] = val_set
        
        # 4. Test Set (1000 samples)
        print("Creating test set (1000 samples)...")
        test_distribution = self.get_difficulty_distribution(1000)
        print(f"Test distribution: {test_distribution}")
        
        test_set = self.sample_by_difficulty(test_distribution, used_ids)
        used_ids.update(test_set['id'].tolist())
        datasets['test_set'] = test_set
        
        return datasets
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame]):
        """Save all datasets to CSV files"""
        print("Saving datasets...")
        
        file_mapping = {
            'cot_train': 'cot_training_set.csv',
            'rl_train': 'rl_training_set.csv',
            'val_set': 'validation_set.csv',
            'test_set': 'test_set.csv'
        }
        
        dataset_info = {}
        
        for dataset_name, df in datasets.items():
            filename = file_mapping[dataset_name]
            filepath = os.path.join(self.output_dir, filename)
            
            df.to_csv(filepath, index=False)
            
            # Collect statistics
            stats = {
                'total_samples': len(df),
                'difficulty_distribution': df['clue_count'].value_counts().to_dict(),
                'clue_range': {
                    'min': int(df['clue_count'].min()),
                    'max': int(df['clue_count'].max()),
                    'mean': float(df['clue_count'].mean())
                }
            }
            
            dataset_info[dataset_name] = {
                'filename': filename,
                'filepath': filepath,
                'stats': stats
            }
            
            print(f"Saved {dataset_name}: {filepath} ({len(df)} samples)")
        
        # Save dataset information
        info_file = os.path.join(self.output_dir, 'dataset_info.json')
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset information saved to: {info_file}")
        return dataset_info
    
    def verify_uniqueness(self, datasets: Dict[str, pd.DataFrame]):
        """Verify that all datasets contain unique puzzles"""
        print("Verifying dataset uniqueness...")
        
        all_ids = set()
        overlaps = {}
        
        for name, df in datasets.items():
            current_ids = set(df['id'].tolist())
            
            # Check for overlaps with previously processed datasets
            overlap_count = len(current_ids.intersection(all_ids))
            if overlap_count > 0:
                overlaps[name] = overlap_count
                print(f"ERROR: {name} has {overlap_count} overlapping puzzles!")
            else:
                print(f"✓ {name} has no overlaps ({len(current_ids)} unique puzzles)")
            
            all_ids.update(current_ids)
        
        if not overlaps:
            print("✓ All datasets are unique!")
        else:
            raise ValueError(f"Dataset overlaps detected: {overlaps}")
        
        return len(overlaps) == 0
    
    def print_summary(self, datasets: Dict[str, pd.DataFrame]):
        """Print a summary of created datasets"""
        print("\n" + "="*60)
        print("DATASET CREATION SUMMARY")
        print("="*60)
        
        total_samples = sum(len(df) for df in datasets.values())
        print(f"Total samples created: {total_samples}")
        print()
        
        for name, df in datasets.items():
            print(f"{name.upper()}:")
            print(f"  Samples: {len(df)}")
            print(f"  Clue range: {df['clue_count'].min()} - {df['clue_count'].max()}")
            print(f"  Mean clues: {df['clue_count'].mean():.1f}")
            
            # Show difficulty distribution
            dist = df['clue_count'].value_counts().sort_index()
            print("  Difficulty distribution:")
            for clues, count in dist.items():
                difficulty = "Very Hard" if clues < 25 else "Hard" if clues < 35 else "Medium" if clues < 50 else "Easy"
                print(f"    {clues} clues: {count} puzzles ({difficulty})")
            print()

def main():
    # Configuration
    OUTPUT_DIR = "data"
    
    # Initialize data preparator (will auto-download dataset)
    preparator = SudokuDataPreparator(OUTPUT_DIR, auto_download=True)
    
    # Load and process data
    preparator.load_and_process_data()
    
    # Create datasets
    datasets = preparator.create_datasets()
    
    # Verify uniqueness
    preparator.verify_uniqueness(datasets)
    
    # Save datasets
    dataset_info = preparator.save_datasets(datasets)
    
    # Print summary
    preparator.print_summary(datasets)
    
    print("Data preparation complete!")
    print(f"All files saved in '{OUTPUT_DIR}' directory")

if __name__ == "__main__":
    main()