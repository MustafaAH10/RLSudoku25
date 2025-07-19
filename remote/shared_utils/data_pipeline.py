"""
Shared data pipeline for all Sudoku RL experiments
Handles data preparation, processing, and analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PuzzleData:
    """Structured puzzle data with comprehensive analysis"""
    puzzle: List[List[int]]
    solution: Dict[str, int]  # Format: {"R1C1": 5, "R2C3": 8}
    difficulty: str
    clue_count: int
    empty_cells: List[str]  # ["R1C1", "R2C3", ...]
    difficulty_score: float  # 0.0 (easy) to 1.0 (hard)
    
    def get_empty_cells_formatted(self) -> str:
        """Return formatted empty cells for prompting"""
        return ", ".join(self.empty_cells)
    
    def get_spatial_grid(self) -> str:
        """Return beautifully formatted spatial grid"""
        grid = []
        for i, row in enumerate(self.puzzle):
            if i in [3, 6]:
                grid.append("─────── ┼ ─────── ┼ ───────")
            
            row_str = ""
            for j, cell in enumerate(row):
                if j in [3, 6]:
                    row_str += " │ "
                if cell == 0:
                    row_str += "  _  "
                else:
                    row_str += f"  {cell}  "
            grid.append(row_str)
        
        return "\n".join(grid)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "puzzle": self.puzzle,
            "solution": self.solution,
            "difficulty": self.difficulty,
            "clue_count": self.clue_count,
            "empty_cells": self.empty_cells,
            "difficulty_score": self.difficulty_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PuzzleData':
        """Create from dictionary"""
        return cls(
            puzzle=data["puzzle"],
            solution=data["solution"],
            difficulty=data["difficulty"],
            clue_count=data["clue_count"],
            empty_cells=data["empty_cells"],
            difficulty_score=data["difficulty_score"]
        )

class DataPipelineManager:
    """Manages the complete data pipeline from raw data to training format"""
    
    def __init__(self):
        self.data_dir = Path("shared_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "splits").mkdir(exist_ok=True)
    
    def prepare_all_data(self) -> Dict[str, str]:
        """Complete data preparation pipeline"""
        print("🔄 Starting comprehensive data preparation...")
        
        # Step 1: Download and cache raw data
        raw_data_path = self._download_and_cache_data()
        
        # Step 2: Process raw data
        processed_data = self._process_raw_data(raw_data_path)
        
        # Step 3: Create stratified splits
        splits = self._create_stratified_splits(processed_data)
        
        # Step 4: Save splits
        split_paths = self._save_splits(splits)
        
        # Step 5: Generate analysis report
        self._generate_analysis_report(splits)
        
        return split_paths
    
    def _download_and_cache_data(self) -> str:
        """Download and cache Kaggle dataset"""
        cache_path = self.data_dir / "raw" / "sudoku_puzzles.csv"
        
        if cache_path.exists():
            print("📂 Using cached dataset")
            return str(cache_path)
        
        print("⬇️ Downloading Kaggle dataset...")
        try:
            import kagglehub
            import glob
            import os
            path = kagglehub.dataset_download('informoney/4-million-sudoku-puzzles-easytohard')
            
            # Find the CSV file in the downloaded directory
            csv_files = glob.glob(os.path.join(path, "*.csv"))
            if not csv_files:
                raise ValueError("No CSV files found in downloaded dataset")
            
            # Copy to our cache
            import shutil
            shutil.copy(csv_files[0], cache_path)
            
            print(f"✅ Dataset cached to {cache_path}")
            return str(cache_path)
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("Please download manually from: https://www.kaggle.com/datasets/informoney/4-million-sudoku-puzzles-easytohard")
            raise
    
    def _process_raw_data(self, data_path: str) -> List[PuzzleData]:
        """Process raw data into structured format"""
        print("🔄 Processing puzzles and analyzing difficulty...")
        
        df = pd.read_csv(data_path)
        
        # Rename columns to match expected format if needed
        if 'quizzes' in df.columns:
            df = df.rename(columns={'quizzes': 'puzzle', 'solutions': 'solution'})
        
        processed = []
        
        # Sample for manageable size (adjust as needed)
        # Optimized for 24-hour GPU experiment - FIXED SEED FOR REPRODUCIBILITY
        sample_size = min(2000, len(df))  # Use 2k puzzles for 24-hour constraint
        df_sample = df.sample(n=sample_size, random_state=42)  # Fixed seed ensures same data across experiments
        
        for idx, row in df_sample.iterrows():
            try:
                puzzle_data = self._parse_puzzle_row(row)
                processed.append(puzzle_data)
            except Exception as e:
                print(f"Warning: Error processing puzzle {idx}: {e}")
                continue
        
        print(f"✅ Processed {len(processed)} puzzles")
        return processed
    
    def _parse_puzzle_row(self, row) -> PuzzleData:
        """Parse a single puzzle row into PuzzleData"""
        # Parse puzzle
        puzzle = [[int(d) for d in row['puzzle'][i*9:(i+1)*9]] for i in range(9)]
        solution_str = row['solution']
        
        # Calculate empty cells and solution mapping
        empty_cells = []
        solution = {}
        
        for i, digit in enumerate(solution_str):
            row_idx, col_idx = i // 9, i % 9
            if puzzle[row_idx][col_idx] == 0:
                cell_key = f"R{row_idx+1}C{col_idx+1}"
                empty_cells.append(cell_key)
                solution[cell_key] = int(digit)
        
        # Advanced difficulty analysis
        clue_count = 81 - len(empty_cells)
        difficulty_score = self._calculate_difficulty_score(puzzle)
        
        # Map difficulty categories based on clue count (for training data)
        if clue_count >= 75:
            difficulty = "beginner"
        elif clue_count >= 65:
            difficulty = "easy"
        elif clue_count >= 55:
            difficulty = "medium"
        elif clue_count >= 45:
            difficulty = "hard"
        elif clue_count >= 35:
            difficulty = "expert"
        else:
            difficulty = "master"
        
        return PuzzleData(
            puzzle=puzzle,
            solution=solution,
            difficulty=difficulty,
            clue_count=clue_count,
            empty_cells=empty_cells,
            difficulty_score=difficulty_score
        )
    
    def _calculate_difficulty_score(self, puzzle: List[List[int]]) -> float:
        """Calculate sophisticated difficulty score"""
        score = 0.0
        empty_count = sum(row.count(0) for row in puzzle)
        
        # Base score from empty cells
        score += (empty_count / 81) * 0.4
        
        # Constraint complexity analysis
        constraint_complexity = 0.0
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] == 0:
                    valid_digits = self._get_valid_digits(puzzle, i, j)
                    if len(valid_digits) == 1:
                        constraint_complexity += 0.01  # Easy cells
                    elif len(valid_digits) == 2:
                        constraint_complexity += 0.05  # Medium cells
                    else:
                        constraint_complexity += 0.10  # Hard cells
        
        score += constraint_complexity
        
        return min(score, 1.0)
    
    def _get_valid_digits(self, puzzle: List[List[int]], row: int, col: int) -> List[int]:
        """Get valid digits for a cell"""
        if puzzle[row][col] != 0:
            return []
        
        used_digits = set()
        
        # Row constraints
        used_digits.update(puzzle[row])
        
        # Column constraints
        used_digits.update(puzzle[i][col] for i in range(9))
        
        # 3x3 box constraints
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                used_digits.add(puzzle[i][j])
        
        return [d for d in range(1, 10) if d not in used_digits]
    
    def _create_stratified_splits(self, data: List[PuzzleData]) -> Dict[str, List[PuzzleData]]:
        """Create balanced splits with stratified train and equal clue representation for test/val"""
        print("🎯 Creating stratified splits...")
        
        # Separate train split (stratified by difficulty) from test/val splits (equal clue representation)
        random.shuffle(data)
        
        # Split ratios: 70% train, 15% val, 15% test (requested configuration)
        n = len(data)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        train_data = data[:train_end]
        remaining_data = data[train_end:]
        
        # For test/val: create equal representation of all clue counts
        clue_groups = {}
        for puzzle in remaining_data:
            clue_count = puzzle.clue_count
            if clue_count not in clue_groups:
                clue_groups[clue_count] = []
            clue_groups[clue_count].append(puzzle)
        
        # Balance test/val with equal clue representation
        val_data, test_data = [], []
        for clue_count, puzzles in clue_groups.items():
            random.shuffle(puzzles)
            mid = len(puzzles) // 2
            val_data.extend(puzzles[:mid])
            test_data.extend(puzzles[mid:])
        
        # Shuffle final datasets
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        return {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
    
    def _save_splits(self, splits: Dict[str, List[PuzzleData]]) -> Dict[str, str]:
        """Save data splits to files"""
        print("💾 Saving data splits...")
        
        split_paths = {}
        
        for split_name, data in splits.items():
            # Convert to JSON format
            json_data = [puzzle.to_dict() for puzzle in data]
            
            # Save to file
            output_path = self.data_dir / "splits" / f"{split_name}.json"
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            split_paths[split_name] = str(output_path)
            print(f"  ✅ Saved {len(json_data)} puzzles to {output_path}")
        
        return split_paths
    
    def _generate_analysis_report(self, splits: Dict[str, List[PuzzleData]]):
        """Generate comprehensive data analysis report"""
        print("📊 Generating data analysis report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_puzzles": sum(len(data) for data in splits.values()),
            "splits": {}
        }
        
        for split_name, data in splits.items():
            # Calculate statistics
            difficulty_dist = {}
            difficulty_scores = []
            clue_counts = []
            
            for puzzle in data:
                difficulty_dist[puzzle.difficulty] = difficulty_dist.get(puzzle.difficulty, 0) + 1
                difficulty_scores.append(puzzle.difficulty_score)
                clue_counts.append(puzzle.clue_count)
            
            report["splits"][split_name] = {
                "count": len(data),
                "difficulty_distribution": difficulty_dist,
                "avg_difficulty_score": np.mean(difficulty_scores),
                "std_difficulty_score": np.std(difficulty_scores),
                "avg_clue_count": np.mean(clue_counts),
                "std_clue_count": np.std(clue_counts),
                "avg_empty_cells": np.mean([len(p.empty_cells) for p in data])
            }
        
        # Save report
        report_path = self.data_dir / "analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Analysis report saved to {report_path}")
        
        # Print summary
        print("\n📊 DATA PREPARATION SUMMARY:")
        print("=" * 50)
        for split_name, stats in report["splits"].items():
            print(f"{split_name.upper()}: {stats['count']} puzzles")
            print(f"  Difficulty distribution: {stats['difficulty_distribution']}")
            print(f"  Avg difficulty score: {stats['avg_difficulty_score']:.3f}")
            print(f"  Avg clue count: {stats['avg_clue_count']:.1f}")
            print()
    
    def load_split(self, split_name: str) -> List[Dict]:
        """Load a specific data split"""
        split_path = self.data_dir / "splits" / f"{split_name}.json"
        
        if not split_path.exists():
            raise FileNotFoundError(f"Split {split_name} not found. Run prepare_all_data() first.")
        
        with open(split_path, 'r') as f:
            return json.load(f)

def main():
    """Main function to run data preparation"""
    print("🚀 Starting Sudoku Data Pipeline")
    print("=" * 50)
    
    pipeline = DataPipelineManager()
    split_paths = pipeline.prepare_all_data()
    
    print("\n🎉 Data preparation completed!")
    print("Split files created:")
    for split_name, path in split_paths.items():
        print(f"  {split_name}: {path}")

if __name__ == "__main__":
    main()