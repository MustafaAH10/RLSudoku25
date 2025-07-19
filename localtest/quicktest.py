"""
Simple test script to verify data preparation and basic functionality
Uses a small model that can run locally for testing purposes
"""

import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_data_preparation():
    """Test if data preparation worked correctly"""
    print("=== Testing Data Preparation ===")
    
    required_files = [
        "data/cot_training_set.csv",
        "data/rl_training_set.csv", 
        "data/validation_set.csv",
        "data/test_set.csv",
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"✅ {file_path}: {len(df)} samples")
            print(f"   Clue range: {df['clue_count'].min()}-{df['clue_count'].max()}")
        else:
            print(f"❌ Missing: {file_path}")
    
    print()

def test_prompt_format():
    """Test the prompt format with a sample puzzle"""
    print("=== Testing Prompt Format ===")
    
    # Sample puzzle (easy one)
    sample_puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    sample_solution = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
    
    def get_empty_cells(puzzle_string):
        empty_cells = []
        for i in range(81):
            if puzzle_string[i] == '0':
                row = i // 9 + 1
                col = i % 9 + 1
                empty_cells.append(f"r{row}c{col}")
        return empty_cells
    
    def format_grid_display(puzzle_string):
        grid = []
        for i in range(9):
            row = []
            for j in range(9):
                digit = int(puzzle_string[i * 9 + j])
                row.append(digit)
            grid.append(row)
        
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
    
    empty_cells = get_empty_cells(sample_puzzle)
    grid_display = format_grid_display(sample_puzzle)
    
    print("Sample puzzle grid:")
    print(grid_display)
    print(f"Empty cells ({len(empty_cells)}): {', '.join(empty_cells[:10])}...")
    
    # Show what the expected solution format looks like
    print("\nExpected solution format:")
    print("SOLUTION:")
    for i, cell in enumerate(empty_cells[:5]):  # Show first 5
        row_idx = int(cell[1]) - 1
        col_idx = int(cell[3]) - 1
        position = row_idx * 9 + col_idx
        expected_digit = sample_solution[position]
        print(f"{cell}: {expected_digit}")
    print("...")
    
    print()

def test_small_model():
    """Test with a small model that can run locally"""
    print("=== Testing Small Model ===")
    
    try:
        # Use a very small model for testing
        model_name = "distilgpt2"
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Simple test prompt
        test_prompt = "Complete this pattern: 1, 2, 3,"
        
        inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=50)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Model working! Test response: {response}")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
    
    print()

def main():
    print("🧪 SUDOKU PROJECT - LOCAL TESTING")
    print("=" * 50)
    
    # Test data preparation
    test_data_preparation()
    
    # Test prompt format
    test_prompt_format()
    
    # Test small model
    test_small_model()
    
    print("=" * 50)
    print("Local testing complete!")
    print()
    print("Next steps:")
    print("1. If data preparation passed: ✅ Ready for GPU rental")
    print("2. If prompt format looks good: ✅ Ready for model training")
    print("3. Rent GPU and run full evaluation with Qwen model")

if __name__ == "__main__":
    main()