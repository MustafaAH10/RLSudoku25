#!/usr/bin/env python3
"""
Model Downloader and Tester
Downloads Qwen/Qwen2.5-14B-Instruct from HuggingFace and saves it locally.
Tests it on a basic Sudoku prompt. Ensures the model is available for benchmarking.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
import time

# Hardcoded paths - standardized for all experiments
BASE_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MODELS_DIR = Path("models")
BASE_MODEL_PATH = MODELS_DIR / "base_model"

# Final model paths for each experiment
EXPERIMENT_1_FINAL_MODEL = MODELS_DIR / "experiment_1_final_model"
EXPERIMENT_2_FINAL_MODEL = MODELS_DIR / "experiment_2_final_model" 
EXPERIMENT_3_FINAL_MODEL = MODELS_DIR / "experiment_3_final_model"

def setup_model_directories():
    """Create model directories"""
    MODELS_DIR.mkdir(exist_ok=True)
    BASE_MODEL_PATH.mkdir(exist_ok=True)
    EXPERIMENT_1_FINAL_MODEL.mkdir(exist_ok=True)
    EXPERIMENT_2_FINAL_MODEL.mkdir(exist_ok=True)
    EXPERIMENT_3_FINAL_MODEL.mkdir(exist_ok=True)
    print(f"📁 Model directories created in: {MODELS_DIR}")

def download_and_save_base_model():
    """Download base model and save it locally"""
    print(f"🔄 Downloading model: {BASE_MODEL_NAME}")
    print(f"💾 Saving to: {BASE_MODEL_PATH}")
    print("=" * 60)
    
    # Download tokenizer
    print("📥 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Save tokenizer locally
    tokenizer.save_pretrained(BASE_MODEL_PATH)
    print(f"✅ Tokenizer saved to {BASE_MODEL_PATH}")
    
    # Download model
    print("📥 Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Save model locally
    model.save_pretrained(BASE_MODEL_PATH)
    print(f"✅ Model saved to {BASE_MODEL_PATH}")
    
    return model, tokenizer

def test_model_with_sudoku(model, tokenizer):
    """Test model with basic Sudoku prompt"""
    print("\n🎯 Testing with basic Sudoku prompt...")
    
    test_puzzle = """You are an expert Sudoku solver. Analyze this puzzle and make the next best move.

  _     _     3   │   _     _     _   │   _     _     8  
  _     4     _   │   7     _     _   │   1     _     _ 
  2     _     _   │   _     _     _   │   _     _     _ 
─────────────────┼─────────────────┼─────────────────
  _     _     _   │   5     2     _   │   _     8     _ 
  _     _     _   │   _     _     _   │   _     _     _ 
  _     3     _   │   _     7     1   │   _     _     _ 
─────────────────┼─────────────────┼─────────────────
  _     _     _   │   _     _     _   │   _     _     7 
  _     _     1   │   _     _     9   │   _     4     _ 
  8     _     _   │   _     _     _   │   3     _     _ 

Rules:
- Numbers 1-9 must appear exactly once in each row, column, and 3x3 box
- Focus on cells with fewest possibilities
- Make logical deductions

Select the next best move and provide your answer in this format:
<answer>R#C#: digit</answer>

Think step by step and choose the most strategic move."""
    
    # Tokenize and generate
    inputs = tokenizer(test_puzzle, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    generation_time = time.time() - start_time
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_response = response[len(test_puzzle):].strip()
    
    print(f"⏱️  Generation time: {generation_time:.2f} seconds")
    print(f"🎯 Generated response:\n{generated_response}")
    
    # Check if response contains valid move format
    import re
    move_pattern = r'<answer>R(\d)C(\d):\s*(\d)</answer>'
    match = re.search(move_pattern, generated_response)
    
    if match:
        row, col, digit = match.groups()
        print(f"✅ Valid move detected: R{row}C{col}: {digit}")
        success = True
    else:
        print("⚠️  No valid move format detected in response")
        success = False
    
    # Get model info
    device_info = str(model.device) if hasattr(model, 'device') else "distributed"
    param_count = sum(p.numel() for p in model.parameters())
    param_count_b = param_count / 1e9
    
    print(f"\n📊 Model Information:")
    print(f"   Device: {device_info}")
    print(f"   Parameters: {param_count_b:.1f}B")
    print(f"   Model path: {BASE_MODEL_PATH}")
    print(f"   Test success: {success}")
    
    return success, generated_response, generation_time

def save_test_results(success, response, generation_time):
    """Save test results"""
    test_results = {
        "model_name": BASE_MODEL_NAME,
        "base_model_path": str(BASE_MODEL_PATH),
        "experiment_1_final_path": str(EXPERIMENT_1_FINAL_MODEL),
        "experiment_2_final_path": str(EXPERIMENT_2_FINAL_MODEL),
        "experiment_3_final_path": str(EXPERIMENT_3_FINAL_MODEL),
        "test_success": success,
        "generation_time": generation_time,
        "generated_response": response,
        "timestamp": time.time()
    }
    
    results_path = Path("model_test_results.json")
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"💾 Test results saved to {results_path}")

def verify_base_model_exists():
    """Check if base model already exists locally"""
    config_path = BASE_MODEL_PATH / "config.json"
    tokenizer_path = BASE_MODEL_PATH / "tokenizer.json"
    
    if config_path.exists() and tokenizer_path.exists():
        print(f"✅ Base model already exists at {BASE_MODEL_PATH}")
        return True
    else:
        print(f"📥 Base model not found at {BASE_MODEL_PATH}")
        return False

def load_local_model():
    """Load model from local path"""
    print(f"🔄 Loading model from {BASE_MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"✅ Model loaded from {BASE_MODEL_PATH}")
    return model, tokenizer

def main():
    """Main function"""
    print("🚀 Qwen Model Downloader and Tester")
    print("=" * 60)
    
    # Setup directories
    setup_model_directories()
    
    # Check if model exists locally
    if verify_base_model_exists():
        model, tokenizer = load_local_model()
    else:
        model, tokenizer = download_and_save_base_model()
    
    # Test model
    success, response, gen_time = test_model_with_sudoku(model, tokenizer)
    
    # Save results
    save_test_results(success, response, gen_time)
    
    # Print standardized paths
    print(f"\n📁 Standardized Model Paths:")
    print(f"   Base model: {BASE_MODEL_PATH}")
    print(f"   Experiment 1 final: {EXPERIMENT_1_FINAL_MODEL}")
    print(f"   Experiment 2 final: {EXPERIMENT_2_FINAL_MODEL}")
    print(f"   Experiment 3 final: {EXPERIMENT_3_FINAL_MODEL}")
    
    if success:
        print("\n🎉 SUCCESS: Model is ready for benchmarking!")
        return 0
    else:
        print("\n❌ FAILURE: Model test failed, check configuration")
        return 1

if __name__ == "__main__":
    exit(main())