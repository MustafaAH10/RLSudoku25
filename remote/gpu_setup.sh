#!/bin/bash

# Simple GPU Setup Script for Sudoku RL Experiments
# Optimized for 24-hour experiment run

set -e

echo "🚀 SUDOKU RL GPU SETUP (24-HOUR OPTIMIZED)"
echo "==========================================="

# Check GPU
echo "📊 Checking GPU availability..."
if ! nvidia-smi &> /dev/null; then
    echo "❌ No GPU detected! Make sure you're on a GPU instance."
    exit 1
fi

echo "✅ GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits

# Install requirements
echo "📦 Installing requirements..."
pip install -q -r requirements.txt

# Setup WandB
echo "🔗 Setting up WandB..."
if [ ! -f ~/.netrc ] || ! grep -q "machine api.wandb.ai" ~/.netrc; then
    echo "Please log in to WandB (get your API key from https://wandb.ai/authorize):"
    wandb login
else
    echo "✅ WandB already configured"
fi

# Verify data
echo "📊 Verifying existing dataset..."
if [ -f "shared_data/splits/train.json" ] && [ -f "shared_data/splits/val.json" ] && [ -f "shared_data/splits/test.json" ]; then
    echo "✅ Data splits found:"
    echo "  - Train: $(jq length shared_data/splits/train.json) samples"
    echo "  - Val: $(jq length shared_data/splits/val.json) samples"
    echo "  - Test: $(jq length shared_data/splits/test.json) samples"
else
    echo "❌ Missing data splits! Run create_balanced_splits.py first."
    exit 1
fi

# Run quick test
echo "🧪 Testing setup..."
python -c "
import torch
from transformers import AutoTokenizer
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
print('✅ Setup test completed!')
"

echo "🎉 Setup completed successfully!"
echo ""
echo "📋 QUICK START GUIDE:"
echo "===================="
echo ""
echo "1. Run all experiments (24-hour optimized):"
echo "   python runexperiment.py"
echo ""
echo "2. Run individual experiments:"
echo "   python benchmark_runner.py  # Run baselines first"
echo "   python experiment_1_strategic_move/strategic_move_rl.py"
echo "   python experiment_2_complete_solution/complete_solution_rl.py"
echo "   python experiment_3_actor_critic/actor_critic_rl.py"
echo ""
echo "3. Monitor progress:"
echo "   - WandB dashboard for real-time metrics"
echo "   - Checkpoints in experiment_*/checkpoints/"
echo "   - Final models in experiment_*/checkpoints/final_model/"
echo ""
echo "🚀 Ready to start 24-hour experiment!"