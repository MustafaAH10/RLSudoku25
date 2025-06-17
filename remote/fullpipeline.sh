#!/bin/bash

# Complete Sudoku RL Training Pipeline
set -e

echo "🚀 SUDOKU RL TRAINING PIPELINE"
echo "=============================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Make sure you're in the project directory."
    exit 1
fi

# Step 1: Setup environment
echo "📦 Step 1: Setting up environment..."
chmod +x setup_environment.sh
./setup_environment.sh

# Step 2: Test model loading
echo "🧪 Step 2: Testing model loading..."
python3 quick_test.py

# Step 3: Prepare data
echo "📊 Step 3: Preparing training data..."
python3 prepare_data.py

# Step 4: Create training config
echo "⚙️  Step 4: Creating training configuration..."
python3 create_training_config.py

# Step 5: Run baseline benchmark
echo "📈 Step 5: Running baseline benchmark..."
python3 sudoku_benchmark.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --test_data "test_data/sudoku_rl_test.json" \
    --output_file "baseline_results.json" \
    --max_tokens 1500 \
    --total_samples 20

echo "📊 Baseline results:"
python3 analyze_results.py --baseline baseline_results.json

# Ask user if they want to proceed with training
echo ""
read -p "🤔 Do you want to proceed with RL training? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "⏸️  Training cancelled. You can run training later with:"
    echo "   python3 sudoku_rl_trainer.py --config rl_config.json"
    exit 0
fi

# Step 6: Run RL training
echo "🎯 Step 6: Starting RL training..."
echo "⏰ This will take 3-4 hours. Starting tmux session..."

# Create tmux session for training
tmux new-session -d -s sudoku_training "python3 sudoku_rl_trainer.py --config rl_config.json"

echo "🔄 Training started in tmux session 'sudoku_training'"
echo "📺 To monitor progress, run: tmux attach-session -t sudoku_training"
echo "🔌 To detach from tmux: Ctrl+B then D"
echo "📊 To monitor GPU: watch nvidia-smi"

# Wait for training to complete or user to stop
echo ""
echo "⏳ Waiting for training to complete..."
echo "   You can safely close this terminal - training will continue in tmux"
echo "   To check if training is done: tmux list-sessions"

# Check every 5 minutes if training is still running
while tmux has-session -t sudoku_training 2>/dev/null; do
    echo "🔄 Training still in progress... ($(date))"
    sleep 300  # Wait 5 minutes
done

echo "✅ Training completed!"

# Step 7: Run post-training benchmark
echo "📈 Step 7: Running post-training benchmark..."
python3 sudoku_benchmark.py \
    --model_name "./sudoku-rl-model" \
    --test_data "test_data/sudoku_rl_test.json" \
    --output_file "post_training_results.json" \
    --max_tokens 1500 \
    --total_samples 20

# Step 8: Analyze results
echo "📊 Step 8: Analyzing results..."
python3 analyze_results.py --baseline baseline_results.json --post_training post_training_results.json

# Step 9: Package results
echo "📦 Step 9: Packaging results..."
mkdir -p final_results
cp baseline_results.json final_results/
cp post_training_results.json final_results/
cp rl_config.json final_results/
cp sudoku-rl-model/training.log final_results/ 2>/dev/null || echo "No training log found"

tar -czf sudoku_rl_results.tar.gz final_results/

echo ""
echo "🎉 PIPELINE COMPLETED!"
echo "====================="
echo "📁 Results packaged in: sudoku_rl_results.tar.gz"
echo "📊 Final analysis above shows the training effectiveness"
echo "💾 Trained model saved in: ./sudoku-rl-model/"
echo ""
echo "🔽 To download results to your local machine:"
echo "   scp root@<server-ip>:/root/sudoku_rl_results.tar.gz ./"