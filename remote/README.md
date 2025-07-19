# Sudoku RL Experiments - 24-Hour GPU Optimized

This directory contains three optimized RL approaches for Sudoku solving, designed for efficient 24-hour GPU training.

## 🚀 Quick Setup

**One-liner setup on any GPU instance:**

```bash
chmod +x gpu_setup.sh
./gpu_setup.sh
```

## 🎯 Run Experiments

### Option 1: Run All Experiments (Recommended)
```bash
python runexperiment.py
```
This runs all three experiments with optimized parameters for 24-hour completion.

### Option 2: Run Individual Experiments
```bash
# Run baseline benchmarks first
python benchmark_runner.py

# Then run individual experiments
python experiment_1_strategic_move/strategic_move_rl.py
python experiment_2_complete_solution/complete_solution_rl.py
python experiment_3_actor_critic/actor_critic_rl.py
```

## 🧠 Three RL Approaches

### 1. Strategic Move RL (Experiment 1)
- **Strategy**: Agent makes one strategic move per turn
- **Episode length**: Dynamic (17-65 moves based on empty cells)
- **Output**: Single move (e.g., "R3C4: 7")
- **Training**: 600 episodes, batch size 4

### 2. Complete Solution RL (Experiment 2)  
- **Strategy**: Agent solves entire puzzle at once
- **Episode length**: Always 1 (single-step)
- **Output**: Complete solution (e.g., "R1C1: 5, R2C3: 8, ...")
- **Training**: 500 episodes, batch size 3

### 3. Actor-Critic RL (Experiment 3)
- **Strategy**: Combines action selection with value estimation
- **Episode length**: Dynamic (17-65 moves based on empty cells)
- **Output**: Move + value (e.g., "R3C4: 7" + "0.75")
- **Training**: 480 episodes, batch size 4

## 📊 Key Optimizations

### Dataset Sizes (24-hour optimized):
- **Train**: 1,500 puzzles
- **Validation**: 500 puzzles  
- **Test**: 500 puzzles

### Dynamic Episode Lengths:
- Episodes now match puzzle complexity (17-65 empty cells)
- No wasted rollouts on easy puzzles
- Adequate training on hard puzzles

### Thinking Mode Compatible:
- Removed explicit `<thinking>` tags
- Increased token limits (300-2000 tokens)
- Relies on Qwen's built-in thinking mode

## 📁 Project Structure

```
remote/
├── experiment_1_strategic_move/
│   └── strategic_move_rl.py
├── experiment_2_complete_solution/
│   └── complete_solution_rl.py
├── experiment_3_actor_critic/
│   └── actor_critic_rl.py
├── shared_utils/
│   ├── data_pipeline.py
│   └── benchmark_manager.py
├── shared_data/
│   ├── raw/sudoku_puzzles.csv
│   └── splits/{train,val,test}.json
├── runexperiment.py        # Main orchestrator
├── benchmark_runner.py     # Baseline benchmarks
├── gpu_setup.sh           # GPU setup script
├── notes.txt              # Comprehensive documentation
└── requirements.txt       # Dependencies
```

## 📈 Expected Timeline (24 hours)

- **Data preparation**: 0.5 hours
- **Baseline benchmarks**: 1 hour
- **Strategic Move training**: 7 hours
- **Complete Solution training**: 8 hours
- **Actor-Critic training**: 6 hours
- **Post-training benchmarks**: 1.5 hours

## 🔧 Monitoring

- **WandB**: Real-time metrics (project: "sudoku-rl-experiments")
- **Checkpoints**: `experiment_*/checkpoints/`
- **Final models**: `experiment_*/checkpoints/final_model/`
- **Benchmarks**: `shared_data/benchmarks/`

## 🛠️ Model Configuration

- **Model**: Qwen/Qwen2.5-14B-Instruct
- **Mixed precision**: FP16 for memory efficiency
- **Thinking mode**: Enabled via inference settings
- **Memory optimized**: Batch sizes tuned for 14B model

## 📊 Key Features

- **Dynamic episode lengths** based on puzzle complexity
- **Consolidated benchmarking** to avoid redundancy
- **Optimized prompts** for thinking mode compatibility
- **Comprehensive logging** and error recovery
- **24-hour time management** with progress tracking

## 🎯 Usage Tips

```bash
# Monitor GPU usage
nvidia-smi

# Check experiment progress
tail -f experiment_final_report.json

# Resume from checkpoint (if needed)
# Experiments automatically resume from latest checkpoint
```

## 🚀 Production Ready

This setup is optimized for:
- **24-hour GPU rentals** (RunPod, Vast.ai, etc.)
- **Memory efficient** training on 14B models
- **Robust error handling** and recovery
- **Comprehensive reporting** and comparison