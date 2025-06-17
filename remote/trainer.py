import json
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    TrainerCallback
)
from trl import PPOTrainer, PPOConfig
from datasets import Dataset
import argparse
from datetime import datetime
import os
import re
from tqdm import tqdm
import wandb
import time
from pathlib import Path

class SudokuRewardModel:
    """Enhanced reward model for Sudoku solving with detailed metrics"""
    
    def __init__(self, 
                 reward_correct_cell=1.0,
                 reward_incorrect_cell=-0.5,
                 reward_format_bonus=0.1,
                 reward_completion_bonus=5.0,
                 reward_partial_progress=0.2):
        self.reward_correct_cell = reward_correct_cell
        self.reward_incorrect_cell = reward_incorrect_cell
        self.reward_format_bonus = reward_format_bonus
        self.reward_completion_bonus = reward_completion_bonus
        self.reward_partial_progress = reward_partial_progress
    
    def parse_solution(self, response_text):
        """Parse model response for cell solutions"""
        solutions = {}
        patterns = [
            r'R(\d)C(\d):\s*(\d)',
            r'R(\d)C(\d)\s*=\s*(\d)',
            r'R(\d)C(\d)\s+(\d)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            for match in matches:
                row, col, digit = int(match[0]), int(match[1]), int(match[2])
                if 1 <= row <= 9 and 1 <= col <= 9 and 1 <= digit <= 9:
                    solutions[f"R{row}C{col}"] = digit
        
        return solutions
    
    def calculate_reward_and_metrics(self, response, puzzle_data):
        """Calculate reward and detailed metrics for WandB logging"""
        expected_solutions = puzzle_data["solution"]
        predicted_solutions = self.parse_solution(response)
        
        total_empty_cells = sum(row.count(0) for row in puzzle_data["puzzle"])
        attempted_cells = len(predicted_solutions)
        
        reward = 0.0
        correct_cells = 0
        incorrect_cells = 0
        
        # Analyze predictions
        for cell, predicted_digit in predicted_solutions.items():
            if cell in expected_solutions:
                if expected_solutions[cell] == predicted_digit:
                    reward += self.reward_correct_cell
                    correct_cells += 1
                else:
                    reward += self.reward_incorrect_cell
                    incorrect_cells += 1
        
        # Format bonus for structured output
        has_valid_format = len(predicted_solutions) > 0
        if has_valid_format:
            reward += self.reward_format_bonus
        
        # Partial progress bonus
        if correct_cells > 0:
            progress_ratio = correct_cells / total_empty_cells
            reward += self.reward_partial_progress * progress_ratio
        
        # Completion bonus for perfect solutions
        is_perfect = (correct_cells == total_empty_cells and incorrect_cells == 0)
        if is_perfect:
            reward += self.reward_completion_bonus
        
        # Calculate metrics
        accuracy = correct_cells / total_empty_cells if total_empty_cells > 0 else 0
        coverage = attempted_cells / total_empty_cells if total_empty_cells > 0 else 0
        precision = correct_cells / attempted_cells if attempted_cells > 0 else 0
        
        # Difficulty-based reward scaling
        difficulty_multipliers = {
            'beginner': 0.8,
            'easy': 0.9,
            'medium': 1.0,
            'hard': 1.2,
            'expert': 1.5
        }
        difficulty = puzzle_data.get('difficulty', 'medium')
        reward *= difficulty_multipliers.get(difficulty, 1.0)
        
        metrics = {
            'reward': reward,
            'accuracy': accuracy,
            'coverage': coverage,
            'precision': precision,
            'correct_cells': correct_cells,
            'incorrect_cells': incorrect_cells,
            'attempted_cells': attempted_cells,
            'total_empty_cells': total_empty_cells,
            'is_perfect': is_perfect,
            'has_valid_format': has_valid_format,
            'difficulty': difficulty,
            'clue_count': puzzle_data.get('clue_count', 0),
            'response_length': len(response)
        }
        
        return reward, metrics

class SudokuRLTrainer:
    """Enhanced RL trainer for Sudoku solving with comprehensive WandB integration"""
    
    def __init__(self, config):
        self.config = config
        self.start_time = time.time()
        
        # Initialize WandB
        self.setup_wandb()
        
        # Setup components
        self.setup_model_and_tokenizer()
        self.setup_reward_model()
        self.setup_ppo_trainer()
        
        # Training state
        self.global_step = 0
        self.best_reward = float('-inf')
        self.training_metrics = []
    
    def setup_wandb(self):
        """Setup WandB logging"""
        if self.config.get("use_wandb", True):
            wandb.init(
                project=self.config.get("wandb_project", "sudoku-rl-experiment"),
                name=f"sudoku-rl-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=self.config,
                tags=["sudoku", "reinforcement-learning", "ppo"],
                notes=f"RL training on {self.config.get('model_name', 'unknown')} model",
                dir="wandb"  # Store wandb files in wandb directory
            )
            
            # Log system info
            wandb.log({
                "system/gpu_available": torch.cuda.is_available(),
                "system/gpu_count": torch.cuda.device_count(),
                "system/gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
                "system/torch_version": torch.__version__,
                "system/cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
            })
            
            print("✅ WandB initialized successfully")
        else:
            print("⚠️  WandB logging disabled")
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        print(f"Loading model: {self.config['model_name']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"Model loaded on device: {self.model.device}")
        
        # Log model info to WandB
        if self.config.get("use_wandb", True):
            model_params = sum(p.numel() for p in self.model.parameters())
            wandb.log({
                "model/total_parameters": model_params,
                "model/trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "model/device": str(self.model.device)
            })
    
    def setup_reward_model(self):
        """Setup reward model"""
        self.reward_model = SudokuRewardModel(
            reward_correct_cell=self.config.get("reward_correct_cell", 1.0),
            reward_incorrect_cell=self.config.get("reward_incorrect_cell", -0.5),
            reward_format_bonus=self.config.get("reward_format_bonus", 0.1),
            reward_completion_bonus=self.config.get("reward_completion_bonus", 5.0),
            reward_partial_progress=self.config.get("reward_partial_progress", 0.2)
        )
        
        print("✅ Reward model initialized")
    
    def setup_ppo_trainer(self):
        """Setup PPO trainer"""
        ppo_config = PPOConfig(
            model_name=self.config["model_name"],
            learning_rate=self.config.get("learning_rate", 1e-5),
            batch_size=self.config.get("per_device_train_batch_size", 1),
            mini_batch_size=self.config.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 8),
            ppo_epochs=self.config.get("ppo_epochs", 4),
            max_grad_norm=self.config.get("max_grad_norm", 1.0),
            vf_coef=self.config.get("vf_coef", 0.1),
            target_kl=self.config.get("target_kl", 0.1),
        )
        
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        print("✅ PPO trainer initialized")
    
    def format_sudoku_grid(self, grid):
        """Format Sudoku grid for display"""
        formatted = []
        for i, row in enumerate(grid):
            if i in [3, 6]:
                formatted.append("──────┼───────┼──────")
            
            row_str = ""
            for j, cell in enumerate(row):
                if j in [3, 6]:
                    row_str += "│ "
                if cell == 0:
                    row_str += "_ "
                else:
                    row_str += f"{cell} "
            formatted.append(row_str.strip())
        
        return "\n".join(formatted)
    
    def create_prompt(self, puzzle_data):
        """Create training prompt"""
        formatted_grid = self.format_sudoku_grid(puzzle_data["puzzle"])
        
        prompt = f"""Solve this Sudoku puzzle completely. Provide ONLY the solution in the exact format shown below.

Puzzle:
```
{formatted_grid}
```

Instructions:
- Fill ALL empty cells (marked with _)
- Each row, column, and 3x3 box must contain digits 1-9 exactly once
- Provide your complete solution using ONLY this format: RnCm: digit
- List every empty cell's solution

SOLUTION:
"""
        return prompt
    
    def load_training_data(self):
        """Load and prepare training data"""
        train_path = self.config.get("train_data_path", "train_data.json")
        
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        
        print(f"📊 Loaded {len(train_data)} training puzzles")
        
        # Log data distribution to WandB
        if self.config.get("use_wandb", True):
            difficulty_counts = {}
            for puzzle in train_data:
                diff = puzzle.get('difficulty', 'unknown')
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            
            wandb.log({f"data/train_{diff}_count": count for diff, count in difficulty_counts.items()})
            wandb.log({"data/total_train_samples": len(train_data)})
        
        # Create prompts
        prompts = []
        puzzle_data_list = []
        
        for puzzle_data in train_data:
            prompt = self.create_prompt(puzzle_data)
            prompts.append(prompt)
            puzzle_data_list.append(puzzle_data)
        
        return prompts, puzzle_data_list
    
    def evaluate_on_validation(self):
        """Evaluate model on validation set"""
        val_path = self.config.get("val_data_path", "val_data.json")
        
        try:
            with open(val_path, 'r') as f:
                val_data = json.load(f)
            
            print(f"🔍 Running validation on {len(val_data[:20])} samples...")  # Limit for speed
            
            val_metrics = {
                'reward': [],
                'accuracy': [],
                'perfect_solutions': 0,
                'by_difficulty': {}
            }
            
            for puzzle_data in val_data[:20]:  # Quick validation
                prompt = self.create_prompt(puzzle_data)
                
                # Generate response
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                input_ids = inputs.input_ids.to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=self.config.get("max_length", 1500),
                        temperature=0.1,  # Low temperature for validation
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                response_ids = outputs[0][len(input_ids[0]):]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                # Calculate metrics
                reward, metrics = self.reward_model.calculate_reward_and_metrics(response, puzzle_data)
                
                val_metrics['reward'].append(reward)
                val_metrics['accuracy'].append(metrics['accuracy'])
                
                if metrics['is_perfect']:
                    val_metrics['perfect_solutions'] += 1
                
                # Track by difficulty
                difficulty = puzzle_data.get('difficulty', 'unknown')
                if difficulty not in val_metrics['by_difficulty']:
                    val_metrics['by_difficulty'][difficulty] = {'count': 0, 'perfect': 0, 'accuracy': []}
                
                val_metrics['by_difficulty'][difficulty]['count'] += 1
                val_metrics['by_difficulty'][difficulty]['accuracy'].append(metrics['accuracy'])
                if metrics['is_perfect']:
                    val_metrics['by_difficulty'][difficulty]['perfect'] += 1
            
            # Calculate summary metrics
            avg_reward = np.mean(val_metrics['reward'])
            avg_accuracy = np.mean(val_metrics['accuracy'])
            perfect_rate = val_metrics['perfect_solutions'] / len(val_data[:20])
            
            # Log to WandB
            if self.config.get("use_wandb", True):
                wandb_log = {
                    "val/avg_reward": avg_reward,
                    "val/avg_accuracy": avg_accuracy,
                    "val/perfect_rate": perfect_rate,
                    "val/perfect_count": val_metrics['perfect_solutions']
                }
                
                # Add difficulty-specific metrics
                for diff, diff_metrics in val_metrics['by_difficulty'].items():
                    if diff_metrics['count'] > 0:
                        diff_acc = np.mean(diff_metrics['accuracy'])
                        diff_perfect_rate = diff_metrics['perfect'] / diff_metrics['count']
                        wandb_log[f"val/{diff}_accuracy"] = diff_acc
                        wandb_log[f"val/{diff}_perfect_rate"] = diff_perfect_rate
                
                wandb.log(wandb_log, step=self.global_step)
            
            print(f"📊 Validation - Avg Reward: {avg_reward:.3f}, Avg Accuracy: {avg_accuracy:.3f}, Perfect Rate: {perfect_rate:.3f}")
            
            return avg_reward
            
        except Exception as e:
            print(f"⚠️  Validation failed: {e}")
            return 0.0
    
    def save_checkpoint(self, epoch, avg_reward):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config["output_dir"]) / f"checkpoint-epoch-{epoch+1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'avg_reward': avg_reward,
            'config': self.config,
            'training_time': time.time() - self.start_time
        }
        
        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)
        
        print(f"💾 Checkpoint saved to {checkpoint_dir}")
        
        # Update best model if this is
    
    def train(self):
        """Main training loop with enhanced WandB logging"""
        print("🚀 Starting RL training...")
        
        # Load training data
        prompts, puzzle_data_list = self.load_training_data()
        
        # Training loop
        for epoch in range(self.config.get("num_train_epochs", 3)):
            epoch_start_time = time.time()
            epoch_rewards = []
            
            print(f"\n📚 Epoch {epoch + 1}/{self.config.get('num_train_epochs', 3)}")
            
            for i, (prompt, puzzle_data) in enumerate(zip(prompts, puzzle_data_list)):
                try:
                    # Generate response
                    response = self.generate_response(prompt)
                    
                    # Calculate reward
                    reward, metrics = self.reward_model.calculate_reward_and_metrics(response, puzzle_data)
                    epoch_rewards.append(reward)
                    
                    # Log to WandB
                    if self.config.get("use_wandb", True):
                        wandb.log({
                            "train/reward": reward,
                            "train/accuracy": metrics['accuracy'],
                            "train/coverage": metrics['coverage'],
                            "train/precision": metrics['precision'],
                            "train/correct_cells": metrics['correct_cells'],
                            "train/incorrect_cells": metrics['incorrect_cells'],
                            "train/attempted_cells": metrics['attempted_cells'],
                            "train/perfect_solution": metrics['is_perfect'],
                            "train/difficulty": metrics['difficulty'],
                            "train/clue_count": metrics['clue_count'],
                            "train/response_length": metrics['response_length'],
                            "train/epoch": epoch,
                            "train/step": i
                        })
                    
                    # Print progress
                    if (i + 1) % 10 == 0:
                        avg_reward = sum(epoch_rewards[-10:]) / 10
                        print(f"Step {i+1}/{len(prompts)} | Avg Reward: {avg_reward:.3f}")
                    
                except Exception as e:
                    print(f"Error in training step {i}: {e}")
                    continue
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
            
            # Log epoch metrics
            if self.config.get("use_wandb", True):
                wandb.log({
                    "epoch/avg_reward": avg_epoch_reward,
                    "epoch/time": epoch_time,
                    "epoch/step": epoch
                })
            
            print(f"✅ Epoch {epoch + 1} completed")
            print(f"   Average reward: {avg_epoch_reward:.3f}")
            print(f"   Time taken: {epoch_time:.2f}s")
            
            # Run validation
            val_reward = self.evaluate_on_validation()
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_reward)
            
            # Log validation metrics
            if self.config.get("use_wandb", True):
                wandb.log({
                    "val/reward": val_reward,
                    "val/epoch": epoch
                })
        
        # Training complete
        if self.config.get("use_wandb", True):
            wandb.finish()
        
        print("🎉 Training completed!")