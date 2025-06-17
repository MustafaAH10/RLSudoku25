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

class SudokuRewardModel:
    """Reward model for Sudoku solving"""
    
    def __init__(self, 
                 reward_correct_cell=1.0,
                 reward_incorrect_cell=-0.5,
                 reward_format_bonus=0.1,
                 reward_completion_bonus=5.0):
        self.reward_correct_cell = reward_correct_cell
        self.reward_incorrect_cell = reward_incorrect_cell
        self.reward_format_bonus = reward_format_bonus
        self.reward_completion_bonus = reward_completion_bonus
    
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
    
    def calculate_reward(self, response, puzzle_data):
        """Calculate reward for a response"""
        expected_solutions = puzzle_data["solution"]
        predicted_solutions = self.parse_solution(response)
        
        total_empty_cells = sum(row.count(0) for row in puzzle_data["puzzle"])
        
        reward = 0.0
        correct_cells = 0
        
        # Reward for correct predictions
        for cell, predicted_digit in predicted_solutions.items():
            if cell in expected_solutions:
                if expected_solutions[cell] == predicted_digit:
                    reward += self.reward_correct_cell
                    correct_cells += 1
                else:
                    reward += self.reward_incorrect_cell
        
        # Format bonus for structured output
        if len(predicted_solutions) > 0:
            reward += self.reward_format_bonus
        
        # Completion bonus for perfect solutions
        if correct_cells == total_empty_cells and len(predicted_solutions) == total_empty_cells:
            reward += self.reward_completion_bonus
        
        # Accuracy-based scaling
        accuracy = correct_cells / total_empty_cells if total_empty_cells > 0 else 0
        reward *= (1 + accuracy)  # Scale reward by accuracy
        
        return reward

class SudokuRLTrainer:
    """RL trainer for Sudoku solving"""
    
    def __init__(self, config):
        self.config = config
        self.setup_model_and_tokenizer()
        self.setup_reward_model()
        self.setup_ppo_trainer()
    
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
    
    def setup_reward_model(self):
        """Setup reward model"""
        self.reward_model = SudokuRewardModel(
            reward_correct_cell=self.config.get("reward_correct_cell", 1.0),
            reward_incorrect_cell=self.config.get("reward_incorrect_cell", -0.5),
            reward_format_bonus=self.config.get("reward_format_bonus", 0.1),
            reward_completion_bonus=self.config.get("reward_completion_bonus", 5.0)
        )
    
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
        with open(self.config["train_data_path"], 'r') as f:
            train_data = json.load(f)
        
        # Create prompts
        prompts = []
        puzzle_data_list = []
        
        for puzzle_data in train_data:
            prompt = self.create_prompt(puzzle_data)
            prompts.append(prompt)
            puzzle_data_list.append(puzzle_data)
        
        return prompts, puzzle_data_list
    
    def train(self):
        """Run RL training"""
        print("Loading training data...")
        prompts, puzzle_data_list = self.load_training_data()
        
        print(f"Training on {len(prompts)} puzzles")
        
        # Training loop
        for epoch in range(self.config.get("num_train_epochs", 3)):
            print(f"\nEpoch {epoch + 1}/{self.config.get('num_train_epochs', 3)}")
            
            epoch_rewards = []
            
            for i, (prompt, puzzle_data) in enumerate(tqdm(zip(prompts, puzzle_data_list), 
                                                          total=len(prompts),
                                                          desc=f"Epoch {epoch + 1}")):
                try:
                    # Tokenize prompt
                    inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                    input_ids = inputs.input_ids.to(self.model.device)
                    
                    # Generate response
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids,
                            max_new_tokens=self.config.get("max_length", 1500),
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    
                    # Get response text
                    response_ids = outputs[0][len(input_ids[0]):]
                    response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    
                    # Calculate reward
                    reward = self.reward_model.calculate_reward(response, puzzle_data)
                    epoch_rewards.append(reward)
                    
                    # PPO step
                    rewards = torch.tensor([reward], dtype=torch.float32)
                    
                    # Run PPO training step
                    stats = self.ppo_trainer.step([input_ids[0]], [outputs[0]], rewards)
                    
                    # Log progress
                    if i % self.config.get("logging_steps", 10) == 0:
                        avg_reward = np.mean(epoch_rewards[-10:]) if epoch_rewards else 0
                        print(f"Step {i}, Avg Reward (last 10): {avg_reward:.3f}")
                        
                        if self.config.get("use_wandb", False):
                            wandb.log({
                                "reward": reward,
                                "avg_reward": avg_reward,
                                "epoch": epoch,
                                "step": i
                            })
                
                except Exception as e:
                    print(f"Error in training step {i}: {e}")
                    continue
            
            # Epoch summary
            avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0
            print(f"Epoch {epoch + 1} completed. Average reward: {avg_epoch_reward:.3f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get("save_every_n_epochs", 1) == 0:
                checkpoint_dir = f"{self.config['output_dir']}/checkpoint-epoch-{epoch + 1}"
                self.model.save_pretrained(checkpoint_dir)
                self.tokenizer.save_pretrained(checkpoint_dir)
                print(f"Checkpoint saved to {checkpoint_dir}")
        
        # Save final model
        print("Saving final model...")
        self.model.save_pretrained(self.config["output_dir"])
        self.tokenizer.save_pretrained(self.config["output_dir"])
        print(f"Final model saved to {self.config['output_dir']}")

def main():
    parser = argparse.ArgumentParser(description="Train Sudoku solver with RL")
    parser.add_argument("--config", type=str, required=True, help="Training config JSON file")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config["use_wandb"] = args.use_wandb
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="sudoku-rl",
            config=config,
            name=f"sudoku-rl-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Initialize trainer and train
    trainer = SudokuRLTrainer(config)
    trainer.train()
    
    print("Training completed!")

if __name__ == "__main__":
    main()