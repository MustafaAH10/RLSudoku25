"""
Experiment 2: Complete Solution RL
Focus: Solve entire puzzle in one attempt, learning complete solution patterns
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))

import torch
import torch.nn.functional as F
import numpy as np
import json
import wandb
# time import removed - not needed
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
# huggingface_hub imports removed - not needed

from data_pipeline import PuzzleData, DataPipelineManager

@dataclass
class CompleteSolutionEpisode:
    """Episode data for complete solution learning"""
    state: str
    action: str
    reward: float
    log_prob: float
    solution_completeness: float  # How complete the solution was
    solution_accuracy: float     # How accurate the completed parts were
    terminal: bool = True  # Always terminal for complete solution
    total_reward: float = 0.0

class CompleteSolutionEnvironment:
    """Environment for complete solution RL"""
    
    def __init__(self):
        self.original_puzzle = None
        self.target_solution = None
        
    def reset(self, puzzle_data: PuzzleData):
        """Reset environment with new puzzle"""
        self.original_puzzle = [row[:] for row in puzzle_data.puzzle]
        self.target_solution = puzzle_data.solution
        return self.get_state()
    
    def get_state(self) -> Dict:
        """Get current state"""
        return {
            "puzzle": self.original_puzzle,
            "spatial_grid": self._format_spatial_grid(),
            "empty_cells": self._get_empty_cells(),
            "total_empty": sum(row.count(0) for row in self.original_puzzle)
        }
    
    def _format_spatial_grid(self) -> str:
        """Format puzzle for display"""
        grid = []
        for i, row in enumerate(self.original_puzzle):
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
    
    def _get_empty_cells(self) -> List[str]:
        """Get list of empty cells"""
        empty_cells = []
        for i in range(9):
            for j in range(9):
                if self.original_puzzle[i][j] == 0:
                    empty_cells.append(f"R{i+1}C{j+1}")
        return empty_cells
    
    def evaluate_complete_solution(self, solution_text: str) -> Tuple[float, Dict]:
        """Evaluate complete solution and return reward"""
        # Parse all moves from solution
        moves = self._parse_complete_solution(solution_text)
        
        # Calculate metrics
        total_empty_cells = sum(row.count(0) for row in self.original_puzzle)
        correct_moves = 0
        incorrect_moves = 0
        
        for move_key, predicted_digit in moves.items():
            if move_key in self.target_solution:
                if self.target_solution[move_key] == predicted_digit:
                    correct_moves += 1
                else:
                    incorrect_moves += 1
        
        # Calculate reward components
        accuracy = correct_moves / len(moves) if len(moves) > 0 else 0
        completeness = len(moves) / total_empty_cells if total_empty_cells > 0 else 0
        
        # Base reward for correctness
        base_reward = correct_moves * 2.0 - incorrect_moves * 1.0
        
        # Bonus for complete correct solution
        if correct_moves == total_empty_cells and incorrect_moves == 0:
            base_reward += 50.0  # Large bonus for perfect solution
        
        # Coverage bonus
        coverage_bonus = completeness * 10.0
        
        # Accuracy bonus
        accuracy_bonus = accuracy * 15.0
        
        total_reward = base_reward + coverage_bonus + accuracy_bonus
        
        return total_reward, {
            "accuracy": accuracy,
            "completeness": completeness,
            "correct_moves": correct_moves,
            "incorrect_moves": incorrect_moves,
            "total_empty_cells": total_empty_cells,
            "attempted_moves": len(moves)
        }
    
    def _parse_complete_solution(self, solution_text: str) -> Dict[str, int]:
        """Parse complete solution from model output"""
        moves = {}
        patterns = [
            r'R(\d)C(\d):\s*(\d)',
            r'R(\d)C(\d)\s*=\s*(\d)',
            r'R(\d)C(\d)\s*,?\s*(\d)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, solution_text, re.IGNORECASE)
            for match in matches:
                try:
                    row, col, digit = int(match[0]), int(match[1]), int(match[2])
                    if 1 <= row <= 9 and 1 <= col <= 9 and 1 <= digit <= 9:
                        moves[f"R{row}C{col}"] = digit
                except (ValueError, IndexError):
                    continue
        
        return moves

class CompleteSolutionRLTrainer:
    """RL trainer for complete solution learning"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Initialize environment
        self.env = CompleteSolutionEnvironment()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 3e-6),
            weight_decay=config.get("weight_decay", 0.01)
        )
        
        # Setup directories
        self.checkpoint_dir = Path("experiment_2_complete_solution/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Final model path (standardized)
        self.final_model_dir = Path("models/experiment_2_final_model")
        self.final_model_dir.mkdir(parents=True, exist_ok=True)
        
        # WandB setup
        if config.get("use_wandb", True):
            wandb.init(
                project="sudoku-rl-experiments",
                name=f"complete-solution-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=config,
                tags=["complete-solution", "rl", "sudoku"]
            )
    
    def train(self, train_data: List[Dict], val_data: List[Dict]) -> Dict:
        """Main training loop"""
        print("🎯 Starting Complete Solution RL Training...")
        
        # Training parameters
        num_episodes = self.config.get("num_episodes", 800)
        batch_size = self.config.get("batch_size", 4)  # Smaller batch for complete solutions
        
        # Training loop
        total_episodes = 0
        batch_episodes = []
        
        for episode_idx in range(num_episodes):
            # Sample random puzzle
            puzzle_dict = np.random.choice(train_data)
            puzzle_data = PuzzleData.from_dict(puzzle_dict)
            
            # Run episode
            episode = self._run_episode(puzzle_data)
            batch_episodes.append(episode)
            
            total_episodes += 1
            
            # Train on batch
            if len(batch_episodes) >= batch_size:
                metrics = self._train_on_batch(batch_episodes)
                
                # Log metrics
                if self.config.get("use_wandb", True):
                    wandb.log({
                        "episode": total_episodes,
                        **metrics
                    })
                
                # Print progress
                if total_episodes % 20 == 0:
                    print(f"Episode {total_episodes}: {metrics}")
                
                # Validation - keep at 100 for this experiment
                if total_episodes % 100 == 0:
                    val_metrics = self._validate(val_data)
                    if self.config.get("use_wandb", True):
                        wandb.log({f"val_{k}": v for k, v in val_metrics.items()})
                    
                    # Save checkpoint
                    self._save_checkpoint(total_episodes, val_metrics)
                
                batch_episodes = []
        
        # Final model save
        self.model.save_pretrained(self.final_model_dir)
        self.tokenizer.save_pretrained(self.final_model_dir)
        
        print(f"🎉 Training completed! Model saved to {self.final_model_dir}")
        
        return {"final_model_path": str(self.final_model_dir)}
    
    def _create_complete_solution_prompt(self, puzzle: List[List[int]]) -> str:
        """Create prompt for complete solution"""
        grid_str = ""
        for i, row in enumerate(puzzle):
            if i in [3, 6]:
                grid_str += "─────── ┼ ─────── ┼ ───────\n"
            
            row_str = ""
            for j, cell in enumerate(row):
                if j in [3, 6]:
                    row_str += " │ "
                if cell == 0:
                    row_str += "  _  "
                else:
                    row_str += f"  {cell}  "
            grid_str += row_str + "\n"
        
        return f"""You are an expert Sudoku solver. Solve this puzzle completely in one attempt.

{grid_str}

Rules:
- Numbers 1-9 must appear exactly once in each row, column, and 3x3 box
- Fill ALL empty cells (_) with correct digits
- Provide complete solution for all empty cells

Analyze the puzzle systematically and provide your complete solution in this exact format:
<answer>R1C1: 5, R2C3: 8, R3C7: 2, R4C2: 1, R5C9: 6</answer>

Example of correct complete answer:
<answer>R1C1: 5, R1C3: 8, R2C2: 9, R3C5: 7</answer>

Think step by step and provide the complete solution."""
    
    def _run_episode(self, puzzle_data: PuzzleData) -> CompleteSolutionEpisode:
        """Run a single episode"""
        # Reset environment
        state = self.env.reset(puzzle_data)
        
        # Create puzzle data for prompt
        current_puzzle_data = PuzzleData(
            puzzle=state["puzzle"],
            solution=puzzle_data.solution,
            difficulty=puzzle_data.difficulty,
            clue_count=puzzle_data.clue_count,
            empty_cells=state["empty_cells"],
            difficulty_score=puzzle_data.difficulty_score
        )
        
        # Generate complete solution
        prompt = self._create_complete_solution_prompt(current_puzzle_data.puzzle)
        response = self._generate_response(prompt)
        
        # Calculate log probability
        log_prob = self._calculate_log_probability(prompt, response)
        
        # Evaluate solution
        reward, solution_info = self.env.evaluate_complete_solution(response)
        
        # Create episode
        episode = CompleteSolutionEpisode(
            state=state["spatial_grid"],
            action=response,
            reward=reward,
            log_prob=log_prob,
            solution_completeness=solution_info["completeness"],
            solution_accuracy=solution_info["accuracy"],
            terminal=True,
            total_reward=reward
        )
        
        return episode
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2000,  # Increased from 1500 for thinking mode
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return response.strip()
    
    def _calculate_log_probability(self, prompt: str, response: str) -> float:
        """Calculate log probability of response"""
        full_text = prompt + response
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=3072)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get response tokens
            response_tokens = self.tokenizer(response, return_tensors="pt")["input_ids"][0]
            log_probs = F.log_softmax(logits[0], dim=-1)
            
            # Calculate average log probability
            total_log_prob = 0.0
            for i, token in enumerate(response_tokens):
                if i < len(log_probs):
                    total_log_prob += log_probs[i][token].item()
            
            return total_log_prob / len(response_tokens) if len(response_tokens) > 0 else 0.0
    
    def _train_on_batch(self, episodes: List[CompleteSolutionEpisode]) -> Dict:
        """Train on batch of episodes using REINFORCE"""
        total_loss = 0.0
        
        for episode in episodes:
            # REINFORCE: loss = -log_prob * reward
            # Scale reward for numerical stability
            scaled_reward = episode.reward / 100.0
            policy_loss = -episode.log_prob * scaled_reward
            total_loss += policy_loss
        
        # Average loss
        avg_loss = total_loss / len(episodes)
        
        # Backpropagation
        self.optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Calculate metrics
        metrics = {
            "loss": avg_loss.item(),
            "avg_episode_reward": np.mean([ep.total_reward for ep in episodes]),
            "avg_completeness": np.mean([ep.solution_completeness for ep in episodes]),
            "avg_accuracy": np.mean([ep.solution_accuracy for ep in episodes]),
            "perfect_solutions": np.sum([ep.solution_completeness == 1.0 and ep.solution_accuracy == 1.0 for ep in episodes])
        }
        
        return metrics
    
    def _validate(self, val_data: List[Dict]) -> Dict:
        """Run validation"""
        val_episodes = []
        
        # Sample validation puzzles
        val_sample = np.random.choice(val_data, min(8, len(val_data)), replace=False)
        
        for puzzle_dict in val_sample:
            puzzle_data = PuzzleData.from_dict(puzzle_dict)
            episode = self._run_episode(puzzle_data)
            val_episodes.append(episode)
        
        # Calculate validation metrics
        val_metrics = {
            "avg_reward": np.mean([ep.total_reward for ep in val_episodes]),
            "avg_completeness": np.mean([ep.solution_completeness for ep in val_episodes]),
            "avg_accuracy": np.mean([ep.solution_accuracy for ep in val_episodes]),
            "perfect_solutions": np.sum([ep.solution_completeness == 1.0 and ep.solution_accuracy == 1.0 for ep in val_episodes])
        }
        
        return val_metrics
    
    def _save_checkpoint(self, episode: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_episode_{episode}"
        checkpoint_path.mkdir(exist_ok=True)
        
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save metrics
        with open(checkpoint_path / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"💾 Checkpoint saved at episode {episode}")

def main():
    """Main function for complete solution RL experiment"""
    print("🚀 Complete Solution RL Experiment")
    print("=" * 50)
    
    # Configuration
    config = {
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "learning_rate": 3e-6,
        "num_episodes": 500,  # Reduced from 800
        "batch_size": 3,  # Reduced from 4 for memory efficiency
        "use_wandb": True,
        "benchmark_samples": 30,  # Reduced from 50
        "weight_decay": 0.01,
        "gradient_clipping": 1.0,
        "checkpoint_frequency": 100  # More frequent checkpoints
    }
    
    # Prepare data
    data_manager = DataPipelineManager()
    
    # Check if data exists, if not prepare it
    try:
        train_data = data_manager.load_split("train")
        val_data = data_manager.load_split("val")
        test_data = data_manager.load_split("test")
    except FileNotFoundError:
        print("📊 Preparing data...")
        data_manager.prepare_all_data()
        train_data = data_manager.load_split("train")
        val_data = data_manager.load_split("val")
        test_data = data_manager.load_split("test")
    
    # Train model
    trainer = CompleteSolutionRLTrainer(config)
    results = trainer.train(train_data, val_data)
    
    print(f"🎉 Training completed! Final model saved to {results['final_model_path']}")
    print("💡 Use the benchmark.py file to evaluate the trained model.")
    
    # Close WandB
    if config.get("use_wandb", True):
        wandb.finish()
    
    print("🎉 Complete Solution RL experiment completed!")

if __name__ == "__main__":
    main()