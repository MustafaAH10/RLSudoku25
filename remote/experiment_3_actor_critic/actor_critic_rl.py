"""
Experiment 3: Actor-Critic RL
Focus: Hybrid approach combining action selection with state value estimation for better learning
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))

import torch
import torch.nn as nn
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
class ActorCriticEpisode:
    """Episode data for actor-critic learning"""
    states: List[str]
    actions: List[str]
    rewards: List[float]
    log_probs: List[float]
    values: List[float]           # Value estimates from critic
    value_targets: List[float]    # Target values for critic training
    advantages: List[float]       # Advantage values for actor training
    terminal: bool = False
    total_reward: float = 0.0

class ActorCriticEnvironment:
    """Environment for actor-critic RL with state value estimation"""
    
    def __init__(self):
        self.original_puzzle = None
        self.current_puzzle = None
        self.target_solution = None
        self.move_history = []
        
    def reset(self, puzzle_data: PuzzleData):
        """Reset environment with new puzzle"""
        self.original_puzzle = [row[:] for row in puzzle_data.puzzle]
        self.current_puzzle = [row[:] for row in puzzle_data.puzzle]
        self.target_solution = puzzle_data.solution
        self.move_history = []
        return self.get_state()
    
    def get_state(self) -> Dict:
        """Get current state with value estimation features"""
        empty_cells = []
        for i in range(9):
            for j in range(9):
                if self.current_puzzle[i][j] == 0:
                    empty_cells.append(f"R{i+1}C{j+1}")
        
        # State value features
        progress_ratio = 1.0 - (len(empty_cells) / 81)
        constraint_density = self._calculate_constraint_density()
        
        return {
            "puzzle": self.current_puzzle,
            "empty_cells": empty_cells,
            "move_count": len(self.move_history),
            "spatial_grid": self._format_spatial_grid(),
            "progress_ratio": progress_ratio,
            "constraint_density": constraint_density,
            "state_value_features": self._get_state_value_features()
        }
    
    def _format_spatial_grid(self) -> str:
        """Format puzzle for display"""
        grid = []
        for i, row in enumerate(self.current_puzzle):
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
    
    def _calculate_constraint_density(self) -> float:
        """Calculate how constrained the puzzle is"""
        total_constraints = 0
        empty_cells = 0
        
        for i in range(9):
            for j in range(9):
                if self.current_puzzle[i][j] == 0:
                    empty_cells += 1
                    valid_digits = self._get_valid_digits(i, j)
                    total_constraints += 9 - len(valid_digits)
        
        return total_constraints / (empty_cells * 9) if empty_cells > 0 else 0
    
    def _get_state_value_features(self) -> Dict:
        """Get features for state value estimation"""
        empty_count = sum(row.count(0) for row in self.current_puzzle)
        
        # Count naked singles and constraint levels
        naked_singles = 0
        high_constraint_cells = 0
        
        for i in range(9):
            for j in range(9):
                if self.current_puzzle[i][j] == 0:
                    valid_digits = self._get_valid_digits(i, j)
                    if len(valid_digits) == 1:
                        naked_singles += 1
                    elif len(valid_digits) <= 3:
                        high_constraint_cells += 1
        
        return {
            "empty_count": empty_count,
            "naked_singles": naked_singles,
            "high_constraint_cells": high_constraint_cells,
            "completion_ratio": 1.0 - (empty_count / 81)
        }
    
    def _get_valid_digits(self, row: int, col: int) -> List[int]:
        """Get valid digits for a cell"""
        if self.current_puzzle[row][col] != 0:
            return []
        
        used_digits = set()
        
        # Row constraints
        used_digits.update(self.current_puzzle[row])
        
        # Column constraints
        used_digits.update(self.current_puzzle[i][col] for i in range(9))
        
        # 3x3 box constraints
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                used_digits.add(self.current_puzzle[i][j])
        
        return [d for d in range(1, 10) if d not in used_digits]
    
    def apply_move(self, move_text: str) -> Tuple[float, bool, Dict]:
        """Apply move and return reward with state value information"""
        # Parse move
        move = self._parse_move(move_text)
        if move is None:
            return -2.0, False, {"error": "invalid_format"}
        
        row, col, digit = move
        
        # Check if valid
        if self.current_puzzle[row][col] != 0:
            return -1.0, False, {"error": "cell_filled"}
        
        # Check if correct according to solution
        cell_key = f"R{row+1}C{col+1}"
        if cell_key not in self.target_solution:
            return -1.0, False, {"error": "cell_not_empty"}
        
        # Calculate reward
        is_correct = self.target_solution[cell_key] == digit
        if not is_correct:
            return -1.0, False, {"error": "incorrect_digit"}
        
        # Apply move
        self.current_puzzle[row][col] = digit
        self.move_history.append((row, col, digit))
        
        # Calculate reward based on progress and strategic value
        base_reward = 2.0
        
        # Bonus for reducing constraint complexity
        old_empty = sum(row.count(0) for row in self.current_puzzle) + 1
        new_empty = sum(row.count(0) for row in self.current_puzzle)
        progress_bonus = (old_empty - new_empty) * 0.5
        
        # Strategic bonus for creating new constraints
        constraint_bonus = self._calculate_constraint_creation_bonus(row, col)
        
        total_reward = base_reward + progress_bonus + constraint_bonus
        
        # Check if complete
        is_complete = sum(row.count(0) for row in self.current_puzzle) == 0
        if is_complete:
            total_reward += 20.0  # Completion bonus
        
        return total_reward, is_complete, {"strategic_value": constraint_bonus}
    
    def _parse_move(self, move_text: str) -> Optional[Tuple[int, int, int]]:
        """Parse move from text"""
        patterns = [
            r'R(\d)C(\d):\s*(\d)',
            r'R(\d)C(\d)\s*=\s*(\d)',
            r'R(\d)C(\d)\s+(\d)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, move_text, re.IGNORECASE)
            if match:
                row, col, digit = int(match.group(1)), int(match.group(2)), int(match.group(3))
                if 1 <= row <= 9 and 1 <= col <= 9 and 1 <= digit <= 9:
                    return (row-1, col-1, digit)  # Convert to 0-indexed
        return None
    
    def _calculate_constraint_creation_bonus(self, row: int, col: int) -> float:
        """Calculate bonus for creating new constraints"""
        # This is a simplified version - in practice would analyze
        # how many new constraints this move creates
        return 0.3  # Base strategic bonus
    
    def estimate_state_value(self) -> float:
        """Estimate the value of current state (for training target)"""
        features = self._get_state_value_features()
        
        # Simple heuristic-based value estimation
        completion_value = features["completion_ratio"] * 5.0
        constraint_value = features["naked_singles"] * 0.5
        progress_value = (81 - features["empty_count"]) / 81 * 3.0
        
        return min(completion_value + constraint_value + progress_value, 10.0)

class ActorCriticRLTrainer:
    """RL trainer for actor-critic learning"""
    
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
        
        # Initialize value head for critic
        self.value_head = nn.Linear(self.model.config.hidden_size, 1).to(self.device)
        
        # Initialize environment
        self.env = ActorCriticEnvironment()
        
        # Optimizers
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 4e-6),
            weight_decay=config.get("weight_decay", 0.01)
        )
        
        self.value_optimizer = torch.optim.AdamW(
            self.value_head.parameters(),
            lr=config.get("learning_rate", 4e-6) * 10  # Higher LR for value head
        )
        
        # Setup directories
        self.checkpoint_dir = Path("experiment_3_actor_critic/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Final model path (standardized)
        self.final_model_dir = Path("models/experiment_3_final_model")
        self.final_model_dir.mkdir(parents=True, exist_ok=True)
        
        # WandB setup
        if config.get("use_wandb", True):
            wandb.init(
                project="sudoku-rl-experiments",
                name=f"actor-critic-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=config,
                tags=["actor-critic", "rl", "sudoku"]
            )
    
    def train(self, train_data: List[Dict], val_data: List[Dict]) -> Dict:
        """Main training loop"""
        print("🎯 Starting Actor-Critic RL Training...")
        
        # Training parameters
        num_episodes = self.config.get("num_episodes", 1200)
        batch_size = self.config.get("batch_size", 6)
        # Note: episode_length is now dynamic based on empty cells per puzzle
        
        # Training loop
        total_episodes = 0
        batch_episodes = []
        
        for episode_idx in range(num_episodes):
            # Sample random puzzle
            puzzle_dict = np.random.choice(train_data)
            puzzle_data = PuzzleData.from_dict(puzzle_dict)
            
            # Run episode with dynamic length based on empty cells
            max_moves = len(puzzle_data.empty_cells)  # Dynamic episode length
            episode = self._run_episode(puzzle_data, max_moves)
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
                if total_episodes % 50 == 0:
                    print(f"Episode {total_episodes}: {metrics}")
                
                # Validation
                if total_episodes % 100 == 0:  # More frequent validation
                    val_metrics = self._validate(val_data)
                    if self.config.get("use_wandb", True):
                        wandb.log({f"val_{k}": v for k, v in val_metrics.items()})
                    
                    # Save checkpoint
                    self._save_checkpoint(total_episodes, val_metrics)
                
                batch_episodes = []
        
        # Final model save
        self.model.save_pretrained(self.final_model_dir)
        self.tokenizer.save_pretrained(self.final_model_dir)
        
        # Save value head
        torch.save(self.value_head.state_dict(), self.final_model_dir / "value_head.pt")
        
        print(f"🎉 Training completed! Model saved to {self.final_model_dir}")
        
        return {"final_model_path": str(self.final_model_dir)}
    
    def _create_actor_critic_prompt(self, puzzle: List[List[int]]) -> str:
        """Create prompt for actor-critic move selection with value estimation"""
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
        
        return f"""You are an expert Sudoku solver with value estimation. Analyze this puzzle and make the best strategic move.

{grid_str}

Rules:
- Numbers 1-9 must appear exactly once in each row, column, and 3x3 box
- Focus on cells with fewest possibilities
- Estimate the value/promise of your move

Make your best move and estimate its value (0.0 to 1.0, where 1.0 is most promising).

Provide your answer in this format:
<answer>R#C#: digit</answer>
<value>0.85</value>

Example of correct answer:
<answer>R3C4: 7</answer>
<value>0.85</value>

Think strategically and provide both the move and its estimated value."""
    
    def _run_episode(self, puzzle_data: PuzzleData, max_moves: int) -> ActorCriticEpisode:
        """Run a single episode"""
        episode = ActorCriticEpisode(
            states=[], actions=[], rewards=[], log_probs=[], values=[],
            value_targets=[], advantages=[]
        )
        
        # Reset environment
        state = self.env.reset(puzzle_data)
        
        for move_idx in range(max_moves):
            if len(state["empty_cells"]) == 0:
                episode.terminal = True
                break
            
            # Create puzzle data for prompt
            current_puzzle_data = PuzzleData(
                puzzle=state["puzzle"],
                solution=puzzle_data.solution,
                difficulty=puzzle_data.difficulty,
                clue_count=puzzle_data.clue_count,
                empty_cells=state["empty_cells"],
                difficulty_score=puzzle_data.difficulty_score
            )
            
            # Generate action and value
            prompt = self._create_actor_critic_prompt(current_puzzle_data.puzzle)
            response = self._generate_response(prompt)
            
            # Parse action and value
            action, value_estimate = self._parse_actor_critic_response(response)
            
            # Calculate log probability
            log_prob = self._calculate_log_probability(prompt, action)
            
            # Apply move
            reward, is_complete, move_info = self.env.apply_move(action)
            
            # Store experience
            episode.states.append(state["spatial_grid"])
            episode.actions.append(action)
            episode.rewards.append(reward)
            episode.log_probs.append(log_prob)
            episode.values.append(value_estimate)
            
            # Update state
            state = self.env.get_state()
            
            if is_complete:
                episode.terminal = True
                break
        
        # Calculate returns and advantages
        self._calculate_returns_and_advantages(episode)
        
        episode.total_reward = sum(episode.rewards)
        return episode
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,  # Increased from 100 for thinking mode
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return response.strip()
    
    def _parse_actor_critic_response(self, response: str) -> Tuple[str, float]:
        """Parse action and value from response"""
        # Look for action
        action_patterns = [
            r'R(\d)C(\d):\s*(\d)',
            r'R(\d)C(\d)\s*=\s*(\d)',
            r'ACTION:\s*(.+?)(?:\s*\||$)',
        ]
        
        action = None
        for pattern in action_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:  # Cell format
                    action = f"R{match.group(1)}C{match.group(2)}: {match.group(3)}"
                else:  # Action format
                    action = match.group(1).strip()
                break
        
        if action is None:
            action = response[:50]  # Fallback
        
        # Look for value
        value_patterns = [
            r'VALUE:\s*(\d+\.?\d*)',
            r'<value>(\d+\.?\d*)</value>',
        ]
        
        value = 5.0  # Default neutral value
        for pattern in value_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    break
                except ValueError:
                    continue
        
        return action, value
    
    def _calculate_log_probability(self, prompt: str, response: str) -> float:
        """Calculate log probability of response"""
        full_text = prompt + response
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
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
    
    def _calculate_returns_and_advantages(self, episode: ActorCriticEpisode):
        """Calculate returns and advantages for actor-critic training"""
        # Calculate returns (discounted rewards)
        returns = []
        running_return = 0.0
        
        for reward in reversed(episode.rewards):
            running_return = reward + self.config.get("gamma", 0.99) * running_return
            returns.append(running_return)
        
        returns.reverse()
        episode.value_targets = returns
        
        # Calculate advantages (returns - values)
        advantages = []
        for i in range(len(returns)):
            advantage = returns[i] - episode.values[i]
            advantages.append(advantage)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = np.array(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        episode.advantages = advantages.tolist()
    
    def _train_on_batch(self, episodes: List[ActorCriticEpisode]) -> Dict:
        """Train on batch of episodes using actor-critic"""
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        for episode in episodes:
            # Actor loss (policy gradient with advantage)
            actor_loss = 0.0
            for log_prob, advantage in zip(episode.log_probs, episode.advantages):
                actor_loss -= log_prob * advantage
            
            # Critic loss (value function approximation)
            values_tensor = torch.tensor(episode.values, dtype=torch.float32, device=self.device)
            targets_tensor = torch.tensor(episode.value_targets, dtype=torch.float32, device=self.device)
            critic_loss = F.mse_loss(values_tensor, targets_tensor)
            
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
        
        # Average losses
        avg_actor_loss = total_actor_loss / len(episodes)
        avg_critic_loss = total_critic_loss / len(episodes)
        
        # Backpropagation for actor
        self.optimizer.zero_grad()
        avg_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Backpropagation for critic
        self.value_optimizer.zero_grad()
        avg_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
        self.value_optimizer.step()
        
        # Calculate metrics
        metrics = {
            "actor_loss": avg_actor_loss.item(),
            "critic_loss": avg_critic_loss.item(),
            "total_loss": avg_actor_loss.item() + avg_critic_loss.item(),
            "avg_episode_reward": np.mean([ep.total_reward for ep in episodes]),
            "avg_episode_length": np.mean([len(ep.rewards) for ep in episodes]),
            "completion_rate": np.mean([ep.terminal for ep in episodes]),
            "avg_value_estimate": np.mean([np.mean(ep.values) for ep in episodes])
        }
        
        return metrics
    
    def _validate(self, val_data: List[Dict]) -> Dict:
        """Run validation"""
        val_episodes = []
        
        # Sample validation puzzles
        val_sample = np.random.choice(val_data, min(8, len(val_data)), replace=False)
        
        for puzzle_dict in val_sample:
            puzzle_data = PuzzleData.from_dict(puzzle_dict)
            max_moves = len(puzzle_data.empty_cells)  # Dynamic validation length
            episode = self._run_episode(puzzle_data, max_moves)
            val_episodes.append(episode)
        
        # Calculate validation metrics
        val_metrics = {
            "avg_reward": np.mean([ep.total_reward for ep in val_episodes]),
            "completion_rate": np.mean([ep.terminal for ep in val_episodes]),
            "avg_value_estimate": np.mean([np.mean(ep.values) for ep in val_episodes])
        }
        
        return val_metrics
    
    def _save_checkpoint(self, episode: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_episode_{episode}"
        checkpoint_path.mkdir(exist_ok=True)
        
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save value head
        torch.save(self.value_head.state_dict(), checkpoint_path / "value_head.pt")
        
        # Save metrics
        with open(checkpoint_path / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"💾 Checkpoint saved at episode {episode}")

def main():
    """Main function for actor-critic RL experiment"""
    print("🚀 Actor-Critic RL Experiment")
    print("=" * 50)
    
    # Configuration
    config = {
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "learning_rate": 4e-6,
        "num_episodes": 480,  # Reduced from 1200
        # episode_length is now dynamic based on empty cells per puzzle
        "batch_size": 4,  # Reduced from 6 for memory efficiency
        "gamma": 0.99,
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
    trainer = ActorCriticRLTrainer(config)
    results = trainer.train(train_data, val_data)
    
    print(f"🎉 Training completed! Final model saved to {results['final_model_path']}")
    print("💡 Use the benchmark.py file to evaluate the trained model.")
    
    # Close WandB
    if config.get("use_wandb", True):
        wandb.finish()
    
    print("🎉 Actor-Critic RL experiment completed!")

if __name__ == "__main__":
    main()