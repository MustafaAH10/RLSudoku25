"""
Experiment 1: Strategic Move RL
Focus: Learn strategic move selection and ordering, not just constraint satisfaction
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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
# huggingface_hub imports removed - not needed

from data_pipeline import PuzzleData, DataPipelineManager

@dataclass
class StrategicEpisode:
    """Episode data for strategic move learning"""
    states: List[str]
    actions: List[str]
    rewards: List[float]
    log_probs: List[float]
    strategic_scores: List[float]  # How strategic each move was
    constraint_benefits: List[float]  # How much each move helped other cells
    terminal: bool = False
    total_reward: float = 0.0

class StrategicMoveEnvironment:
    """Environment that rewards strategic thinking over brute force"""
    
    def __init__(self):
        self.original_puzzle = None
        self.current_puzzle = None
        self.target_solution = None
        self.move_history = []
        self.eliminated_possibilities = {}  # Track what each move eliminated
        
    def reset(self, puzzle_data: PuzzleData):
        """Reset environment with new puzzle"""
        self.original_puzzle = [row[:] for row in puzzle_data.puzzle]
        self.current_puzzle = [row[:] for row in puzzle_data.puzzle]
        self.target_solution = puzzle_data.solution
        self.move_history = []
        self.eliminated_possibilities = {}
        return self.get_state()
    
    def get_state(self) -> Dict:
        """Get current state with strategic information"""
        empty_cells = []
        for i in range(9):
            for j in range(9):
                if self.current_puzzle[i][j] == 0:
                    empty_cells.append(f"R{i+1}C{j+1}")
        
        return {
            "puzzle": self.current_puzzle,
            "empty_cells": empty_cells,
            "move_count": len(self.move_history),
            "spatial_grid": self._format_spatial_grid(),
            "strategic_info": self._analyze_strategic_opportunities()
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
    
    def _analyze_strategic_opportunities(self) -> Dict:
        """Analyze strategic opportunities for learning"""
        naked_singles = []
        hidden_singles = []
        constraint_cells = []
        
        for i in range(9):
            for j in range(9):
                if self.current_puzzle[i][j] == 0:
                    cell_key = f"R{i+1}C{j+1}"
                    valid_digits = self._get_valid_digits(i, j)
                    
                    if len(valid_digits) == 1:
                        naked_singles.append(cell_key)
                    elif len(valid_digits) == 2:
                        constraint_cells.append(cell_key)
                    
                    # Check for hidden singles (simplified)
                    if self._is_hidden_single(i, j, valid_digits):
                        hidden_singles.append(cell_key)
        
        return {
            "naked_singles": naked_singles,
            "hidden_singles": hidden_singles,
            "constraint_cells": constraint_cells,
            "total_empty": sum(row.count(0) for row in self.current_puzzle)
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
    
    def _is_hidden_single(self, row: int, col: int, valid_digits: List[int]) -> bool:
        """Check if any digit is a hidden single (simplified check)"""
        # This is a simplified version - in real implementation would check
        # if any digit can only go in this cell within row/col/box
        return len(valid_digits) == 1
    
    def apply_move(self, move_text: str) -> Tuple[float, bool, Dict]:
        """Apply move and return strategic reward"""
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
        
        # Calculate strategic reward
        reward, strategic_info = self._calculate_strategic_reward(row, col, digit)
        
        # Apply move
        self.current_puzzle[row][col] = digit
        self.move_history.append((row, col, digit))
        
        # Check if complete
        is_complete = sum(row.count(0) for row in self.current_puzzle) == 0
        
        return reward, is_complete, strategic_info
    
    def _parse_move(self, move_text: str) -> Optional[Tuple[int, int, int]]:
        """Parse move from text"""
        import re
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
    
    def _calculate_strategic_reward(self, row: int, col: int, digit: int) -> Tuple[float, Dict]:
        """Calculate reward based on strategic value, not just correctness"""
        cell_key = f"R{row+1}C{col+1}"
        
        # Base reward for correctness
        is_correct = (cell_key in self.target_solution and 
                     self.target_solution[cell_key] == digit)
        
        if not is_correct:
            return -1.0, {"strategic_score": 0.0, "constraint_benefit": 0.0}
        
        # Strategic analysis
        strategic_score = 0.0
        constraint_benefit = 0.0
        
        # Analyze move type
        valid_digits = self._get_valid_digits(row, col)
        
        if len(valid_digits) == 1:
            # Naked single - good but not super strategic
            strategic_score += 1.0
        elif len(valid_digits) == 2:
            # Strategic choice between two options
            strategic_score += 2.0
        else:
            # Strategic choice among many options
            strategic_score += 3.0
        
        # Calculate constraint benefit (how much this helps other cells)
        before_constraints = self._count_total_constraints()
        
        # Temporarily apply move to check benefit
        temp_puzzle = [row[:] for row in self.current_puzzle]
        temp_puzzle[row][col] = digit
        
        after_constraints = self._count_total_constraints_with_puzzle(temp_puzzle)
        constraint_benefit = after_constraints - before_constraints
        
        # Combine into final reward
        base_reward = 2.0  # Correct move
        strategic_bonus = strategic_score * 0.5
        constraint_bonus = constraint_benefit * 0.3
        
        total_reward = base_reward + strategic_bonus + constraint_bonus
        
        return total_reward, {
            "strategic_score": strategic_score,
            "constraint_benefit": constraint_benefit,
            "move_type": "naked_single" if len(valid_digits) == 1 else "strategic_choice"
        }
    
    def _count_total_constraints(self) -> int:
        """Count total constraints in current puzzle"""
        return self._count_total_constraints_with_puzzle(self.current_puzzle)
    
    def _count_total_constraints_with_puzzle(self, puzzle: List[List[int]]) -> int:
        """Count total constraints with given puzzle state"""
        total_constraints = 0
        
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] == 0:
                    # Count how many digits are eliminated for this cell
                    used_digits = set()
                    
                    # Row constraints
                    used_digits.update(puzzle[i])
                    
                    # Column constraints
                    used_digits.update(puzzle[k][j] for k in range(9))
                    
                    # Box constraints
                    box_row, box_col = 3 * (i // 3), 3 * (j // 3)
                    for bi in range(box_row, box_row + 3):
                        for bj in range(box_col, box_col + 3):
                            used_digits.add(puzzle[bi][bj])
                    
                    total_constraints += len(used_digits) - 1  # Exclude 0
        
        return total_constraints

class StrategicMoveRLTrainer:
    """RL trainer focused on strategic move learning"""
    
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
        self.env = StrategicMoveEnvironment()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 5e-6),
            weight_decay=config.get("weight_decay", 0.01)
        )
        
        # Setup directories
        self.checkpoint_dir = Path("experiment_1_strategic_move/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Final model path (standardized)
        self.final_model_dir = Path("models/experiment_1_final_model")
        self.final_model_dir.mkdir(parents=True, exist_ok=True)
        
        # WandB setup
        if config.get("use_wandb", True):
            wandb.init(
                project="sudoku-rl-experiments",
                name=f"strategic-move-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=config,
                tags=["strategic-move", "rl", "sudoku"]
            )
    
    def train(self, train_data: List[Dict], val_data: List[Dict]) -> Dict:
        """Main training loop"""
        print("🎯 Starting Strategic Move RL Training...")
        
        # Training parameters
        num_episodes = self.config.get("num_episodes", 1000)
        batch_size = self.config.get("batch_size", 8)
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
        
        print(f"🎉 Training completed! Model saved to {self.final_model_dir}")
        
        return {"final_model_path": str(self.final_model_dir)}
    
    def _create_strategic_move_prompt(self, puzzle: List[List[int]]) -> str:
        """Create prompt for strategic move selection"""
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
        
        return f"""You are an expert Sudoku solver. Analyze this puzzle and make strategic moves.

{grid_str}

Rules:
- Numbers 1-9 must appear exactly once in each row, column, and 3x3 box
- Focus on cells with fewest possibilities
- Make logical deductions

Select the next best move and provide your answer in this format:
<answer>R#C#: digit</answer>

Example of correct answer:
<answer>R3C4: 7</answer>

Think step by step and choose the most strategic move."""
    
    def _run_episode(self, puzzle_data: PuzzleData, max_moves: int) -> StrategicEpisode:
        """Run a single episode"""
        episode = StrategicEpisode(
            states=[], actions=[], rewards=[], log_probs=[],
            strategic_scores=[], constraint_benefits=[]
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
            
            # Generate move
            prompt = self._create_strategic_move_prompt(current_puzzle_data.puzzle)
            response = self._generate_response(prompt)
            
            # Calculate log probability
            log_prob = self._calculate_log_probability(prompt, response)
            
            # Apply move
            reward, is_complete, strategic_info = self.env.apply_move(response)
            
            # Store experience
            episode.states.append(state["spatial_grid"])
            episode.actions.append(response)
            episode.rewards.append(reward)
            episode.log_probs.append(log_prob)
            episode.strategic_scores.append(strategic_info.get("strategic_score", 0.0))
            episode.constraint_benefits.append(strategic_info.get("constraint_benefit", 0.0))
            
            # Update state
            state = self.env.get_state()
            
            if is_complete:
                episode.terminal = True
                break
        
        episode.total_reward = sum(episode.rewards)
        return episode
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,  # Increased from 50 for thinking mode
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return response.strip()
    
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
    
    def _train_on_batch(self, episodes: List[StrategicEpisode]) -> Dict:
        """Train on batch of episodes using policy gradient"""
        total_loss = 0.0
        
        for episode in episodes:
            # Calculate returns with strategic bonuses
            returns = []
            running_return = 0.0
            
            for i in reversed(range(len(episode.rewards))):
                reward = episode.rewards[i]
                strategic_bonus = episode.strategic_scores[i] * 0.1
                constraint_bonus = episode.constraint_benefits[i] * 0.05
                
                total_step_reward = reward + strategic_bonus + constraint_bonus
                running_return = total_step_reward + self.config.get("gamma", 0.99) * running_return
                returns.append(running_return)
            
            returns.reverse()
            
            # Normalize returns
            if len(returns) > 1:
                returns = torch.tensor(returns, dtype=torch.float32)
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Policy gradient loss
            policy_loss = 0.0
            for log_prob, return_val in zip(episode.log_probs, returns):
                policy_loss -= log_prob * return_val
            
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
            "avg_episode_length": np.mean([len(ep.rewards) for ep in episodes]),
            "completion_rate": np.mean([ep.terminal for ep in episodes]),
            "avg_strategic_score": np.mean([np.mean(ep.strategic_scores) for ep in episodes if ep.strategic_scores])
        }
        
        return metrics
    
    def _validate(self, val_data: List[Dict]) -> Dict:
        """Run validation"""
        val_episodes = []
        
        # Sample validation puzzles
        val_sample = np.random.choice(val_data, min(10, len(val_data)), replace=False)
        
        for puzzle_dict in val_sample:
            puzzle_data = PuzzleData.from_dict(puzzle_dict)
            max_moves = len(puzzle_data.empty_cells)  # Dynamic validation length
            episode = self._run_episode(puzzle_data, max_moves)
            val_episodes.append(episode)
        
        # Calculate validation metrics
        val_metrics = {
            "avg_reward": np.mean([ep.total_reward for ep in val_episodes]),
            "completion_rate": np.mean([ep.terminal for ep in val_episodes]),
            "avg_strategic_score": np.mean([np.mean(ep.strategic_scores) for ep in val_episodes if ep.strategic_scores])
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
    """Main function for strategic move RL experiment"""
    print("🚀 Strategic Move RL Experiment")
    print("=" * 50)
    
    # Configuration - Optimized for 24-hour GPU run
    config = {
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "learning_rate": 5e-6,
        "num_episodes": 600,  # Reduced from 1000
        # episode_length is now dynamic based on empty cells per puzzle
        "batch_size": 4,  # Reduced from 8 for memory efficiency
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
    trainer = StrategicMoveRLTrainer(config)
    results = trainer.train(train_data, val_data)
    
    print(f"🎉 Training completed! Final model saved to {results['final_model_path']}")
    print("💡 Use the benchmark.py file to evaluate the trained model.")
    
    # Close WandB
    if config.get("use_wandb", True):
        wandb.finish()
    
    print("🎉 Strategic Move RL experiment completed!")

if __name__ == "__main__":
    main()