#!/usr/bin/env python3
"""
Final RL Training Script for Sudoku with Qwen3-1.7B
===================================================

This script performs RL training on Qwen3-1.7B with thinking mode
using 8-bit quantization for optimal local performance.
"""

import json
import torch
import numpy as np
import random
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Core imports
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    print("✅ All required libraries imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Install: pip install transformers trl torch accelerate bitsandbytes")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Qwen3RLConfig:
    """Configuration for Qwen3-1.7B RL training"""
    
    # Model settings
    model_name: str = "Qwen/Qwen3-1.7B"  # Latest Qwen3 with thinking mode
    use_8bit: bool = True                # Enable 8-bit quantization
    max_length: int = 1024
    enable_thinking: bool = True         # Enable Qwen3 thinking mode
    
    # PPO settings
    learning_rate: float = 1e-5
    batch_size: int = 4                  # Good for 6GB VRAM
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    ppo_epochs: int = 4
    
    # Reward settings  
    perfect_solution_reward: float = 15.0
    correct_cell_reward: float = 0.3
    incorrect_penalty: float = -0.5
    coverage_bonus: float = 3.0
    thinking_bonus: float = 1.0          # Bonus for using thinking mode
    
    # Training settings
    num_training_steps: int = 100        # Reasonable for local testing
    eval_steps: int = 20
    save_steps: int = 50
    log_steps: int = 5
    
    # Generation settings (optimized for Qwen3)
    temperature: float = 0.6             # Recommended for thinking mode
    top_p: float = 0.95
    top_k: int = 20
    max_new_tokens: int = 200

class Qwen3SudokuEnvironment:
    """Sudoku environment optimized for Qwen3 with thinking mode"""
    
    def __init__(self, config: Qwen3RLConfig):
        self.config = config
        
    def create_qwen3_prompt(self, puzzle: str, clue_count: int) -> str:
        """Create Qwen3-optimized prompt with thinking mode"""
        
        # Format puzzle grid
        formatted_grid = self.format_sudoku_grid(puzzle)
        empty_positions = self.find_empty_positions(puzzle)
        
        # Create messages for Qwen3 chat template
        messages = [
            {
                "role": "system",
                "content": "You are an expert Sudoku solver. Use step-by-step reasoning to solve puzzles systematically. Think through each constraint carefully."
            },
            {
                "role": "user", 
                "content": f"""Solve this Sudoku puzzle with {clue_count} given clues.

PUZZLE:
{formatted_grid}

EMPTY CELLS TO FILL: {', '.join(empty_positions)}

Rules: Each row, column, and 3×3 box must contain digits 1-9 exactly once.

Provide your solution in this EXACT format:
SOLUTION:
R1C1: digit
R2C3: digit
...

Think step by step about constraints and logical deductions."""
            }
        ]
        
        return messages
    
    def format_sudoku_grid(self, quiz_string: str) -> str:
        """Format Sudoku for optimal LLM comprehension"""
        if len(quiz_string) != 81:
            raise ValueError(f"Invalid puzzle length: {len(quiz_string)}")
        
        grid_lines = []
        for row in range(9):
            line_chars = []
            for col in range(9):
                char = quiz_string[row * 9 + col]
                line_chars.append('_' if char == '0' else char)
            
            # Create visually clear 3x3 blocks
            line = f"{' '.join(line_chars[0:3])} │ {' '.join(line_chars[3:6])} │ {' '.join(line_chars[6:9])}"
            grid_lines.append(line)
            
            # Add separators after rows 2 and 5
            if row == 2 or row == 5:
                grid_lines.append("──────┼───────┼──────")
        
        return '\n'.join(grid_lines)
    
    def find_empty_positions(self, quiz_string: str) -> List[str]:
        """Find empty cell positions"""
        empty_positions = []
        for i in range(81):
            if quiz_string[i] == '0':
                row, col = (i // 9) + 1, (i % 9) + 1
                empty_positions.append(f"R{row}C{col}")
        return empty_positions
    
    def parse_qwen3_response(self, response: str) -> Dict[str, int]:
        """Parse Qwen3 response including thinking content"""
        
        solutions = {}
        
        # Extract final response after thinking block
        if '<think>' in response and '</think>' in response:
            # Get content after thinking block
            think_end = response.rfind('</think>')
            final_response = response[think_end + 8:] if think_end != -1 else response
        else:
            final_response = response
        
        # Look for SOLUTION: section
        if 'SOLUTION:' in final_response.upper():
            solution_start = final_response.upper().find('SOLUTION:') + len('SOLUTION:')
            solution_text = final_response[solution_start:].strip()
        else:
            solution_text = final_response
        
        # Extract R1C1: digit patterns
        pattern = r'R(\d+)C(\d+):\s*(\d)'
        matches = re.findall(pattern, solution_text, re.IGNORECASE)
        
        for match in matches:
            row, col, digit = int(match[0]), int(match[1]), int(match[2])
            if 1 <= row <= 9 and 1 <= col <= 9 and 1 <= digit <= 9:
                cell_key = f"R{row}C{col}"
                solutions[cell_key] = digit
        
        return solutions
    
    def check_thinking_quality(self, response: str) -> float:
        """Assess quality of thinking content"""
        
        if '<think>' not in response or '</think>' not in response:
            return 0.0  # No thinking content
        
        # Extract thinking content
        think_start = response.find('<think>') + 7
        think_end = response.find('</think>')
        thinking_content = response[think_start:think_end].lower()
        
        # Look for reasoning indicators
        reasoning_patterns = [
            'constraint', 'eliminate', 'possible', 'cannot', 'must be',
            'row', 'column', 'box', 'block', 'only option',
            'rule out', 'deduce', 'logic', 'because'
        ]
        
        score = sum(1 for pattern in reasoning_patterns if pattern in thinking_content)
        return min(score / 8.0, 1.0)  # Normalize to 0-1
    
    def calculate_comprehensive_reward(self, puzzle: str, solution: str, response: str) -> Tuple[float, Dict]:
        """Calculate reward with thinking mode bonus"""
        
        # Get expected solutions
        expected_solutions = {}
        for i in range(81):
            if puzzle[i] == '0':
                row, col = (i // 9) + 1, (i % 9) + 1
                cell_key = f"R{row}C{col}"
                expected_solutions[cell_key] = int(solution[i])
        
        # Parse model predictions
        predicted_solutions = self.parse_qwen3_response(response)
        
        # Calculate correctness
        correct_cells = 0
        incorrect_cells = 0
        total_cells = len(expected_solutions)
        attempted_cells = len(predicted_solutions)
        
        for cell_key, expected_digit in expected_solutions.items():
            if cell_key in predicted_solutions:
                if predicted_solutions[cell_key] == expected_digit:
                    correct_cells += 1
                else:
                    incorrect_cells += 1
        
        # Reward components
        rewards = {}
        
        # Perfect solution bonus
        if correct_cells == total_cells and attempted_cells == total_cells:
            rewards['perfect'] = self.config.perfect_solution_reward
        else:
            rewards['perfect'] = 0.0
        
        # Cell-level rewards
        rewards['correct_cells'] = correct_cells * self.config.correct_cell_reward
        rewards['incorrect_penalty'] = incorrect_cells * self.config.incorrect_penalty
        
        # Coverage bonus
        if attempted_cells == total_cells:
            rewards['coverage'] = self.config.coverage_bonus
        else:
            rewards['coverage'] = 0.0
        
        # Thinking quality bonus
        thinking_score = self.check_thinking_quality(response)
        rewards['thinking'] = thinking_score * self.config.thinking_bonus
        
        # Total reward
        total_reward = sum(rewards.values())
        
        # Metrics for logging
        metrics = {
            'total_reward': total_reward,
            'reward_breakdown': rewards,
            'correct_cells': correct_cells,
            'total_cells': total_cells,
            'attempted_cells': attempted_cells,
            'accuracy': correct_cells / total_cells if total_cells > 0 else 0,
            'coverage': attempted_cells / total_cells if total_cells > 0 else 0,
            'thinking_score': thinking_score,
            'perfect_solution': correct_cells == total_cells and attempted_cells == total_cells
        }
        
        return total_reward, metrics

class Qwen3RLTrainer:
    """RL trainer for Qwen3-1.7B with thinking mode"""
    
    def __init__(self, config: Qwen3RLConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🔧 Initializing Qwen3 RL Trainer")
        logger.info(f"   Model: {config.model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   8-bit loading: {config.use_8bit}")
        logger.info(f"   Thinking mode: {config.enable_thinking}")
        
        # Setup quantization config
        if config.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
        else:
            quantization_config = None
        
        # Load tokenizer
        logger.info("📥 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with value head
        logger.info("📥 Loading model with value head...")
        try:
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                config.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if not config.use_8bit else None,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
        
        # Initialize environment
        self.env = Qwen3SudokuEnvironment(config)
        
        # PPO configuration
        ppo_config = PPOConfig(
            model_name=config.model_name,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            mini_batch_size=config.mini_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            ppo_epochs=config.ppo_epochs,
            max_grad_norm=1.0,
            target_kl=0.1,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
            log_with=None,
        )
        
        # Initialize PPO trainer
        logger.info("🎯 Initializing PPO trainer...")
        try:
            self.ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=self.model,
                tokenizer=self.tokenizer,
                dataset=None
            )
            logger.info("✅ PPO trainer initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing PPO: {e}")
            raise
        
        # Training history
        self.training_history = {
            'step': [], 'reward': [], 'accuracy': [], 'coverage': [],
            'perfect_solutions': [], 'thinking_scores': []
        }
    
    def load_training_data(self, data_path: str) -> List[Dict]:
        """Load training dataset"""
        logger.info(f"📁 Loading training data from {data_path}")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✅ Loaded {len(data)} training samples")
            return data
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def generate_qwen3_response(self, messages: List[Dict]) -> str:
        """Generate response using Qwen3 with thinking mode"""
        
        # Apply chat template with thinking enabled
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking
        )
        
        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Generate with optimized parameters for thinking mode
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract only the new generated content
        input_length = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0][input_length:]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return response
    
    def training_step(self, batch_data: List[Dict]) -> Dict:
        """Execute one PPO training step"""
        
        queries = []
        responses = []
        rewards = []
        batch_metrics = []
        
        for sample in batch_data:
            # Create Qwen3 prompt
            messages = self.env.create_qwen3_prompt(sample['puzzle'], sample['clue_count'])
            
            # Apply chat template
            query_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.enable_thinking
            )
            
            # Tokenize query
            query_tensor = self.tokenizer.encode(
                query_text, 
                return_tensors="pt", 
                max_length=800, 
                truncation=True
            ).squeeze().to(self.device)
            
            # Generate response using PPO model
            response_tensor = self.ppo_trainer.generate(
                query_tensor.unsqueeze(0),
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            ).squeeze()
            
            # Decode response
            full_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
            response_text = full_text[len(query_text):].strip()
            
            # Calculate reward
            reward, metrics = self.env.calculate_comprehensive_reward(
                sample['puzzle'], sample['solution'], response_text
            )
            
            # Store for batch processing
            queries.append(query_tensor)
            responses.append(response_tensor)
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            batch_metrics.append(metrics)
        
        # PPO training step
        try:
            stats = self.ppo_trainer.step(queries, responses, rewards)
        except Exception as e:
            logger.warning(f"PPO step failed: {e}")
            stats = {}
        
        # Aggregate metrics
        avg_metrics = {
            'reward': float(np.mean([r.item() for r in rewards])),
            'accuracy': float(np.mean([m['accuracy'] for m in batch_metrics])),
            'coverage': float(np.mean([m['coverage'] for m in batch_metrics])),
            'perfect_solutions': float(np.mean([m['perfect_solution'] for m in batch_metrics])),
            'thinking_scores': float(np.mean([m['thinking_score'] for m in batch_metrics])),
        }
        
        # Add PPO stats
        if stats:
            avg_metrics.update({k: float(v) for k, v in stats.items() if isinstance(v, (int, float))})
        
        return avg_metrics
    
    def evaluate(self, eval_data: List[Dict], num_samples: int = 10) -> Dict:
        """Evaluate model performance"""
        
        logger.info(f"📊 Evaluating on {num_samples} samples...")
        
        eval_samples = random.sample(eval_data, min(num_samples, len(eval_data)))
        
        metrics_list = []
        
        for sample in eval_samples:
            messages = self.env.create_qwen3_prompt(sample['puzzle'], sample['clue_count'])
            response = self.generate_qwen3_response(messages)
            
            _, metrics = self.env.calculate_comprehensive_reward(
                sample['puzzle'], sample['solution'], response
            )
            
            metrics_list.append(metrics)
        
        # Aggregate results
        eval_metrics = {
            'eval_reward': float(np.mean([m['total_reward'] for m in metrics_list])),
            'eval_accuracy': float(np.mean([m['accuracy'] for m in metrics_list])),
            'eval_coverage': float(np.mean([m['coverage'] for m in metrics_list])),
            'eval_perfect_rate': float(np.mean([m['perfect_solution'] for m in metrics_list])),
            'eval_thinking_score': float(np.mean([m['thinking_score'] for m in metrics_list])),
        }
        
        logger.info(f"   Reward: {eval_metrics['eval_reward']:.3f}")
        logger.info(f"   Accuracy: {eval_metrics['eval_accuracy']:.3f}")
        logger.info(f"   Perfect Rate: {eval_metrics['eval_perfect_rate']:.3f}")
        logger.info(f"   Thinking Score: {eval_metrics['eval_thinking_score']:.3f}")
        
        return eval_metrics
    
    def train(self, train_data_path: str, eval_data_path: str):
        """Main training loop"""
        
        logger.info("🚀 Starting Qwen3 RL training...")
        
        # Load data
        train_data = self.load_training_data(train_data_path)
        eval_data = self.load_training_data(eval_data_path)
        
        logger.info(f"📚 Training samples: {len(train_data)}")
        logger.info(f"📊 Evaluation samples: {len(eval_data)}")
        
        step = 0
        best_eval_reward = -float('inf')
        
        # Training loop
        while step < self.config.num_training_steps:
            try:
                # Sample batch
                batch_data = random.sample(train_data, min(self.config.batch_size, len(train_data)))
                
                # Training step
                metrics = self.training_step(batch_data)
                step += 1
                
                # Record metrics
                for key, value in metrics.items():
                    if key in self.training_history:
                        self.training_history[key].append(value)
                self.training_history['step'].append(step)
                
                # Logging
                if step % self.config.log_steps == 0:
                    logger.info(f"Step {step:3d}: "
                               f"R={metrics.get('reward', 0):.3f}, "
                               f"Acc={metrics.get('accuracy', 0):.3f}, "
                               f"Cov={metrics.get('coverage', 0):.3f}, "
                               f"Think={metrics.get('thinking_scores', 0):.3f}")
                
                # Evaluation
                if step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(eval_data, num_samples=5)
                    
                    if eval_metrics['eval_reward'] > best_eval_reward:
                        best_eval_reward = eval_metrics['eval_reward']
                        self.save_model(f"best_qwen3_rl_step_{step}")
                        logger.info(f"🎯 New best model! Reward: {best_eval_reward:.3f}")
                
                # Save checkpoint
                if step % self.config.save_steps == 0:
                    self.save_model(f"qwen3_rl_checkpoint_step_{step}")
                    self.save_training_history()
                    logger.info(f"💾 Checkpoint saved at step {step}")
                    
            except Exception as e:
                logger.error(f"❌ Error at step {step}: {e}")
                continue
        
        # Final save
        logger.info("🏁 Training completed!")
        self.save_model("qwen3_rl_final")
        self.save_training_history()
        
        final_eval = self.evaluate(eval_data, num_samples=10)
        logger.info(f"📈 Final Results:")
        logger.info(f"   Best Reward: {best_eval_reward:.3f}")
        logger.info(f"   Final Accuracy: {final_eval['eval_accuracy']:.3f}")
        logger.info(f"   Perfect Rate: {final_eval['eval_perfect_rate']:.3f}")
    
    def save_model(self, name: str):
        """Save model checkpoint"""
        save_dir = Path("models") / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            logger.info(f"💾 Model saved to {save_dir}")
        except Exception as e:
            logger.error(f"⚠️  Error saving model: {e}")
    
    def save_training_history(self):
        """Save training history"""
        try:
            with open("qwen3_rl_training_history.json", 'w') as f:
                json.dump(self.training_history, f, indent=2)
        except Exception as e:
            logger.error(f"⚠️  Error saving history: {e}")

def main():
    """Main execution function"""
    
    print("🎲 Qwen3-1.7B RL Training for Sudoku")
    print("=" * 50)
    
    # Configuration
    config = Qwen3RLConfig(
        model_name="Qwen/Qwen3-1.7B",
        use_8bit=True,              # Enable for 6GB VRAM
        enable_thinking=True,       # Enable thinking mode
        num_training_steps=100,     # Reasonable for local
        batch_size=4,
        eval_steps=20,
        save_steps=50
    )
    
    print(f"🔧 Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   8-bit quantization: {config.use_8bit}")
    print(f"   Thinking mode: {config.enable_thinking}")
    print(f"   Training steps: {config.num_training_steps}")
    
    # Check data files
    train_path = "training_data/sudoku_rl_train.json"
    eval_path = "test_data/sudoku_rl_test.json"
    
    if not Path(train_path).exists():
        print(f"❌ Training data not found: {train_path}")
        print("💡 Run the data preparation script first!")
        return
    
    if not Path(eval_path).exists():
        print(f"❌ Test data not found: {eval_path}")
        print("💡 Run the data preparation script first!")
        return
    
    try:
        # Initialize trainer
        trainer = Qwen3RLTrainer(config)
        
        # Start training
        trainer.train(train_path, eval_path)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Check 'models/' for saved checkpoints")
        print(f"📊 Check 'qwen3_rl_training_history.json' for metrics")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()