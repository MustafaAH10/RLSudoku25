import json

def create_config():
    """Create training configuration file optimized for RTX 4090"""
    
    config = {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "train_data_path": "data/train_data.json",
        "val_data_path": "data/val_data.json",
        "output_dir": "./sudoku-rl-model",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,  # Increased for RTX 4090
        "gradient_accumulation_steps": 4,  # Adjusted for batch size
        "learning_rate": 1e-5,
        "max_length": 2500,  # Increased to accommodate full reasoning and solutions
        "max_new_tokens": 2000,  # Maximum new tokens to generate
        "logging_steps": 5,
        "save_every_n_epochs": 1,
        "ppo_epochs": 4,
        "max_grad_norm": 1.0,
        "vf_coef": 0.1,
        "target_kl": 0.1,
        "reward_correct_cell": 1.0,
        "reward_incorrect_cell": -0.5,
        "reward_format_bonus": 0.1,
        "reward_completion_bonus": 5.0,
        
        # WandB settings
        "use_wandb": True,
        "wandb_project": "sudoku-rl-experiment",
        "wandb_entity": None,  # Set your WandB username here
        "wandb_run_name": None,  # Will be auto-generated
        
        # Hardware settings
        "use_8bit": True,
        "use_4bit": False,
        "device_map": "auto",
        "torch_dtype": "float16",
        
        # Training optimization
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "val_reward",
        "greater_is_better": True
    }
    
    with open("rl_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Training config created: rl_config.json")
    print("📝 Config contents:")
    print(json.dumps(config, indent=2))

if __name__ == "__main__":
    create_config()