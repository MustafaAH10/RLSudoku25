import json

def create_config():
    """Create training configuration file"""
    
    config = {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "train_data_path": "train_data.json",
        "val_data_path": "val_data.json",
        "output_dir": "./sudoku-rl-model",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-5,
        "max_length": 1500,
        "logging_steps": 5,
        "save_every_n_epochs": 1,
        "ppo_epochs": 4,
        "max_grad_norm": 1.0,
        "vf_coef": 0.1,
        "target_kl": 0.1,
        "reward_correct_cell": 1.0,
        "reward_incorrect_cell": -0.5,
        "reward_format_bonus": 0.1,
        "reward_completion_bonus": 5.0
    }
    
    with open("rl_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Training config created: rl_config.json")
    print("📝 Config contents:")
    print(json.dumps(config, indent=2))

if __name__ == "__main__":
    create_config()