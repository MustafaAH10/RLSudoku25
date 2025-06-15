"""
Model Memory Analysis for Local Sudoku RL Training
==================================================

This script analyzes memory requirements for different models
and provides realistic options for local training.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
import psutil
import gc

class ModelMemoryAnalyzer:
    """Analyze memory requirements for different models"""
    
    def __init__(self):
        self.available_ram = psutil.virtual_memory().total / (1024**3)  # GB
        self.available_vram = self.get_gpu_memory()
        
    def get_gpu_memory(self):
        """Get available GPU memory"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        return 0
    
    def estimate_model_memory(self, model_name: str, use_8bit: bool = False, use_4bit: bool = False):
        """Estimate memory requirements for a model"""
        
        models_info = {
            # Small models (good for local testing)
            "microsoft/DialoGPT-small": {"params": 117e6, "layers": 12},
            "microsoft/DialoGPT-medium": {"params": 345e6, "layers": 24},
            "distilgpt2": {"params": 82e6, "layers": 6},
            "gpt2": {"params": 124e6, "layers": 12},
            
            # Medium models (possible on good laptops)
            "microsoft/DialoGPT-large": {"params": 762e6, "layers": 36},
            "gpt2-medium": {"params": 345e6, "layers": 24},
            "EleutherAI/gpt-neo-125M": {"params": 125e6, "layers": 12},
            "EleutherAI/gpt-neo-1.3B": {"params": 1.3e9, "layers": 24},
            
            # Large models (need good hardware)
            "Qwen/Qwen2.5-1.5B": {"params": 1.5e9, "layers": 28},
            "Qwen/Qwen2.5-3B": {"params": 3e9, "layers": 36},
            "Qwen/Qwen2.5-7B": {"params": 7e9, "layers": 32},
            "Qwen/Qwen2.5-8B": {"params": 8e9, "layers": 32},
        }
        
        if model_name not in models_info:
            print(f"❌ Model {model_name} not in database")
            return None
        
        info = models_info[model_name]
        params = info["params"]
        
        # Calculate memory requirements
        if use_4bit:
            bytes_per_param = 0.5  # 4-bit quantization
            precision = "4-bit"
        elif use_8bit:
            bytes_per_param = 1    # 8-bit quantization
            precision = "8-bit"
        else:
            bytes_per_param = 4    # float32
            precision = "float32"
        
        # Memory breakdown
        model_weights = (params * bytes_per_param) / (1024**3)  # GB
        value_head = 0.05  # ~50MB for value head
        gradients = model_weights if not (use_8bit or use_4bit) else model_weights * 0.5
        optimizer_states = model_weights * 2 if not (use_8bit or use_4bit) else model_weights
        activations = 0.5  # Estimate for batch_size=2
        
        total_memory = model_weights + value_head + gradients + optimizer_states + activations
        
        return {
            "model_name": model_name,
            "parameters": f"{params/1e6:.1f}M" if params < 1e9 else f"{params/1e9:.1f}B",
            "precision": precision,
            "model_weights_gb": model_weights,
            "value_head_gb": value_head,
            "gradients_gb": gradients,
            "optimizer_states_gb": optimizer_states,
            "activations_gb": activations,
            "total_memory_gb": total_memory,
            "fits_in_ram": total_memory < self.available_ram * 0.8,  # Leave 20% buffer
            "fits_in_vram": total_memory < self.available_vram * 0.8 if self.available_vram > 0 else False
        }
    
    def get_local_recommendations(self):
        """Get model recommendations for local training"""
        
        print(f"💻 System Analysis:")
        print(f"   Available RAM: {self.available_ram:.1f} GB")
        print(f"   Available VRAM: {self.available_vram:.1f} GB" if self.available_vram > 0 else "   GPU: Not available")
        
        models_to_test = [
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium", 
            "gpt2",
            "gpt2-medium",
            "EleutherAI/gpt-neo-1.3B",
            "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen2.5-3B",
        ]
        
        recommendations = {
            "safe_cpu": [],
            "safe_gpu": [],
            "risky_cpu": [],
            "risky_gpu": [],
            "too_large": []
        }
        
        for model_name in models_to_test:
            # Test different precision levels
            for use_8bit, use_4bit, desc in [(False, False, "fp32"), (True, False, "8bit"), (False, True, "4bit")]:
                if use_8bit and use_4bit:
                    continue
                    
                analysis = self.estimate_model_memory(model_name, use_8bit, use_4bit)
                if not analysis:
                    continue
                
                memory_req = analysis["total_memory_gb"]
                model_desc = f"{model_name} ({desc})"
                
                if self.available_vram > 0 and memory_req < self.available_vram * 0.6:
                    recommendations["safe_gpu"].append((model_desc, memory_req))
                elif self.available_vram > 0 and memory_req < self.available_vram * 0.8:
                    recommendations["risky_gpu"].append((model_desc, memory_req))
                elif memory_req < self.available_ram * 0.6:
                    recommendations["safe_cpu"].append((model_desc, memory_req))
                elif memory_req < self.available_ram * 0.8:
                    recommendations["risky_cpu"].append((model_desc, memory_req))
                else:
                    recommendations["too_large"].append((model_desc, memory_req))
        
        return recommendations
    
    def print_recommendations(self):
        """Print model recommendations"""
        
        recs = self.get_local_recommendations()
        
        print(f"\n🎯 Model Recommendations for Local Training:")
        print("=" * 60)
        
        if recs["safe_gpu"]:
            print(f"\n✅ SAFE GPU OPTIONS (use these!):")
            for model, memory in sorted(recs["safe_gpu"], key=lambda x: x[1]):
                print(f"   {model:<40} {memory:.1f} GB")
        
        if recs["safe_cpu"]:
            print(f"\n✅ SAFE CPU OPTIONS (slower but reliable):")
            for model, memory in sorted(recs["safe_cpu"], key=lambda x: x[1]):
                print(f"   {model:<40} {memory:.1f} GB")
        
        if recs["risky_gpu"]:
            print(f"\n⚠️  RISKY GPU OPTIONS (might run out of memory):")
            for model, memory in sorted(recs["risky_gpu"], key=lambda x: x[1]):
                print(f"   {model:<40} {memory:.1f} GB")
        
        if recs["risky_cpu"]:
            print(f"\n⚠️  RISKY CPU OPTIONS (might be slow/crash):")
            for model, memory in sorted(recs["risky_cpu"], key=lambda x: x[1]):
                print(f"   {model:<40} {memory:.1f} GB")
        
        if recs["too_large"]:
            print(f"\n❌ TOO LARGE (don't try locally):")
            for model, memory in sorted(recs["too_large"], key=lambda x: x[1])[:3]:
                print(f"   {model:<40} {memory:.1f} GB")
    
    def test_model_loading(self, model_name: str, use_8bit: bool = False, use_4bit: bool = False):
        """Actually test loading a model to verify memory usage"""
        
        print(f"\n🧪 Testing model loading: {model_name}")
        
        try:
            # Record initial memory
            initial_memory = psutil.virtual_memory().used / (1024**3)
            
            # Load tokenizer
            print("   Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with quantization if specified
            print("   Loading model...")
            load_kwargs = {}
            if use_8bit:
                load_kwargs["load_in_8bit"] = True
            elif use_4bit:
                load_kwargs["load_in_4bit"] = True
            else:
                load_kwargs["torch_dtype"] = torch.float32
            
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            
            # Add value head for PPO
            print("   Adding value head...")
            ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, **load_kwargs)
            
            # Record final memory
            final_memory = psutil.virtual_memory().used / (1024**3)
            memory_used = final_memory - initial_memory
            
            print(f"✅ Success! Memory used: {memory_used:.1f} GB")
            
            # Test a small generation
            print("   Testing generation...")
            inputs = tokenizer("Test input", return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   Generation test: '{result_text[:50]}...'")
            
            # Cleanup
            del model, ppo_model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True, memory_used
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            
            # Cleanup on failure
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return False, 0

def main():
    """Main analysis function"""
    
    analyzer = ModelMemoryAnalyzer()
    
    # Print system info and recommendations
    analyzer.print_recommendations()
    
    # Test some models if user wants
    print(f"\n🧪 Would you like to test loading a specific model? (y/n): ", end="")
    test_choice = input().strip().lower()
    
    if test_choice == 'y':
        print(f"\nAvailable models to test:")
        test_models = [
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium",
            "gpt2",
            "EleutherAI/gpt-neo-125M"
        ]
        
        for i, model in enumerate(test_models):
            print(f"{i+1}. {model}")
        
        try:
            choice = int(input("Enter number (1-4): ")) - 1
            if 0 <= choice < len(test_models):
                model_name = test_models[choice]
                
                # Test with different precisions
                for use_8bit, use_4bit, desc in [(False, False, "fp32"), (True, False, "8bit")]:
                    if use_8bit and use_4bit:
                        continue
                    print(f"\n--- Testing {desc} ---")
                    success, memory = analyzer.test_model_loading(model_name, use_8bit, use_4bit)
                    if not success:
                        break
            else:
                print("Invalid choice")
        except ValueError:
            print("Invalid input")
    
    print(f"\n💡 Recommendations:")
    print(f"   - For quick testing: microsoft/DialoGPT-small")
    print(f"   - For better results: microsoft/DialoGPT-medium or gpt2-medium")  
    print(f"   - If you have good GPU: Qwen/Qwen2.5-1.5B with 8-bit loading")

if __name__ == "__main__":
    main()