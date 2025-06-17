import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model_loading(model_name):
    """Quick test to verify model can be loaded and run inference"""
    
    print(f"Testing model: {model_name}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"Model loaded on device: {model.device}")
        
        # Test inference
        print("Testing inference...")
        test_prompt = "Hello, this is a test. Please respond:"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test response: {response[len(test_prompt):].strip()}")
        
        print("✅ Model test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    # Test with Qwen model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    success = test_model_loading(model_name)
    
    if success:
        print("\n🎉 Ready to proceed with training!")
    else:
        print("\n⚠️  Please check your setup before proceeding.")