import torch
import transformers
import trl
import pandas as pd
import numpy as np

def main():
    print("🧪 Quick Test Suite")
    print("=" * 30)
    
    # Test PyTorch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    # Test basic tensor operations
    x = torch.randn(5, 5)
    y = torch.mm(x, x.t())
    print(f"   Tensor operations: OK")
    
    # Test Transformers
    print(f"✅ Transformers {transformers.__version__}")
    
    # Test TRL
    print(f"✅ TRL {trl.__version__}")
    
    # Test data libraries
    print(f"✅ Pandas {pd.__version__}")
    print(f"✅ NumPy {np.__version__}")
    
    print("\n🎉 All tests passed! Ready for RL training.")

if __name__ == "__main__":
    main()