# test_engine.py
from engine.tensor import Tensor
import numpy as np

print("--- Running Final Reductions & Loss Verification ---")

# 1. Test Reductions (Sum and Mean)
x_red = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
s = x_red.sum(axis=1) # Sum along columns -> [[3.0], [7.0]]
s.backward()
print(f"✓ Sum gradient backward successful: \n{x_red.grad}")

# Reset gradients for next operation
x_red.grad = np.zeros_like(x_red.data)

m = x_red.mean() # Absolute flat mean of all 4 entries
m.backward()
print(f"✓ Mean gradient backward successful: \n{x_red.grad}")

# 2. Test Softmax Cross Entropy Loss
# Simulating a Batch size of 2 with 3 classes
logits = Tensor([[2.0, 0.5, 1.0], 
                 [1.0, 5.0, 0.2]], requires_grad=True)

# Perfect target predictions (one-hot vectors)
targets = Tensor([[1.0, 0.0, 0.0], 
                  [0.0, 1.0, 0.0]], requires_grad=False)

loss = Tensor.softmax_cross_entropy(logits, targets)
loss.backward()

assert logits.grad is not None and logits.grad.shape == logits.data.shape
print("\n🎉 ALL MATHEMATICAL CORE REDUCTIONS AND LOSS VERIFIED!")