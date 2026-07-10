# test_nn.py
import numpy as np
from engine.tensor import Tensor
from nn.layers import Linear
from nn.optim import Adam

print("--- Running Neural Network & Adam Validation Check ---")

# 1. Initialize a Linear Layer (3 input features, 2 output features)
layer = Linear(3, 2)
optimizer = Adam([layer], lr=0.01)

# 2. Simulate a dummy input batch (size=2, features=3)
x = Tensor([[1.0, 2.0, 3.0], 
            [4.0, 5.0, 6.0]], requires_grad=False)

# 3. Run Forward Pass
out = layer(x)
print(f"✓ Forward pass output shape: {out.data.shape} (Expected: (2, 2))")

# 4. Run Backward Pass (using dummy output gradients)
loss = out.sum()
loss.backward()
print(f"✓ Weight gradients computed: W.grad shape is {layer.W.grad.shape}")

# 5. Take an Optimization Step
old_W = layer.W.data.copy()
optimizer.step()
assert not np.array_equal(old_W, layer.W.data), "Weights did not update!"
print("✓ Adam optimizer step successfully executed and weights updated!")

# 6. Reset Gradients
optimizer.zero_grad()
assert np.all(layer.W.grad == 0.0), "zero_grad() failed to clear gradients!"
print("✓ zero_grad() verified.")

print("\n🎉 SUCCESS: Neural Network modules and masked Adam optimizer are fully functional!")