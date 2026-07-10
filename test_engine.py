# test_engine.py
from engine.tensor import Tensor
import numpy as np

print("--- Running Autodiff Engine Sanity Check ---")

# 1. Initialize parent Tensors
x = Tensor([2.0, 3.0], requires_grad=True)
y = Tensor([4.0, 5.0], requires_grad=True)

# 2. Perform a forward pass operation (z = x + y)
z = x + y
print(f"Forward Pass Result (z): {z}")

# 3. Trigger backpropagation
z.backward()

# 4. Print out accumulated gradients
print(f"Gradient of x (dz/dx): {x.grad}")
print(f"Gradient of y (dz/dy): {y.grad}")

# 5. Automatically verify calculations match manual calculus
expected_grad = np.array([1.0, 1.0], dtype=np.float32)

assert np.allclose(x.grad, expected_grad), f"x.grad failed! Expected {expected_grad}, got {x.grad}"
assert np.allclose(y.grad, expected_grad), f"y.grad failed! Expected {expected_grad}, got {y.grad}"

print("\n🎉 SUCCESS: Core Tensor engine backprop and gradients are perfectly accurate!")