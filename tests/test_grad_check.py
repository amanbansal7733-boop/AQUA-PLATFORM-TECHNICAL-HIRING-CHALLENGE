import numpy as np
from engine.tensor import Tensor

def relative_error(a, b):
    return np.linalg.norm(a - b) / max(1e-8, (np.linalg.norm(a) + np.linalg.norm(b)))

def test_operations_grad_check():
    # ... inside your test configuration setup ...
    epsilon = 1e-3
    tolerance = 1e-3  # Adjusted to perfectly align with float32 numerical precision limit
    
    # --- TEST CASE 1: MATRIX MULTIPLICATION & ADDITION ---
    x_init = np.random.randn(2, 3).astype(np.float32)
    w_init = np.random.randn(3, 2).astype(np.float32)
    
    # Analytical backward pass
    x = Tensor(x_init, requires_grad=True)
    w = Tensor(w_init, requires_grad=True)
    out = x @ w
    out.backward()
    
    grad_x_analytical = x.grad.copy()
    
    # Numerical gradient estimation for x
    grad_x_numerical = np.zeros_like(x_init)
    for i in range(x_init.shape[0]):
        for j in range(x_init.shape[1]):
            # x + epsilon
            x_pos = Tensor(x_init.copy(), requires_grad=False)
            x_pos.data[i, j] += epsilon
            out_pos = x_pos @ Tensor(w_init, requires_grad=False)
            
            # x - epsilon
            x_neg = Tensor(x_init.copy(), requires_grad=False)
            x_neg.data[i, j] -= epsilon
            out_neg = x_neg @ Tensor(w_init, requires_grad=False)
            
            # Symmetric difference quotient
            grad_x_numerical[i, j] = np.sum(out_pos.data - out_neg.data) / (2 * epsilon)
            
    err_x = relative_error(grad_x_analytical, grad_x_numerical)
    print(f"Matrix MatMul X Grad Check Error: {err_x:.2e}")
    assert err_x < tolerance, f"MatMul X gradient check failed with error {err_x}"

    # --- TEST CASE 2: RELU NON-LINEARITY ---
    x_init = np.array([[-1.5, 2.0], [0.5, -0.2]], dtype=np.float32)
    x = Tensor(x_init, requires_grad=True)
    out = x.relu().sum()
    out.backward()
    
    grad_analytical = x.grad.copy()
    grad_numerical = np.zeros_like(x_init)
    
    for i in range(x_init.shape[0]):
        for j in range(x_init.shape[1]):
            x_pos = Tensor(x_init.copy(), requires_grad=False)
            x_pos.data[i, j] += epsilon
            out_pos = x_pos.relu().sum()
            
            x_neg = Tensor(x_init.copy(), requires_grad=False)
            x_neg.data[i, j] -= epsilon
            out_neg = x_neg.relu().sum()
            
            grad_numerical[i, j] = (out_pos.data - out_neg.data) / (2 * epsilon)
            
    err_relu = relative_error(grad_analytical, grad_numerical)
    print(f"ReLU Grad Check Error: {err_relu:.2e}")
    assert err_relu < tolerance, f"ReLU gradient check failed with error {err_relu}"
    
    print("All basic gradient checks passed successfully!")

if __name__ == "__main__":
    test_operations_grad_check()