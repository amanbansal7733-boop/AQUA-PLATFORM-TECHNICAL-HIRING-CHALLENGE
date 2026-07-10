import numpy as np
from engine.tensor import Tensor

def test_masked_weight_gradients():
    """
    Verifies that a masked weight node drops out of the active computational graph,
    produces zero gradient for masked coordinates, and preserves structural correctness.
    """
    # 1. Initialize deterministic inputs and weights
    np.random.seed(42)
    X_init = np.random.randn(2, 3).astype(np.float32)
    W_init = np.random.randn(3, 2).astype(np.float32)
    
    # Define a static binary mask where element [1, 0] is pruned (set to 0)
    mask = np.ones_like(W_init)
    mask[1, 0] = 0.0  # Force-pruned connection
    
    X = Tensor(X_init, requires_grad=False)
    W = Tensor(W_init, requires_grad=True)
    
    # 2. Forward pass tracking active weights explicitly
    # W_active = W * Mask
    W_active = W * Tensor(mask, requires_grad=False)
    out = X @ W_active
    loss = out.sum()
    
    # 3. Backward pass execution
    loss.backward()
    
    # --- VERIFICATION CRITERIA ---
    # Criterion A: The gradient at the masked coordinate must be exactly 0.0
    pruned_grad = W.grad[1, 0]
    print(f"Gradient at pruned connection [1, 0]: {pruned_grad}")
    assert pruned_grad == 0.0, f"Expected 0.0 gradient for pruned node, got {pruned_grad}"
    
    # Criterion B: Modifying the masked coordinate manually in the source weight tensor
    # must have absolutely zero impact on the forward output.
    W_corrupt = W_init.copy()
    W_corrupt[1, 0] += 999.0  # Massively change the underlying dead weight
    
    X_check = Tensor(X_init, requires_grad=False)
    W_check = Tensor(W_corrupt, requires_grad=False)
    W_active_check = W_check * Tensor(mask, requires_grad=False)
    out_check = X_check @ W_active_check
    
    forward_diff = np.abs(out.data - out_check.data).sum()
    print(f"Forward output deviation when corrupting underlying dead weight: {forward_diff}")
    assert forward_diff == 0.0, "Corrupting an underlying masked weight changed the forward pass calculation!"
    
    print("Masked-weight gradient verification suite passed perfectly!")

if __name__ == "__main__":
    test_masked_weight_gradients()