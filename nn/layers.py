import numpy as np
from engine.tensor import Tensor

class Linear:
    def __init__(self, in_features, out_features, use_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        
        # He (Kaiming) Normal Initialization for stable variance
        limit = np.sqrt(2.0 / in_features)
        self.W = Tensor(np.random.randn(in_features, out_features) * limit, requires_grad=True)
        
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) if use_bias else None
        
        # Initialize an active binary mask (1 = active, 0 = pruned)
        self.mask = np.ones((in_features, out_features), dtype=np.float32)

    def forward(self, x):
        # Apply the mask directly to the weights during computation
        W_active = self.W * Tensor(self.mask, requires_grad=False)
        out = x @ W_active
        if self.bias is not None:
            out = out + self.bias
        return out

    def __call__(self, x):
        return self.forward(x)

    def get_parameters(self):
        return [self.W, self.bias] if self.bias is not None else [self.W]