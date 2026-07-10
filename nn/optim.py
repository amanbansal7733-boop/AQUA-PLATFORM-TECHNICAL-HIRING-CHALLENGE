import numpy as np

class Adam:
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Initialize moment vectors tracking each parameter tensor
        self.m = {}
        self.v = {}
        
        for layer in self.layers:
            for p in layer.get_parameters():
                self.m[p] = np.zeros_like(p.data)
                self.v[p] = np.zeros_like(p.data)

    def step(self):
        self.t += 1
        for layer in self.layers:
            # Mask tracking array configuration
            weight_mask = layer.mask
            
            for p in layer.get_parameters():
                if p.grad is None:
                    continue
                
                # Determine if this parameter is the weight matrix subject to pruning
                is_weight = (p.data.shape == weight_mask.shape)
                
                # Compute bias-corrected moments
                if is_weight:
                    # Trapped Bug Fixed: Only update optimizer moments where mask == 1
                    # Historical moments are frozen exactly in time for pruned elements
                    active = (weight_mask == 1.0)
                    
                    self.m[p][active] = self.beta1 * self.m[p][active] + (1 - self.beta1) * p.grad[active]
                    self.v[p][active] = self.beta2 * self.v[p][active] + (1 - self.beta2) * (p.grad[active] ** 2)
                else:
                    # Standard unmasked parameter update (e.g., biases)
                    self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * p.grad
                    self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (p.grad ** 2)
                
                # Bias correction calculations
                m_hat = self.m[p] / (1 - self.beta1 ** self.t)
                v_hat = self.v[p] / (1 - self.beta2 ** self.t)
                
                # Update parameter values safely
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for layer in self.layers:
            for p in layer.get_parameters():
                if p.grad is not None:
                    p.grad.fill(0.0)