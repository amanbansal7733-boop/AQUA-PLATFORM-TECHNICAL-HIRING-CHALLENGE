import numpy as np

class TaylorPruningScheduler:
    def __init__(self, layers, initial_sparsity=0.0, final_sparsity=0.90, start_step=1, end_step=30):
        """
        Manages gradual structural pruning across target network layers using a cubic ramp.
        """
        self.layers = layers
        self.s_i = initial_sparsity
        self.s_f = final_sparsity
        self.t_0 = start_step
        self.n_steps = end_step - start_step
        self.current_step = 0

    def compute_target_sparsity(self):
        """Calculates the target sparsity for the current training milestone using a cubic curve."""
        if self.current_step < self.t_0:
            return self.s_i
        if self.current_step >= (self.t_0 + self.n_steps):
            return self.s_f
            
        # Cubic schedule progression calculation
        progress = (self.current_step - self.t_0) / self.n_steps
        sparsity = self.s_f + (self.s_i - self.s_f) * ((1.0 - progress) ** 3)
        return sparsity

    def step(self):
        self.current_step += 1
        target_sparsity = self.compute_target_sparsity()
        
        if target_sparsity <= 0.0:
            return target_sparsity

        # 1. Gather global saliency scores across all target network layers
        all_scores = []
        for layer in self.layers:
            # Taylor expansion importance estimate: |Weight * Gradient|
            # We use the raw accumulated analytical gradient before any mask overrides
            if layer.W.grad is not None:
                saliency = np.abs(layer.W.data * layer.W.grad)
                all_scores.append(saliency.flatten())
                
        if not all_scores:
            return target_sparsity
            
        global_scores = np.concatenate(all_scores)
        
        # 2. Determine global threshold value for current sparsity milestone
        k = int(len(global_scores) * target_sparsity)
        if k >= len(global_scores):
            k = len(global_scores) - 1
            
        threshold = np.partition(global_scores, k)[k]

        # 3. Apply threshold to update masks for individual layers
        for layer in self.layers:
            if layer.W.grad is not None:
                saliency = np.abs(layer.W.data * layer.W.grad)
                # Connections falling below global importance threshold are masked out
                layer.mask = (saliency >= threshold).astype(np.float32)
                
                # Instantly clear dead data to enforce structural honesty
                layer.W.data *= layer.mask
                
        return target_sparsity