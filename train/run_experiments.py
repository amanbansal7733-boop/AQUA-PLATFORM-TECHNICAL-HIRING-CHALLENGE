import numpy as np
import matplotlib.pyplot as plt
from engine.tensor import Tensor
from nn.layers import Linear
from nn.optim import Adam
from prune.scheduler import TaylorPruningScheduler
from train.train_baseline import generate_synthetic_data

def train_experiment(criterion="taylor", target_sparsity=0.90, seed=42):
    """Trains an MLP using either Taylor or Magnitude pruning and returns final accuracy and FLOP cost."""
    np.random.seed(seed)
    X_data, Y_data = generate_synthetic_data(num_samples=500)
    
    layer1 = Linear(2, 16)
    layer2 = Linear(16, 2)
    layers = [layer1, layer2]
    optimizer = Adam(layers, lr=0.01)
    
    epochs = 50
    batch_size = 32
    num_samples = X_data.shape[0]
    
    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(num_samples)
        X_shuff, Y_shuff = X_data[indices], Y_data[indices]
        num_batches = int(np.ceil(num_samples / batch_size))
        
        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, num_samples)
            xb = Tensor(X_shuff[start:end], requires_grad=False)
            yb = Tensor(Y_shuff[start:end], requires_grad=False)
            
            h1 = layer1(xb).relu()
            logits = layer2(h1)
            loss = Tensor.softmax_cross_entropy(logits, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # --- PRUNING MILESTONE UPDATE ---
        # Linearly/cubically step up target sparsity towards final value
        progress = min(1.0, epoch / 40.0)
        curr_sparsity = target_sparsity * (progress ** 3)
        
        if curr_sparsity > 0:
            # Collect all matrix weights globally
            all_w = np.concatenate([l.W.data.flatten() for l in layers])
            
            if criterion == "taylor":
                all_scores = np.concatenate([np.abs(l.W.data * l.W.grad).flatten() for l in layers if l.W.grad is not None])
            else:  # Magnitude Pruning Baseline
                all_scores = np.abs(all_w)
                
            k = int(len(all_scores) * curr_sparsity)
            if k < len(all_scores):
                threshold = np.partition(all_scores, k)[k]
                for l in layers:
                    score = np.abs(l.W.data * l.W.grad) if criterion == "taylor" else np.abs(l.W.data)
                    l.mask = (score >= threshold).astype(np.float32)
                    l.W.data *= l.mask

    # Evaluate validation pass metrics
    xb_val = Tensor(X_data, requires_grad=False)
    h1_val = layer1(xb_val).relu()
    logits_val = layer2(h1_val)
    preds = np.argmax(logits_val.data, axis=1)
    acc = np.mean(preds == np.argmax(Y_data, axis=1))
    
    # Calculate Real Multiplications (FLOP count proxy)
    # Dense Forward Layer 1: X (N x 2) @ W (2 x 16) -> 2 * 16 entries. 
    # Sparse Forward Layer 1 uses exactly: (active inputs) * (non-zero weights per input)
    active_ops_l1 = np.sum(layer1.mask) * num_samples
    active_ops_l2 = np.sum(layer2.mask) * num_samples
    total_sparse_flops = active_ops_l1 + active_ops_l2
    
    return acc, total_sparse_flops

def run_pareto_sweep():
    sparsities = [0.0, 0.50, 0.75, 0.90, 0.95]
    seeds = [42, 101, 2023]
    
    results = {"taylor": {"acc": [], "flops": []}, "magnitude": {"acc": [], "flops": []}}
    
    print("Running Part 4 Experimental Comparison Suite across multiple random seeds...")
    for crit in ["taylor", "magnitude"]:
        for s in sparsities:
            acc_runs = []
            flop_runs = []
            for seed in seeds:
                acc, flops = train_experiment(criterion=crit, target_sparsity=s, seed=seed)
                acc_runs.append(acc)
                flop_runs.append(flops)
            results[crit]["acc"].append(np.mean(acc_runs))
            results[crit]["flops"].append(np.mean(flop_runs))
            print(f"[{crit.upper()}] Sparsity: {s*100}%, Mean Acc: {np.mean(acc_runs)*100:.1f}%, Mean FLOPs: {np.mean(flop_runs)}")

    # Plot Pareto Frontier
    plt.figure(figsize=(10, 5))
    plt.plot(sparsities, results["taylor"]["acc"], 'o-', label='Taylor Saliency Pruning', color='blue')
    plt.plot(sparsities, results["magnitude"]["acc"], 's--', label='Magnitude Pruning Baseline', color='red')
    plt.title("Sparsity-Accuracy Pareto Frontier")
    plt.xlabel("Achieved Structural Sparsity")
    plt.ylabel("Mean Accuracy (N=3 Seeds)")
    plt.grid(True)
    plt.legend()
    plt.savefig("pareto_curve.png")
    print("Pareto Curve plot generated and exported safely as 'pareto_curve.png'")

if __name__ == "__main__":
    run_pareto_sweep()