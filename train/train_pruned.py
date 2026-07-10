import numpy as np
from engine.tensor import Tensor
from nn.layers import Linear
from nn.optim import Adam
from prune.scheduler import TaylorPruningScheduler
from train.train_baseline import generate_synthetic_data

def train_self_pruning():
    X_data, Y_data = generate_synthetic_data(num_samples=500)
    
    layer1 = Linear(2, 16)
    layer2 = Linear(16, 2)
    layers = [layer1, layer2]
    
    optimizer = Adam(layers, lr=0.01)
    # Give the model plenty of steps to gradually prune and re-equilibriate
    epochs = 60 
    batch_size = 32
    num_samples = X_data.shape[0]
    
    # Configure the scheduler to hit a punishing 90% final structural sparsity
    prune_scheduler = TaylorPruningScheduler(layers, initial_sparsity=0.0, final_sparsity=0.90, start_step=5, end_step=45)
    
    print("Starting Part 3 Taylor-Expansion Self-Pruning Run...")
    print(f"Epoch\tLoss\t\tAccuracy\tSparsity Status")
    print("-" * 60)
    
    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(num_samples)
        X_shuffled = X_data[indices]
        Y_shuffled = Y_data[indices]
        
        epoch_loss = 0.0
        correct_predictions = 0
        num_batches = int(np.ceil(num_samples / batch_size))
        
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            xb = Tensor(X_shuffled[start_idx:end_idx], requires_grad=False)
            yb = Tensor(Y_shuffled[start_idx:end_idx], requires_grad=False)
            
            h1 = layer1(xb).relu()
            logits = layer2(h1)
            loss = Tensor.softmax_cross_entropy(logits, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.data * (end_idx - start_idx)
            correct_predictions += np.sum(np.argmax(logits.data, axis=1) == np.argmax(yb.data, axis=1))
            
        # Update pruning schedule at the end of each training epoch
        current_sparsity = prune_scheduler.step()
        
        total_loss = epoch_loss / num_samples
        total_acc = correct_predictions / num_samples
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch}\t{total_loss:.4f}\t\t{total_acc * 100:.1f}%\t\tTarget: {current_sparsity * 100:.1f}%")
            
    # Calculate exactly how many structural zero weights remain across the model
    total_weights = layer1.W.data.size + layer2.W.data.size
    active_weights = np.sum(layer1.mask) + np.sum(layer2.mask)
    achieved_sparsity = 1.0 - (active_weights / total_weights)
    
    print("-" * 60)
    print(f"Pruning completed! Verified True Sparse Matrix Sparsity: {achieved_sparsity * 100:.2f}%")

if __name__ == "__main__":
    train_self_pruning()