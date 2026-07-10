import numpy as np
from engine.tensor import Tensor
from nn.layers import Linear
from nn.optim import Adam

def generate_synthetic_data(num_samples=400):
    """Generates a non-linear concentric circles binary classification dataset."""
    np.random.seed(42)
    samples_per_class = num_samples // 2
    
    # Class 0: Inner Circle
    r1 = np.random.uniform(0.0, 0.4, samples_per_class)
    theta1 = np.random.uniform(0, 2 * np.pi, samples_per_class)
    X0 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)
    Y0 = np.zeros((samples_per_class, 1))
    
    # Class 1: Outer Ring
    r2 = np.random.uniform(0.6, 1.0, samples_per_class)
    theta2 = np.random.uniform(0, 2 * np.pi, samples_per_class)
    X1 = np.stack([r2 * np.cos(theta2), r2 * np.sin(theta2)], axis=1)
    Y1 = np.ones((samples_per_class, 1))
    
    X = np.vstack([X0, X1]).astype(np.float32)
    Y_labels = np.vstack([Y0, Y1]).astype(np.int32)
    
    # Convert labels to one-hot encoding for Softmax Cross-Entropy
    Y_one_hot = np.zeros((num_samples, 2), dtype=np.float32)
    Y_one_hot[np.arange(num_samples), Y_labels.ravel()] = 1.0
    
    return X, Y_one_hot

def train_baseline():
    X_data, Y_data = generate_synthetic_data(num_samples=500)
    
    # Define a 2-layer MLP architecture (2 -> 16 -> 2)
    hidden_dim = 16
    layer1 = Linear(2, hidden_dim)
    layer2 = Linear(hidden_dim, 2)
    layers = [layer1, layer2]
    
    optimizer = Adam(layers, lr=0.01)
    epochs = 40
    batch_size = 32
    num_samples = X_data.shape[0]
    
    print("Starting Part 2 Dense Baseline Training Loop...")
    print(f"Epoch\tLoss\t\tAccuracy")
    print("-" * 35)
    
    for epoch in range(1, epochs + 1):
        # Shuffle dataset per epoch
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
            
            # Forward Pass
            h1 = layer1(xb).relu()
            logits = layer2(h1)
            
            # Loss Calculation
            loss = Tensor.softmax_cross_entropy(logits, yb)
            
            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics Evaluation
            epoch_loss += loss.data * (end_idx - start_idx)
            predictions = np.argmax(logits.data, axis=1)
            targets_idx = np.argmax(yb.data, axis=1)
            correct_predictions += np.sum(predictions == targets_idx)
            
        total_loss = epoch_loss / num_samples
        total_acc = correct_predictions / num_samples
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch}\t{total_loss:.4f}\t\t{total_acc * 100:.1f}%")
            
    print("-" * 35)
    print("Baseline Dense Training verification completed successfully!")

if __name__ == "__main__":
    train_baseline()