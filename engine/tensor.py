import numpy as np

class Tensor:
    def __init__(self, data, parents=(), op="", requires_grad=True):
        # Convert inputs securely to float32 NumPy arrays
        self.data = np.asarray(data, dtype=np.float32)
        self.parents = parents
        self.op = op
        self.requires_grad = requires_grad
        
        # Initialize gradient storage matching the exact shape of data
        self.grad = None
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
            
        # Internal backward function placeholder for specific operations
        self._backward = lambda: None

    def backward(self):
        """Executes reverse-mode automatic differentiation using topological sorting."""
        if not self.requires_grad:
            return

        # 1. Build a strict topological order of the graph (Post-order traversal)
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build_topo(parent)
                topo.append(v)
                
        build_topo(self)
        
        # 2. Set the seed gradient for the root node to 1.0
        self.grad = np.ones_like(self.data)
        
        # 3. Traverse backward through the sorted nodes and accumulate gradients
        for v in reversed(topo):
            if v.requires_grad and v.op != "":
                v._backward()

    # --- ELEMENT-WISE ADDITION OPERATIONS WITH BROADCASTING ---
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, parents=(self, other), op="+")
        
        def _backward():
            # Standard addition gradient passes through directly
            # We must sum out any dimensions added or expanded by NumPy broadcasting
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(out.grad, other.data.shape)
                
        out._backward = _backward
        return out

    def _unbroadcast(self, grad, target_shape):
        """Reduces broadcasted dimensions to match the parent's target shape."""
        if grad.shape == target_shape:
            return grad
        
        # Sum along axes that were added on the left side
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
            
        # Sum along axes where target shape has a size of 1
        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
                
        return grad

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        return f"Tensor({self.data.tolist()}, op='{self.op}', shape={self.data.shape})"
    
    # --- SUBTRACTION ---
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data - other.data, parents=(self, other), op="-")
        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(-out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return other.__sub__(self)

    # --- ELEMENT-WISE MULTIPLICATION ---
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, parents=(self, other), op="*")
        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    # --- MATRIX MULTIPLICATION ---
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Matrix multiplication requires another Tensor instance.")
        out = Tensor(self.data @ other.data, parents=(self, other), op="@")
        
        def _backward():
            if self.requires_grad:
                # Gradient w.r.t self: out.grad @ other.T
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                # Gradient w.r.t other: self.T @ out.grad
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    # --- RELU NON-LINEARITY ---
    def relu(self):
        out = Tensor(np.maximum(0, self.data), parents=(self,), op="relu")
        def _backward():
            if self.requires_grad:
                # Gradient flows only through elements greater than zero
                self.grad += out.grad * (self.data > 0)
        out._backward = _backward
        return out