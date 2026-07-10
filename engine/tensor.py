import numpy as np

class Tensor:
    def __init__(self, data, parents=(), op="", requires_grad=True):
        self.data = np.asarray(data, dtype=np.float32)
        self.parents = parents
        self.op = op
        self.requires_grad = requires_grad
        
        self.grad = None
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
            
        self._backward = lambda: None

    def backward(self):
        """Executes reverse-mode automatic differentiation using topological sorting."""
        if not self.requires_grad:
            return

        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build_topo(parent)
                topo.append(v)
                
        build_topo(self)
        
        self.grad = np.ones_like(self.data)
        
        for v in reversed(topo):
            if v.requires_grad and v.op != "":
                v._backward()

    # --- ELEMENT-WISE ADDITION ---
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, parents=(self, other), op="+")
        
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.grad = np.zeros_like(self.data)
                self.grad += self._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                if other.grad is None: other.grad = np.zeros_like(other.data)
                other.grad += self._unbroadcast(out.grad, other.data.shape)
                
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    # --- SUBTRACTION ---
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data - other.data, parents=(self, other), op="-")
        
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.grad = np.zeros_like(self.data)
                self.grad += self._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                if other.grad is None: other.grad = np.zeros_like(other.data)
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
                if self.grad is None: self.grad = np.zeros_like(self.data)
                self.grad += self._unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                if other.grad is None: other.grad = np.zeros_like(other.data)
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
                if self.grad is None: 
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad @ other.data.T
                
            if other.requires_grad:
                if other.grad is None: 
                    other.grad = np.zeros_like(other.data)
                other.grad += self.data.T @ out.grad
                
        out._backward = _backward
        return out

    # --- RELU NON-LINEARITY ---
    def relu(self):
        out = Tensor(np.maximum(0, self.data), parents=(self,), op="relu")
        
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.grad = np.zeros_like(self.data)
                self.grad += out.grad * (self.data > 0)
                
        out._backward = _backward
        return out

    # --- REDUCTIONS ---
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), parents=(self,), op="sum")
        
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.grad = np.zeros_like(self.data)
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.data.shape)
                
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), parents=(self,), op="mean")
        
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.grad = np.zeros_like(self.data)
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                num_elements = self.data.size if axis is None else np.prod(np.array(self.data.shape)[axis])
                self.grad += np.broadcast_to(grad, self.data.shape) / num_elements
                
        out._backward = _backward
        return out

    # --- SOFTMAX CROSS-ENTROPY LOSS ---
    @staticmethod
    def softmax_cross_entropy(logits, targets):
        shifted_logits = logits.data - np.max(logits.data, axis=-1, keepdims=True)
        exps = np.exp(shifted_logits)
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
        
        eps = 1e-15
        loss_val = -np.sum(targets.data * np.log(probs + eps)) / logits.data.shape[0]
        out = Tensor(loss_val, parents=(logits, targets), op="softmax_ce")
        
        def _backward():
            if logits.requires_grad:
                if logits.grad is None: logits.grad = np.zeros_like(logits.data)
                logits.grad += (probs - targets.data) / logits.data.shape[0] * out.grad
                
        out._backward = _backward
        return out

    def _unbroadcast(self, grad, target_shape):
        if grad.shape == target_shape:
            return grad
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __repr__(self):
        return f"Tensor({self.data.tolist()}, op='{self.op}', shape={self.data.shape})"