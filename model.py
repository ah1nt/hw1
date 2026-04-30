import numpy as np

class Layer:
    def forward(self, inputs):
        raise NotImplementedError
        
    def backward(self, grad_output):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, in_features, out_features):
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros(out_features)
        
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.W) + self.b
        
    def backward(self, grad_output):
        self.grad_W = np.dot(self.inputs.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self.W.T)

class ReLU(Layer):
    def __init__(self):
        self.inputs = None
        
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)
        
    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.inputs <= 0] = 0
        return grad_input

class Sigmoid(Layer):
    def __init__(self):
        self.out = None
        
    def forward(self, inputs):
        # Clip to prevent overflow
        inputs_clipped = np.clip(inputs, -500, 500)
        self.out = 1 / (1 + np.exp(-inputs_clipped))
        return self.out
        
    def backward(self, grad_output):
        return grad_output * self.out * (1 - self.out)

class Tanh(Layer):
    def __init__(self):
        self.out = None
        
    def forward(self, inputs):
        self.out = np.tanh(inputs)
        return self.out
        
    def backward(self, grad_output):
        return grad_output * (1 - self.out ** 2)

class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.targets = None
        
    def forward(self, logits, targets):
        self.targets = targets
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        batch_size = logits.shape[0]
        log_likelihood = -np.log(self.probs[np.arange(batch_size), targets] + 1e-8)
        return np.sum(log_likelihood) / batch_size
        
    def backward(self):
        batch_size = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(batch_size), self.targets] -= 1
        return grad / batch_size

class SGDOptimizer:
    def __init__(self, model, lr=0.01, weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        
    def step(self):
        for layer in self.model.layers:
            if isinstance(layer, Linear):
                layer.W -= self.lr * (layer.grad_W + self.weight_decay * layer.W)
                layer.b -= self.lr * layer.grad_b

    def zero_grad(self):
        for layer in self.model.layers:
            if isinstance(layer, Linear):
                layer.grad_W.fill(0)
                layer.grad_b.fill(0)

class MLP:
    def __init__(self, input_dim, hidden_dim, num_classes, activation='relu'):
        self.layers = []
        self.layers.append(Linear(input_dim, hidden_dim))
        
        if activation.lower() == 'relu':
            self.layers.append(ReLU())
        elif activation.lower() == 'sigmoid':
            self.layers.append(Sigmoid())
        elif activation.lower() == 'tanh':
            self.layers.append(Tanh())
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self.layers.append(Linear(hidden_dim, num_classes))
        
    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
        
    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def get_weights(self):
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers if isinstance(layer, Linear)]

    def set_weights(self, weights):
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W = weights[idx][0].copy()
                layer.b = weights[idx][1].copy()
                idx += 1
