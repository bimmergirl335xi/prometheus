# neural.py
import numpy as np
from config import GPU_AVAILABLE, xp  # Uses CuPy if GPU is on, otherwise NumPy

class SimpleNeuralNet:
    """
    A lightweight Multi-Layer Perceptron (MLP).
    Structure: Input -> Hidden Layer -> Output Layer
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize Weights (random small numbers)
        # W1 connects Input -> Hidden
        self.W1 = xp.random.randn(input_size, hidden_size) * 0.1
        # b1 is the bias for the Hidden layer
        self.b1 = xp.zeros((1, hidden_size))
        
        # W2 connects Hidden -> Output
        self.W2 = xp.random.randn(hidden_size, output_size) * 0.1
        # b2 is the bias for the Output layer
        self.b2 = xp.zeros((1, output_size))

    def forward(self, X):
        """
        The 'Guessing' Phase.
        X: The input data vector
        Returns: The network's prediction
        """
        # Ensure input is 2D array (batch_size, input_size)
        self.X = xp.array(X).reshape(1, -1)

        # 1. Input -> Hidden
        # Z1 is the raw math: (Input * Weights) + Bias
        self.z1 = xp.dot(self.X, self.W1) + self.b1
        
        # 2. Activation (ReLU)
        # ReLU: "If negative, become 0. If positive, stay positive."
        # This allows the net to learn non-linear patterns.
        self.a1 = xp.maximum(0, self.z1)

        # 3. Hidden -> Output
        self.z2 = xp.dot(self.a1, self.W2) + self.b2
        
        # 4. Final Activation (Linear for regression, or Sigmoid/Softmax for classification)
        # We'll use Linear here (raw numbers) for predicting state vectors.
        output = self.z2
        return output

    def train(self, inputs, targets, modulation: float = 1.0):
        """
        The 'Learning' Phase (Backpropagation).
        inputs: What the agent saw
        targets: What actually happened (the truth)
        """
        # 1. Make a guess (Forward pass)
        prediction = self.forward(inputs)
        
        # 2. Calculate Error (How wrong were we?)
        targets = xp.array(targets).reshape(1, -1)
        error = prediction - targets
        
        # Calculate Loss (Mean Squared Error) - just for tracking
        loss = xp.mean(error ** 2)

        # 3. Backpropagation (The "Blame Game")
        # We calculate gradients (derivatives) to see which weights contributed to the error.
        
        # Gradient for Output Layer
        # (Derivative of MSE with linear activation is just 2 * error)
        delta_output = 2 * error
        
        # Calculate change for W2 and b2
        d_W2 = xp.dot(self.a1.T, delta_output)
        d_b2 = xp.sum(delta_output, axis=0, keepdims=True)

        # DYNAMIC LEARNING RATE
        # The agent can now "focus" harder (modulation > 1.0) or cement memories (modulation < 0.1)
        effective_lr = self.learning_rate * float(modulation)

        # Gradient for Hidden Layer
        # We propagate the error backwards through W2, passing through the ReLU derivative
        error_hidden = xp.dot(delta_output, self.W2.T)
        
        # Derivative of ReLU: 1 if positive, 0 if negative
        relu_derivative = (self.z1 > 0).astype(float)
        delta_hidden = error_hidden * relu_derivative

        # Calculate change for W1 and b1
        d_W1 = xp.dot(self.X.T, delta_hidden)
        d_b1 = xp.sum(delta_hidden, axis=0, keepdims=True)

        # 4. Update Weights (Gradient Descent)
        # "Move the weights slightly in the opposite direction of the error"
        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1
        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2

        return float(loss) # Return loss so the agent knows how confused it is
    
    def inject_noise(self, noise_level: float = 0.01):
        """
        Simulates 'Temperature' in LLMs.
        Jiggles weights slightly to break out of local minima (creative rut).
        """
        self.W1 += xp.random.randn(*self.W1.shape) * noise_level
        self.W2 += xp.random.randn(*self.W2.shape) * noise_level