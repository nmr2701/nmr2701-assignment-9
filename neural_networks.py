import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights


        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

    def _activate(self, z, derivative=False):
        if self.activation_fn == 'tanh':
            if derivative:
                return 1 - np.tanh(z) ** 2
            return np.tanh(z)
        elif self.activation_fn == 'relu':
            if derivative:
                return (z > 0).astype(float)
            return np.maximum(0, z)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            if derivative:
                return sig * (1 - sig)
            return sig
        else:
            raise ValueError("Unsupported activation function")
        

    def forward(self, X):
        # Forward pass
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self._activate(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self._activate(self.z2)
        return self.a2

    def backward(self, X, y):
        # Compute gradients
        m = y.shape[0]
        dz2 = self.a2 - y
        self.dW2 = (1/m) * self.a1.T.dot(dz2)  # Store as attribute
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2.dot(self.W2.T)
        dz1 = da1 * self._activate(self.z1, derivative=True)
        self.dW1 = (1/m) * X.T.dot(dz1)  # Store as attribute
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)

        # Update weights and biases
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * db2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps by calling forward and backward function
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
        
    # Plot hidden features
    hidden_features = mlp.a1  # shape (n_samples, hidden_dim)
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)

    x = np.linspace(-1, 1, 10)
    y_grid = np.linspace(-1, 1, 10)
    X_grid, Y_grid = np.meshgrid(x, y_grid)
    if mlp.W2.shape[0] > 2:  # Ensure the third dimension exists
        Z_grid = -(mlp.W2[0, 0] * X_grid + mlp.W2[1, 0] * Y_grid + mlp.b2[0, 0]) / mlp.W2[2, 0]
        ax_hidden.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, color='blue')

    transformed_X = mlp.a1  # Hidden layer activations
    ax_input.scatter(transformed_X[:, 0], transformed_X[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7, edgecolor='k')

    # Plot input layer decision boundary
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(grid)  # Pass grid through the network
    predictions = predictions.reshape(xx.shape)
    ax_input.contourf(xx, yy, predictions, levels=50, cmap='bwr', alpha=0.7)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')


    input_pos = [(0, 0.5), (0, 1)]
    hidden_pos = [(1, 0.3), (1, 0.6), (1, 0.9)]
    output_pos = [(2, 0.5)]

    for x, y in input_pos:
        ax_gradient.scatter(x, y, color='lightcoral', s=100, zorder=2)
    for x, y in hidden_pos:
        ax_gradient.scatter(x, y, color='lightblue', s=100, zorder=2)
    for x, y in output_pos:
        ax_gradient.scatter(x, y, color='lightgreen', s=100, zorder=2)

    for idx_input, (x_start, y_start) in enumerate(input_pos):
        for idx_hidden, (x_end, y_end) in enumerate(hidden_pos):
            line_thickness = np.abs(mlp.dW1[idx_input, idx_hidden])
            ax_gradient.plot(
                [x_start, x_end], [y_start, y_end],
                linewidth=line_thickness * 5, color='gray', alpha=0.8, zorder=1
            )

    for idx_hidden, (x_start, y_start) in enumerate(hidden_pos):
        for idx_output, (x_end, y_end) in enumerate(output_pos):
            line_thickness = np.abs(mlp.dW2[idx_hidden, idx_output])
            ax_gradient.plot(
                [x_start, x_end], [y_start, y_end],
                linewidth=line_thickness  * 5, color='gray', alpha=0.8, zorder=1
            )


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)