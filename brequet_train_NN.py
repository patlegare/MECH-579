import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


#Brequet range model

def density(h):
    """Air density at altitude h (m)."""
    return 1.2 * (1.0 - h * 0.0065 / 288.0) ** 5.26


def mass_fuel_rate(v, h):
    """Mass fuel rate using mass flow of air through the turbine."""
    turbine_area = np.pi * 0.92 ** 2 / 2.0
    FAR = 1e-1
    return density(h) * v * turbine_area * FAR


def C_L(v, h, weight, S):
    """Lift coefficient."""
    return 9.81 * weight / (0.5 * density(h) * v ** 2 * S)


def C_Dw(v, h):
    """Wave-drag coefficient."""
    return 10.0 * (np.arctan(10.0 * ((v / (343.0 * 0.7)) ** 2 - 1.0)) + np.pi / 2.0)


def C_D(v, h, weight, S):
    """Total drag coefficient (parasite + induced + wave)."""
    e = 0.8
    AR = 10.0
    C_D0 = 0.0083
    return C_D0 + C_L(v, h, weight, S) ** 2 / (np.pi * e * AR) + C_Dw(v, h)


def ct(v, h, weight, S):
    """Specific fuel consumption."""
    c_t_value = mass_fuel_rate(v, h) / (0.5 * density(h) * v ** 2 * S * C_D(v, h, weight, S))
    return c_t_value + 1e-5


def breguet_range(v, h):
    """
    Breguet range (km) as a function of velocity v (m/s) and altitude h (m).
    v and h can be NumPy arrays of the same shape.
    """
    S = 100.0
    W_empty = 162_400.0
    W_fuel_total = 146_571.0

    # Representative cruise weight: 75% of fuel remaining
    fuel_percentage = 0.75
    weight_used = W_empty + fuel_percentage * W_fuel_total

    # Initial and final weights in the Breguet formula
    W_i = W_empty + W_fuel_total
    W_f = W_empty

    rng = (
        v / ct(v, h, weight_used, S)
        * C_L(v, h, weight_used, S) / C_D(v, h, weight_used, S)
        * np.log(W_i / W_f)
    )
    return rng / 1e3  # convert to km


#dataset generation

V_MIN, V_MAX = 10.0, 300.0        # m/s (avoid V=0 to keep formula well-defined)
H_MIN, H_MAX = 0.0, 25_000.0      # m


def generate_dataset(n_samples, seed=0):
    """
    Generate n_samples of (V, h) and corresponding Breguet range values.

    Returns:
        X_tensor : [n_samples, 2] normalized inputs in [0,1]^2 (torch.float32)
        y_tensor : [n_samples, 1] normalized ranges (torch.float32)
        y_scale  : scaling factor used for the targets
    """
    rng = np.random.default_rng(seed)

    v = rng.uniform(V_MIN, V_MAX, size=(n_samples, 1))
    h = rng.uniform(H_MIN, H_MAX, size=(n_samples, 1))

    # Normalise inputs to [0,1] for better conditioning
    v_norm = (v - V_MIN) / (V_MAX - V_MIN)
    h_norm = (h - H_MIN) / (H_MAX - H_MIN)
    X_norm = np.hstack([v_norm, h_norm])

    # True Breguet range (km)
    y = breguet_range(v, h).reshape(-1, 1)

    # Scale target to O(1) to help optimization
    y_scale = 15_000.0  # typical order of magnitude of the range
    y_norm = y / y_scale

    X_tensor = torch.from_numpy(X_norm).float()
    y_tensor = torch.from_numpy(y_norm).float()

    return X_tensor, y_tensor, y_scale


# NN model

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_sizes=None, output_dim=1):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 32]

        layers = []
        prev_dim = input_dim
        for hdim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.Tanh())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


#train

def train_network(n_samples, hidden_sizes, num_epochs=10_000, lr=1e-3, seed=0):
    """
    Train an MLP to approximate the Breguet range for a given
    sample size and architecture.

    Returns:
        model        : trained MLP
        loss_history : NumPy array of MSE loss per epoch
        y_scale      : target scaling factor
    """
    torch.manual_seed(seed)
    X, y, y_scale = generate_dataset(n_samples, seed=seed)

    model = MLP(input_dim=2, hidden_sizes=hidden_sizes, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 1000 == 0:
            print(
                f"Epoch {epoch+1:5d} | "
                f"samples={n_samples:4d} | hidden={hidden_sizes} | "
                f"loss={loss.item():.3e}"
            )

    return model, np.array(loss_history), y_scale


#for 3a

def main():
    base_samples = 50
    num_epochs = 10_000

    # 1) Base case: 50 samples, 2 hidden layers with 32 neurons each
    base_hidden = [32, 32]
    print("Training base model (50 samples, hidden layers [32, 32])...")
    base_model, base_loss, y_scale = train_network(
        base_samples, base_hidden, num_epochs=num_epochs, lr=1e-3, seed=0
    )

    # 2) Change size of the initial sample set: 200 samples, same architecture
    large_samples = 200
    print("Training with larger sample set (200 samples, hidden layers [32, 32])...")
    _, loss_200, _ = train_network(
        large_samples, base_hidden, num_epochs=num_epochs, lr=1e-3, seed=1
    )

    # 3) Change the neural network model itself (same N = 50)
    #    a) Smaller network: one hidden layer of 16 neurons
    small_hidden = [16]
    print("Training smaller network (50 samples, hidden layer [16])...")
    _, loss_small, _ = train_network(
        base_samples, small_hidden, num_epochs=num_epochs, lr=1e-3, seed=2
    )

    #    b) Larger network: two hidden layers of 64 neurons each
    large_hidden = [64, 64]
    print("Training larger network (50 samples, hidden layers [64, 64])...")
    _, loss_large_net, _ = train_network(
        base_samples, large_hidden, num_epochs=num_epochs, lr=1e-3, seed=3
    )

    epochs = np.arange(1, num_epochs + 1)

    # Plot A: convergence of loss for different sample sizes (same architecture)
    plt.figure(figsize=(7, 5))
    plt.semilogy(epochs, base_loss, label=f"N = {base_samples}, hidden {base_hidden}")
    plt.semilogy(epochs, loss_200, label=f"N = {large_samples}, hidden {base_hidden}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (scaled targets)")
    plt.title("Convergence of loss for different sample sizes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot B: convergence of loss for different architectures (same N = 50)
    plt.figure(figsize=(7, 5))
    plt.semilogy(epochs, loss_small, label=f"hidden {small_hidden}")
    plt.semilogy(epochs, base_loss, label=f"hidden {base_hidden}")
    plt.semilogy(epochs, loss_large_net, label=f"hidden {large_hidden}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (scaled targets)")
    plt.title(f"Convergence of loss for different architectures (N = {base_samples})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
