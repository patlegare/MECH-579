import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# import from your part (a) file
from brequet_train_NN import (
    MLP,
    breguet_range,
    V_MIN, V_MAX,
    H_MIN, H_MAX,
    train_network,
)

# optimize using the trained NN

def optimize_with_surrogate(model, y_scale,
                            num_epochs_opt=2000, lr_opt=5e-3, seed=0):
    """
    Optimize the design variables (V, h) using the trained NN surrogate.

    model       : trained MLP that takes normalized [v_norm, h_norm] in [0,1]^2
                  and outputs normalized range (scaled by y_scale).
    y_scale     : factor used to unscale target from training.
    num_epochs_opt : number of gradient-based optimization steps
    lr_opt      : learning rate for Adam optimizer
    """

    torch.manual_seed(seed)

    # design variables are the normalized inputs [v_norm, h_norm]
    x = torch.rand(1, 2, requires_grad=True)  # shape (1, 2), random start in [0,1]

    optimizer = torch.optim.Adam([x], lr=lr_opt)

    range_pred_history = []
    range_true_history = []

    t_start = time.time()

    for epoch in range(num_epochs_opt):
        optimizer.zero_grad()

        # keep x inside [0,1]^2 so that V, h stay in the design box
        with torch.no_grad():
            x.clamp_(0.0, 1.0)

        # predicted normalized range from surrogate NN
        y_hat_norm = model(x)
        y_hat = y_scale * y_hat_norm  # km

        # we want to maximize range, so we minimize the negative range
        loss = -y_hat.mean()
        loss.backward()
        optimizer.step()

        # record surrogate-predicted range
        range_pred = float(y_hat.detach().numpy())
        range_pred_history.append(range_pred)

        # map normalized x back to physical V, h
        v = V_MIN + (V_MAX - V_MIN) * float(x[0, 0].detach().numpy())
        h = H_MIN + (H_MAX - H_MIN) * float(x[0, 1].detach().numpy())

        # compute the true Breguet range at this design
        range_true = float(breguet_range(np.array([[v]]), np.array([[h]])))
        range_true_history.append(range_true)

    t_end = time.time()
    avg_time_per_epoch = (t_end - t_start) / num_epochs_opt

    # final design
    with torch.no_grad():
        x.clamp_(0.0, 1.0)
    v_opt = V_MIN + (V_MAX - V_MIN) * float(x[0, 0].detach().numpy())
    h_opt = H_MIN + (H_MAX - H_MIN) * float(x[0, 1].detach().numpy())

    return (
        v_opt,
        h_opt,
        np.array(range_pred_history),
        np.array(range_true_history),
        avg_time_per_epoch,
    )


#scipy optimization for time comparison

def scipy_reference():
    """
    Run a classical optimizer (SQP-like) on the true Breguet range
    and return average time per iteration and the resulting optimum.
    """

    def neg_range_from_norm(x_norm):
        v = V_MIN + (V_MAX - V_MIN) * x_norm[0]
        h = H_MIN + (H_MAX - H_MIN) * x_norm[1]
        return -breguet_range(np.array([[v]]), np.array([[h]]))[0, 0]

    x0 = np.array([0.5, 0.5])  # center of the box in normalized space
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    t0 = time.time()
    # can swtihc method
    result = minimize(neg_range_from_norm, x0, method="L-BFGS-B", bounds=bounds)
    t1 = time.time()

    avg_time = (t1 - t0) / result.nit
    v_star = V_MIN + (V_MAX - V_MIN) * result.x[0]
    h_star = H_MIN + (H_MAX - H_MIN) * result.x[1]
    range_star = breguet_range(np.array([[v_star]]), np.array([[h_star]]))[0, 0]

    return v_star, h_star, range_star, avg_time


# Question 3 b 

def main():
    # for simplicity, retrain with the base setup from Q3(a):
    n_samples = 50
    hidden_sizes = [32, 32]
    num_epochs_sur = 10_000

    print("Training surrogate NN for Breguet range (reuse of part a setup)...")
    surrogate, loss_hist, y_scale = train_network(
        n_samples, hidden_sizes, num_epochs=num_epochs_sur, lr=1e-3, seed=0
    )
    surrogate.eval()
    print(f"Final training loss (scaled targets): {loss_hist[-1]:.3e}")

    # 2) optimize (V, h) using the trained surrogate
    num_epochs_opt = 2000
    print("Optimizing design variables using surrogate model...")
    v_opt, h_opt, range_pred_hist, range_true_hist, t_nn = optimize_with_surrogate(
        surrogate, y_scale,
        num_epochs_opt=num_epochs_opt,
        lr_opt=5e-3,
        seed=1,
    )
    print(f"NN-based optimum (surrogate):  V* ≈ {v_opt:.3f} m/s, h* ≈ {h_opt:.1f} m")
    print(f"Predicted final range (surrogate) ≈ {range_pred_hist[-1]:.2f} km")
    print(f"Average NN optimization time per epoch ≈ {t_nn*1e3:.3f} ms")

    # 3) SciPy reference for timing comparison
    print("Running SciPy reference optimization on true Breguet range...")
    v_sci, h_sci, range_sci, t_sci = scipy_reference()
    print(f"SciPy optimum:  V* ≈ {v_sci:.3f} m/s, h* ≈ {h_sci:.1f} m")
    print(f"True range at SciPy optimum ≈ {range_sci:.2f} km")
    print(f"Average SciPy time per iteration ≈ {t_sci*1e3:.3f} ms")

    # 4) plot convergence of objective vs epochs
    epochs_opt = np.arange(1, num_epochs_opt + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs_opt, range_pred_hist, "b-", label="Predicted range (NN surrogate)")
    plt.plot(epochs_opt, range_true_hist, "r--", label="True Breguet range at (V,h)")
    plt.xlabel("Epoch")
    plt.ylabel("Range (km)")
    plt.title("Convergence of objective vs epochs\nNN-based optimization of Breguet range")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
