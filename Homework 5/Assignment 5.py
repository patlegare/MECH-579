import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Problem parameters
T = 15.0
N = 45
vmax = 1.5
dt = T / (N - 1)


def unpack(z):
    #split into x and y arrays
    x = z[:N]
    y = z[N:]
    return x, y


def cost_field(x, y):
    #cost function
    return 1.0 / ((x - 5.0) ** 2 + (y - 5.0) ** 2 + 1.0)


def objective(z):
    #objective function
    x, y = unpack(z)
    term1 = np.sum(cost_field(x, y))
    dx = np.diff(x)
    dy = np.diff(y)
    term2 = np.sum(dx * dx + dy * dy)
    return term1 + term2


def objective_grad(z):
    #analytical gradiet of the objective function
    x, y = unpack(z)

    denom = (x - 5.0) ** 2 + (y - 5.0) ** 2 + 1.0
    dC_dx = -2.0 * (x - 5.0) / (denom ** 2)
    dC_dy = -2.0 * (y - 5.0) / (denom ** 2)

    grad2_x = np.zeros_like(x)
    grad2_y = np.zeros_like(y)

    for j in range(N):
        if j > 0:
            grad2_x[j] += 2.0 * (x[j] - x[j - 1])
            grad2_y[j] += 2.0 * (y[j] - y[j - 1])
        if j < N - 1:
            grad2_x[j] -= 2.0 * (x[j + 1] - x[j])
            grad2_y[j] -= 2.0 * (y[j + 1] - y[j])

    grad_x = dC_dx + grad2_x
    grad_y = dC_dy + grad2_y

    return np.concatenate([grad_x, grad_y])


# Constraints
constraints = []


def add_eq_constraint(name, indices, coeffs, rhs):
    #coefficients and indices equals RHS
    indices = np.asarray(indices, dtype=int)
    coeffs = np.asarray(coeffs, dtype=float)

    def fun(z):
        return float(np.dot(coeffs, z[indices]) - rhs)

    def jac(z):
        g = np.zeros_like(z)
        g[indices] = coeffs
        return g

    constraints.append({"type": "eq", "fun": fun, "jac": jac, "name": name})


# Boundary conditions on position
add_eq_constraint("x_start", [0], [1.0], 0.0)
add_eq_constraint("y_start", [N], [1.0], 0.0)
add_eq_constraint("x_end", [N - 1], [1.0], 10.0)
add_eq_constraint("y_end", [2 * N - 1], [1.0], 10.0)

# Initial velocity equal to zero: x2 = x1 and y2 = y1
add_eq_constraint("vx_initial", [1, 0], [1.0, -1.0], 0.0)
add_eq_constraint("vy_initial", [N + 1, N], [1.0, -1.0], 0.0)


def make_speed_constraint(k):
    """
    Speed constraint for segment k:
    vmax^2 - (u_k^2 + v_k^2) >= 0
    """
    def fun(z):
        x, y = unpack(z)
        u = (x[k + 1] - x[k]) / dt
        v = (y[k + 1] - y[k]) / dt
        return vmax ** 2 - (u * u + v * v)

    def jac(z):
        x, y = unpack(z)
        u = (x[k + 1] - x[k]) / dt
        v = (y[k + 1] - y[k]) / dt

        g = np.zeros_like(z)

        du_dxk1 = 1.0 / dt
        du_dxk = -1.0 / dt
        dv_dyk1 = 1.0 / dt
        dv_dyk = -1.0 / dt

        g[k] += -2.0 * u * du_dxk
        g[k + 1] += -2.0 * u * du_dxk1
        g[N + k] += -2.0 * v * dv_dyk
        g[N + k + 1] += -2.0 * v * dv_dyk1

        return g

    return {"type": "ineq", "fun": fun, "jac": jac, "name": f"speed_{k}"}


# Add speed constraints for all segments
for k in range(N - 1):
    constraints.append(make_speed_constraint(k))


# Bounds on position: 0 <= x_i, y_i <= 10
bounds = [(0.0, 10.0)] * (2 * N)


# Initial guess: just a traight line from (0,0) to (10,10)
x_init = np.linspace(0.0, 10.0, N)
y_init = np.linspace(0.0, 10.0, N)
z0 = np.concatenate([x_init, y_init])


# History for plots
iter_history = []
grad_norm_history = []
eq_violation_history = []
ineq_violation_history = []


def callback(xk):
    #stores gradients and constraints values for each iteration
    k = len(iter_history)
    iter_history.append(k)

    g = objective_grad(xk)
    grad_norm_history.append(np.linalg.norm(g))

    eq_vals = []
    ineq_vals = []

    for c in constraints:
        val = c["fun"](xk)
        if c["type"] == "eq":
            eq_vals.append(val)
        else:
            ineq_vals.append(val)

    if eq_vals:
        eq_violation_history.append(float(np.max(np.abs(eq_vals))))
    else:
        eq_violation_history.append(0.0)

    if ineq_vals:
        ineq_violation_history.append(float(np.min(ineq_vals)))
    else:
        ineq_violation_history.append(0.0)


def main():
    # Solve optimization problem
    result = minimize(
        objective,
        z0,
        method="SLSQP",
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={"maxiter": 200, "ftol": 1e-6, "disp": True},
    )

    print("\nSuccess:", result.success)
    print("Message:", result.message)
    print("Final objective value:", result.fun)

    z_opt = result.x
    x_opt, y_opt = unpack(z_opt)

    # a) Path on contour plot of cost field
    grid_x = np.linspace(0.0, 10.0, 200)
    grid_y = np.linspace(0.0, 10.0, 200)
    X, Y = np.meshgrid(grid_x, grid_y)
    C = cost_field(X, Y)

    plt.figure(figsize=(6, 5))
    cs = plt.contourf(X, Y, C, levels=40)
    plt.colorbar(cs, label="C(x, y)")
    plt.plot(x_init, y_init, "o--", label="Initial guess")
    plt.plot(x_opt, y_opt, "o-", label="Optimized path")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.title("Drone path on cost field")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # b) Convergence of gradient norm
    plt.figure(figsize=(6, 4))
    plt.semilogy(iter_history, grad_norm_history, "o-")
    plt.xlabel("Iteration")
    plt.ylabel("Norm of gradient of objective")
    plt.title("Gradient convergence (proxy for Lagrangian gradient)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # c) Constraint values as function of iterations
    plt.figure(figsize=(6, 4))
    plt.semilogy(iter_history, eq_violation_history, "o-", label="Max abs equality")
    plt.plot(iter_history, ineq_violation_history, "s-", label="Min inequality")
    plt.axhline(0.0, color="k", linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Constraint metrics")
    plt.title("Constraint convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Final constraint table
    print("\nFinal constraint function values")
    print("(Equality: h(x) = 0, Inequality: g(x) >= 0)\n")
    print("{:<20s} {:>15s}".format("Constraint", "value"))
    for c in constraints:
        val = c["fun"](z_opt)
        name = c.get("name", "unnamed")
        print("{:<20s} {:>15.6e}".format(name, val))


if __name__ == "__main__":
    main()
