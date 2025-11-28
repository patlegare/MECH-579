import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Problem parameters
T = 15.0
N = 45
vmax = 1.5
dt = T / (N - 1)


def unpack(z):
    # split into x and y arrays
    x = z[:N]
    y = z[N:]
    return x, y


def cost_field(x, y):
    # cost function question 1
    return 1.0 / ((x - 5.0) ** 2 + (y - 5.0) ** 2 + 1.0)
    # cost function question 2 and 3
    #return np.cos(x) ** 2 * np.cos(y) ** 2


def cost_grad(x, y):
    # analytical gradient of the cost function
    # for question 1
    denom = (x - 5.0) ** 2 + (y - 5.0) ** 2 + 1.0
    dC_dx = -2.0 * (x - 5.0) / (denom ** 2)
    dC_dy = -2.0 * (y - 5.0) / (denom ** 2)

    # for question 2 and 3
    #dC_dx = -np.sin(2.0 * x) * (np.cos(y) ** 2)
    #dC_dy = -(np.cos(x) ** 2) * np.sin(2.0 * y)

    return dC_dx, dC_dy


def objective(z):
    # objective function
    x, y = unpack(z)
    term1 = np.sum(cost_field(x, y))
    dx = np.diff(x)
    dy = np.diff(y)
    term2 = np.sum(dx * dx + dy * dy)
    return term1 + term2


def objective_grad(z):
    # analytical gradient of the objective function
    x, y = unpack(z)

    dC_dx, dC_dy = cost_grad(x, y)

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
    # coefficients and indices equals RHS
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
    # Speed constraint for segment k: vmax^2 - (u_k^2 + v_k^2) >= 0
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

# Initial guess: slightly perturbed line from (0,0) to (10,10)
x_init = np.linspace(0.0, 10.0, N)
t = np.linspace(0.0, np.pi, N)
y_init = np.linspace(0.0, 10.0, N) + 0.5 * np.sin(t)
# enforce box bounds
y_init = np.clip(y_init, 0.0, 10.0)
z0 = np.concatenate([x_init, y_init])


#Question 1/2: single drone

def main_single_drone():
    #histories for SLSQP (objective gradient & constraints)
    iter_history = []
    grad_norm_history = []
    eq_violation_history = []
    ineq_violation_history = []

    def callback_slsqp(xk):
        # stores objective gradients and constraint violations for each iteration (SLSQP)
        k = len(iter_history)
        iter_history.append(k)

        g = objective_grad(xk)
        grad_norm_history.append(np.linalg.norm(g))

        max_eq_violation = 0.0
        max_ineq_violation = 0.0

        for c in constraints:
            val = c["fun"](xk)
            if c["type"] == "eq":
                max_eq_violation = max(max_eq_violation, abs(val))
            else:
                # inequality is g(z) >= 0, violation only if negative
                max_ineq_violation = max(max_ineq_violation, max(0.0, -val))

        eq_violation_history.append(max_eq_violation)
        ineq_violation_history.append(max_ineq_violation)

    # 1) Solve with SLSQP
    result_slsqp = minimize(
        objective,
        z0,
        method="SLSQP",
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        callback=callback_slsqp,
        options={"maxiter": 10000, "ftol": 1e-6, "disp": True},
    )

    print("\nSLSQP solve")
    print("Success:", result_slsqp.success)
    print("Message:", result_slsqp.message)
    print("Final objective value:", result_slsqp.fun)

    z_opt = result_slsqp.x
    x_opt, y_opt = unpack(z_opt)

    #a) Show the drone's path around the cost equation (using SLSQP solution)
    grid_x = np.linspace(0.0, 10.0, 200)
    grid_y = np.linspace(0.0, 10.0, 200)
    X, Y = np.meshgrid(grid_x, grid_y)
    C = cost_field(X, Y)

    plt.figure(figsize=(7, 6))
    cs = plt.contourf(X, Y, C, levels=40, cmap='viridis')
    plt.colorbar(cs, label="C(x, y)")
    plt.plot(x_init, y_init, "k--", alpha=0.6, label="Initial guess")
    plt.scatter(x_opt, y_opt, c='red', s=15, zorder=5, label="Drone locations")
    plt.plot(x_opt, y_opt, "r-", alpha=0.8, label="Optimized path (SLSQP)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.title("(a) Drone path on cost field")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    #c) Constraint convergence (still from SLSQP)
    eps = 1e-16
    ineq_plot = np.maximum(ineq_violation_history, eps)

    plt.figure(figsize=(6, 4))
    # equality violations are basically at machine precision, so we skip them
    # eq_plot = np.maximum(eq_violation_history, eps)
    # plt.semilogy(iter_history, eq_plot, "r-", label="Max abs equality violation")
    plt.semilogy(iter_history, ineq_plot, "g--", label="Max inequality violation")
    plt.xlabel("Iteration")
    plt.ylabel("Constraint Violation")
    plt.title("(c) Constraint convergence (SLSQP)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Final constraint table (using SLSQP solution)
    print("\n" + "=" * 45)
    print("FINAL CONSTRAINT TABLE (SLSQP solution)")
    print("=" * 45)
    print("{:<25s} | {:>15s}".format("Constraint Name", "Value"))
    print("-" * 45)

    for c in constraints:
        if "speed" not in c.get("name", ""):
            val = c["fun"](z_opt)
            name = c.get("name", "unnamed")
            print("{:<25s} | {:>15.4e}".format(name, val))

    # Summarize speed constraints
    speed_vals = []
    for c in constraints:
        if "speed" in c.get("name", ""):
            speed_vals.append(c["fun"](z_opt))

    min_speed_margin = min(speed_vals)
    print("-" * 45)
    print("{:<25s} | {:>15.4e}".format("Min Speed Margin (>=0)", min_speed_margin))
    print("(Positive margin means all speed limits are satisfied)")
    print("=" * 45)

    # solving with trust-constr for Lagrangian Plot

    lag_iter_history = []
    lag_grad_norm_history = []

    def callback_trust(xk, state=None):
        """
        Callback for trust-constr
        """
        if state is None:
            return

        k = len(lag_iter_history)
        lag_iter_history.append(k)

        # Prefer the full Lagrangian gradient if provided
        if "lagrangian_grad" in state and state["lagrangian_grad"] is not None:
            grad_L = state["lagrangian_grad"]
            lag_grad_norm_history.append(np.linalg.norm(grad_L))
        else:
            # scalar optimality measure
            lag_grad_norm_history.append(state["optimality"])

    result_trust = minimize(
        objective,
        z0,  # same starting guess
        method="trust-constr",
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        callback=callback_trust,
        options={
            "maxiter": 10000,
            "verbose": 1,     
            "gtol": 1e-8    
        },
    )

    print("\ntrust-constr solve (for Lagrangian gradient history)")
    print("Success:", result_trust.success)
    print("Message:", result_trust.message)
    print("Final objective value:", result_trust.fun)

    # b) plot the convergence of the gradient of the Lagrangian (trust-constr)
    plt.figure(figsize=(6, 4))
    plt.semilogy(lag_iter_history, lag_grad_norm_history, "b-", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel(r"$||\nabla \mathcal{L}(x)||$ (optimality)")
    plt.title("(b) Lagrangian gradient convergence (trust-constr)")
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()
    plt.show()


#question 3: 2 drones

def unpack_two(z):
    # split into x1, y1, x2, y2 arrays
    x1 = z[0:N]
    y1 = z[N:2 * N]
    x2 = z[2 * N:3 * N]
    y2 = z[3 * N:4 * N]
    return x1, y1, x2, y2


def objective_two(z):
    x1, y1, x2, y2 = unpack_two(z)
    term1 = np.sum(cost_field(x1, y1)) + np.sum(cost_field(x2, y2))
    dx1 = np.diff(x1)
    dy1 = np.diff(y1)
    dx2 = np.diff(x2)
    dy2 = np.diff(y2)
    term2 = np.sum(dx1 * dx1 + dy1 * dy1 + dx2 * dx2 + dy2 * dy2)
    return term1 + term2


def objective_grad_two(z):
    x1, y1, x2, y2 = unpack_two(z)

    dC1_dx, dC1_dy = cost_grad(x1, y1)
    dC2_dx, dC2_dy = cost_grad(x2, y2)

    def path_grad(coord):
        g = np.zeros_like(coord)
        for j in range(N):
            if j > 0:
                g[j] += 2.0 * (coord[j] - coord[j - 1])
            if j < N - 1:
                g[j] -= 2.0 * (coord[j + 1] - coord[j])
        return g

    grad2_x1 = path_grad(x1)
    grad2_y1 = path_grad(y1)
    grad2_x2 = path_grad(x2)
    grad2_y2 = path_grad(y2)

    gx1 = dC1_dx + grad2_x1
    gy1 = dC1_dy + grad2_y1
    gx2 = dC2_dx + grad2_x2
    gy2 = dC2_dy + grad2_y2

    return np.concatenate([gx1, gy1, gx2, gy2])


constraints_two = []
d_min = 1.0


def add_eq_constraint_two(name, indices, coeffs, rhs):
    indices = np.asarray(indices, dtype=int)
    coeffs = np.asarray(coeffs, dtype=float)

    def fun(z):
        return float(np.dot(coeffs, z[indices]) - rhs)

    def jac(z):
        g = np.zeros_like(z)
        g[indices] = coeffs
        return g

    constraints_two.append({"type": "eq", "fun": fun, "jac": jac, "name": name})


def idx_x1(i):
    return i


def idx_y1(i):
    return N + i


def idx_x2(i):
    return 2 * N + i


def idx_y2(i):
    return 3 * N + i


# Boundary conditions for drone 1: (0,0) to (10,10)
add_eq_constraint_two("x1_start", [idx_x1(0)], [1.0], 0.0)
add_eq_constraint_two("y1_start", [idx_y1(0)], [1.0], 0.0)
add_eq_constraint_two("x1_end", [idx_x1(N - 1)], [1.0], 10.0)
add_eq_constraint_two("y1_end", [idx_y1(N - 1)], [1.0], 10.0)

# Boundary conditions for drone 2: (10,10) to (0,0)
add_eq_constraint_two("x2_start", [idx_x2(0)], [1.0], 10.0)
add_eq_constraint_two("y2_start", [idx_y2(0)], [1.0], 10.0)
add_eq_constraint_two("x2_end", [idx_x2(N - 1)], [1.0], 0.0)
add_eq_constraint_two("y2_end", [idx_y2(N - 1)], [1.0], 0.0)

# Zero initial velocities for both drones
add_eq_constraint_two("vx1_initial", [idx_x1(1), idx_x1(0)], [1.0, -1.0], 0.0)
add_eq_constraint_two("vy1_initial", [idx_y1(1), idx_y1(0)], [1.0, -1.0], 0.0)
add_eq_constraint_two("vx2_initial", [idx_x2(1), idx_x2(0)], [1.0, -1.0], 0.0)
add_eq_constraint_two("vy2_initial", [idx_y2(1), idx_y2(0)], [1.0, -1.0], 0.0)


def make_speed_constraint_two(drone, k):
    # Speed constraint for segment k of drone 1 or 2
    def fun(z):
        x1, y1, x2, y2 = unpack_two(z)
        if drone == 1:
            x = x1
            y = y1
        else:
            x = x2
            y = y2
        u = (x[k + 1] - x[k]) / dt
        v = (y[k + 1] - y[k]) / dt
        return vmax ** 2 - (u * u + v * v)

    def jac(z):
        x1, y1, x2, y2 = unpack_two(z)
        if drone == 1:
            base_x = 0
            base_y = N
            x = x1
            y = y1
        else:
            base_x = 2 * N
            base_y = 3 * N
            x = x2
            y = y2

        u = (x[k + 1] - x[k]) / dt
        v = (y[k + 1] - y[k]) / dt

        g = np.zeros_like(z)

        du_dxk1 = 1.0 / dt
        du_dxk = -1.0 / dt
        dv_dyk1 = 1.0 / dt
        dv_dyk = -1.0 / dt

        g[base_x + k] += -2.0 * u * du_dxk
        g[base_x + k + 1] += -2.0 * u * du_dxk1
        g[base_y + k] += -2.0 * v * dv_dyk
        g[base_y + k + 1] += -2.0 * v * dv_dyk1

        return g

    return {"type": "ineq", "fun": fun, "jac": jac, "name": f"speed_d{drone}_{k}"}


for k in range(N - 1):
    constraints_two.append(make_speed_constraint_two(1, k))
    constraints_two.append(make_speed_constraint_two(2, k))


def make_collision_constraint(i):
    # collision avoidance: keep drones at least d_min apart
    def fun(z):
        x1, y1, x2, y2 = unpack_two(z)
        dx = x1[i] - x2[i]
        dy = y1[i] - y2[i]
        return dx * dx + dy * dy - d_min ** 2

    def jac(z):
        x1, y1, x2, y2 = unpack_two(z)
        dx = x1[i] - x2[i]
        dy = y1[i] - y2[i]
        g = np.zeros_like(z)
        g[idx_x1(i)] += 2.0 * dx
        g[idx_y1(i)] += 2.0 * dy
        g[idx_x2(i)] += -2.0 * dx
        g[idx_y2(i)] += -2.0 * dy
        return g

    return {"type": "ineq", "fun": fun, "jac": jac, "name": f"collision_{i}"}


for i in range(N):
    constraints_two.append(make_collision_constraint(i))

# Bounds and initial guess for question 3
bounds_two = [(0.0, 10.0)] * (4 * N)

x1_init_q3 = np.linspace(0.0, 10.0, N)
t_q3 = np.linspace(0.0, np.pi, N)
y1_init_q3 = np.linspace(0.0, 10.0, N) + 0.5 * np.cos(t_q3)
y1_init_q3 = np.clip(y1_init_q3, 0.0, 10.0)

x2_init_q3 = np.linspace(10.0, 0.0, N)
y2_init_q3 = np.linspace(10.0, 0.0, N) - 0.5 * np.sin(t_q3)
y2_init_q3 = np.clip(y2_init_q3, 0.0, 10.0)

z0_two = np.concatenate([x1_init_q3, y1_init_q3, x2_init_q3, y2_init_q3])


def main_two_drones():
    result = minimize(
        objective_two,
        z0_two,
        method="SLSQP",
        jac=objective_grad_two,
        bounds=bounds_two,
        constraints=constraints_two,
        options={"maxiter": 300, "ftol": 1e-6, "disp": True},
    )

    print("\nTwo-drone SLSQP solve")
    print("Success:", result.success)
    print("Message:", result.message)
    print("Final objective value:", result.fun)

    x1_opt, y1_opt, x2_opt, y2_opt = unpack_two(result.x)

    grid_x = np.linspace(0.0, 10.0, 200)
    grid_y = np.linspace(0.0, 10.0, 200)
    X, Y = np.meshgrid(grid_x, grid_y)
    C = cost_field(X, Y)

    plt.figure(figsize=(7, 6))
    cs = plt.contourf(X, Y, C, levels=40, cmap='viridis')
    plt.colorbar(cs, label="C(x, y)")

    # initial guesses
    plt.plot(x1_init_q3, y1_init_q3, "k--", alpha=0.5, label="Drone 1 initial")
    plt.plot(
        x2_init_q3,
        y2_init_q3,
        linestyle="--",
        color="magenta",
        alpha=0.7,
        label="Drone 2 initial",
    )

    # optimized paths
    plt.plot(x1_opt, y1_opt, "r-", label="Drone 1 path")
    plt.plot(x2_opt, y2_opt, color="c", label="Drone 2 path")

    # arrows to indicate path
    mid = N // 2

    plt.annotate(
        "",
        xy=(x1_opt[mid + 1], y1_opt[mid + 1]),
        xytext=(x1_opt[mid], y1_opt[mid]),
        arrowprops=dict(arrowstyle="->", color="r", lw=2),
    )

    plt.annotate(
        "",
        xy=(x2_opt[mid + 1], y2_opt[mid + 1]),
        xytext=(x2_opt[mid], y2_opt[mid]),
        arrowprops=dict(arrowstyle="->", color="c", lw=2),
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.title("Two-drone paths with collision avoidance")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # use cost_field/cost_grad versions for question 1 or 2
    # set scenario = 1 or 2 for single drone, scenario = 3 for two drones
    scenario = 1
    if scenario == 3:
        main_two_drones()
    else:
        main_single_drone()
