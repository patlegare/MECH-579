import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import jax
import jax.numpy as jnp

# Professor Nadarajah’s Breguet-Range model (NumPy)
def density(h):
    return 1.2 * (1 - h * 0.0065 / 288) ** 5.26

def mass_fuel_rate(v, h):
    turbine_area = np.pi * 0.92**2 / 2
    FAR = 1E-1
    return density(h) * v * turbine_area * FAR

def C_L(v, h, weight, S):
    return 9.81 * weight / (0.5 * density(h) * v**2 * S)

def C_Dw(v, h):
    return 10 * (np.atan(10 * ((v / (343 * 0.7))**2 - 1)) + np.pi / 2)

def C_D(v, h, weight, S):
    e, AR = 0.8, 10
    coeff_drag = 0.5 / 60
    coeff_drag += C_L(v, h, weight, S)**2 / (np.pi * e * AR)
    return coeff_drag + C_Dw(v, h)

def ct(v, h, weight, S):
    ct_value = mass_fuel_rate(v, h) / (0.5 * density(h) * v**2 * S * C_D(v, h, weight, S))
    return ct_value + 1e-5

def aircraft_range(v, h, weight, S):
    # Breguet range in km
    W_empty = 162_400.0
    W_fuel_total = 146_571.0
    W_i = W_empty + W_fuel_total
    W_f = W_empty
    return (v / ct(v, h, weight, S)
            * C_L(v, h, weight, S) / C_D(v, h, weight, S)
            * np.log(W_i / W_f)) / 1e3

# constants and design-space bounds
S = 100.0
W_empty = 162_400.0
W_fuel_total = 146_571.0
fuel_percentage = 0.75 # 75% of fuel remaining
weight_used = W_empty + fuel_percentage * W_fuel_total  # representative cruise weight

v_min, v_max = 10.0, 540.0 / 3.6 # m/s (assuming cant be below 10m/s)
h_min, h_max = 0.0, 2.0e4 # m (cant be below 0m of altitude)

# JAX versions of the model (for gradients)
def density_jax(h):
    return 1.2 * (1.0 - h * 0.0065 / 288.0) ** 5.26

def mass_fuel_rate_jax(v, h):
    turbine_area = jnp.pi * 0.92**2 / 2.0
    FAR = 1e-1
    return density_jax(h) * v * turbine_area * FAR

def C_L_jax(v, h, weight, S):
    return 9.81 * weight / (0.5 * density_jax(h) * v**2 * S)

def C_Dw_jax(v, h):
    return 10.0 * (jnp.arctan(10.0 * ((v / (343.0 * 0.7))**2 - 1.0)) + jnp.pi / 2.0)

def C_D_jax(v, h, weight, S):
    e, AR = 0.8, 10.0
    coeff_drag = 0.5 / 60.0
    coeff_drag += C_L_jax(v, h, weight, S)**2 / (jnp.pi * e * AR)
    return coeff_drag + C_Dw_jax(v, h)

def ct_jax(v, h, weight, S):
    ct_value = mass_fuel_rate_jax(v, h) / (0.5 * density_jax(h) * v**2 * S * C_D_jax(v, h, weight, S))
    return ct_value + 1e-5

def aircraft_range_jax(v, h, weight, S):
    W_empty = 162_400.0
    W_fuel_total = 146_571.0
    W_i = W_empty + W_fuel_total
    W_f = W_empty
    return (v / ct_jax(v, h, weight, S)
            * C_L_jax(v, h, weight, S) / C_D_jax(v, h, weight, S)
            * jnp.log(W_i / W_f)) / 1e3

# objective for SciPy: minimize negative range
def objective_jax(x):
    v, h = x[0], x[1]
    return -aircraft_range_jax(v, h, weight_used, S)

grad_objective_jax = jax.grad(objective_jax)

def objective(x):
    return float(objective_jax(jnp.array(x)))

def objective_grad(x):
    return np.array(grad_objective_jax(jnp.array(x)))

# constraint functions for plotting convergence
def g1(x):  # v - v_min >= 0
    v, h = x
    return v - v_min

def g2(x):  # v_max - v >= 0
    v, h = x
    return v_max - v

def g3(x):  # h - h_min >= 0
    v, h = x
    return h - h_min

def g4(x):  # h_max - h >= 0
    v, h = x
    return h_max - h

# optimization with SciPy (L-BFGS-B + gradient)
bounds = [(v_min, v_max), (h_min, h_max)]
x0 = np.array([120.0, 8000.0])

history = {'x': [], 'grad_norm': [], 'range': [], 'constraints': []}

def callback(xk):
    history['x'].append(np.copy(xk))
    gk = objective_grad(xk)
    history['grad_norm'].append(np.linalg.norm(gk))
    v, h = xk
    history['range'].append(aircraft_range(v, h, weight_used, S))
    history['constraints'].append([g1(xk), g2(xk), g3(xk), g4(xk)])

result = minimize(objective, x0,
                  method='L-BFGS-B',
                  jac=objective_grad,
                  bounds=bounds,
                  callback=callback)

print("Optimal design (v*, h*):", result.x)
print("  v* [m/s] =", result.x[0], "  (km/h =", result.x[0] * 3.6, ")")
print("  h* [m]   =", result.x[1])
print("Max range [km] =", -result.fun)

# (i) convergence of gradient norm
iters = np.arange(1, len(history['grad_norm']) + 1)

plt.figure()
plt.semilogy(iters, history['grad_norm'], 'o-')
plt.xlabel('Iteration')
plt.ylabel(r'$\log\|\nabla J(x_k)\|_2$')
plt.title('Convergence of Gradient of Lagrangian (Breguet Range)')
plt.grid(True)
plt.tight_layout()
plt.show()

# (ii) convergence of range and constraints
g_vals = np.array(history['constraints'])  # shape (n_iter, 4)

plt.figure(figsize=(7, 5))

# Objective
plt.plot(iters, history['range'], 'k-o', label='Range (km)')

# Constraint margins (velocity taken out since it lives on the boundary:active constraint)
plt.plot(iters, g_vals[:, 2], 'g--', label='Altitude margin: h - h_min')
plt.plot(iters, g_vals[:, 3], 'm--', label='Altitude margin: h_max - h')

plt.xlabel('Iteration')
plt.ylabel('Value (constraint margins, range)')
plt.title('Convergence of Range and Constraint Margins vs Iteration')
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# (iii) design-space contour plot with optimization path
x_star = result.x

v_axis_ext = np.linspace(v_min, 250.0, 300)
h_axis_ext = np.linspace(h_min, h_max, 200)
V_ext, H_ext = np.meshgrid(v_axis_ext, h_axis_ext)

R_ext = aircraft_range(V_ext, H_ext, weight_used, S)

plt.figure(figsize=(7, 6))
contour = plt.contourf(V_ext, H_ext, R_ext, levels=40, cmap="viridis")
cbar = plt.colorbar(contour)
cbar.set_label("Range (km)")

path = np.vstack([x0] + history['x'])
plt.plot(path[:, 0], path[:, 1],
         'w-o', linewidth=2, markersize=4, label='Optimization path')

plt.scatter(x0[0], x0[1], c='red', s=60, label='Start')
plt.scatter(x_star[0], x_star[1], c='lime', s=60, label='Optimum')

plt.xlabel("Velocity (m/s)")
plt.ylabel("Altitude (m)")
plt.title("Maximum Breguet Range with Constraints")
plt.legend()
plt.xlim(0, 250)
plt.ylim(h_min, h_max)
plt.tight_layout()
plt.show()
