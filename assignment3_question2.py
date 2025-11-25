import numpy as np
import matplotlib.pyplot as plt

#Professor Nadarajah’s Breguet-Range model
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
    W_empty = 162_400.0
    W_fuel_total = 146_571.0
    W_i = W_empty + W_fuel_total
    W_f = W_empty
    return (v / ct(v, h, weight, S)
            * C_L(v, h, weight, S) / C_D(v, h, weight, S)
            * np.log(W_i / W_f)) / 1e3

#constants
S = 100
W_empty = 162_400.0
W_fuel_total = 146_571.0
fuel_percentage = 0.75
weight_used = W_empty + fuel_percentage * W_fuel_total

v_min, v_max = 10.0, 540 / 3.6
h_min, h_max = 0.0, 2.0e4

#gradient
def grad_numeric(func, x, eps=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = 1.0
        g[i] = (func(x + eps*e) - func(x - eps*e)) / (2*eps)
    return g

def f_vec(x):
    return aircraft_range(x[0], x[1], weight_used, S)

#gradient ascent
def projected_gradient_ascent(x0, max_iter=150, tol=1e-6):
    x = x0.astype(float).copy()
    history, fvals = [x.copy()], [f_vec(x)]
    iters = 0
    for k in range(max_iter):
        g = grad_numeric(f_vec, x)
        if np.linalg.norm(g) < tol:
            break
        p = g  # scaled step direction
        alpha = 0.05  # initial step size

        # Backtracking line search
        while True:
            x_new = x + alpha * p
            x_new[0] = np.clip(x_new[0], v_min, v_max)
            x_new[1] = np.clip(x_new[1], h_min, h_max)
            if f_vec(x_new) >= f_vec(x) + 1e-5 * alpha * np.dot(g, p):
                break
            alpha *= 0.5
            if alpha < 1e-8:
                break

        x = x_new
        history.append(x.copy())
        fvals.append(f_vec(x))
        iters = k + 1

    return np.array(history), np.array(fvals), iters

# start point
#x0 = np.array([130.0, 11000.0])  # you can adjust this
#start point using a grid search for initial guess
v_axis = np.linspace(v_min, v_max, 200)
h_axis = np.linspace(h_min, h_max, 200)
V, H = np.meshgrid(v_axis, h_axis)
R = aircraft_range(V, H, weight_used, S)

i, j = np.unravel_index(np.argmax(R), R.shape)
x0 = np.array([V[i, j], H[i, j]])

#run
trajectory, fvals, iterations = projected_gradient_ascent(x0)
x_star = trajectory[-1]
R_star = fvals[-1]

#print
print("\n===============================")
print("MAXIMUM BREQUET RANGE RESULTS")
print("===============================")
print(f"Optimal velocity v*  = {x_star[0]:.4f} m/s  ({x_star[0]*3.6:.2f} km/h)")
print(f"Optimal altitude h*  = {x_star[1]:.4f} m")
print(f"Maximum range R*     = {R_star:.4f} km")
print(f"Converged in {iterations} iterations")

#design space plot
v_axis_ext = np.linspace(v_min, 250, 300)
h_axis_ext = np.linspace(h_min, h_max, 200)
V_ext, H_ext = np.meshgrid(v_axis_ext, h_axis_ext)
R_ext = aircraft_range(V_ext, H_ext, weight_used, S)

plt.figure(figsize=(7,6))
contour = plt.contourf(V_ext, H_ext, R_ext, levels=40, cmap="viridis")
plt.colorbar(contour, label="Range (km)")
plt.scatter(x0[0], x0[1], c='red', s=60, label='Start')
plt.scatter(x_star[0], x_star[1], c='lime', s=60, label='Optimum')
plt.xlabel("Velocity (m/s)")
plt.ylabel("Altitude (m)")
plt.title("Maximum Breguet Range with Manufacturer Constraints")
plt.legend()
plt.xlim(0, 250)
plt.tight_layout()
plt.show()

#convergence plot
plt.figure(figsize=(7,5))
plt.plot(fvals, '-o')
plt.xlabel("Iteration")
plt.ylabel("Range (km)")
plt.title("Convergence of Objective Function (Range)")
plt.grid(True)
plt.tight_layout()
plt.show()