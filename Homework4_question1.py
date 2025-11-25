import numpy
from scipy.optimize import minimize
import jax
import jax.numpy
from matplotlib import pyplot as plt


# Objective function: Rosenbrock
# f(x, y) = (1 - x)^2 + 100 (y - x^2)^2
def objective(x):
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2


# Analytic gradient of the objective
def objective_grad(x):
    df_dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0]**2)
    df_dy = 200.0 * (x[1] - x[0]**2)
    return numpy.array([df_dx, df_dy])


# Constraint: inside the unit disk
def constraint(x):
    return 1.0 - x[0]**2 - x[1]**2


# Analytic gradient of the constraint
def constraint_grad(x):
    dc_dx = -2.0 * x[0]
    dc_dy = -2.0 * x[1]
    return numpy.array([dc_dx, dc_dy])


# JAX versions of the functions for automatic differentiation
def objective_jax(x):
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2


def constraint_jax(x):
    return 1.0 - x[0]**2 - x[1]**2


# JAX gradients (automatic differentiation)
objective_grad_jax_raw = jax.grad(objective_jax)
constraint_grad_jax_raw = jax.grad(constraint_jax)


# Wrappers converting JAX arrays to NumPy arrays for scipy.optimize
def objective_grad_jax(x):
    return numpy.array(objective_grad_jax_raw(x))


def constraint_grad_jax(x):
    return numpy.array(constraint_grad_jax_raw(x))


# Initial point
x0 = numpy.array([-1, 1])
bounds = [(-1.5, 2.5), (-1.0, 3.0)]

# Constraints without gradient
cons = {'type': 'ineq', 'fun': constraint}

# Constraints with analytic gradient
cons_withgrad = {'type': 'ineq', 'fun': constraint, 'jac': constraint_grad}

# Constraints with gradient from automatic differentiation
cons_withgrad_jax = {'type': 'ineq', 'fun': constraint, 'jac': constraint_grad_jax}


# Initialize histories
history_no = {'x': [], 'grad_norm': []}
history_an = {'x': [], 'grad_norm': []}
history_jax = {'x': [], 'grad_norm': []}


# Callbacks to store iterates and gradient norms
def callback_no_grad(xk):
    gk = objective_grad(xk)  # true gradient, even though SLSQP is not using it
    history_no['x'].append(numpy.copy(xk))
    history_no['grad_norm'].append(numpy.linalg.norm(gk))


def callback_an_grad(xk):
    gk = objective_grad(xk)
    history_an['x'].append(numpy.copy(xk))
    history_an['grad_norm'].append(numpy.linalg.norm(gk))


def callback_jax_grad(xk):
    gk = objective_grad_jax(xk)
    history_jax['x'].append(numpy.copy(xk))
    history_jax['grad_norm'].append(numpy.linalg.norm(gk))


# Minimize Rosenbrock using SLSQP
result1 = minimize(objective, x0, method='SLSQP',
                   constraints=[cons],
                   bounds=bounds,
                   callback=callback_no_grad)

# Minimize with analytic gradients supplied
result2 = minimize(objective, x0, method='SLSQP',
                   jac=objective_grad, constraints=[cons_withgrad],
                   bounds=bounds,
                   callback=callback_an_grad)

# Minimize with automatic differentiation (JAX) gradients supplied
result3 = minimize(objective, x0, method='SLSQP',
                   jac=objective_grad_jax, constraints=[cons_withgrad_jax],
                   bounds=bounds,
                   callback=callback_jax_grad)                        


# Output
print("Optimal solution (x, y) without gradients:                 ", result1.x)
print("Optimal solution (x, y) with analytic gradients:           ", result2.x)
print("Optimal solution (x, y) with automatic differentiation:    ", result3.x)
print("Objective values:")
print("  f(x*) no grad        =", result1.fun)
print("  f(x*) analytic grad  =", result2.fun)
print("  f(x*) autodiff (JAX) =", result3.fun)


# Convergence plot
plt.figure()
iters_no = numpy.arange(1, len(history_no['grad_norm']) + 1)
iters_an = numpy.arange(1, len(history_an['grad_norm']) + 1)
iters_jax = numpy.arange(1, len(history_jax['grad_norm']) + 1)

plt.semilogy(iters_no, history_no['grad_norm'], 'o-', label='SLSQP gradient')
plt.semilogy(iters_an, history_an['grad_norm'], 's-', label='Analytic gradient')
plt.semilogy(iters_jax, history_jax['grad_norm'], 'x-', label='JAX gradient')

plt.xlabel('Iteration')
plt.ylabel('log(||∇f(x)||)')
plt.title('Convergence of gradient norm')
plt.legend()
plt.grid(True)


# contour plot of Rosenbrock with optimization paths
x_vals = numpy.linspace(-1.5, 2.5, 400)
y_vals = numpy.linspace(-1.2, 2.4, 400)
X, Y = numpy.meshgrid(x_vals, y_vals)
Z = (1.0 - X)**2 + 100.0 * (Y - X**2)**2

plt.figure(figsize=(9, 4.5))
contours = plt.contour(X, Y, Z,
                       levels=numpy.logspace(-1, 3, 20),
                       colors='black',
                       linewidths=0.8)

# Paths of the three optimization runs
path_no  = numpy.vstack((x0, numpy.array(history_no['x'])))
path_an  = numpy.vstack((x0, numpy.array(history_an['x'])))
path_jax = numpy.vstack((x0, numpy.array(history_jax['x'])))

# Make styles clearly different so overlapping paths are still visible
plt.plot(path_no[:, 0],  path_no[:, 1],  'r--o',  label='Path (SLSQP)')
plt.plot(path_an[:, 0],  path_an[:, 1],  'b-.s',  label='Path (analytic grad)')
plt.plot(path_jax[:, 0], path_jax[:, 1], 'g-^',   label='Path (JAX grad)')

# Constraint boundary: x^2 + y^2 = 1
theta = numpy.linspace(0.0, 2.0 * numpy.pi, 400)
plt.plot(numpy.cos(theta), numpy.sin(theta), 'k--', label=r'$1 - x^2 - y^2 = 0$')

# Starting point and unconstrained minimum
plt.plot(x0[0],  x0[1],  'ms', markersize=8,  label='Initial guess')
plt.plot(1.0, 1.0, 'y*', markersize=10, label='Unconstrained minimum (1,1)')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour Plot with SLSQP Optimization Paths')
plt.xlim(-1.8, 1.8)  
plt.ylim(-1.2, 2.4)
plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.show()