import numpy
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plot
import jax
import jax.numpy


# Objective function
def objective(x):
    # return x[0]**2 + x[1]**2
    # Example: Rosenbrock function
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# Gradient of the objective
def objective_grad(x):
    # return numpy.array([2*x[0], 2*x[1]])
    # Gradient of the Rosenbrock function
    return np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
                     200 * (x[1] - x[0]**2)])

# Define constraints
def constraint(x):
    # Constraint: x^2 + y^2 - 1 = 0
    # return x[0]**2 + x[1]**2 - 1
    # Constraint: x[0] + 2*x[1] <= 1 (written as c(x) <= 0: x[0] + 2*x[1] - 1 <= 0)
    return [x[0] + 2*x[1]]

# Gradient of the constraint
def constraint_grad(x):
    # return numpy.array([2*x[0], 2*x[1]])
    # Jacobian of the constraint
    return [[1, 2]]


# Compute gradients using JAX
objective_grad = jax.grad(objective)
constraint_grad = jax.grad(constraint)

# Convert JAX arrays to numPy for scipy
def objective_grad_jax(x):
    return numpy.array(objective_grad(x))

def constraint_grad_jax(x):
    return numpy.array(constraint_grad(x))

# Initial guess
x0 = numpy.array([0.5, 0.5])

# Assemble Constraint
cons = {'type': 'eq', 'fun': constraint}
cons_withgrad = {'type': 'eq', 'fun': constraint, 'jac': constraint_grad}
cons_withgrad_jax = {'type': 'eq', 'fun': constraint, 'jac': constraint_grad_jax}


# Callback function to print x at each iteration
def iteration_callback(xk):
    print(f"Current x: {xk}")


# Minimize using SLSQP
minimize_function1 = minimize(objective, x0, method='trust-constr', constraints=[cons])


# Output
print("Optimal solution (x, y):", minimize_function1.x)



# --- Configuration ---
# Use a method that provides lagrangian_grad in its callback, like 'trust-constr'
OPTIMIZATION_METHOD = 'trust-constr'

# Data structures to store history
lagrangian_gradient_norms = []
iterations = []
iteration_count = 0

def callback_function(xk, intermediate_result):
    global iteration_count
    # 'trust-constr' provides the gradient of the Lagrangian in intermediate_result.lagrangian_grad
    if hasattr(intermediate_result, 'lagrangian_grad'):
        grad_L = intermediate_result.lagrangian_grad
        grad_L_norm = numpy.linalg.norm(grad_L)
        lagrangian_gradient_norms.append(grad_L_norm)
        iterations.append(iteration_count)
        iteration_count += 1
    else:
        # Handle cases where the attribute might not exist (e.g., other methods or versions)
        print(f"Warning: '{OPTIMIZATION_METHOD}' callback did not provide 'lagrangian_grad'")

# Set bounds and create NonlinearConstraint object
bounds = Bounds([0, -0.5], [1, 2.0])


minimize_function2 = minimize(
    objective,
    x0,
    method=OPTIMIZATION_METHOD,
    jac=objective_grad,
    bounds=bounds,
    constraints=[cons],
    callback=callback_function,
    options={'verbose': 1, 'gtol': 1e-10} # gtol is the tolerance for the Lagrangian gradient norm
)

# Plot the Convergence of the Lagrangian Gradient Norm
plot.figure(figsize=(10, 6))
plot.semilogy(iterations, lagrangian_gradient_norms, marker='o', linestyle='-')
plot.xlabel('Iteration Number')
plot.ylabel('Lagrangian Gradient Norm (log scale)')
plot.title(f'Convergence of the Lagrangian Gradient Norm ({OPTIMIZATION_METHOD})')
plot.grid(True, which="both", ls="--")
plot.show()

print(f"Optimization successful: {minimize_function2.success}")
print(f"Optimal solution: {minimize_function2.x}")
# The minimize_function2 object also has the final lagrangian_grad norm available
final_lag_grad_norm = numpy.linalg.norm(minimize_function2.lagrangian_grad) if hasattr(minimize_function2, 'lagrangian_grad') else "N/A"
print(f"Final Lagrangian gradient norm: {final_lag_grad_norm}")