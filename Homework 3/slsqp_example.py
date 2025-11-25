import numpy
from scipy.optimize import minimize
import jax
import jax.numpy


# Objective function
def objective(x):
    return x[0]**2 + x[1]**2

# Gradient of the objective
def objective_grad(x):
    return numpy.array([2*x[0], 2*x[1]])

# Constraint: x^2 + y^2 - 1 = 0
def constraint(x):
    return x[0]**2 + x[1]**2 - 1

# Gradient of the constraint
def constraint_grad(x):
    return numpy.array([2*x[0], 2*x[1]])


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

# Minimize using SLSQP
minimize_function1 = minimize(objective, x0, method='SLSQP', constraints=[cons])
minimize_function2 = minimize(objective, x0, method='SLSQP', jac=objective_grad, constraints=[cons_withgrad])
minimize_function3 = minimize(objective, x0, method='SLSQP', jac=objective_grad_jax, constraints=[cons_withgrad_jax])


# Output
print("Optimal solution (x, y):                              ", minimize_function1.x)
print("Optimal solution with Analytical Gradient(x, y):      ", minimize_function2.x)
print("Optimal solution with Automatic Differentiation(x, y):", minimize_function3.x)
