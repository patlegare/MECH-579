import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#Rosenbrock Function
def f(x,y):
  return (1-x)**2+100*(y-x**2)**2

#Steepest Descent
#General form: xk+1=xk+ alpha*p

#Initialize
x=np.array([-1.2, 1.0])
alpha=1
rho=0.5
c=1e-4
max_iterations=100000
tol=1e-6

#Gradient of f(x,y)
def grad_f(x,y):
  df_dx=2*x-2-400*x*y+400*x**3
  df_dy=200*y-200*x**2
  return np.array([df_dx, df_dy])

#For plots
path=[x.copy()]
grad_norms=[]

#Steph length alpha (Using simple backtracking line search) 
#Start with alpha=1 and decrease
for k in range(max_iterations):
  grad = grad_f(x[0], x[1])
  grad_norm = np.linalg.norm(grad)
  grad_norms.append(grad_norm)

    # Stop if gradient is small
  if grad_norm < tol:
      print(f"Converged after {k} iterations.")
      break

    # Search direction for steepest descent
  p= -grad

    # backtracking line search
  alpha= 1.0
  while f(*(x + alpha*p)) > f(*x) + c*alpha*np.dot(grad, p):
      alpha*= rho

    #update
  x = x + alpha * p
  path.append(x.copy())
print(f"Final point: x = {x[0]:.6f}, y = {x[1]:.6f}")
print(f"Final f(x,y) = {f(x[0], x[1]):.6e}")


# convert path to array for plotting
path = np.array(path)

# convergence of plot
plt.figure(figsize=(6,4))
plt.plot(np.log10(grad_norms),linewidth=1)
plt.title("Convergence of Steepest Descent on Rosenbrock Function")
plt.xlabel("Iteration")
plt.ylabel("log(∇f)")
plt.grid(True)
plt.show()

#Contour plot of path
X = np.linspace(-2, 2, 400)
Y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)

plt.figure(figsize=(7,6))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
plt.plot(path[:,0], path[:,1], 'ro-', label='Descent Path',linewidth=1)
plt.scatter(path[-1,0], path[-1,1], color='blue', s=40, label='Final point')
plt.annotate(
    'Global minimum (1, 1)',
    xy=(1, 1), xycoords='data',
    xytext=(0.4, 2), textcoords='data',
    arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8),
    horizontalalignment='center', fontsize=10
)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Steepest Descent Path on Rosenbrock Function')
plt.legend()
plt.grid(True)
plt.show()

#Hestenes-Stiefel Nonlinear Conjugate Gradient
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def f(x):
    #x is a length-2 numpy array
    X, Y = x[0], x[1]
    return (1 - X)**2 + 100.0 * (Y - X**2)**2

def grad_f(x):
    #Gradient
    X, Y = x[0], x[1]
    df_dx = 2*(X - 1) - 400*X*(Y - X**2)
    df_dy = 200*(Y - X**2)
    return np.array([df_dx, df_dy])

#backtracking line search
def backtracking_line_search(x, p, grad, alpha0=1.0, rho=0.5, c=1e-4, max_ls_iter=50):
    alpha = alpha0
    fx = f(x)
    grad_dot_p = np.dot(grad, p)
    for _ in range(max_ls_iter):
        x_new = x + alpha * p
        if f(x_new) <= fx + c * alpha * grad_dot_p:
            return alpha
        alpha *= rho
    return alpha

# CG
def nonlinear_cg_hs(x0, tol=1e-6, max_iter=5000, restart_freq=50, verbose=True):
    """
    Nonlinear Conjugate Gradient using Hestenes-Stiefel beta formula.
    Returns: path (Nx2), grad_norms (N), final_x, final_grad
    """
    x = x0.copy()
    g = grad_f(x)
    p = -g #initial search direction
    path = [x.copy()]
    grad_norms = [np.linalg.norm(g)]

    for k in range(max_iter):
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break

        # Line search to obtain alpha
        alpha = backtracking_line_search(x, p, g, alpha0=1.0, rho=0.5, c=1e-4, max_ls_iter=60)

        # Step
        x_new = x + alpha * p
        g_new = grad_f(x_new)

        # Compute HS beta
        yk = g_new - g
        denom = np.dot(p, yk)
        if abs(denom) < 1e-12:
            beta_hs = 0.0
        else:
            beta_hs = np.dot(g_new, yk) / denom

        # Restart if beta negative or at restart frequency (keeps conjugacy stable)
        if beta_hs < 0 or (restart_freq is not None and (k+1) % restart_freq == 0):
            beta_hs = 0.0

        # New direction
        p = -g_new + beta_hs * p

        # Save iteration data
        x = x_new
        g = g_new
        path.append(x.copy())
        grad_norms.append(np.linalg.norm(g))

        if verbose and (k % 1000 == 0) and k > 0:
            print(f"Iter {k:5d} ||grad|| = {np.linalg.norm(g):.3e}  alpha={alpha:.2e}  beta={beta_hs:.3e}")

    else:
        # max_iter reached
        if verbose:
            print("Reached maximum iterations without full convergence.")

    return np.array(path), np.array(grad_norms), x, g

if __name__ == "__main__":
    x0 = np.array([-1.2, 1.0])

    # Parameters
    tol = 1e-8
    max_iter = 5000
    restart_freq = 50  

    path, grad_norms, x_final, g_final = nonlinear_cg_hs(
        x0, tol=tol, max_iter=max_iter, restart_freq=restart_freq, verbose=True
    )

    print()
    print(f"Converged after {len(grad_norms)-1} iterations.")
    print(f"Final point: x = {x_final[0]:.8f}, y = {x_final[1]:.8f}")
    print(f"Final f(x,y) = {f(x_final):.4e}")

   
# plot number of iterations
plt.figure(figsize=(8,4))
plt.plot(np.log10(grad_norms), linewidth=1)
plt.xlabel("Iteration")
plt.ylabel("log(∇f)")
plt.title("Convergence of Nonlinear CG (Hestenes–Stiefel) on Rosenbrock")
plt.grid(True)
plt.tight_layout()
plt.show()

#contour plot with path
X = np.linspace(-2.0, 2.0, 400)
Y = np.linspace(-1.0, 3.0, 400)
X, Y = np.meshgrid(X, Y)
Z = (1 - X)**2 + 100.0*(Y - X**2)**2

plt.figure(figsize=(7,6))
levels = np.logspace(-1, 3, 20)
cs = plt.contour(X, Y, Z, levels=levels, norm=LogNorm(), cmap='viridis')
plt.plot(path[:,0], path[:,1], marker='o', markersize=3, linewidth=1, color='red', label='Conjugate Gradient-HS path')
plt.annotate(
    'Global minimum (1, 1)',
    xy=(1, 1), xycoords='data',
    xytext=(0.4, 2), textcoords='data',
    arrowprops=dict(facecolor='darkblue', shrink=0.05, width=1.5, headwidth=8),
    horizontalalignment='center', fontsize=10
)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Nonlinear CG (Hestenes–Stiefel) Path on Rosenbrock')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
import numpy as np

def quasi_newton_dfp(x0, tol=1e-8, max_iter=5000, verbose=True):
    """
    Quasi-Newton optimization using DFP (Davidon-Fletcher-Powell) update.
    Returns: path (Nx2), grad_norms (N), final_x, final_grad
    """
    x = x0.copy()
    g = grad_f(x)
    n = len(x)
    H = np.eye(n)  # initial Hessian inverse approximation

    path = [x.copy()]
    grad_norms = [np.linalg.norm(g)]

    for k in range(max_iter):
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break

        # Search direction
        p = -H @ g

        # Line search
        alpha = backtracking_line_search(x, p, g)

        # Update step
        x_new = x + alpha * p
        g_new = grad_f(x_new)

        # Compute DFP update terms
        s = x_new - x
        y = g_new - g
        rho = 1.0 / np.dot(y, s)

        # Update H (DFP)
        Hy = H @ y
        H = H + np.outer(s, s) * rho - np.outer(Hy, Hy) / np.dot(y, Hy)

        # Save iteration data
        x = x_new
        g = g_new
        path.append(x.copy())
        grad_norms.append(np.linalg.norm(g))

        if verbose and (k % 500 == 0) and k > 0:
            print(f"Iter {k:5d} ||grad|| = {np.linalg.norm(g):.3e}  alpha={alpha:.2e}")

    else:
        if verbose:
            print("Reached maximum iterations without full convergence.")

    return np.array(path), np.array(grad_norms), x, g

# main
if __name__ == "__main__":
    x0 = np.array([-1.2, 1.0])

    tol = 1e-8
    max_iter = 5000

    path, grad_norms, x_final, g_final = quasi_newton_dfp(
        x0, tol=tol, max_iter=max_iter, verbose=True
    )

    print()
    print(f"Converged after {len(grad_norms)-1} iterations.")
    print(f"Final point: x = {x_final[0]:.8f}, y = {x_final[1]:.8f}")
    print(f"Final f(x,y) = {f(x_final):.4e}")

    #convergence plot
    plt.figure(figsize=(8,4))
    plt.plot(np.log10(grad_norms), linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("log(∇f)")
    plt.title("Convergence of Quasi-Newton DFP on Rosenbrock")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # contour path
    X = np.linspace(-2.0, 2.0, 400)
    Y = np.linspace(-1.0, 3.0, 400)
    X, Y = np.meshgrid(X, Y)
    Z = (1 - X)**2 + 100.0*(Y - X**2)**2

    plt.figure(figsize=(7,6))
    levels = np.logspace(-1, 3, 20)
    cs = plt.contour(X, Y, Z, levels=levels, norm=LogNorm(), cmap='viridis')

    plt.plot(path[:,0], path[:,1], marker='o', markersize=3, linewidth=1,
             color='red', label='Quasi-Newton DFP path')

    plt.annotate(
        'Global minimum (1, 1)',
        xy=(1, 1), xycoords='data',
        xytext=(0.4, 2.0), textcoords='data',
        arrowprops=dict(facecolor='darkblue', shrink=0.05, width=1.5, headwidth=8),
        horizontalalignment='center', fontsize=10
    )

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Quasi-Newton DFP Path on Rosenbrock Function')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Rosenbrock function
def f(x):
    X, Y = x[0], x[1]
    return (1 - X)**2 + 100.0 * (Y - X**2)**2

# Gradient
def grad_f(x):
    X, Y = x[0], x[1]
    df_dx = 2*(X - 1) - 400*X*(Y - X**2)
    df_dy = 200*(Y - X**2)
    return np.array([df_dx, df_dy])

# Hessian
def hess_f(x):
    X, Y = x[0], x[1]
    d2f_dx2 = 2 - 400*Y + 1200*X**2
    d2f_dxdy = -400*X
    d2f_dydx = -400*X
    d2f_dy2 = 200
    return np.array([[d2f_dx2, d2f_dxdy],
                     [d2f_dydx, d2f_dy2]])

# Backtracking line search
def backtracking_line_search(x, p, grad, alpha0=1.0, rho=0.5, c=1e-4, max_ls_iter=50):
    alpha = alpha0
    fx = f(x)
    grad_dot_p = np.dot(grad, p)
    for _ in range(max_ls_iter):
        x_new = x + alpha * p
        if f(x_new) <= fx + c * alpha * grad_dot_p:
            return alpha
        alpha *= rho
    return alpha

#%%s
# Newton's method
def newtons_method(x0, tol=1e-8, max_iter=1000, verbose=True):
    """
    Newton's method for unconstrained optimization.
    Returns: path (Nx2), grad_norms (N), final_x, final_grad
    """
    x = x0.copy()
    g = grad_f(x)
    path = [x.copy()]
    grad_norms = [np.linalg.norm(g)]

    for k in range(max_iter):
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break

        # Compute Hessian and search direction
        H = hess_f(x)
        try:
            p = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # If Hessian is singular or not positive definite
            p = -g
            if verbose:
                print(f"Warning: Hessian not invertible at iteration {k}, using gradient direction.")

        # Backtracking line search
        alpha = backtracking_line_search(x, p, g, alpha0=1.0, rho=0.5, c=1e-4)

        # Update step
        x_new = x + alpha * p
        g_new = grad_f(x_new)

        # Save data
        x = x_new
        g = g_new
        path.append(x.copy())
        grad_norms.append(np.linalg.norm(g))

        if verbose and (k % 10 == 0) and k > 0:
            print(f"Iter {k:5d} ||grad|| = {np.linalg.norm(g):.3e}  alpha={alpha:.2e}")

    else:
        if verbose:
            print("Reached maximum iterations without full convergence.")

    return np.array(path), np.array(grad_norms), x, g


# ------------------ MAIN EXECUTION ------------------
if __name__ == "__main__":
    x0 = np.array([-1.2, 1.0])

    tol = 1e-8
    max_iter = 1000

    path, grad_norms, x_final, g_final = newtons_method(
        x0, tol=tol, max_iter=max_iter, verbose=True
    )

    print()
    print(f"Converged after {len(grad_norms)-1} iterations.")
    print(f"Final point: x = {x_final[0]:.8f}, y = {x_final[1]:.8f}")
    print(f"Final f(x,y) = {f(x_final):.4e}")

    # ---------- Convergence plot ----------
    plt.figure(figsize=(8,4))
    plt.plot(np.log10(grad_norms), linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("log(∇f)")
    plt.title("Convergence of Newton's Method on Rosenbrock")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------- Contour plot ----------
    X = np.linspace(-2.0, 2.0, 400)
    Y = np.linspace(-1.0, 3.0, 400)
    X, Y = np.meshgrid(X, Y)
    Z = (1 - X)**2 + 100.0*(Y - X**2)**2

    plt.figure(figsize=(7,6))
    levels = np.logspace(-1, 3, 20)
    cs = plt.contour(X, Y, Z, levels=levels, norm=LogNorm(), cmap='viridis')

    plt.plot(path[:,0], path[:,1], marker='o', markersize=3, linewidth=1,
             color='red', label="Newton's Method path")

    plt.annotate(
        'Global minimum (1, 1)',
        xy=(1, 1), xycoords='data',
        xytext=(0.4, 2.0), textcoords='data',
        arrowprops=dict(facecolor='darkblue', shrink=0.05, width=1.5, headwidth=8),
        horizontalalignment='center', fontsize=10
    )

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Newton's Method Path on Rosenbrock Function")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()