import numpy as np
import matplotlib.pyplot as plt

#rosenbrock
def f(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_f(x):
    dfdx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dfdy = 200*(x[1] - x[0]**2)
    return np.array([dfdx, dfdy])

def hess_f(x):
    d2fxx = 2 - 400*x[1] + 1200*x[0]**2
    d2fxy = -400*x[0]
    d2fyy = 200
    return np.array([[d2fxx, d2fxy],
                     [d2fxy, d2fyy]])

# constraints
def c1(x): return np.array([1 - x[0] - x[1]])
def grad_c1(x): return np.array([[-1, -1]])

def c2(x): return np.array([1 - x[0]**2 - x[1]**2])
def grad_c2(x): return np.array([[-2*x[0], -2*x[1]]])

# backtracking line search
def backtracking(f, x, p, grad_f, alpha=1.0, rho=0.5, c=1e-4):
    fx = f(x)
    while f(x + alpha*p) > fx + c*alpha*np.dot(grad_f, p):
        alpha *= rho
    return alpha

# SQP
def SQP(x0, c_fun, grad_c_fun, max_iter=50, tol=1e-6):
    x = x0.copy()
    lam = np.array([0.0])
    grad_norms = []
    trajectory = [x.copy()]

    for k in range(max_iter):
        g = grad_f(x)
        c = c_fun(x)
        A = grad_c_fun(x)
        H = hess_f(x)

        # Solve KKT system
        KKT = np.block([
            [H, A.T],
            [A, np.zeros((1,1))]
        ])
        rhs = -np.concatenate([g, c])
        sol = np.linalg.solve(KKT, rhs)
        p = sol[:2]
        lam_new = sol[2:]

        # Line search
        alpha = backtracking(f, x, p, g)
        x_new = x + alpha*p

        grad_norms.append(np.linalg.norm(g))
        trajectory.append(x_new.copy())

        # Check convergence
        if np.linalg.norm(p) < tol and np.linalg.norm(c) < tol:
            print(f"Converged at iteration {k}")
            break

        x, lam = x_new, lam_new

    return np.array(trajectory), grad_norms, x, lam

#starting point (-1.2,1)
x0 = np.array([-1.2, 1.0])
traj1, grad_norms1, x_opt1, lam1 = SQP(x0, c1, grad_c1)
traj2, grad_norms2, x_opt2, lam2 = SQP(x0, c2, grad_c2)

# gradconvergence plot
plt.figure(figsize=(7,5))
plt.semilogy(grad_norms1, label=r'Constraint $1 - x - y = 0$')
plt.semilogy(grad_norms2, label=r'Constraint $1 - x^2 - y^2 = 0$')
plt.xlabel("Iteration", fontsize=12)
plt.ylabel(r"$\log(\|\nabla f(x)\|)$", fontsize=12)
plt.title("Convergence of the Gradient Norm")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# contour plot of path
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = (1 - X)**2 + 100*(Y - X**2)**2

plt.figure(figsize=(7,6))
plt.contour(X, Y, Z, levels=np.logspace(-1,3,20), cmap='viridis')
plt.plot(traj1[:,0], traj1[:,1], 'r-o', label=r'Path ($1-x-y=0$)')
plt.plot(traj2[:,0], traj2[:,1], 'b-o', label=r'Path ($1-x^2-y^2=0$)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour Plot with SQP Optimization Paths')
plt.legend()
plt.tight_layout()
plt.show()

