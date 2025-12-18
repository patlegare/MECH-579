# Imports
import numpy as np  # numpy for vectorization
from collections.abc import Callable  # For type hints
import matplotlib.pyplot as plt
from scipy import optimize
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
class HeatEquation2D:
    """Heat Equation Solver for MECH 579 Final Project

    This class will construct and solve the unsteady heat equation
    with Robin BCs as described in the assignment.
    """
    def __init__(self, x:float, y:float, height:float , n_x:int, n_y:int,
                     k:float=1.0, rho:float=1.0, cp:float=1.0,
                     CFL:float=0.1, init_condition:Callable[[np.ndarray,np.ndarray], np.ndarray] = lambda x,y: np.sin(x+y)):
        """Intializition function for the heat equation

        Parameters

        ------

        x (float): Physical Size of CPU in x-direction [m]

        y (float): Physical Size of CPU in y-direction [m]

        n_x (int): Number of grid points in x-direction [m]

        n_y (int): Number of grid points in y-direction [m]

        k (float): The heat transfer coefficient of the CPU [W/[mK]]

        rho (float): Constant density of CPU [kg/m^3]

        cp (float): Specific heat capacity of CPU [kJ/[kgK]]

        CFL (float): Courant-Friedrichs-Lewy Number

        init_condition (function(x,y)): Initial condition of the CPU
        """
        ## MESHING variables
        self.n_x = n_x
        self.n_y = n_y
        self.boundary_conditions = []
        # Physical locations
        x_axis = np.linspace(0, x, self.n_x)
        y_axis = np.linspace(0, y, self.n_y)
        self.X, self.Y = np.meshgrid(x_axis, y_axis, indexing='ij')
        self.dx = x_axis[1] - x_axis[0]
        self.dy = y_axis[1] - y_axis[0]
        # Variables of Mesh size
        self.u = np.zeros((self.n_x, self.n_y))
        self.h_top_values = np.zeros((self.n_x, self.n_y))
        self.h_boundary_values = np.zeros((self.n_x, self.n_y))

        ## Heat Generation Properties
        self.heat_generation_function = lambda x, y, a, b, c: a * x + b * y + c  # Can be changed
        self.heat_gen_a = 0
        self.heat_gen_b = 0
        self.heat_gen_c = 0
        self.heat_generation_total = 0

        ## Material Properties
        self.k = k
        self.rho = rho
        self.cp = cp
        self.thermal_alpha = self.k / (self.rho * self.cp)
        self.height = height  # m

        ## Temporal Properties
        self.CFL = CFL
        self.dt = self.CFL * (self.dx * self.dy) / self.thermal_alpha
        self.current_time = 0
        self.steady_state_error = 1E2  # Large inital number to ensure that the problem will continue
        self.max_iter = 5E4
        self.init_condition = init_condition
        self.apply_initial_conditions()

        ## External Variables of Air
        self.ext_k = 0.02772  # W/m/K Thermal Coeffcient
        self.ext_Pr = 0.7215  # Prantl Number
        self.ext_nu = 1.506 * 10 ** (-5)  # m^2/s Kinematic Viscosity
        self.ext_T = 273 + 20  # K Temperature

        ## Fan Variables
        self.v = 10  # m/s Air Velocity
        self.fan_efficiency_func = lambda v: -0.002 * v**2 + 0.08 * v
        self.fan_efficiency = self.fan_efficiency_func(self.v)

        self.verbose = False

    def set_initial_conditions(self, initial_conditions:Callable[[np.ndarray,np.ndarray],np.ndarray]):
        """Sets the initial condition

        Parameters

        ------

        initial_conditions(function(x,y)): Initial condition of the CPU
        """
        self.init_condition = initial_conditions

    def apply_initial_conditions(self):
        """Applies the initial condition into self.u"""
        self.u = self.init_condition(self.X, self.Y)

    def reset(self):
        """Resets the heat equation"""
        self.apply_initial_conditions()
        self.current_time = 0
        self.steady_state_error = 1E2

    def set_heat_generation(self, heat_generation_function: Callable[[np.ndarray,np.ndarray,float,float,float], np.ndarray],
                            a: float, b: float, c: float):
        """Sets the heat generation function and associated variables

        Parameters

        ------

        heat_generation_function (function(x,y,a,b,c)): Function that dictates the heat generation by the CPU

        integrated_total (float): Total integrated value

        a, b, c (float): Variables associated with the heat generation function
        """
        self.heat_generation_function = heat_generation_function
        self.heat_gen_a = a
        self.heat_gen_b = b
        self.heat_gen_c = c
        heat_generation_matrix = self.heat_generation_function(self.X, self.Y, self.heat_gen_a, self.heat_gen_b, self.heat_gen_c) * self.dx * self.dy * self.height
        i0, iN, j0, jN = 0, self.n_x - 1, 0, self.n_y - 1
        # Boundaries with one side
        heat_generation_matrix[i0, :] /= 2
        heat_generation_matrix[iN, :] /= 2
        heat_generation_matrix[j0, :] /= 2
        heat_generation_matrix[jN, :] /= 2
        # Boundaries with two sides
        heat_generation_matrix[i0, j0] /= 2
        heat_generation_matrix[iN, jN] /= 2
        heat_generation_matrix[iN, j0] /= 2
        heat_generation_matrix[i0, jN] /= 2
        self.heat_generation_total = np.sum(np.sum(heat_generation_matrix))

    def set_fan_velocity(self, v: float):
        """Sets the fan velocity

        Parameters

        ------

        v (float): Variable associated with the fan velocity
        """
        self.v = v
        self.fan_efficiency = self.fan_efficiency_func(self.v)

    def h_boundary(self, u: np.ndarray):
        """Calculates the convective heat transfer coefficient at the boundaries

        Parameters

        ------

        u (np.ndarray): Current Temperature Mesh
        """
        beta = 1 / ((u + self.ext_T) / 2)
        #added absolute value so its never negative 
        rayleigh = 9.81 * beta * np.abs(u - self.ext_T) * self.dx**3 / (self.ext_nu**2) * self.ext_Pr
        nusselt = (0.825 + (0.387 * rayleigh**(1/6)) /
                   (1 + (0.492 / self.ext_Pr)**(9/16))**(8/27))**2
        return nusselt * self.ext_k / self.dx

    def h_top(self, x: np.ndarray, u):
        """Calculates the convective heat transfer coefficient from the fan velocity

        Parameters

        ------

        x (np.ndarray): x position

        u (np.ndarray): UNUSED
        """
        Rex = self.v * x / self.ext_nu
        r, c = Rex.shape
        Nux = np.zeros((r, c))
        for i in range(r):
            for j in range(c):
                if Rex[i, j] < 5E5:
                    Nux[i, j] = 0.332 * Rex[i, j]**0.5 * self.ext_Pr**(1/3)
                else:
                    Nux[i, j] = 0.0296 * Rex[i, j]**0.8 * self.ext_Pr**(1/3)
        h = Nux * self.ext_k / (x + 1E-5)
        return h

    def calculate_h(self):
        """Calculates all necessary convective heat transfer coefficients"""
        self.h_top_values = self.h_top(self.X, self.u)
        self.h_boundary_values = self.h_boundary(self.u)

    def apply_boundary_conditions(self, old_u):
        """Calculates the change in temperature at the boundary.

        Parameters

        -----

        old_u (np.ndarray): Current Temperature Mesh
        """
        e_dot = self.heat_generation_function(self.X, self.Y, self.heat_gen_a, self.heat_gen_b, self.heat_gen_c)
        tau = self.thermal_alpha * self.dt / (self.dx * self.dy)
        i0, j0, iN, jN = 0, 0, self.n_x - 1, self.n_y - 1
        # Left
        self.u[i0, 1:-1] = (old_u[i0, 1:-1] +
                            2 * tau * self.h_boundary_values[i0, 1:-1] / self.k * self.dy * (self.ext_T - old_u[i0, 1:-1]) +
                            tau * self.dx * (old_u[i0, 2:] - old_u[i0, 1:-1]) / self.dy +
                            tau * self.dx * (old_u[i0, 1:-1] - old_u[i0, 2:]) / self.dy +
                            2 * tau * self.dy * (old_u[i0 + 1, 1:-1] - old_u[i0, 1:-1]) / self.dx +
                            tau * self.h_top_values[i0, 1:-1] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[i0, 1:-1]) +
                            tau * e_dot[i0, 1:-1] / self.k * self.dx * self.dy)

        # Right
        self.u[iN, 1:-1] = (old_u[iN, 1:-1] +
                            2 * tau * self.h_boundary_values[iN, 1:-1] / self.k * self.dy * (self.ext_T - old_u[iN, 1:-1]) +
                            tau * self.dx * (old_u[iN, 2:] - old_u[iN, 1:-1]) / self.dy +
                            tau * self.dx * (old_u[iN, 1:-1] - old_u[iN, 2:]) / self.dy +
                            2 * tau * self.dy * (old_u[iN - 1, 1:-1] - old_u[iN, 1:-1]) / self.dx +
                            tau * self.h_top_values[iN, 1:-1] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[iN, 1:-1]) +
                            tau * e_dot[iN, 1:-1] / self.k * self.dx * self.dy)

        # Bottom
        self.u[1:-1, j0] = (old_u[1:-1, j0] +
                            2 * tau * self.h_boundary_values[1:-1, j0] / self.k * self.dx * (self.ext_T - old_u[1:-1, j0]) +
                            tau * self.dy * (old_u[2:, j0] - old_u[1:-1, j0]) / self.dx +
                            tau * self.dy * (old_u[1:-1, j0] - old_u[2:, j0]) / self.dx +
                            2 * tau * self.dx * (old_u[1:-1, j0 + 1] - old_u[1:-1, j0]) / self.dy +
                            tau * self.h_top_values[1:-1, j0] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[1:-1, j0]) +
                            tau * e_dot[1:-1, j0] / self.k * self.dx * self.dy)

        # Top
        self.u[1:-1, jN] = (old_u[1:-1, jN] +
                            2 * tau * self.h_boundary_values[1:-1, jN] / self.k * self.dx * (self.ext_T - old_u[1:-1, jN]) +
                            tau * self.dy * (old_u[2:, jN] - old_u[1:-1, jN]) / self.dx +
                            tau * self.dy * (old_u[1:-1, jN] - old_u[2:, jN]) / self.dx +
                            2 * tau * self.dx * (old_u[1:-1, jN - 1] - old_u[1:-1, jN]) / self.dy +
                            tau * self.h_top_values[1:-1, jN] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[1:-1, jN]) +
                            tau * e_dot[1:-1, jN] / self.k * self.dx * self.dy)

        ## Bottom Left Corner
        self.u[i0, j0] = (old_u[i0, j0] +
                          2 * tau * self.h_boundary_values[i0, j0] * self.dy / self.k * (self.ext_T - old_u[i0, j0]) +
                          2 * tau * self.h_boundary_values[i0, j0] * self.dx / self.k * (self.ext_T - old_u[i0, j0]) +
                          2 * tau * self.dx * (old_u[i0, j0 + 1] - old_u[i0, j0]) / self.dy +
                          2 * tau * self.dy * (old_u[i0 + 1, j0] - old_u[i0, j0]) / self.dx +
                          tau * self.h_top_values[i0, j0] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[i0, j0]) +
                          tau * e_dot[i0, j0] / self.k * self.dx * self.dy)
        ## Bottom Right Corner
        self.u[iN, j0] = (old_u[iN, j0] +
                          2 * tau * self.h_boundary_values[iN, j0] * self.dy / self.k * (self.ext_T - old_u[iN, j0]) +
                          2 * tau * self.h_boundary_values[iN, j0] * self.dx / self.k * (self.ext_T - old_u[iN, j0]) +
                          2 * tau * self.dx * (old_u[iN, j0 + 1] - old_u[iN, j0]) / self.dy +
                          2 * tau * self.dy * (old_u[iN - 1, j0] - old_u[iN, j0]) / self.dx +
                          tau * self.h_top_values[iN, j0] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[iN, j0]) +
                          tau * e_dot[iN, j0] / self.k * self.dx * self.dy)
        ## Top Left Corner
        self.u[i0, jN] = (old_u[i0, jN] +
                          2 * tau * self.h_boundary_values[i0, jN] * self.dy / self.k * (self.ext_T - old_u[i0, jN]) +
                          2 * tau * self.h_boundary_values[i0, jN] * self.dx / self.k * (self.ext_T - old_u[i0, jN]) +
                          2 * tau * self.dx * (old_u[i0, jN - 1] - old_u[i0, jN]) / self.dy +
                          2 * tau * self.dy * (old_u[i0 + 1, jN] - old_u[i0, jN]) / self.dx +
                          tau * self.h_top_values[i0, jN] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[i0, jN]) +
                          tau * e_dot[i0, jN] / self.k * self.dx * self.dy)
        ## Top Right Corner
        self.u[iN, jN] = (old_u[iN, jN] +
                          2 * tau * self.h_boundary_values[iN, jN] * self.dy / self.k * (self.ext_T - old_u[iN, jN]) +
                          2 * tau * self.h_boundary_values[iN, jN] * self.dx / self.k * (self.ext_T - old_u[iN, jN]) +
                          2 * tau * self.dx * (old_u[iN, jN - 1] - old_u[iN, jN]) / self.dy +
                          2 * tau * self.dy * (old_u[iN - 1, jN] - old_u[iN, jN]) / self.dx +
                          tau * self.h_top_values[iN, jN] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[iN, jN]) +
                          tau * e_dot[iN, jN] / self.k * self.dx * self.dy)
        return

    def step_forward_in_time(self):
        """Steps forward in time 1 timestep"""
        self.calculate_h()
        old_u = self.u.copy()
        self.apply_boundary_conditions(old_u)
        tau = self.thermal_alpha * self.dt / (self.dx * self.dy)
        self.u[1:-1, 1:-1] = (old_u[1:-1, 1:-1] +
                              tau * (
                                  self.dy * (old_u[2:, 1:-1] - 2 * old_u[1:-1, 1:-1] + old_u[0:-2, 1:-1]) / self.dx +
                                  self.dx * (old_u[1:-1, 2:] - 2 * old_u[1:-1, 1:-1] + old_u[1:-1, 0:-2]) / self.dy
                              ) + tau * (
                                  self.h_top_values[1:-1, 1:-1] / self.k * self.dx * self.dy / self.height * (self.ext_T - old_u[1:-1, 1:-1]) +
                                  self.dx * self.dy / self.k * self.heat_generation_function(
                                      self.X[1:-1, 1:-1], self.Y[1:-1, 1:-1],
                                      self.heat_gen_a, self.heat_gen_b, self.heat_gen_c
                                  )
                              ))
        self.steady_state_error = np.linalg.norm(self.u - old_u, np.inf)
        self.current_time += self.dt

    def solve_until_steady_state(self, tol: float = 1e-3):
        """Solves until steady state is reached

        Parameters

        ------

        tol (float, optional): Tolerance until steady state
        """
        iter = 0
        self.step_forward_in_time()
        while self.steady_state_error > tol and iter < self.max_iter:
            self.step_forward_in_time()
            iter += 1
            if (iter % 1000) == 0 and self.verbose:
                print(f"Iteration: {iter}, Error: {self.steady_state_error}")

    def solve_until_time(self, final_time: float):
        """Solves until time is reached

        Parameters

        final_time (float): Final time of sim ul a ti o n
        """
        iter = 0
        while self.current_time < final_time:
            self.step_forward_in_time()
            iter += 1
            if (iter % 1000) == 0 and self.verbose:
                print(f"Iteration: {iter}, Time: {self.current_time}")


if __name__ == "__main__":
    # Physical Dimensions
    cpu_x = 0.04  # m
    cpu_y = 0.04  # m
    cpu_z = 0.04  # m
    N = 25

    # Temporal Parameters
    CFL = 0.5
    # Silicon Constants
    k_si = 149
    rho_si = 2323
    c_si = 19.789 / 28.085 * 1000  # J/(kgK)

    #Create global storage
    history = {
        "obj": [],          # objective value
        "Tmax": [],         # max temperature
        "eta": [],          # fan efficiency
        "constraint": [],   # power constraint residual (should → 0)
        "x": [],            # design variables [v, a, b, c]
        "grad_L": []        # norm of gradient of Lagrangian
    }

    def initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r, c = x.shape
        u = np.zeros([r, c])
        ## Cosine Case
        u = 70 * np.sin(x * np.pi / cpu_x) * np.sin(y * np.pi / cpu_y) + 293
        return u

    def heat_generation_function(x: np.ndarray, y: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * x + b * y + c

    ## Problem Set up
    heq = HeatEquation2D(cpu_x, cpu_y, cpu_z, N, N,
                        k=k_si, rho=rho_si, cp=c_si,
                        init_condition=initial_condition)
    # Test values for a,b,c
    test_a = 1 * 10**6
    test_b = 1 * 10**6
    test_c = (1.5625 * 10**5 - 0.02 * test_b - 0.02 * test_a)
    ## Fan velocity for test
    fan_velocity = 10.0
    heq.set_heat_generation(heat_generation_function, test_a, test_b, test_c)
    heq.set_fan_velocity(fan_velocity)
    ## plotting initial conditions
    fig, ax = plt.subplots()
    contour1 = ax.contourf(heq.X, heq.Y, heq.u - 273)
    fig.colorbar(contour1, ax=ax)
    plt.show()
    ## Setting objective function
    heq.max_iter = 5E5
    #w1 = 0.5
    w1 = 0.2
    w2 = 1 - w1
    global_tolerance = 1E-3

    #cache so the callback can reuse results
    cache = {"x": None, "Tmax": None, "eta": None, "con": None, "obj": None}

    def objective_function(x):
        """Objective Function
#put in objective functions
        Parameters

        ------

        x[0] (float): Velocity of Fan
        x[1] (float): a coefficient of heat generation
        x[2] (float): b coefficient of heat generation
        x[3] (float): c coefficient of heat generation

        """
        # Extract values
        v, a, b, c = x
        heq.reset()

        # Set variables
        heq.set_fan_velocity(v)
        heq.set_heat_generation(heat_generation_function, a, b, c)

        # Solve PDE
        heq.solve_until_steady_state(tol=global_tolerance)

        # Store the results
        Tmax = np.max(heq.u)
        eta = heq.fan_efficiency
        obj = w1 * (Tmax / 273.0) - w2 * eta #normalizing for conditioning
        con = heq.heat_generation_total - 10.0

        #cache instead of history
        cache["obj"] = obj
        cache["Tmax"] = Tmax
        cache["eta"] = eta
        cache["con"] = con
        cache["x"] = np.array(x, dtype=float).copy()

        return obj
    
    #jax AD gradient for objective function
    Xj = jnp.asarray(heq.X, dtype=jnp.float64)
    Yj = jnp.asarray(heq.Y, dtype=jnp.float64)
    dx = float(heq.dx)
    dy = float(heq.dy)
    dt = float(heq.dt)
    alp = float(heq.thermal_alpha)

    #constant 
    k_air = float(heq.ext_k)
    Pr = float(heq.ext_Pr)
    nu = float(heq.ext_nu)
    Tinf = float(heq.ext_T)
    g0 = 9.81

    k_s = float(heq.k)
    H = float(heq.height)

    n = int(heq.n_x)  # assume a square grid
    tau = alp * dt / (dx * dy)

    #functions (same as before but using jax.numpy)
    def u0_jax():
        return 70.0 * jnp.sin(Xj * jnp.pi / cpu_x) * jnp.sin(Yj * jnp.pi / cpu_y) + 293.0
    def eta_jax(v):
        return -0.002*v*v+0.08*v
    def q_jax(a, b, c):
        return a * Xj + b * Yj + c
    #adding small term so i dont get nan in output 
    EPS=1e-12
    def h_top_jax(v):
        Rex = v * Xj / nu
        Rex=jnp.maximum(Rex,0.0)+EPS
        Nu_lam = 0.332 * (Rex**0.5) * (Pr ** (1.0 / 3.0))
        Nu_tur = 0.0296 * (Rex ** 0.8) * (Pr ** (1.0 / 3.0))
        Nu = jnp.where(Rex < 5e5, Nu_lam, Nu_tur)
        return Nu * k_air / (Xj + 1e-5)

    def h_side_jax(u):
        denom = (u + Tinf) / 2.0
        denom = jnp.where(jnp.abs(denom) < EPS, jnp.sign(denom) * EPS, denom)
        beta= 1.0/denom
        Ra = g0 * beta * jnp.abs(u - Tinf) * (dx ** 3) / (nu ** 2) * Pr
        Ra=jnp.maximum(Ra,0.0)+EPS

        Nu = (0.825 + (0.387 * (Ra ** (1.0 / 6.0))) /
              ((1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0))) ** 2
        return Nu * k_air / dx

    def step_jax(u, v, a, b, c):
        ht = h_top_jax(v)
        hb = h_side_jax(u)
        e = q_jax(a, b, c)

        u0 = u
        un = u0

        i0, iN, j0, jN = 0, n - 1, 0, n - 1

        # Left edge
        un = un.at[i0, 1:-1].set(
            u0[i0, 1:-1]
            + 2 * tau * hb[i0, 1:-1] / k_s * dy * (Tinf - u0[i0, 1:-1])
            + tau * dx * (u0[i0, 2:] - u0[i0, 1:-1]) / dy
            + tau * dx * (u0[i0, 1:-1] - u0[i0, 2:]) / dy
            + 2 * tau * dy * (u0[i0 + 1, 1:-1] - u0[i0, 1:-1]) / dx
            + tau * ht[i0, 1:-1] / k_s * dx * dy / H * (Tinf - u0[i0, 1:-1])
            + tau * e[i0, 1:-1] / k_s * dx * dy
        )

        # Right edge
        un = un.at[iN, 1:-1].set(
            u0[iN, 1:-1]
            + 2 * tau * hb[iN, 1:-1] / k_s * dy * (Tinf - u0[iN, 1:-1])
            + tau * dx * (u0[iN, 2:] - u0[iN, 1:-1]) / dy
            + tau * dx * (u0[iN, 1:-1] - u0[iN, 2:]) / dy
            + 2 * tau * dy * (u0[iN - 1, 1:-1] - u0[iN, 1:-1]) / dx
            + tau * ht[iN, 1:-1] / k_s * dx * dy / H * (Tinf - u0[iN, 1:-1])
            + tau * e[iN, 1:-1] / k_s * dx * dy
        )

        # Bottom edge
        un = un.at[1:-1, j0].set(
            u0[1:-1, j0]
            + 2 * tau * hb[1:-1, j0] / k_s * dx * (Tinf - u0[1:-1, j0])
            + tau * dy * (u0[2:, j0] - u0[1:-1, j0]) / dx
            + tau * dy * (u0[1:-1, j0] - u0[2:, j0]) / dx
            + 2 * tau * dx * (u0[1:-1, j0 + 1] - u0[1:-1, j0]) / dy
            + tau * ht[1:-1, j0] / k_s * dx * dy / H * (Tinf - u0[1:-1, j0])
            + tau * e[1:-1, j0] / k_s * dx * dy
        )

        # Top edge
        un = un.at[1:-1, jN].set(
            u0[1:-1, jN]
            + 2 * tau * hb[1:-1, jN] / k_s * dx * (Tinf - u0[1:-1, jN])
            + tau * dy * (u0[2:, jN] - u0[1:-1, jN]) / dx
            + tau * dy * (u0[1:-1, jN] - u0[2:, jN]) / dx
            + 2 * tau * dx * (u0[1:-1, jN - 1] - u0[1:-1, jN]) / dy
            + tau * ht[1:-1, jN] / k_s * dx * dy / H * (Tinf - u0[1:-1, jN])
            + tau * e[1:-1, jN] / k_s * dx * dy
        )

        # Corners
        un = un.at[i0, j0].set(
            u0[i0, j0]
            + 2 * tau * hb[i0, j0] * dy / k_s * (Tinf - u0[i0, j0])
            + 2 * tau * hb[i0, j0] * dx / k_s * (Tinf - u0[i0, j0])
            + 2 * tau * dx * (u0[i0, j0 + 1] - u0[i0, j0]) / dy
            + 2 * tau * dy * (u0[i0 + 1, j0] - u0[i0, j0]) / dx
            + tau * ht[i0, j0] / k_s * dx * dy / H * (Tinf - u0[i0, j0])
            + tau * e[i0, j0] / k_s * dx * dy
        )

        un = un.at[iN, j0].set(
            u0[iN, j0]
            + 2 * tau * hb[iN, j0] * dy / k_s * (Tinf - u0[iN, j0])
            + 2 * tau * hb[iN, j0] * dx / k_s * (Tinf - u0[iN, j0])
            + 2 * tau * dx * (u0[iN, j0 + 1] - u0[iN, j0]) / dy
            + 2 * tau * dy * (u0[iN - 1, j0] - u0[iN, j0]) / dx
            + tau * ht[iN, j0] / k_s * dx * dy / H * (Tinf - u0[iN, j0])
            + tau * e[iN, j0] / k_s * dx * dy
        )

        un = un.at[i0, jN].set(
            u0[i0, jN]
            + 2 * tau * hb[i0, jN] * dy / k_s * (Tinf - u0[i0, jN])
            + 2 * tau * hb[i0, jN] * dx / k_s * (Tinf - u0[i0, jN])
            + 2 * tau * dx * (u0[i0, jN - 1] - u0[i0, jN]) / dy
            + 2 * tau * dy * (u0[i0 + 1, jN] - u0[i0, jN]) / dx
            + tau * ht[i0, jN] / k_s * dx * dy / H * (Tinf - u0[i0, jN])
            + tau * e[i0, jN] / k_s * dx * dy
        )

        un = un.at[iN, jN].set(
            u0[iN, jN]
            + 2 * tau * hb[iN, jN] * dy / k_s * (Tinf - u0[iN, jN])
            + 2 * tau * hb[iN, jN] * dx / k_s * (Tinf - u0[iN, jN])
            + 2 * tau * dx * (u0[iN, jN - 1] - u0[iN, jN]) / dy
            + 2 * tau * dy * (u0[iN - 1, jN] - u0[iN, jN]) / dx
            + tau * ht[iN, jN] / k_s * dx * dy / H * (Tinf - u0[iN, jN])
            + tau * e[iN, jN] / k_s * dx * dy
        )

        # Interior
        un = un.at[1:-1, 1:-1].set(
            u0[1:-1, 1:-1]
            + tau * (
                dy * (u0[2:, 1:-1] - 2 * u0[1:-1, 1:-1] + u0[0:-2, 1:-1]) / dx
                + dx * (u0[1:-1, 2:] - 2 * u0[1:-1, 1:-1] + u0[1:-1, 0:-2]) / dy
            )
            + tau * (
                ht[1:-1, 1:-1] / k_s * dx * dy / H * (Tinf - u0[1:-1, 1:-1])
                + dx * dy / k_s * e[1:-1, 1:-1]
            )
        )

        err = jnp.max(jnp.abs(un - u0))
        return un, err
    
    def obj_jax(x):
        v, a, b, c = x
        u = u0_jax()
        err0 = jnp.array(1e9)
        it0 = jnp.array(0)

        nsteps = 3000  # small for testing, increase after

        def cond(state):
            _, _, it = state
            return it < nsteps  # Only depend on iteration count: differentiable

        def body(state):
            u, err, it = state
            un, ern = step_jax(u, v, a, b, c)
            return (un, ern, it + 1)

        u, _, _ = jax.lax.while_loop(cond, body, (u, err0, it0))

        Tmax = jnp.max(u)
        return w1 * (Tmax / 273.0) - w2 * eta_jax(v)
    
    #using forward mode (AD) instead of classic jac.grad() because its taking too long to run
    #jax.jacfwd returns jacobian isntead of a scalar
    grad_obj_jax = jax.grad(obj_jax) 

    #central difference for derivative comparison
    def fd_central(fun, x, i, h):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        return (fun(xp) - fun(xm)) / (2.0 * h)
    
    ## Bounds for inputs
    bounds = [
        (0, 30),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (0, np.inf),
    ]

    def constraint_one(x):
        """Constraint for total power generation by the CPU

        Parameters
        #put in constraints
        -------

        x[1] (float): a coefficient of heat generation
        x[2] (float): b coefficient of heat generation
        x[3] (float): c coefficient of heat generation
        """

        _, a, b, c = x  # Extract Variables

        #Calculate heat generation
        heq.set_heat_generation(heat_generation_function, a, b, c)

        return heq.heat_generation_total - 10.0

    ## Setting the constraints as a nonlinear constrant list (needed for trust-constr)
    nlc = optimize.NonlinearConstraint(constraint_one, 0.0, 0.0, jac="2-point")

    ## Creating the initial guess
    v0 = 10
    x0_heat = 0
    x0 = [v0,
          x0_heat * 10 ** 5,
          x0_heat * 10 ** 5,
          (156250 - 0.02 * x0_heat * 10 ** 5 - 0.02 * x0_heat * 10 ** 5)]
    heq.verbose = False

    #adding callback to store iteration history and compute lagrangian (for plots)
    def callback(xk, state):
        xk = np.array(xk, dtype=float)
        if cache["x"] is None or not np.allclose(xk, cache["x"], rtol=0, atol=0):
            objective_function(xk)

        g = np.asarray(state.grad, dtype=float)
        Jc = np.asarray(state.jac[0], dtype=float)
        lam = np.asarray(state.v[0], dtype=float).ravel()

        grad_L = g + Jc.T @ lam
        grad_L_norm = np.linalg.norm(grad_L)

        #append
        history["obj"].append(state.fun)
        history["Tmax"].append(cache["Tmax"])
        history["eta"].append(cache["eta"])
        history["constraint"].append(cache["con"])
        history["x"].append(xk.copy())
        history["grad_L"].append(grad_L_norm)

    # Optimize (using trust-constr)
    optimization_result = optimize.minimize(
        objective_function,
        x0,
        method="trust-constr", #takes a long time to run 
        jac="2-point", #2-point computes jacboian using finite differences (forward difference)
        bounds=bounds,
        constraints=[nlc],
        callback=callback,
        options={"maxiter": 30}
    )
    # Build optimal solution
    heq.set_fan_velocity(optimization_result.x[0])
    heq.set_heat_generation(heat_generation_function, optimization_result.x[1], optimization_result.x[2],
                            optimization_result.x[3])
    print(
        f"Optimization result: {objective_function(optimization_result.x)}\n"
        f"v: {optimization_result.x[0]} m/s, "
        f"a: {optimization_result.x[1]}, "
        f"b: {optimization_result.x[2]}, "
        f"c: {optimization_result.x[3]}"
        f"\n"
        f"Constraints:\n"
        f"Total Heat Generation: {heq.heat_generation_total} Constraint: {constraint_one(optimization_result.x)}\n"
    )
    #table for comparing FD vs AD
    x_eval = np.array(optimization_result.x, dtype=float)

    idx = 0  #in this case, comparing v. 0=v, 1=a, 2=b, 3=c
    name = ["v", "a", "b", "c"][idx]

    hs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    fd_vals = [fd_central(objective_function, x_eval, idx, h) for h in hs]

    fd_ref = fd_vals[-1]
    for k_ in range(2, len(fd_vals)):
        if abs(fd_vals[-k_] - fd_vals[-k_ + 1]) < 1e-10 * max(1.0, abs(fd_vals[-k_ + 1])):
            fd_ref = fd_vals[-k_]
            break

    ad_val = float(grad_obj_jax(jnp.array(x_eval, dtype=jnp.float64))[idx])
    #finite difference convergence "sweep"
    print("\nFD convergence for dJ/d%s:" % name)
    for h, vfd in zip(hs, fd_vals):
        print(f"h = {h:.1e}  FD = {vfd:.16e}")

    #table comparison for AD vs FD
    print("\nDerivative comparison (all digits shown):")
    print(f"{'parameter':>6}  {'FD (converged)':>22}  {'AD (JAX)':>22}  {'absolute difference':>22}")
    print(f"{name:>6}  {fd_ref:>22.16e}  {ad_val:>22.16e}  {abs(fd_ref-ad_val):>22.16e}")

    g_ad = np.array(grad_obj_jax(jnp.array(x_eval, dtype=jnp.float64)), dtype=float)
    print("\nFull AD gradient at x_eval [dJ/dv, dJ/da, dJ/db, dJ/dc]:")
    print(g_ad)

    '''#plots for part b
    it = np.arange(1, len(history["obj"]) + 1)
    x_hist = np.array(history["x"])

    #Lagrangian gradient
    plt.figure()
    plt.plot(it, history["grad_L"])
    plt.xlabel("Iteration")
    plt.ylabel(r"$\|\nabla \mathcal{L}\|_2$")
    plt.title("Lagrangian Gradient Norm Convergence")
    plt.grid(True)

    #objective function convergence
    plt.figure()
    plt.plot(it, history["obj"])
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("Objective Convergence")
    plt.grid(True)

    #maximum temperature convergence
    plt.figure()
    plt.plot(it, history["Tmax"])
    plt.xlabel("Iteration")
    plt.ylabel(r"$T_{\max}$ (K)")
    plt.title("Maximum Temperature Convergence")
    plt.grid(True)

    #efficiency convergence
    plt.figure()
    plt.plot(it, history["eta"])
    plt.xlabel("Iteration")
    plt.ylabel(r"$\eta$")
    plt.title("Fan Efficiency Convergence")
    plt.grid(True)

    #constraint convergence (heat=10W)
    plt.figure()
    plt.plot(it, history["constraint"])
    plt.xlabel("Iteration")
    plt.ylabel("Power Constraint Residual (W)")
    plt.title("Constraint Convergence")
    plt.grid(True)

    #fan velocity convergence
    plt.figure()
    plt.plot(it, x_hist[:, 0])
    plt.xlabel("Iteration")
    plt.ylabel(r"$v$ (m/s)")
    plt.title("Fan Velocity Convergence")
    plt.grid(True)

    #plot for design parameters a and b
    plt.figure()
    plt.plot(it, x_hist[:, 1], label="a")
    plt.plot(it, x_hist[:, 2], label="b")
    plt.xlabel("Iteration")
    plt.ylabel("Heat generation slope parameters")
    plt.title("Design Parameter Convergence: a and b")
    plt.legend()
    plt.grid(True)

    #plot for design parameter c
    plt.figure()
    plt.plot(it, x_hist[:, 3])
    plt.xlabel("Iteration")
    plt.ylabel("c")
    plt.title("Design Parameter Convergence: c")
    plt.grid(True) 
    plt.show()

    ## Plot optimal solution
    fig, ax = plt.subplots()
    contour3 = ax.contourf(heq.X, heq.Y, heq.u - 273)
    fig.colorbar(contour3, ax=ax)
    plt.show()'''
