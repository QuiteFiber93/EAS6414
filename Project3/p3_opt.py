import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sin, cos
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

'''
EAS 6414 Project 3
Dylan D'Silva
Optimized Version
'''

# Configuration
config = {
    'sigma': 0.1,
    'maxiter': 30,
    'scale_factor': 0.9,
    'doPlot': False,
    'tol': 1E-3,
    'rtol': 1E-6,
    'atol': 1E-6,
    'glsdc_rtol': 1E-6,
    'glsdc_atol': 1E-6,
    'seed': 2025
}

# Constants
delim_equals = '=' * 80
delim_dash = '-' * 80

# Given Values
X0 = np.array([2, 0])
P_TRUE = np.array([0.05, 4, 0.2, -0.5, 10, pi/2])
TSPAN = [0, 300]
TEVAL = np.linspace(0.1, 300, 3000)  # More efficient than list comprehension

# Initial variational matrices (constants)
PHI0 = np.eye(2)
PSI0 = np.zeros((2, 6))


def dynamics(t: float, y: NDArray[np.floating], p: NDArray[np.floating]) -> NDArray:
    """Integrates the system dynamics and variational matrices.
    
    Args:
        t: Time variable
        y: State vector [x, phi.flat, psi.flat] (length 18)
        p: Parameter vector (length 6)
    
    Returns:
        Time derivative of state vector
    """
    # Extract components (views, not copies)
    x = y[:2]
    phi = y[2:6].reshape(2, 2)
    psi = y[6:].reshape(2, 6)
    
    # Unpack parameters
    p1, p2, p3, p4, p5, p6 = p
    
    # Pre-compute common terms
    x0_sq = x[0] ** 2
    x0_cu = x0_sq * x[0]
    phase = p5 * t + p6
    sin_phase = sin(phase)
    cos_phase = cos(phase)
    
    # Jacobian matrix A = df/dx
    A = np.array([[0, 1],
                  [-(p2 + 3*p3*x0_sq), -p1]])
    
    # Parameter sensitivity matrix dfdp
    dfdp = np.array([[0, 0, 0, 0, 0, 0],
                     [-x[1], -x[0], -x0_cu, -sin_phase, 
                      -p4*t*cos_phase, -p4*cos_phase]])
    
    # State derivative
    xdot = np.array([x[1], 
                     -(p1*x[1] + p2*x[0] + p3*x0_cu + p4*sin_phase)])
    
    # Variational equations
    phidot = A @ phi
    psidot = A @ psi + dfdp
    
    return np.concatenate([xdot, phidot.ravel(), psidot.ravel()])


def generate_measurements(state_solution, sigma: float, seed: int = None) -> NDArray:
    """Generate noisy measurements from true state."""
    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(0, sigma, len(state_solution.t))
    return state_solution.y[0, :] + noise


def glsdc(dynamics_func, z_init: NDArray, ytilde: NDArray, 
          teval: NDArray, sigma: float, tspan: list,
          tol: float = 1E-3, maxiter: int = 30,
          rtol: float = 1E-8, atol: float = 1E-8,
          verbose: bool = True) -> tuple:
    """Least Squares Differential Correction algorithm.
    
    Args:
        dynamics_func: System dynamics function
        z_init: Initial guess for [x0, p]
        ytilde: Noisy measurements
        teval: Measurement times
        sigma: Measurement standard deviation
        tspan: Time span for integration
        tol: Convergence tolerance
        maxiter: Maximum iterations
        rtol: Relative tolerance for integration
        atol: Absolute tolerance for integration
        verbose: Print iteration details
    
    Returns:
        Tuple of (coverged_estimate, trajectory, information_matrix, converged_flag)
    """
    z = z_init.copy()
    old_cost = np.inf
    n_meas = len(teval)
    
    # Pre-allocate arrays
    Lambda = np.zeros((8, 8))
    N = np.zeros(8)
    err = np.zeros(n_meas)
    
    # Pre-compute weight matrices (diagonal, time-dependent)
    W_inv = np.array([(1 + t) * sigma**2 for t in teval])
    
    if verbose:
        print(f'\n{"Iter":>5} {"x(0)":>8} {"xdot(0)":>8} {"p1":>8} {"p2":>8} '
              f'{"p3":>8} {"p4":>8} {"p5":>8} {"p6":>8} {"Cost":>12}')
        print(delim_dash)
    
    converged = False
    trajectory = None
    
    for iteration in range(maxiter):
        # Extract current guess
        x0_guess, p_guess = z[:2], z[2:]
        
        # Initialize for iteration
        Lambda.fill(0)
        N.fill(0)
        
        # Initial state with variational matrices
        state0 = np.concatenate([x0_guess, PHI0.ravel(), PSI0.ravel()])
        
        # Integrate trajectory
        trajectory = solve_ivp(
            dynamics_func, tspan, state0,
            t_eval=teval,
            rtol=rtol, atol=atol, args=(p_guess,)
        )
        
        if trajectory.status != 0:
            print(f"\nWarning: Integration failed - {trajectory.message}")
            break
        
        # Compute residuals
        err = ytilde - trajectory.y[0, :]
        
        # Build information matrix and normal equations vector
        for k in range(n_meas):
            # Extract and reshape variational matrices
            phi = trajectory.y[2:6, k].reshape(2, 2)
            psi = trajectory.y[6:, k].reshape(2, 6)
            
            # Observation matrix H = C * [Phi, Psi] where C = [1, 0]
            H = np.concatenate([phi[0, :], psi[0, :]])  # First row only
            
            # Weight for this measurement
            w = 1.0 / W_inv[k]
            
            # Accumulate normal equations
            Lambda += w * np.outer(H, H)
            N += w * H * err[k]
        
        # Compute cost
        new_cost = np.sum((err**2) / W_inv)
        
        if verbose:
            print(f'{iteration:5d} {z[0]:8.4f} {z[1]:8.4f} {z[2]:8.4f} '
                  f'{z[3]:8.4f} {z[4]:8.4f} {z[5]:8.4f} {z[6]:8.4f} '
                  f'{z[7]:8.4f} {new_cost:12.6e}')
        
        # Check convergence
        if iteration > 0 and abs(new_cost - old_cost) / old_cost <= tol:
            converged = True
            break
        
        # Update state estimate
        try:
            delta_z = np.linalg.solve(Lambda, N)
        except np.linalg.LinAlgError:
            print("\nWarning: Singular information matrix")
            break
        
        z += delta_z
        z[7] = z[7] % (2 * pi)  # Keep phase angle in [0, 2π]
        old_cost = new_cost
    
    return z, trajectory, Lambda, converged


def plot_results(sim_motion, meas_states, ytilde, tspan, save: bool = False):
    """Generate all plots for the results."""
    
    # Solution state variables vs time
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    
    ax1.plot(sim_motion.t, sim_motion.y[0, :])
    ax1.set_title('System Solution Using RKF45')
    ax1.set_ylabel(r'$x(t)$')
    ax1.set_xlim(tspan)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(sim_motion.t, sim_motion.y[1, :])
    ax2.set_xlabel('t')
    ax2.set_ylabel(r'$\dot{x}(t)$')
    ax2.grid(True, alpha=0.3)
    
    fig1.tight_layout()
    
    # Phase portrait
    fig2, ax3 = plt.subplots(figsize=(6, 6))
    ax3.plot(sim_motion.y[0, :], sim_motion.y[1, :])
    ax3.set_title('Phase Portrait of System')
    ax3.set_xlabel(r'$x(t)$')
    ax3.set_ylabel(r'$\dot{x}(t)$')
    ax3.grid(True, alpha=0.3)
    
    # Measurements overlay
    fig3, ax4 = plt.subplots(figsize=(10, 3))
    ax4.plot(sim_motion.t, sim_motion.y[0, :], linewidth=1, label='True Motion')
    ax4.scatter(meas_states.t, ytilde, s=1, c='orange', label='Measured Position')
    ax4.set_title("Solution Position vs Time")
    ax4.set_xlim(tspan)
    ax4.set_xlabel(r"$t$")
    ax4.set_ylabel(r'$x(t)$')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    fig3.tight_layout()
    
    if save:
        fig1.savefig('Images/solution_fig.png', format='png', dpi=1440, bbox_inches='tight')
        fig2.savefig('Images/phase_portrait.png', format='png', dpi=1440, bbox_inches='tight')
        fig3.savefig('Images/measurements_fig.png', format='png', dpi=1440, bbox_inches='tight')
        print('Plots saved to Images/')
    
    return fig1, fig2, fig3


def main():
    """Main execution function."""
    
    # Print header
    print(delim_equals)
    print('EAS 6414 Project 3: Initial State and Parameter Estimation')
    print(delim_equals)
    
    print('\nGiven Values')
    print(delim_dash)
    print(f'x0                              = {X0}')
    print(f'p                               = {P_TRUE}')
    print(f'Measurement Covariance          = {config["sigma"]}^2')
    print(f'Maximum Iteration for GLSDC     = {config["maxiter"]}')
    print(f'Scale factor for initial guess  = {config["scale_factor"]}\n')
    
    # Task 1: Simulate motion
    print(delim_equals)
    print('Task 1: Simulating Motion, Measurements, and Validating Matrices')
    print(delim_equals)
    print("\nIntegrating System Dynamics ...")
    
    state0 = np.concatenate([X0, PHI0.ravel(), PSI0.ravel()])
    
    # Simulate true motion
    sim_motion = solve_ivp(
        dynamics, TSPAN, state0, args=(P_TRUE,),
        rtol=config['rtol'], atol=config['atol']
    )
    
    # Generate measurements
    meas_states = solve_ivp(
        dynamics, TSPAN, state0, t_eval=TEVAL, args=(P_TRUE,),
        rtol=config['rtol'], atol=config['atol']
    )
    
    # Add measurement noise
    ytilde = generate_measurements(meas_states, config['sigma'], config['seed'])
    
    # Generate plots
    print('Generating plots ...')
    plot_results(sim_motion, meas_states, ytilde, TSPAN, save=config['doPlot'])
    
    # Task 2: GLSDC
    print('\n' + delim_equals)
    print('Task 2: GLSDC Algorithm')
    print(delim_equals)
    
    # Initial guess
    z_true = np.concatenate([X0, P_TRUE])
    z_init = config['scale_factor'] * z_true
    
    # Run GLSDC
    z_est, traj_est, Lambda, converged = glsdc(
        dynamics, z_init, ytilde, TEVAL, config['sigma'], TSPAN,
        tol=config['tol'], maxiter=config['maxiter'],
        rtol=config['glsdc_rtol'], atol=config['glsdc_atol']
    )
    
    print(delim_dash)
    if not converged:
        print(f'\nGLSDC failed to converge within {config["maxiter"]} iterations')
    else:
        print('\nGLSDC converged successfully!')
    
    print('\nFinal Estimate')
    print(f'x0 = {z_est[:2]}')
    print(f'p  = {z_est[2:]}')
    
    print('\nTrue Values')
    print(f'x0 = {X0}')
    print(f'p  = {P_TRUE}')
    
    print('\nEstimation Error')
    print(f'Δx0 = {z_est[:2] - X0}')
    print(f'Δp  = {z_est[2:] - P_TRUE}')
    
    # Covariance analysis
    P0 = np.linalg.inv(Lambda)
    print('\nCovariance Matrix P(0):')
    print(f'σ_x0    = {np.sqrt(P0[0, 0]):.6e}')
    print(f'σ_xdot0 = {np.sqrt(P0[1, 1]):.6e}')
    for i, label in enumerate(['p1', 'p2', 'p3', 'p4', 'p5', 'p6']):
        print(f'σ_{label}    = {np.sqrt(P0[i+2, i+2]):.6e}')
    
    if config['doPlot']:
        plt.show()


if __name__ == '__main__':
    main()