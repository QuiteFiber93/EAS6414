import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sin, cos
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from matplotlib.patches import Ellipse
from multiprocessing import Pool, cpu_count
from functools import partial

'''
EAS 6414 Project 3
Optimized Complete Solution with Tasks 1-4
'''

# Configuration
config = {
    'sigma': 0.1,
    'maxiter': 30,
    'scale_factor': 0.9,
    'doPlot': True,
    'tol': 1E-3,
    'rtol': 1E-6,
    'atol': 1E-6,
    'glsdc_rtol': 1E-6,
    'glsdc_atol': 1E-6,
    'seed': 2025,
    'n_monte_carlo': 1000,
    'use_parallel': True,
    'n_processes': None  # None = use all available cores
}

# Constants
delim_equals = '=' * 80
delim_dash = '-' * 80

# Given Values
X0 = np.array([2, 0])
P_TRUE = np.array([0.05, 4, 0.2, -0.5, 10, pi/2])
TSPAN = [0, 300]
TEVAL = np.linspace(0.1, 300, 3000)

# Initial variational matrices
PHI0 = np.eye(2)
PSI0 = np.zeros((2, 6))

# Epochs for Task 3 and 4
EPOCHS = [0, 100, 200, 300]
EPOCH_INDICES = [0, 1000, 2000, 3000]  # Corresponding to t = 0, 100, 200, 300


def dynamics(t: float, y: NDArray[np.floating], p: NDArray[np.floating]) -> NDArray:
    """Integrates the system dynamics and variational matrices."""
    x = y[:2]
    phi = y[2:6].reshape(2, 2)
    psi = y[6:].reshape(2, 6)
    
    p1, p2, p3, p4, p5, p6 = p
    
    x0_sq = x[0] ** 2
    x0_cu = x0_sq * x[0]
    phase = p5 * t + p6
    sin_phase = sin(phase)
    cos_phase = cos(phase)
    
    # Jacobian A = df/dx
    A = np.array([[0, 1],
                  [-(p2 + 3*p3*x0_sq), -p1]])
    
    # Parameter sensitivity dfdp
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
          rtol: float = 1E-6, atol: float = 1E-6,
          verbose: bool = True) -> tuple:
    """Generalized Least Squares Differential Correction algorithm."""
    z = z_init.copy()
    old_cost = np.inf
    n_meas = len(teval)
    
    Lambda = np.zeros((8, 8))
    N = np.zeros(8)
    err = np.zeros(n_meas)
    
    # Weight matrices (time-dependent)
    W_inv = np.array([(1 + t) * sigma**2 for t in teval])
    
    if verbose:
        print(f'\n{"Iter":>5} {"x(0)":>8} {"xdot(0)":>8} {"p1":>8} {"p2":>8} '
              f'{"p3":>8} {"p4":>8} {"p5":>8} {"p6":>8} {"Cost":>12}')
        print(delim_dash)
    
    converged = False
    trajectory = None
    
    for iteration in range(maxiter):
        x0_guess, p_guess = z[:2], z[2:]
        
        Lambda.fill(0)
        N.fill(0)
        
        state0 = np.concatenate([x0_guess, PHI0.ravel(), PSI0.ravel()])
        
        trajectory = solve_ivp(
            dynamics_func, tspan, state0,
            t_eval=teval,
            rtol=rtol, atol=atol, args=(p_guess,)
        )
        
        if trajectory.status != 0:
            print(f"\nWarning: Integration failed - {trajectory.message}")
            break
        
        err = ytilde - trajectory.y[0, :]
        
        for k in range(n_meas):
            phi = trajectory.y[2:6, k].reshape(2, 2)
            psi = trajectory.y[6:, k].reshape(2, 6)
            
            H = np.concatenate([phi[0, :], psi[0, :]])
            w = 1.0 / W_inv[k]
            
            Lambda += w * np.outer(H, H)
            N += w * H * err[k]
        
        new_cost = np.sum((err**2) / W_inv)
        
        if verbose:
            print(f'{iteration:5d} {z[0]:8.4f} {z[1]:8.4f} {z[2]:8.4f} '
                  f'{z[3]:8.4f} {z[4]:8.4f} {z[5]:8.4f} {z[6]:8.4f} '
                  f'{z[7]:8.4f} {new_cost:12.6e}')
        
        if iteration > 0 and abs(new_cost - old_cost) / old_cost <= tol:
            converged = True
            break
        
        try:
            delta_z = np.linalg.solve(Lambda, N)
        except np.linalg.LinAlgError:
            print("\nWarning: Singular information matrix")
            break
        
        z += delta_z
        z[7] = z[7] % (2 * pi)
        old_cost = new_cost
    
    return z, trajectory, Lambda, converged


def propagate_covariance(trajectory, P0, epoch_indices):
    """Propagate covariance using linear error theory (Task 3)."""
    Pz_propagated = []
    Px_propagated = []
    
    for idx in epoch_indices:
        # Extract Phi and Psi at this time
        phi = trajectory.y[2:6, idx].reshape(2, 2)
        psi = trajectory.y[6:, idx].reshape(2, 6)
        
        # Build sensitivity matrix dz(t)/dz(t0)
        S = np.block([[phi, psi],
                      [np.zeros((6, 2)), np.eye(6)]])
        
        # Propagate covariance: P(t) = S * P(t0) * S^T
        Pz_t = S @ P0 @ S.T
        Pz_propagated.append(Pz_t)
        
        # Extract position-velocity covariance (upper-left 2x2)
        Px_t = Pz_t[:2, :2]
        Px_propagated.append(Px_t)
    
    return Pz_propagated, Px_propagated


def plot_error_ellipses(Px_list, epochs, z_est, trajectory, ax=None):
    """Plot 1σ, 2σ, and 3σ error ellipses."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['red', 'blue', 'green', 'purple']
    
    for i, (Px, t, color) in enumerate(zip(Px_list, epochs, colors)):
        # Get estimated state at this epoch
        if t == 0:
            x_est = z_est[:2]
        else:
            idx = EPOCH_INDICES[i]
            x_est = trajectory.y[:2, idx]
        
        # Eigenvalues and eigenvectors for ellipse
        eigvals, eigvecs = np.linalg.eig(Px)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        
        # Plot 1σ, 2σ, 3σ ellipses
        for n_sigma, alpha, lw in [(1, 0.3, 1.5), (2, 0.2, 1.2), (3, 0.1, 1.0)]:
            width = 2 * n_sigma * np.sqrt(eigvals[0])
            height = 2 * n_sigma * np.sqrt(eigvals[1])
            
            ellipse = Ellipse(xy=x_est, width=width, height=height,
                            angle=angle, facecolor=color, alpha=alpha,
                            edgecolor=color, linewidth=lw,
                            label=f't={t}s, {n_sigma}σ' if n_sigma == 1 else None)
            ax.add_patch(ellipse)
        
        # Mark center point
        ax.plot(x_est[0], x_est[1], 'o', color=color, markersize=6, 
               label=f't={t}s center')
    
    ax.set_xlabel(r'$x(t)$')
    ax.set_ylabel(r'$\dot{x}(t)$')
    ax.set_title('Error Ellipses at Different Epochs (1σ, 2σ, 3σ)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='best', fontsize=8)
    
    return ax


def monte_carlo_simulation(n_runs=1000):
    """Task 4: Monte Carlo simulation."""
    print('\n' + delim_equals)
    print(f'Task 4: Monte Carlo Simulation ({n_runs} runs)')
    print(delim_equals)
    
    # Storage for results
    z_estimates = []
    converged_runs = 0
    
    # True measurements
    state0 = np.concatenate([X0, PHI0.ravel(), PSI0.ravel()])
    meas_states = solve_ivp(
        dynamics, TSPAN, state0, t_eval=TEVAL, args=(P_TRUE,),
        rtol=config['rtol'], atol=config['atol']
    )
    
    print(f'\nRunning {n_runs} Monte Carlo simulations...')
    
    for run in range(n_runs):
        if (run + 1) % 100 == 0:
            print(f'Completed {run + 1}/{n_runs} runs')
        
        # Generate new measurement noise
        ytilde = generate_measurements(meas_states, config['sigma'], seed=config['seed'] + run)
        
        # Initial guess
        z_true = np.concatenate([X0, P_TRUE])
        z_init = config['scale_factor'] * z_true
        
        # Run GLSDC
        z_est, traj_est, Lambda, converged = glsdc(
            dynamics, z_init, ytilde, TEVAL, config['sigma'], TSPAN,
            tol=config['tol'], maxiter=config['maxiter'],
            rtol=config['glsdc_rtol'], atol=config['glsdc_atol'],
            verbose=False
        )
        
        if converged:
            z_estimates.append(z_est)
            converged_runs += 1
    
    print(f'\nConverged runs: {converged_runs}/{n_runs}')
    
    if converged_runs == 0:
        print('No converged runs - cannot compute statistics')
        return None
    
    z_estimates = np.array(z_estimates)
    z_true = np.concatenate([X0, P_TRUE])
    
    # Compute estimation errors
    errors = z_estimates - z_true
    
    # Statistical mean and covariance
    error_mean = np.mean(errors, axis=0)
    error_cov = np.cov(errors.T)
    
    print('\nMonte Carlo Statistics:')
    print(delim_dash)
    print('Mean Estimation Error:')
    print(f'  E[Δx(0)]    = {error_mean[0]:.6e}')
    print(f'  E[Δẋ(0)]    = {error_mean[1]:.6e}')
    for i in range(6):
        print(f'  E[Δp{i+1}]     = {error_mean[i+2]:.6e}')
    
    print('\nStandard Deviations:')
    print(f'  σ_x(0)  = {np.sqrt(error_cov[0, 0]):.6e}')
    print(f'  σ_ẋ(0)  = {np.sqrt(error_cov[1, 1]):.6e}')
    for i in range(6):
        print(f'  σ_p{i+1}    = {np.sqrt(error_cov[i+2, i+2]):.6e}')
    
    return z_estimates, error_mean, error_cov


def plot_results(sim_motion, meas_states, ytilde, tspan, save: bool = False):
    """Generate all plots for Task 1."""
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax1.plot(sim_motion.t, sim_motion.y[0, :])
    ax1.set_title('System Solution Using RKF45')
    ax1.set_ylabel(r'$x(t)$')
    ax1.set_xlim(tspan)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(sim_motion.t, sim_motion.y[1, :])
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel(r'$\dot{x}(t)$')
    ax2.grid(True, alpha=0.3)
    
    fig1.tight_layout()
    
    # Phase portrait
    fig2, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(sim_motion.y[0, :], sim_motion.y[1, :])
    ax3.set_title('Phase Portrait of System')
    ax3.set_xlabel(r'$x(t)$')
    ax3.set_ylabel(r'$\dot{x}(t)$')
    ax3.grid(True, alpha=0.3)
    
    # Measurements overlay
    fig3, ax4 = plt.subplots(figsize=(12, 4))
    ax4.plot(sim_motion.t, sim_motion.y[0, :], linewidth=1.5, label='True Motion')
    ax4.scatter(meas_states.t, ytilde, s=0.5, c='orange', alpha=0.5, label='Measurements')
    ax4.set_title("Solution Position vs Time with Measurements")
    ax4.set_xlim(tspan)
    ax4.set_xlabel(r"$t$ [s]")
    ax4.set_ylabel(r'$x(t)$')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    fig3.tight_layout()
    
    return fig1, fig2, fig3


def main():
    """Main execution function."""
    print(delim_equals)
    print('EAS 6414 Project 3: Initial State and Parameter Estimation')
    print(delim_equals)
    
    print('\nGiven Values')
    print(delim_dash)
    print(f'x0                              = {X0}')
    print(f'p                               = {P_TRUE}')
    print(f'Measurement Std Dev             = {config["sigma"]}')
    print(f'Maximum Iteration for GLSDC     = {config["maxiter"]}')
    print(f'Scale factor for initial guess  = {config["scale_factor"]}\n')
    
    # Task 1: Simulate motion
    print(delim_equals)
    print('Task 1: Simulating Motion, Measurements, and Validating Matrices')
    print(delim_equals)
    print("\nIntegrating System Dynamics...")
    
    state0 = np.concatenate([X0, PHI0.ravel(), PSI0.ravel()])
    
    sim_motion = solve_ivp(
        dynamics, TSPAN, state0, args=(P_TRUE,),
        rtol=config['rtol'], atol=config['atol']
    )
    
    meas_states = solve_ivp(
        dynamics, TSPAN, state0, t_eval=TEVAL, args=(P_TRUE,),
        rtol=config['rtol'], atol=config['atol']
    )
    
    ytilde = generate_measurements(meas_states, config['sigma'], config['seed'])
    
    print('Generating plots...')
    plot_results(sim_motion, meas_states, ytilde, TSPAN, save=config['doPlot'])
    
    # Task 2: GLSDC
    print('\n' + delim_equals)
    print('Task 2: GLSDC Algorithm')
    print(delim_equals)
    
    z_true = np.concatenate([X0, P_TRUE])
    z_init = config['scale_factor'] * z_true
    
    z_est, traj_est, Lambda, converged = glsdc(
        dynamics, z_init, ytilde, TEVAL, config['sigma'], TSPAN,
        tol=config['tol'], maxiter=config['maxiter'],
        rtol=config['glsdc_rtol'], atol=config['glsdc_atol']
    )
    
    print(delim_dash)
    if not converged:
        print(f'\nGLSDC failed to converge within {config["maxiter"]} iterations')
        return
    else:
        print('\nGLSDC converged successfully!')
    
    print('\nFinal Estimate:')
    print(f'x0 = [{z_est[0]:.6f}, {z_est[1]:.6f}]')
    print(f'p  = {z_est[2:]}')
    
    print('\nTrue Values:')
    print(f'x0 = {X0}')
    print(f'p  = {P_TRUE}')
    
    print('\nEstimation Error:')
    print(f'Δx0 = {z_est[:2] - X0}')
    print(f'Δp  = {z_est[2:] - P_TRUE}')
    
    # Covariance at t0
    P0 = np.linalg.inv(Lambda)
    print('\nCovariance Matrix P(0) - Standard Deviations:')
    print(f'σ_x(0)   = {np.sqrt(P0[0, 0]):.6e}')
    print(f'σ_ẋ(0)   = {np.sqrt(P0[1, 1]):.6e}')
    for i, label in enumerate(['p1', 'p2', 'p3', 'p4', 'p5', 'p6']):
        print(f'σ_{label}     = {np.sqrt(P0[i+2, i+2]):.6e}')
    
    # Task 3: Covariance Propagation
    print('\n' + delim_equals)
    print('Task 3: Covariance Propagation and Error Ellipses')
    print(delim_equals)
    
    Pz_list, Px_list = propagate_covariance(traj_est, P0, EPOCH_INDICES)
    
    print('\nPropagated Standard Deviations at Epochs:')
    for i, t in enumerate(EPOCHS):
        print(f'\nEpoch t = {t}s:')
        Px = Px_list[i]
        print(f'  σ_x(t)  = {np.sqrt(Px[0, 0]):.6e}')
        print(f'  σ_ẋ(t)  = {np.sqrt(Px[1, 1]):.6e}')
        
        # Compare with actual errors
        if t == 0:
            x_true = X0
            x_est_at_t = z_est[:2]
        else:
            idx = EPOCH_INDICES[i]
            x_true = meas_states.y[:2, idx]
            x_est_at_t = traj_est.y[:2, idx]
        
        actual_error = x_est_at_t - x_true
        print(f'  Actual Δx(t)  = {actual_error[0]:.6e}')
        print(f'  Actual Δẋ(t)  = {actual_error[1]:.6e}')
    
    # Plot error ellipses
    fig_ellipse, ax_ellipse = plt.subplots(figsize=(10, 8))
    plot_error_ellipses(Px_list, EPOCHS, z_est, traj_est, ax=ax_ellipse)
    
    # Task 4: Monte Carlo
    if config['n_monte_carlo'] > 0:
        mc_results = monte_carlo_simulation(config['n_monte_carlo'])
        
        if mc_results is not None:
            z_estimates, error_mean, error_cov_mc = mc_results
            
            print('\nComparison: Linear Theory vs Monte Carlo')
            print(delim_dash)
            print('Standard Deviations at t=0:')
            print(f'{"Parameter":<10} {"Linear Theory":<15} {"Monte Carlo":<15}')
            print(delim_dash)
            print(f'{"x(0)":<10} {np.sqrt(P0[0,0]):.6e}      {np.sqrt(error_cov_mc[0,0]):.6e}')
            print(f'{"ẋ(0)":<10} {np.sqrt(P0[1,1]):.6e}      {np.sqrt(error_cov_mc[1,1]):.6e}')
            for i in range(6):
                print(f'{"p"+str(i+1):<10} {np.sqrt(P0[i+2,i+2]):.6e}      {np.sqrt(error_cov_mc[i+2,i+2]):.6e}')
    
    if config['doPlot']:
        plt.show()
    
    print('\n' + delim_equals)
    print('Project 3 Complete!')
    print(delim_equals)


if __name__ == '__main__':
    main()