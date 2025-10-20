# Standard Library Imports
from collections import namedtuple
import time
import argparse

# Third Party Imports
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from numpy import pi, sin, cos
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm

'''
EAS 6414 Project 3
Dylan D'Silva
AUTONOMOUS FORMULATION WITH θ = p₅t + p₆ AS STATE VARIABLE
'''

# Command line arguments
parser = argparse.ArgumentParser(description='Runs Monte Carlo Methods for GLSDC Algorithm')
parser.add_argument('--sigma', type=float, default=0.1, help='Standard Deviation of x(t) measurements')
parser.add_argument('--maxiter', type=int, default=30, help='Maximum Number of iterations for GLSDC Algorithm')
parser.add_argument('--ntrials', type=int, default=16, help='Number of Monte Carlo Trials')
parser.add_argument('--scalefactor', type=float, default=0.99, help='Scale factor for GLSDC Guess')
parser.add_argument('--tol', type=float, default=1E-5, help='Error Tolerance for GLSDC')

args = parser.parse_args()

# Settings
sigma           = args.sigma
maxiter         = args.maxiter
scale_factor    = args.scalefactor
tol             = args.tol

# Delimeter strings
delim_equals    = '='*90
delim_dash      = '-'*90

'''
********************************************************************
Given Values
********************************************************************
'''
x0              = np.array([2.0, 0.0])  # Initial Conditions [x(0), xdot(0)]
p               = np.array([0.05, 4, 0.2, -0.5, 10, pi/2])  # True Parameters
tspan           = [0, 300]  # Initial and Final Times
teval           = [i/10 for i in range(1, 3001)]  # Measurement times

'''
********************************************************************
System Dynamics - AUTONOMOUS FORMULATION
********************************************************************
'''
@njit(cache=True)
def dynamics(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Autonomous System Dynamics with θ as state variable
    
    State vector: [x, xdot, theta, vec(phi_2x2), vec(psi_2x6), psi_theta5]
    Total size: 20 elements
    
    System equations:
    ẋ = ẋ
    ẍ = -(p₁ẋ + p₂x + p₃x³ + p₄sin(θ))
    θ̇ = p₅
    
    Args:
        t: Integration time (not used in autonomous system)
        y: State vector [x, xdot, theta, vec(phi), vec(psi), psi_theta5]
        p: Parameter vector [p1, p2, p3, p4, p5, p6]
    
    Returns:
        Time derivative of state vector
    """
    # Extract states
    x = y[0]
    xdot = y[1]
    theta = y[2]
    
    # Extract parameters
    p1, p2, p3, p4, p5, p6 = p
    
    # Precompute trig functions
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    
    # State derivatives
    dx = xdot
    dxdot = -(p1 * xdot + p2 * x + p3 * x**3 + p4 * sin_theta)
    dtheta = p5
    
    # Jacobian A = ∂f/∂[x, ẋ] (2×2)
    A11 = 0.0
    A12 = 1.0
    A21 = -(p2 + 3.0 * p3 * x**2)
    A22 = -p1
    
    # Extract Φ (2×2)
    phi = y[3:7].reshape((2, 2))
    
    # Compute Φ̇ = A @ Φ
    phidot = np.zeros((2, 2))
    phidot[0, 0] = A11 * phi[0, 0] + A12 * phi[1, 0]
    phidot[0, 1] = A11 * phi[0, 1] + A12 * phi[1, 1]
    phidot[1, 0] = A21 * phi[0, 0] + A22 * phi[1, 0]
    phidot[1, 1] = A21 * phi[0, 1] + A22 * phi[1, 1]
    
    # Extract ∂θ/∂p₅ state
    psi_theta5 = y[19]
    dpsi_theta5 = 1.0  # d(∂θ/∂p₅)/dt = 1, so ∂θ/∂p₅ = t
    
    # ∂f/∂p (2×6)
    dfdp = np.zeros((2, 6))
    dfdp[1, 0] = -xdot
    dfdp[1, 1] = -x
    dfdp[1, 2] = -x**3
    dfdp[1, 3] = -sin_theta
    dfdp[1, 4] = -p4 * cos_theta * psi_theta5  # Chain rule: -p₄cos(θ)·∂θ/∂p₅
    dfdp[1, 5] = -p4 * cos_theta
    
    # Extract Ψ (2×6)
    psi = y[7:19].reshape((2, 6))
    
    # Compute Ψ̇ = A @ Ψ + ∂f/∂p
    psidot = np.zeros((2, 6))
    for j in range(6):
        psidot[0, j] = A11 * psi[0, j] + A12 * psi[1, j] + dfdp[0, j]
        psidot[1, j] = A21 * psi[0, j] + A22 * psi[1, j] + dfdp[1, j]
    
    # Concatenate results
    statedot = np.empty(20)
    statedot[0] = dx
    statedot[1] = dxdot
    statedot[2] = dtheta
    statedot[3:7] = phidot.ravel()
    statedot[7:19] = psidot.ravel()
    statedot[19] = dpsi_theta5
    
    return statedot


'''
********************************************************************
Validation Functions
********************************************************************
'''
def validate_variational_matrices(simulated_motion, measured_states, x0, p, tspan, teval):
    """Validates Φ and Ψ using multiple methods"""
    
    print('\n' + delim_equals)
    print('VALIDATING VARIATIONAL MATRICES Φ(t,t₀) AND Ψ(t,t₀)')
    print(delim_equals)
    
    # Method 2: Finite differences for Φ
    print('\n' + delim_dash)
    print('Method 2: Finite Differences for Φ')
    print(delim_dash)
    
    epsilon = 1e-8
    t_test = 100
    idx_test = 999
    phi_analytical = measured_states.y[3:7, idx_test].reshape(2, 2)
    
    phi0 = np.eye(2)
    psi0 = np.zeros((2, 6))
    state0_nominal = np.concatenate([x0, [p[5]], phi0.flatten(), psi0.flatten(), [0.0]])
    
    print(f'{"Variable":>10} {"Max Error":>15} {"Status":>10}')
    print(delim_dash)
    
    for i in range(2):
        x0_pert = x0.copy()
        x0_pert[i] += epsilon
        state0_pert = np.concatenate([x0_pert, [p[5]], phi0.flatten(), psi0.flatten(), [0.0]])
        
        sol_pert = solve_ivp(dynamics, tspan, state0_pert, t_eval=[t_test], 
                           args=(p,), rtol=1E-10, atol=1E-10)
        
        dx_numerical = (sol_pert.y[:2, 0] - measured_states.y[:2, idx_test]) / epsilon
        dx_analytical = phi_analytical[:, i]
        error = np.max(np.abs(dx_numerical - dx_analytical))
        status = 'PASS' if error < 1e-5 else '✗ FAIL'
        
        var_name = 'x(0)' if i == 0 else 'ẋ(0)'
        print(f'{var_name:>10} {error:15.6e} {status:>10}')
    
    # Method 3: Finite differences for Ψ
    print('\n' + delim_dash)
    print('Method 3: Finite Differences for Ψ')
    print(delim_dash)
    
    psi_analytical = measured_states.y[7:19, idx_test].reshape(2, 6)
    
    print(f'{"Parameter":>10} {"Max Error":>15} {"Status":>10}')
    print(delim_dash)
    
    for j in range(6):
        p_pert = p.copy()
        p_pert[j] += epsilon
        
        # If perturbing p6, must update theta0
        theta0_pert = p_pert[5]
        state0_pert = np.concatenate([x0, [theta0_pert], phi0.flatten(), psi0.flatten(), [0.0]])
        
        sol_pert = solve_ivp(dynamics, tspan, state0_pert, t_eval=[t_test],
                           args=(p_pert,), rtol=1E-10, atol=1E-10)
        
        dx_numerical = (sol_pert.y[:2, 0] - measured_states.y[:2, idx_test]) / epsilon
        dx_analytical = psi_analytical[:, j]
        error = np.max(np.abs(dx_numerical - dx_analytical))
        status = 'PASS' if error < 1e-5 else 'FAIL'
        
        print(f'p{j+1:1d}{" "*8} {error:15.6e} {status:>10}')
    
    print('\n' + delim_equals)


'''
********************************************************************
GLSDC Algorithm
********************************************************************
'''
GLSDC_SOL = namedtuple('GLSDC_SOL', ['z', 'traj', 'Lambda'])

def glsdc(dynamics, 
          z: NDArray,
          ytilde: NDArray, 
          teval: list = teval,
          tspan: list = tspan,
          sigma: float = sigma,
          tol: float = 1e-3,
          maxiter: int = 30,
          verbose: bool = False
          ):
    """
    GLSDC Algorithm for autonomous system
    
    Args:
        dynamics: Dynamics function
        z: Initial guess [x0, xdot0, p1, ..., p6]
        ytilde: Measurements
        teval: Evaluation times
        tspan: Integration span
        sigma: Measurement standard deviation
        tol: Convergence tolerance
        maxiter: Maximum iterations
        verbose: Print progress
    
    Returns:
        namedtuple: (z_estimate, trajectory, Lambda)
    """
    
    # Initialize
    phi0 = np.eye(2)
    psi0 = np.zeros((2, 6))
    psi_theta5_0 = 0.0
    
    old_cost = np.inf
    new_cost = 1
    
    # Weight matrix
    teval_arr = np.array(teval)
    scale_with_t = 0
    W_diag = 1.0 / ((1 + scale_with_t * teval_arr) * sigma**2)
    
    # Measurement matrix H = [1, 0] (measuring position only)
    dgdx = np.array([[1.0, 0.0]])
    
    if verbose:
        print(f'\n{"Iter":>5} {"x(0)":>8} {"xdot(0)":>8} {"p1":>8} {"p2":>8} '
              f'{"p3":>8} {"p4":>8} {"p5":>8} {"p6":>8} {"Cost":>12}')
        print(delim_dash)
    
    for i in range(maxiter):
        # Extract estimates
        x0_guess = z[:2]
        p_guess = z[2:]
        
        old_cost = new_cost
        
        # Initialize normal equation matrices
        Lambda = np.zeros((8, 8))
        N = np.zeros(8)
        
        # Set up initial state: [x, xdot, theta0, Phi, Psi, psi_theta5]
        theta0 = p_guess[5]  # θ(0) = p₆
        initial_state_guess = np.concatenate([x0_guess, [theta0], phi0.flatten(), 
                                             psi0.flatten(), [psi_theta5_0]])
        
        # Integrate
        glsdc_guess_traj = solve_ivp(
            dynamics,
            tspan,
            initial_state_guess,
            t_eval=teval,
            rtol=1e-6,
            atol=1e-6,
            args=(p_guess,)
        )
        
        if glsdc_guess_traj.status == -1:
            print(f"Integration failed: {glsdc_guess_traj.message}")
            return GLSDC_SOL(z, glsdc_guess_traj, Lambda)
        
        # Compute errors
        err = ytilde - glsdc_guess_traj.y[0, :]
        new_cost = np.sum(W_diag * err**2)
        
        # Extract variational matrices
        phi_data = glsdc_guess_traj.y[3:7, :].T    # Shape: (n_times, 4)
        psi_data = glsdc_guess_traj.y[7:19, :].T   # Shape: (n_times, 12)
        
        for k in range(len(teval)):
            phi_k = phi_data[k].reshape(2, 2)
            psi_k = psi_data[k].reshape(2, 6)
            
            # H_i = [dgdx @ Phi, dgdx @ Psi]
            H_i = np.concatenate([dgdx @ phi_k, dgdx @ psi_k], axis=1)
            
            # Accumulate normal equations
            Lambda += (H_i.T * W_diag[k]) @ H_i
            N += (H_i.T * W_diag[k]).squeeze() * err[k]
        
        if verbose:
            print(f'{i:5d} {z[0]:8.4f} {z[1]:8.4f} {z[2]:8.4f} {z[3]:8.4f} ' +
                  f'{z[4]:8.4f} {z[5]:8.4f} {z[6]:8.4f} {z[7]:8.4f} {new_cost:12.6e}')
        
        # Solve for update
        delta_z = np.linalg.solve(Lambda, N)
        
        # Check convergence
        if abs(new_cost - old_cost) / old_cost <= tol or np.linalg.norm(delta_z) <= tol:
            break
        
        # Update state
        z += delta_z
        z[-1] = z[-1] % (2 * pi)  # Keep p6 in [0, 2π]
    
    if verbose:
        print(delim_dash)
    
    if abs(new_cost - old_cost) / old_cost > tol and np.linalg.norm(delta_z) > tol:
        print(f'\nGLSDC failed to converge (tol={tol}) in {maxiter} iterations')
    
    return GLSDC_SOL(z, glsdc_guess_traj, Lambda)


'''
********************************************************************
Covariance Propagation
********************************************************************
'''
def create_cov_ellipse(P: np.ndarray, mu: np.ndarray, scale: float = 1.0, npoints=100) -> np.ndarray:
    """Create covariance ellipse points"""
    t = np.linspace(0, 2*pi, npoints)    
    ellipse = np.array([cos(t), sin(t)])
    
    D, V = np.linalg.eig(P)
    cov_ellipse = V @ (scale * np.sqrt(D) * ellipse.T).T + mu.reshape(2, 1)
    
    return cov_ellipse


def propagate_cov(z: np.ndarray, glsdc_traj, Lambda: np.ndarray):
    """Propagate covariance to t = 0, 100, 200, 300"""
    
    sample_times = [100, 200, 300]
    
    xt = np.zeros((2, 4))
    Pt = np.zeros((4, 8, 8))
    
    # At t=0
    xt[:, 0] = z[:2]
    P0 = np.linalg.inv(Lambda)
    Pt[0, :, :] = P0
    
    for k, t in enumerate(sample_times):
        # Extract state at time t
        xt[:, k + 1] = glsdc_traj.y[:2, t * 10 - 1]
        
        # Extract Phi and Psi
        PhiPsiVec = glsdc_traj.y[3:19, t * 10 - 1]
        Phi = PhiPsiVec[:4].reshape(2, 2)
        Psi = PhiPsiVec[4:].reshape(2, 6)
        
        # Build dz/dz0 matrix
        dzdz0 = np.block([[Phi, Psi], 
                         [np.zeros((6, 2)), np.eye(6)]])
        
        # Propagate covariance
        Pt[k+1, :, :] = dzdz0 @ P0 @ dzdz0.T
    
    return xt, Pt


'''
********************************************************************
Monte Carlo Simulation
********************************************************************
'''
def monte_carlo_sim(dynamics, z_true, measured_states, niter=1000, n_jobs=-1):
    """Monte Carlo simulation of GLSDC"""
    
    print(f'\nRunning Monte Carlo simulation with {niter} iterations...')
    progress_bar = tqdm(range(niter), desc='Monte Carlo Simulation')
    
    MonteCarloTrial = namedtuple('MonteCarloTrial', ['t0', 't100', 't200', 't300', 'p'])
    
    def single_glsdc_run(seed):
        rng = np.random.default_rng(seed=seed)
        
        # Add noise to measurements
        ytilde_trial = measured_states.y[0, :] + rng.normal(loc=0, scale=sigma, size=len(teval))
        
        # Run GLSDC
        z_trial, glsdc_trial, _ = glsdc(dynamics, scale_factor*z_true, ytilde_trial)
        
        # Extract results
        return MonteCarloTrial(
            z_trial[:2],
            np.squeeze(glsdc_trial.y[:2, 999]),
            np.squeeze(glsdc_trial.y[:2, 1999]),
            np.squeeze(glsdc_trial.y[:2, 2999]),
            z_trial[2:]
        )
    
    # Parallel execution
    trial_results = Parallel(n_jobs=n_jobs)(
        delayed(single_glsdc_run)(2025 + n) for n in progress_bar
    )
    
    # Parse results
    x_at_0 = np.array([trial.t0 for trial in trial_results])
    x_at_100 = np.array([trial.t100 for trial in trial_results])
    x_at_200 = np.array([trial.t200 for trial in trial_results])
    x_at_300 = np.array([trial.t300 for trial in trial_results])
    
    # Calculate statistics
    monte_carlo_stats = {
        't0': {'mean': np.mean(x_at_0, axis=0), 'cov': np.cov(x_at_0.T)},
        't100': {'mean': np.mean(x_at_100, axis=0), 'cov': np.cov(x_at_100.T)},
        't200': {'mean': np.mean(x_at_200, axis=0), 'cov': np.cov(x_at_200.T)},
        't300': {'mean': np.mean(x_at_300, axis=0), 'cov': np.cov(x_at_300.T)}
    }
    
    return monte_carlo_stats, x_at_0, x_at_100, x_at_200, x_at_300


'''
********************************************************************
Plotting Functions
********************************************************************
'''
def plot_cov_ellipse(sample_cov, projected_cov, sample_states, projected_state, 
                     true_states, saveplot=False, filename='Images/cov_ellipse.png'):
    """Plot covariance ellipses comparing theory and Monte Carlo"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='tight')
    
    times = [0, 100, 200, 300]
    
    for n in range(2):
        for i in range(2):
            k = 2*n + i
            
            # Get covariances and means
            sample_mu = sample_states[:, k]
            projected_mu = projected_state[:, k]
            sample_Px = sample_cov[k, :, :]
            projected_Px = projected_cov[k, :, :]
            
            # Plot true location
            axes[n, i].scatter(true_states[0, k], true_states[1, k], 
                             50, 'k', marker='x', linewidths=2, label='True State')
            
            # Plot ellipses for 1σ, 2σ, 3σ
            for scale in range(1, 4):
                sample_ellipse = create_cov_ellipse(sample_Px, sample_mu, scale, npoints=70)
                projected_ellipse = create_cov_ellipse(projected_Px, projected_mu, scale, npoints=70)
                
                label_mc = f'{scale}σ Monte Carlo' if scale == 1 else None
                label_theory = f'{scale}σ Theory' if scale == 1 else None
                
                axes[n, i].plot(sample_ellipse[0], sample_ellipse[1], 
                              color='blue', linewidth=1.5, label=label_mc)
                axes[n, i].plot(projected_ellipse[0], projected_ellipse[1], 
                              '--', color='orange', linewidth=1.5, label=label_theory)
            
            axes[n, i].set_xlabel('x')
            axes[n, i].set_ylabel('ẋ')
            axes[n, i].set_title(f't = {times[k]}s')
            axes[n, i].grid(True, alpha=0.3)
            axes[n, i].legend(loc='best')
            axes[n, i].axis('equal')
    
    if saveplot:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f'Covariance ellipse plot saved to {filename}')
    
    return fig


'''
********************************************************************
MAIN EXECUTION
********************************************************************
'''
if __name__ == '__main__':
    print(delim_equals)
    print('EAS 6414 Project 3: Initial State and Parameter Estimation')
    print('AUTONOMOUS FORMULATION: θ = p₅t + p₆ as state variable')
    print(delim_equals)
    print('\nGiven Values')
    print(delim_dash)
    print(f'x0                              = {np.array2string(x0)}')
    print(f'p                               = {np.array2string(p)}')
    print(f'Measurement Covariance          = {sigma}²')
    print(f'Maximum Iterations for GLSDC    = {maxiter}')
    print(f'Scale factor for initial guess  = {scale_factor}\n')
    
    '''
    ********************************************************************
    TASK 1: Integrate System and Validate Variational Matrices
    ********************************************************************
    '''
    print(delim_equals)
    print('TASK 1: Simulating Motion and Validating Matrices')
    print(delim_equals)
    print("\nIntegrating autonomous system...")
    
    # Initial state: [x, xdot, theta, Phi, Psi, psi_theta5]
    phi0 = np.eye(2)
    psi0 = np.zeros((2, 6))
    theta0 = p[5]  # θ(0) = p₆
    psi_theta5_0 = 0.0
    state0 = np.concatenate([x0, [theta0], phi0.flatten(), psi0.flatten(), [psi_theta5_0]])
    
    # Integrate
    simulated_motion = solve_ivp(dynamics, tspan, state0, args=(p,), rtol=1E-10, atol=1E-10)
    measured_states = solve_ivp(dynamics, tspan, state0, t_eval=teval, args=(p,), rtol=1E-10, atol=1E-10)
    
    print(f"Integration complete: {len(simulated_motion.t)} time points")
    
    # Validate variational matrices
    validate_variational_matrices(simulated_motion, measured_states, x0, p, tspan, teval)
    
    # Add measurement noise
    rng = np.random.default_rng(seed=2025)
    ytilde = measured_states.y[0, :] + rng.normal(loc=0, scale=sigma, size=len(teval))
    
    '''
    ********************************************************************
    TASK 2: GLSDC Algorithm
    ********************************************************************
    '''
    print('\n' + delim_equals)
    print('TASK 2: GLSDC Algorithm')
    print(delim_equals)
    
    # Initial guess
    z_true = np.concatenate([x0, p])
    z = scale_factor * z_true
    
    # Run GLSDC
    z, glsdc_traj, Lambda = glsdc(dynamics, z, ytilde, verbose=True)
    
    print('\nFinal Estimate:')
    print(f'x0 = {np.array2string(z[:2])}')
    print(f'p  = {np.array2string(z[2:])}')
    
    '''
    ********************************************************************
    TASK 3: Covariance Propagation
    ********************************************************************
    '''
    print('\n' + delim_equals)
    print('TASK 3: Covariance Propagation')
    print(delim_equals)
    print('\nCalculating P(t) at t = 0, 100, 200, 300')
    
    xt, Pt = propagate_cov(z, glsdc_traj, Lambda)
    
    # Get true states at sample times
    true_states = np.zeros((2, 4))
    true_states[:, 0] = x0
    true_states[:, 1] = measured_states.y[:2, 999]
    true_states[:, 2] = measured_states.y[:2, 1999]
    true_states[:, 3] = measured_states.y[:2, 2999]
    
    print()
    for k in range(4):
        t = k * 100
        print(f'x(t = {t})  = {np.array2string(xt[:, k])}')
        print(f'Px(t = {t}) = \n{np.array2string(Pt[k, :2, :2])}\n')
    
    '''
    ********************************************************************
    TASK 4: Monte Carlo Simulation
    ********************************************************************
    '''
    print(delim_equals)
    print('TASK 4: Monte Carlo Simulation')
    print(delim_equals)
    
    monte_carlo_stats, x_at_0, x_at_100, x_at_200, x_at_300 = monte_carlo_sim(
        dynamics, z_true, measured_states, niter=args.ntrials
    )
    
    # Print statistics
    print('\n' + delim_dash)
    print('Monte Carlo Statistics:')
    print(delim_dash)
    for key, value in monte_carlo_stats.items():
        t_val = key[1:] if key != 't0' else '0'
        print(f'\nStatistics at t = {t_val}s:')
        print(f'  Mean: {value["mean"]}')
        print(f'  Cov:\n{np.array2string(value["cov"])}')
    
    # Prepare sample covariance matrices
    sample_cov = np.zeros((4, 2, 2))
    sample_cov[0] = monte_carlo_stats['t0']['cov']
    sample_cov[1] = monte_carlo_stats['t100']['cov']
    sample_cov[2] = monte_carlo_stats['t200']['cov']
    sample_cov[3] = monte_carlo_stats['t300']['cov']
    
    # Sample means
    sample_states = np.zeros((2, 4))
    sample_states[:, 0] = monte_carlo_stats['t0']['mean']
    sample_states[:, 1] = monte_carlo_stats['t100']['mean']
    sample_states[:, 2] = monte_carlo_stats['t200']['mean']
    sample_states[:, 3] = monte_carlo_stats['t300']['mean']
    
    # Compare diagonal elements (standard deviations)
    print('\n' + delim_dash)
    print('Comparison: Theory vs Monte Carlo Standard Deviations')
    print(delim_dash)
    print(f'{"Time":>6} {"σ_x (Theory)":>15} {"σ_x (MC)":>15} {"σ_ẋ (Theory)":>15} {"σ_ẋ (MC)":>15}')
    print(delim_dash)
    
    for k, t_val in enumerate([0, 100, 200, 300]):
        sigma_theory = np.sqrt(np.diag(Pt[k, :2, :2]))
        sigma_mc = np.sqrt(np.diag(sample_cov[k]))
        print(f'{t_val:6d} {sigma_theory[0]:15.6e} {sigma_mc[0]:15.6e} '
              f'{sigma_theory[1]:15.6e} {sigma_mc[1]:15.6e}')
    
    # Plot covariance ellipses
    print('\n' + delim_dash)
    print('Generating Covariance Ellipse Plots...')
    print(delim_dash)
    
    fig_cov = plot_cov_ellipse(
        sample_cov, 
        Pt[:, :2, :2], 
        sample_states, 
        xt, 
        true_states,
        saveplot=True, 
        filename='Images/cov_ellipse_autonomous.png'
    )
    
    # Additional plots for Task 1
    print('\nGenerating solution plots...')
    
    # Plot 1: State variables vs time
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    ax1.plot(simulated_motion.t, simulated_motion.y[0, :])
    ax1.set_ylabel('x(t)')
    ax1.set_title('System Solution Using Autonomous Formulation')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(simulated_motion.t, simulated_motion.y[1, :])
    ax2.set_ylabel('ẋ(t)')
    ax2.set_xlabel('t')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Images/solution_states.png', dpi=300, bbox_inches='tight')
    
    # Plot 2: Phase portrait
    fig2, ax = plt.subplots(figsize=(4, 3))
    ax.plot(simulated_motion.y[0, :], simulated_motion.y[1, :])
    ax.set_xlabel('x(t)')
    ax.set_ylabel('ẋ(t)')
    ax.set_title('Phase Portrait')
    ax.grid(True, alpha=0.3)
    plt.savefig('Images/phase_portrait.png', dpi=400, bbox_inches='tight')
    
    # Plot 3: Measurements overlay
    fig3, ax = plt.subplots(figsize=(10, 3))
    ax.plot(simulated_motion.t, simulated_motion.y[0, :], 
            linewidth=1, label='True Motion')
    ax.scatter(measured_states.t, ytilde, s=1, c='orange', 
              label='Measured Position')
    ax.set_title("Solution Position vs Time with Measurements")
    ax.set_xlim(tspan)
    ax.set_xlabel('t')
    ax.set_ylabel('x(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('Images/measurements_overlay.png', dpi=300, bbox_inches='tight')
    
    # Plot 4: Verification of θ state
    fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    theta_expected = p[4] * simulated_motion.t + p[5]
    ax1.plot(simulated_motion.t, simulated_motion.y[2, :], 
            label='θ(t) computed', linewidth=2)
    ax1.plot(simulated_motion.t, theta_expected, '--', 
            label='θ(t) = p₅t + p₆', alpha=0.7, linewidth=2)
    ax1.set_ylabel('θ(t)')
    ax1.set_title('Verification of Autonomous State Variables')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(simulated_motion.t, simulated_motion.y[19, :], 
            label='∂θ/∂p₅ computed', linewidth=2)
    ax2.plot(simulated_motion.t, simulated_motion.t, '--', 
            label='t (expected)', alpha=0.7, linewidth=2)
    ax2.set_ylabel('∂θ/∂p₅')
    ax2.set_xlabel('t')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Images/theta_verification.png', dpi=300, bbox_inches='tight')
    
    print('\nAll plots generated and saved to Images/ directory')
    
    # Final summary
    print('\n' + delim_equals)
    print('SUMMARY')
    print(delim_equals)
    print('\nEstimation Errors:')
    print(f'True x0:      {x0}')
    print(f'Estimated x0: {z[:2]}')
    print(f'Error x0:     {z[:2] - x0}')
    print(f'\nTrue p:       {p}')
    print(f'Estimated p:  {z[2:]}')
    print(f'Error p:      {z[2:] - p}')
    
    print('\n' + delim_equals)
    print('PROJECT COMPLETE')
    print(delim_equals)
    
    plt.show()