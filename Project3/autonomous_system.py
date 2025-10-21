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
'''


# This makes it so that key values can be changed as arguments from the command line
parser = argparse.ArgumentParser(description='Runs Monte Carlo Methods for GLSDC Algorithm')

parser.add_argument('--sigma', type = float, default = 0.1, help='Standard Deviation of x(t) measurements')
parser.add_argument('--maxiter', type = int, default = 30, help = 'Maximum Number of iterations for GLSDC Algorithm')
parser.add_argument('--ntrials', type = int, default = 32, help = 'Number of Monte Carlo Trials')
parser.add_argument('--scalefactor', type = float,  default = 0.99, help = 'Scale factor for GLSDC Guess')
parser.add_argument('--tol', type = float, default = 1E-5, help = 'Error Tolerance for GLSDC')
parser.add_argument('--scalewitht', type = float, default = 0, help = 'How much is weight matrix affected by tk. Set to 0 for no affect.')

args = parser.parse_args()

# Settings to change how file runes
sigma           = args.sigma # Given Standard Deviation 
maxiter         = args.maxiter # Max iteration count for GLSDC 
scale_factor    = args.scalefactor # Scale factor for GLSDC Guess
tol             = args.tol # Error Tolerance for GLSDC
scale_with_t    = args.scalewitht

# Delimeter strings for prinmt statements
delim_equals    = '='*90
delim_dash      = '-'*90

'''
********************************************************************
Given Values
********************************************************************
'''
x0              = np.array([2, 0]) # Initial Conditions
p               = np.array([0.05, 4, 0.2, -0.5, 10, pi/2]) # True Parameters
tspan           = [0, 300] # Initial and Final Times of Simulation
teval           = [i/10 for i in range(1, 3001)] # Times at which system is sampled for measurement

'''
********************************************************************
System Dynamics
********************************************************************
'''

@njit(cache=True)
def dynamics_autonomous(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Autonomous System Dynamics Including Variational Matrices
    Augmented state: [x, xdot, theta, vec(Phi), vec(Psi)], theta = p5*t + p6
    Uses Numba Just-In-Time machine code compilation and caching to speed up function calls
    
    Args:
        t: Time variable (not used in autonomous formulation)
        y: State vector [x, xdot, theta, vec(phi_3x3), vec(psi_3x6)]
        p: Parameter vector [p1, p2, p3, p4, p5, p6]
    
    Returns:
        Time derivative of state vector
    """
    # Extract states
    x = y[0:3]  # [x, xdot, theta]
    
    # Extract parameters
    p1, p2, p3, p4, p5, p6 = p
    
    # theta is now a state variable
    theta = x[2]
    
    # df/dp (3x6) - partial derivatives of dynamics w.r.t. parameters
    dfdp = np.zeros((3, 6))
    dfdp[1, 0] = -x[1]
    dfdp[1, 1] = -x[0]
    dfdp[1, 2] = -x[0]**3  
    dfdp[1, 3] = -sin(theta)
    dfdp[2, 4] = 1
    
    # State derivatives
    xdot = np.empty(3)
    xdot[0] = x[1]
    xdot[1] = -(p1 * x[1] + p2 * x[0] + p3 * x[0]**3 + p4 * sin(theta))
    xdot[2] = p5  # theta_dot = p5
    
    # Phi (3x3)
    # Phidot = df/dx @ Phi
    # A = [
    #     [0,                1,              0            ]
    #     [-p2 - 3*p3*x^2,  -p1,            -p4*cos(theta)]
    #     [0,                0,              0            ]
    # ]
    phi = y[3:12].reshape((3, 3))
    phidot = np.zeros((3, 3))
    
    # First row
    phidot[0, :] = phi[1, :]
    
    # Second row
    phidot[1, :] = -(p2 + 3.0 * p3 * x[0]**2) * phi[0, :] - p1 * phi[1, :] + - p4 * cos(theta) * phi[2, :]
    
    # Psi (3x6)
    # Psidot = A @ Psi + df/dp
    psi = y[12:30].reshape((3, 6))
    psidot = np.empty((3, 6))
    
    for j in range(6):
        # First row
        psidot[0, j] = psi[1, j] + dfdp[0, j]
        
        # Second row
        psidot[1, j] = -(p2 + 3.0 * p3 * x[0]**2) * psi[0, j] - p1 * psi[1, j] + - p4 * cos(theta) * psi[2, j] + dfdp[1, j]
        
        # Third row
        psidot[2, j] = dfdp[2, j]
    
    # Concatenate results
    statedot = np.empty(30)
    statedot[0:3] = xdot
    statedot[3:12] = phidot.ravel()
    statedot[12:30] = psidot.ravel()
    
    return statedot

'''
********************************************************************
Implementing GLSDC from Tapley, Shultz, and Born 2004
********************************************************************
'''

GLSDC_SOL = namedtuple('GLSDC_SOL', ['z', 'traj', 'Lambda'])
def glsdc_autonomous(dynamics, 
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
    Implements the GLSDC Algorithm for AUTONOMOUS system formulation
    State vector: [x, xdot, theta, vec(Phi_3x3), vec(Psi_3x6)]

    Args:
        dynamics (function): Function containing autonomous system dynamics
        z (NDArray): Parameters to be estimated [x(0), xdot(0), p1, p2, p3, p4, p5, p6]
        ytilde (NDArray): Measurements
        teval (list, optional): Times at which solution to dynamics must be returned. Defaults to teval.
        tspan (list, optional): Interval over which system is integrated. Defaults to tspan.
        sigma (float, optional): Standard Deviation of ytilde. Defaults to sigma.
        tol (float, optional): Tolerance for solution convergence from GLSDC. Defaults to 1e-3.
        maxiter (int, optional): Maximum number of iterations for GLSDC Algorithm. Defaults to 30.
        verbose (bool, optional): Determines whether to print statements concerning algorithm progress. Defaults to False.

    Returns:
        namedtuple: namedtuple containing estimate for z, the trajectory using z as the initial condition, and Information Matrix
    """
    
    # Initializing / declaring relevant values
    phi0    = np.eye(3)
    psi0    = np.zeros((3, 6))
    psi0[2, 5] = 1
    
    old_cost = np.inf   # old_cost is the cost of the most recently run iteration
    new_cost = 1        # new_cost is the cost of the current iteration
    
    # Precompute weights for all time steps (vectorized)
    teval_arr   = np.array(teval) # Numpy array of all times for ytilde
    
    # Weight Matrix
    W_diag      = 1.0 / ((1 + scale_with_t * teval_arr) * sigma**2)  # Diagonal weight matrix elements
    
    if verbose:
        print(f'\n{"Iter":>5} {"x(0)":>8} {"xdot(0)":>8} {"p1":>8} {"p2":>8} '
              f'{"p3":>8} {"p4":>8} {"p5":>8} {"p6":>8} {"Cost":>12}')
        print('-' * 80)
    
    for i in range(maxiter):
        
        # Initialize for iteration
        # Extracting x0 and p estimates from z variable
        x0_guess    = z[0]
        xdot0_guess = z[1]
        p_guess     = z[2:]
        theta0_guess = p_guess[5]
        x_aug_0 = np.array([x0_guess, xdot0_guess, theta0_guess])
        
        # Record cost of most recent iteration
        old_cost    = new_cost
        
        # Initializing matrices to solve normal equation
        Lambda  = np.zeros((8, 8))
        N       = np.zeros(8)
        
        # Set up initial state according to current estimate of z
        initial_state_guess = np.concatenate([x_aug_0, phi0.flatten(), psi0.flatten()])
        
        # Integrate trajectory
        glsdc_guess_traj    = solve_ivp(
            dynamics,
            tspan,
            initial_state_guess,
            t_eval=teval,
            rtol=1e-6,
            atol=1e-6,
            args=(p_guess,)
        )
        
        # Checking to see if the solve_ivp function actually integrated the entire interval
        if glsdc_guess_traj.status == -1:
            print(f"Integration failed: {glsdc_guess_traj.message}")
            return z, glsdc_guess_traj, Lambda
        
        # Vectorized error computation (measure x only, first component)
        err = ytilde - glsdc_guess_traj.y[0, :]  # Shape: (n_measurements,)
        
        # The performance index is given as J = err^T @ W @ err
        new_cost = np.sum(W_diag * err**2)
        
        # Extracting Phi and Psi data from solution
        phi_data = glsdc_guess_traj.y[3:12, :].T # dim = len(teval) x 9
        psi_data = glsdc_guess_traj.y[12:30, :].T # dim = len(teval) x 18
        
        for k in range(len(teval)):
            
            # Reshape variational matrices
            phi_k = phi_data[k].reshape(3, 3) # Is now (3x3)
            psi_k = psi_data[k].reshape(3, 6) # Is now (3x6)
            
            # H_i
            H_i = np.zeros((1, 8))
            H_i[0, :2] = phi_k[0, :2]
            H_i[0, 2:8] = psi_k[0, :6]
            H_i[0, 7] += phi_k[0, 2]
            
            # Accumulate normal equations
            Lambda += (H_i.T * W_diag[k]) @ H_i  # (8, 8)
            N += (H_i.T * W_diag[k]).squeeze() * err[k]  # (8,)
        
        if verbose:
            print(f'{i:5d} {z[0]:8.4f} {z[1]:8.4f} {z[2]:8.4f} {z[3]:8.4f} ' +
                  f'{z[4]:8.4f} {z[5]:8.4f} {z[6]:8.4f} {z[7]:8.4f} {new_cost:12.6e}')
        
        # Solve for update step (using solve instead of inv for numerical stability)
        delta_z = np.linalg.solve(Lambda, N)
        
        # Check convergence
        if abs(new_cost - old_cost) / old_cost <= tol or np.linalg.norm(delta_z) <= tol:
            break
        
        # Update state using calculated step
        z += delta_z
        
        # Keep p6 in [0, 2pi]
        z[-1] = z[-1] % (2 * pi)
    
    if verbose:
        print('-' * 80)
    
    # Again, checking convergence of solution
    if abs(new_cost - old_cost) / old_cost > tol and np.linalg.norm(delta_z) > tol:
        print(f'\nGLSDC failed to converge (tol={tol}) in {maxiter} iterations')
    
    return GLSDC_SOL(z, glsdc_guess_traj, Lambda)

def create_cov_ellipse(P: np.ndarray, mu: np.ndarray, scale: float = 1.0, npoints = 100) -> np.ndarray:
    """Create covariance ellipse points

    Args:
        P (np.ndarray): Covariance Matrix
        mu (np.ndarray): Mean array
        scale (np.ndarray, optional): Number of standard deviations from the mean. Defaults to 1.
        npoints (int, optional): Number of points to return. Defaults to 100.

    Returns:
        np.ndarray: x-y pairs of cov ellipse
    """
    t       = np.linspace(0, 2*pi, npoints)    
    ellipse = np.array([cos(t), sin(t)])

    D, V = np.linalg.eig(P)
    cov_ellispe = V @ (scale * np.sqrt(D) * ellipse.T).T + mu.reshape(2, 1)
    
    return cov_ellispe

def propagate_cov_autonomous(z: np.ndarray, glsdc_traj, Lambda: np.ndarray):
    """
    Propagates the covariance of a GLSDC estimate for autonomous system
    
    Args:
        z: Estimated parameters [x(0), xdot(0), p1, ..., p6]
        glsdc_traj: Solution from GLSDC with augmented state
        Lambda: Information matrix from GLSDC
    
    Returns:
        Tuple(xt, Pt): The state [x, xdot] and covariance at t=0, 100, 200, 300
    """
    # Times at which system is sampled
    sample_times = [100, 200, 300]
    
    # Creating variable to hold values at each time (only x and xdot, not theta)
    xt = np.zeros((2, 4))
    Pt = np.zeros((4, 8, 8))

    # Assigning t=0
    xt[:, 0] = z[:2] 
    P0 = np.linalg.inv(Lambda)
    Pt[0, :, :] = P0

    # Using solution calculated from GLSDC to construct Phi(t) and Psi(t)
    for k, t in enumerate(sample_times):
        
        # Extracting state variable (x and xdot only, not theta)
        xt[:, k + 1] = glsdc_traj.y[:2, t * 10 - 1]
        
        # Extracting Phi (3x3) and Psi (3x6) at t from glsdc estimate trajectory
        PhiPsiVec = glsdc_traj.y[3:, t * 10 - 1]
        
        # Reshaping Phi and Psi to their matrix forms
        Phi_aug = PhiPsiVec[:9].reshape(3, 3)   # 3x3 for [x, xdot, theta]
        Psi_aug = PhiPsiVec[9:].reshape(3, 6)   # 3x6
        
        # We want ∂[x, xdot](t)/∂z where z = [x(0), xdot(0), p1, ..., p6]
        # 
        # From augmented system:
        # ∂[x, xdot, theta](t)/∂[x(0), xdot(0), theta(0)] = Phi_aug
        # ∂[x, xdot, theta](t)/∂[p1, ..., p6] = Psi_aug
        #
        # Since theta(0) = p6, we have:
        # ∂[x, xdot](t)/∂[x(0), xdot(0)] = Phi_aug[0:2, 0:2]
        # ∂[x, xdot](t)/∂p_j = Psi_aug[0:2, j] + Phi_aug[0:2, 2] * ∂theta(0)/∂p_j
        
        # Build the sensitivity matrix for [x, xdot] w.r.t. z
        Phi_reduced = Phi_aug[0:2, 0:2]  # 2x2: ∂[x,xdot]/∂[x(0),xdot(0)]
        
        Psi_reduced = Psi_aug[0:2, :].copy()  # 2x6: Start with ∂[x,xdot]/∂p
        Psi_reduced[:, 5] += Phi_aug[0:2, 2]  # Add chain rule term for p6
        
        # dz/dz0 = [[dx/dx0, dx/dp], [dp/dx0, dp/dp]]
        # where z = [x(0), xdot(0), p1, ..., p6]
        dxdz0 = np.block([[Phi_reduced, Psi_reduced], 
                          [np.zeros((6, 2)), np.eye(6)]])  # 8x8
        
        Pt[k+1, :, :] = dxdz0 @ P0 @ dxdz0.T
    
    return xt, Pt


def monte_carlo_sim_autonomous(dynamics, z_true, measured_states, niter=1000, n_jobs=-1):
    """
    Performs a Monte Carlo Simulation of the GLSDC Algorithm for autonomous system.
    This function will use joblib to multithread.

    Args:
        dynamics (function): Function containing autonomous system dynamics
        z_true (np.ndarray): True parameters to be estimated [x(0), xdot(0), p1, ..., p6]
        measured_states: Solution containing the measured states from the true trajectory
        niter (int, optional): Number of Monte Carlo Trials to perform. Defaults to 1000.
        n_jobs (int, optional): Number of processors to use. Defaults to -1.

    Returns:
        monte_carlo_stats: Dictionary with statistics at t=0, 100, 200, 300
        x_at_*: Arrays of state estimates from all trials
    """
    
    print(f'Running Monte Carlo simulation with {niter} iterations...')
    progress_bar = tqdm(range(niter), desc='Monte Carlo Simulation of GLSDC Results')
    
    # MonteCarloTrial is a namedtuple to contain the results we care about from a single trial
    MonteCarloTrial = namedtuple('MonteCarloTrial', ['t0', 't100', 't200', 't300', 'p'])
    
    # This function will be used to help implement multi processing of 1000 glsdc function calls
    def single_glsdc_run(seed):
        
        # Setting seed for repeatability
        rng = np.random.default_rng(seed=seed)
        
        # Create Noise Measurements for single monte carlo trial
        ytilde_trial = measured_states.y[0, :] + rng.normal(loc=0, scale=sigma, size=len(teval))
        
        # Run GLSDC algorithm for single monte carlo trial
        z_trial, glsdc_trial, _ = glsdc_autonomous(dynamics, scale_factor*z_true, ytilde_trial)
        
        # Extract x and xdot (not theta) at t = 0, 100, 200, 300, and p
        # Note: glsdc_trial.y has augmented state [x, xdot, theta, ...]
        return MonteCarloTrial(z_trial[:2],                      # x(0), xdot(0)
                               np.squeeze(glsdc_trial.y[:2, 999]),   # x(100), xdot(100)
                               np.squeeze(glsdc_trial.y[:2, 1999]),  # x(200), xdot(200)
                               np.squeeze(glsdc_trial.y[:2, 2999]),  # x(300), xdot(300)
                               z_trial[2:])                          # p
        
    
    # Perform Monte Carlo Parallelization
    trial_results = Parallel(n_jobs=n_jobs)(delayed(single_glsdc_run)(2025 + n) for n in progress_bar)
    
    # Parsing results
    x_at_0   = np.array([trial.t0 for trial in trial_results])
    x_at_100 = np.array([trial.t100 for trial in trial_results])
    x_at_200 = np.array([trial.t200 for trial in trial_results])
    x_at_300 = np.array([trial.t300 for trial in trial_results])
    
    # Calculate Sample Statistics
    monte_carlo_stats = {
        't0'   : {'mean': np.mean(x_at_0, axis=0),   'cov': np.cov(x_at_0.T)},
        't100' : {'mean': np.mean(x_at_100, axis=0), 'cov': np.cov(x_at_100.T)},
        't200' : {'mean': np.mean(x_at_200, axis=0), 'cov': np.cov(x_at_200.T)},
        't300' : {'mean': np.mean(x_at_300, axis=0), 'cov': np.cov(x_at_300.T)}
    }
    
    return monte_carlo_stats, x_at_0, x_at_100, x_at_200, x_at_300

def make_sol_plots(saveplot = False):
    
    '''
    Function to create plots needed for project. Creates x(t) vs t, xdot(t) vs t, xdot(t) vs x(t), measurements over x(t) vs t
    '''
    print('Plotting Solution ...')
    # Plotting Solution State Variables vs Time
    solution_fig, axs = plt.subplots(2, 1, figsize = (10, 4), sharex='col')
    xvt, xdotvt = axs

    # Plotting Position vs Time
    xvt.plot(simulated_motion.t, simulated_motion.y[0, :])

    # Plotting Velocity vs Time
    xdotvt.plot(simulated_motion.t, simulated_motion.y[1, :])

    # Setting Figure Title
    xvt.set_title('System Solution Using RKF45')
    xvt.grid()

    # Changing Axis Properties
    xvt.set_xlim(tspan)
    xdotvt.set_xlabel('t')
    xdotvt.grid()

    xvt.set_ylabel(r'$x(t)$ vs $t$')
    xdotvt.set_ylabel(r'$\dot{x}(t)$ vs $t$')

    solution_fig.tight_layout()

    print('Generating Phase Portrait ...')
    # Plotting phase portrait
    phase_portrait = plt.figure()
    plase_portrait_ax = phase_portrait.gca()
    plase_portrait_ax.plot(simulated_motion.y[0, :], simulated_motion.y[1, :])

    # Changing plot properties
    plase_portrait_ax.set_title('Phase Portrait of System')
    plase_portrait_ax.set_xlabel(r'$x(t)$ vs $t$')
    plase_portrait_ax.set_ylabel(r'$\dot{x}(t)$ vs $t$')
    plase_portrait_ax.grid()

    print('Overlaying Measurements ...')
    # Plotting measurements against x(t)
    measurement_fig = plt.figure(figsize = (10, 3))
    measurement_ax = measurement_fig.gca()

    measurement_ax.plot(simulated_motion.t, simulated_motion.y[0, :], linewidth=1, label = 'True Motion')
    measurement_ax.scatter(measured_states.t, ytilde, s = 1, c= 'orange', label = 'Measured Position')

    measurement_ax.set_title("Solution Position vs Time")
    measurement_ax.set_xlim(tspan)
    measurement_ax.set_xlabel(r"$t$")
    measurement_ax.set_ylabel(r'$x(t)$ vs $t$')
    measurement_ax.grid()
    measurement_ax.legend()

    if saveplot:
        solution_fig.savefig('Images/solution_fig.png', format='png', dpi=1440)
        phase_portrait.savefig('Images/phase_portrait.png', format='png', dpi=1440)
        measurement_fig.savefig('Images/measurements_fig.png', format='png', dpi=1440)
        print('Plots Generated and Saved\n')
        
    return solution_fig, phase_portrait, measurement_fig

def plot_cov_ellipse(sample_cov, projected_cov, sample_states, projected_state, true_states, saveplot=False, filename = 'Images/cov_ellipse.png'):
    fig, axs = plt.subplots(2, 2, figsize = (12, 10), layout='tight')
    
    times = [0, 100, 200, 300]
    
    for n in range(2):
        for i in range(2):
            
            # Counting variable
            k = 2*n + i
            
            # Get mean and cov at t
            sample_mu    = sample_states[:, k]
            projected_mu = projected_state[:, k]
            sample_Px    = sample_cov[k, :, :]
            projected_Px = projected_cov[k, :, :]
            
            # Plot true state
            axs[n, i].scatter(true_states[0, k], true_states[1, k], color = 'k', marker = 'x', label = 'True State')
            
            # Plot 1 sigma, 2 sima, 3 sigma ellipses
            for scale in range(1, 4):
                sample_ellipse      = create_cov_ellipse(sample_Px, sample_mu, scale, npoints=70)
                projected_ellipse   = create_cov_ellipse(projected_Px, projected_mu, scale, npoints=70)
                
                # This is so only 1 entry for each shows up in legend
                if scale == 1: 
                    projected_label = 'Linear Theory'
                    sample_label    = 'Monte Carlo'
                else: 
                    projected_label = None
                    sample_label    = None
                
                axs[n, i].plot(projected_ellipse[0], projected_ellipse[1], color = 'blue', label = projected_label)
                axs[n, i].plot(sample_ellipse[0], sample_ellipse[1], '--', color = 'orange', label = sample_label)

            axs[n, i].set_xlabel(r'$x(t)$')
            axs[n, i].set_ylabel(r'$\dot{x}(t)$')
            axs[n, i].set_title(r'$t$' f' = {times[k]}')
            axs[n, i].legend()
            axs[n, i].axis('equal')
            axs[n, i].grid()
            
    if saveplot:
        fig.savefig(filename, dpi = 300)
            
    return fig

# ============================================================================
# MAIN EXECUTION BLOCK FOR AUTONOMOUS SYSTEM
# ============================================================================

if __name__ == '__main__':
    print(delim_equals + '\nEAS 6414 Project 3: Initial State and Parameter Estimation\n' + delim_equals)
    print('\nGiven Values\n' + delim_dash)
    print(f'x0                              = {np.array2string(x0)}')
    print(f'p                               = {np.array2string(p)}')
    print(f'Measurement Covariance          = {sigma}^2')
    print(f'Maximum Iteration for GLSDC     = {maxiter}')
    print(f'Scale factor for initial guess  = {scale_factor}\n')
    
    '''
    ********************************************************************
    Integrating Autonomous System Dynamics
    ********************************************************************
    '''

    # Initial State for Autonomous System: [x, xdot, theta]
    theta0 = p[5]  # theta(0) = p6
    x0_aug = np.array([x0[0], x0[1], theta0])
    
    phi0 = np.eye(3)        # 3x3 for autonomous system
    psi0 = np.zeros((3, 6))
    psi0[2, 4] = 1
    state0 = np.concatenate([x0_aug, phi0.flatten(), psi0.flatten()])

    print(delim_equals + '\nTask 1: Simulating Motion, Measurements, and Validating Matrices\n' + delim_equals)
    print("\nIntegrating Autonomous System Dynamics ...")

    # Integrating Autonomous System
    simulated_motion = solve_ivp(dynamics_autonomous, tspan, state0, args=(p,), 
                                 rtol=1E-10, atol=1E-10)
    measured_states = solve_ivp(dynamics_autonomous, tspan, state0, t_eval=teval, 
                                args=(p,), rtol=1E-10, atol=1E-10)
    
    # Validating State Transition Matrix
    print('\nValidating Variational Matrices ...\n' + delim_dash)
    
    # Validate variational matrices for autonomous system

    """
    ********************************************************************
    Adding Measurement Noise
    ********************************************************************
    """

    # Setting numpy seed
    rng = np.random.default_rng(seed=2025)

    # adding gaussian noise to measurements
    ytilde = measured_states.y[0, :] + rng.normal(loc=0, scale=sigma, size=len(teval))
    
    # Estimating z = [x(0), xdot(0), p^T]^T
    z_true = np.concatenate([x0, p])

    # Adjusting to create an initial guess
    z = scale_factor * z_true

    print(delim_equals + '\nTask 2: GLSDC Algorithm (Autonomous Formulation)\n' + delim_equals)
    
    '''
    ********************************************************************
    Implementing GLSDC for Autonomous System
    ********************************************************************
    '''
    
    z, glsdc_traj, Lambda = glsdc_autonomous(dynamics_autonomous, z, ytilde, verbose=True)

    print('Final Estimate')
    x0_guess = z[:2]
    p_guess = z[2:]

    print(f'x0 = {np.array2string(x0_guess)}')
    print(f'p  = {np.array2string(p_guess)}') 
    
    '''
    ********************************************************************
    Covariance Propagation for Autonomous System
    ********************************************************************
    '''
    print('\n' + delim_equals + '\nProducing Covariance Ellipses\n' + delim_equals)
    print(f'\nCalculating P(t) at t = 0, 100, 200, 300')
    
    xt, Pt = propagate_cov_autonomous(z, glsdc_traj, Lambda)
    
    '''
    ********************************************************************
    Monte Carlo Simulation for Autonomous System
    ********************************************************************
    '''
    
    print('\n' + delim_equals + '\nMonte Carlo Simulation\n' + delim_equals + '\n')
    
    monte_carlo_stats, x_at_0, x_at_100, x_at_200, x_at_300 = monte_carlo_sim_autonomous(
        dynamics_autonomous, z_true, measured_states, niter=args.ntrials)
    
    # Printing Statistics to Console
    print(delim_dash + '\nMonte Carlo Statistics')
       
    for key, value in monte_carlo_stats.items():
        print(f'Statistics for t = ' + key)
        print(f'    - Mean: {value["mean"]}')
        print(f'    - Cov:\n{np.array2string(value["cov"])}')
        print(np.linalg.eig(value['cov']))
        
    # Collecting sample statistics for plotting
    sample_states = np.zeros((2, 4))
    sample_states[:, 0] = monte_carlo_stats['t0']['mean']
    sample_states[:, 1] = monte_carlo_stats['t100']['mean']
    sample_states[:, 2] = monte_carlo_stats['t200']['mean']
    sample_states[:, 3] = monte_carlo_stats['t300']['mean']
    
    sample_cov = np.zeros((4, 2, 2))
    sample_cov[0] = monte_carlo_stats['t0']['cov']
    sample_cov[1] = monte_carlo_stats['t100']['cov']
    sample_cov[2] = monte_carlo_stats['t200']['cov']
    sample_cov[3] = monte_carlo_stats['t300']['cov']
    
    # Get the true states at sample times, used for errors and plotting
    true_states = np.zeros((2, 4))
    true_states[:, 0] = x0
    true_states[:, 1] = measured_states.y[:2, 999]
    true_states[:, 2] = measured_states.y[:2, 1999]
    true_states[:, 3] = measured_states.y[:2, 2999]
    
    # Comparing Projected vs Sample Statistics
    for k in range(4):
        print(f'\nt = {int(k*100)}\n' + delim_dash)
        
        print(f'Predicted x({int(k*100)})  = {np.array2string(xt[:, k])}')
        print(f'Sample Mean x({int(k*100)})  = {np.array2string(sample_states[:, k])}')
        
        abserr = xt[:, k] - true_states[:, k]
        print(f'Predicted Error = {abserr}, error mag: {np.linalg.norm(abserr)}')
        abserr = sample_states[:, k] - true_states[:, k]
        print(f'Sample Mean Error = {abserr}, error mag: {np.linalg.norm(abserr)}')
        
        print(f'Predicted sigma_x = {np.sqrt(Pt[k, 0, 0])}, Predicted sigma_xdot = {np.sqrt(Pt[k, 1, 1])}')
        print(f'Sample sigma_x = {np.sqrt(sample_cov[k, 0, 0])}, Sample sigma_xdot = {np.sqrt(sample_cov[k, 1, 1])}')
        
        print(f'Predicted Px({int(k*100)}) = \n{np.array2string(Pt[k, :2, :2])}')
        print(f'Sample Px({int(k*100)}) = \n{np.array2string(sample_cov[k])}')
        
        D, V = np.linalg.eig(Pt[k, :2, :2])
        print(f'Predicted D = {np.array2string(D)}')
        print(f'Predicted V = \n{np.array2string(V)}')   
        
        D, V = np.linalg.eig(sample_cov[k])
        print(f'Sample D = {np.array2string(D)}')
        print(f'Sample V = \n{np.array2string(V)}')
    
    '''
    ********************************************************************
    Plotting
    ********************************************************************
    '''
    
    # Plotting Covariance Ellipses
    cov_plot = plot_cov_ellipse(sample_cov, Pt[:, :2, :2], sample_states, xt, true_states, 
                                 saveplot=True, filename='Images/cov_ellipse_autonomous.png')
    
    task1 = make_sol_plots(True)
    
    plt.show()