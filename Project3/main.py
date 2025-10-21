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
parser.add_argument('--dosetseed', type = bool, default = False, help = 'Toggles setting a seed for repeatable results')
parser.add_argument('--baseseed', type = int, default = 2025, help = 'Base seed for RNG functions')

args = parser.parse_args()

# Settings to change how file runes
sigma           = args.sigma # Given Standard Deviation 
maxiter         = args.maxiter # Max iteration count for GLSDC 
scale_factor    = args.scalefactor # Scale factor for GLSDC Guess
tol             = args.tol # Error Tolerance for GLSDC
scale_with_t    = args.scalewitht
setseed         = args.dosetseed

if setseed:
    baseseed = args.baseseed
else:
    basesee = np.random.randint(low = 0, high = 100000)

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
def dynamics(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    System Dynamics Including Variational Matrices
    Uses Numba Just-In-Time machine code compilation and caching to speed up function calls
    
    Args:
        t: Time variable
        y: State vector [x, vec(phi), vec(psi)]
        p: Parameter vector
    
    Returns:
        Time derivative of state vector
    """
    # Extract states
    x = y[0:2]
    
    # Extract parameters
    p1, p2, p3, p4, p5, p6 = p
    
    # Computing a often repeated value
    theta = p5 * t + p6
    
    # df/dp (2x6)
    dfdp        = np.zeros((2, 6))
    dfdp[1, 0]  = -x[1]
    dfdp[1, 1]  = -x[0]
    dfdp[1, 2]  = -x[0]**3
    dfdp[1, 3]  = -sin(theta)
    dfdp[1, 4]  = -p4 * t * cos(theta)
    dfdp[1, 5]  = -p4 * cos(theta)
    
    # xdot and xddot
    xdot    = np.empty(2)
    xdot[0] = x[1]
    xdot[1] = -(p1 * x[1] + p2 * x[0] + p3 * x[0]**3 + p4 * sin(theta))
    
    # Phi (2x2)
    # Phidot = df/dx @ Phi
    # A = [
    #     [0,               1]
    #     [-p2 - 3p3x**2, -p1]
    # ]
    phi             = y[2:6].reshape((2, 2))
    phidot          = np.empty((2, 2))
    phidot[0, 0]    = phi[1, 0]
    phidot[0, 1]    = phi[1, 1]
    phidot[1, 0]    = -(p2 + 3.0 * p3 * x[0]**2) * phi[0, 0] - p1 * phi[1, 0]
    phidot[1, 1]    = -(p2 + 3.0 * p3 * x[0]**2) * phi[0, 1] - p1 * phi[1, 1]
    
    # Psi (2x6)
    # Psidot = df/dx @ Psi + df/dp
    # df/dp = [
    #     [0        0       0       0           0                   0      ]
    #     [-xdot, -x,   -x**3, -sin(theta), -p4*t*cos(theta), -p4cos(theta)]
    # ]
    psi = y[6:18].reshape((2, 6))
    psidot = np.empty((2, 6))
    for j in range(6):
        psidot[0, j] = psi[1, j]
        psidot[1, j] = -(p2 + 3.0 * p3 * x[0]**2) * psi[0, j] - p1 * psi[1, j] + dfdp[1, j]
    
    # Concatenate results
    statedot          = np.empty(18)
    statedot[0:2]     = xdot
    statedot[2:6]     = phidot.ravel()
    statedot[6:18]    = psidot.ravel()
    
    return statedot

'''
********************************************************************
Implementing GLSDC from Tapley, Shultz, and Born 2004
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
    Implements the GLSDC Algorithm as described in 
    Statistical Orbit Determination by Tapley, Schultz, and Born 2004

    Args:
        dynamics (function): Function Containintg system dynamics
        z (NDArray): Parameters to be estimated
        ytilde (NDArray): Measurements
        teval (list, optional): Times at which solution to dynamics must be returned. Defaults to teval.
        tspan (list, optional): Interval over which system is integrated. Defaults to tspan.
        sigma (float, optional): Standard Deviation of ytilde. Defaults to sigma.
        tol (float, optional): Tolerance for solution convergence from GLSDC. Defaults to 1e-3.
        maxiter (int, optional): Maximum number of iterations for GLSDC Algorithm. Defaults to 30.
        verbose (bool, optional): Determines whether to print statements concerning algorithm progress. Defaults to False.

    Returns:
        namedtuple: namedtuple containing estimate for z, the trajectory using z as the initial coniditon, and Information Matrix
    """
    
    # Initializng / declaring relevant values
    phi0    = np.eye(2)
    psi0    = np.zeros((2, 6))
    
    old_cost = np.inf   # old_cost is the cost of the most recently run iteration
    new_cost = 1        # new_cost is the cost of the current iteration
    
    # Precompute weights for all time steps (vectorized)
    teval_arr   = np.array(teval) # Numpy array of all times for ytilde
    
    # Weight Matrix
    W_diag      = 1.0 / ((1 + scale_with_t * teval_arr) * sigma**2)  # Diagonal weight matrix elements
    
    # H matrix selector (avoid repeated array creation)
    dgdx = np.array([[1.0, 0.0]])
    
    if verbose:
        print(f'\n{"Iter":>5} {"x(0)":>8} {"xdot(0)":>8} {"p1":>8} {"p2":>8} '
              f'{"p3":>8} {"p4":>8} {"p5":>8} {"p6":>8} {"Cost":>12}')
        print('-' * 80)
    
    for i in range(maxiter):
        
        # Initialize for iteration
        # Extracting x0 and p estimates from z variable
        x0_guess    = z[:2]
        p_guess     = z[2:]
        
        # Record cost of most recent iteration
        old_cost    = new_cost
        
        # Initializing matrices to solve normal equation
        Lambda  = np.zeros((8, 8))
        N       = np.zeros(8)
        
        # Set up initial state according to current estimate of z
        initial_state_guess = np.concatenate([x0_guess, phi0.flatten(), psi0.flatten()])
        
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
        # Sometimes I ran into errors about required step sizes being too small and the 
        # function exited integration prematurely. This just checks and lets me know that is the case
        if glsdc_guess_traj.status == -1:
            print(f"Integration failed: {glsdc_guess_traj.message}")
            return z, glsdc_guess_traj, Lambda
        
        # Vectorized error computation
        err = ytilde - glsdc_guess_traj.y[0, :]  # Shape: (n_measurements,)
        
        # The performance index is given as J = err^T @ W @ err
        new_cost = np.sum(W_diag * err**2)
        
        # Extracting Phi and Psi data from solution
        phi_data = glsdc_guess_traj.y[2:6, :].T # dim = len(teval) x 4
        psi_data = glsdc_guess_traj.y[6:18, :].T # dim = len(teval) x 12
        
        for k in range(len(teval)):
            
            # Reshape variational matrices
            phi_k = phi_data[k].reshape(2, 2) # Is now (2x2)
            psi_k = psi_data[k].reshape(2, 6) # Is now (2x6)
            
            # H_i = dg_i / dz = [dg/dx * dx/dx(0), dg/dx * dx/dp]
            # H_i = [[1, 0] * Phi, [1, 0] * Psi]
            H_i = np.concatenate([ dgdx @ phi_k, dgdx @ psi_k], axis=1)
            
            # Accumulate normal equations
            Lambda += (H_i.T * W_diag[k]) @ H_i  # (8, 8)
            N += (H_i.T * W_diag[k]).squeeze() * err[k]  # (8,)
        
        if verbose:
            print(f'{i:5d} {z[0]:8.4f} {z[1]:8.4f} {z[2]:8.4f} {z[3]:8.4f} ' +
                  f'{z[4]:8.4f} {z[5]:8.4f} {z[6]:8.4f} {z[7]:8.4f} {new_cost:12.6e}')
        
        # Solve for update step (use solve instead of inv for numerical stability)
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
 
'''
********************************************************************
Covariance Matrix Calculations
********************************************************************
'''

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
    

def propagate_cov(z: np.ndarray, glsdc_traj, Lambda: np.ndarray):
    """Propogates the covariancce of a glsdc estimate

    Returns:
        Tuple(xt, Pt): The state, xt, and the covariance matrix, Pt, at t=0, 100, 200, 300
    """
    # Times at which system is sampled
    sample_times = [100, 200, 300]
    
    # Creating variable to hold values at each time
    xt = np.zeros((2, 4))
    Pt = np.zeros((4, 8, 8))

    # Assining t=0
    xt[:, 0] = z[:2] 
    P0 = np.linalg.inv(Lambda)
    Pt[0, :, :] = P0

    # Using solution calculated from GLSDC to construct Phi(t) and Psi(t)
    for k,t in enumerate(sample_times):
        
        # Extracting state variable
        xt[:, k + 1] = glsdc_traj.y[:2, t * 10 - 1]
        
        # Extracting Phi and Psi at t from glsdc estimate trajectory
        PhiPsiVec = glsdc_traj.y[2:, t * 10 - 1]
        
        # Reshaping Phi and Psi to their matrix forms
        Phi = PhiPsiVec[:4].reshape(2,2)
        Psi = PhiPsiVec[4:].reshape(2, 6)
        
        # dz/dz0 = [[dx/dx0, dx/dp], [dp/dx, dp/dp]]
        # dx/dx0 = Phi(t)
        # dx/dp = Psi(t)
        # dp/dx = 0
        # dp/dp = I_(6x6)
        dzdz0 = np.block([[Phi, Psi], [np.zeros((6, 2)), np.eye(6)]])
        Pt[k+1, :, :] = dzdz0 @ P0 @ dzdz0.T
    
    return xt, Pt

'''
********************************************************************
Monte Carlo Simulation and Sample Statistics
********************************************************************
'''

def monte_carlo_sim(dynamics, z_true, measured_states, niter = 1000, n_jobs = -1):
    """Performs a Monte Carlo Simulation of the GLSDC Algorithm. This function will use joblib to multithread.

    Args:
        dynamics (function): Function containing sytstem dynamcis
        z_true (np.ndarray): True parameters to be estimated
        measured_states (_type_): Solution containing the measured states from the true trajectory
        niter (int, optional): Number of Monte Carlo Trials to perform. Defaults to 1000.
        n_jobs (int, optional): Number of processors to use. Defaults to -1.

    Returns:
        _type_: _description_
    """
    
    print(f'Running Monte Carlo simulation with {niter} iterations...')
    progress_bar = tqdm(range(niter), desc = 'Monte Carlo Simulation of GLSDC Results')
    
    # MonteCarloTrial is a namedtuple to containt the results we care about from a single trial
    MonteCarloTrial = namedtuple('MonteCarloTrial', ['t0', 't100', 't200', 't300', 'p'])
    
    # This function will be used to help implement multi processing of 1000 glsdc functin calls
    def single_glsdc_run(seed):
        
        # Setting seed for repeatability
        rng = np.random.default_rng(seed=seed)
        
        # Create Noise Measurements for single monte carlo trial
        ytilde_trial = measured_states.y[0, :] + rng.normal(loc=0, scale=sigma, size = len(teval))
        
        # Run GLSDC algorithm for single monte carlo trial
        z_trial, glsdc_trial, _ = glsdc(dynamics, scale_factor*z_true, ytilde_trial)
        
        # Extract x at t = 0, 100, 200, 300, and p
        return MonteCarloTrial(z_trial[:2], # x(0)
                               np.squeeze(glsdc_trial.y[:2, 999]), # x(100)
                               np.squeeze(glsdc_trial.y[:2, 1999]), # x(200)
                               np.squeeze(glsdc_trial.y[:2, 2999]), # x(300)
                               z_trial[2:]) # p
        
    
    # Perform Monte Carlo Parallelization
    trial_results = Parallel(n_jobs = n_jobs)(delayed(single_glsdc_run)(2025 + n) for n in progress_bar)
    
    # Parsing results
    x_at_0      = np.array([trial.t0 for trial in trial_results])
    x_at_100    = np.array([trial.t100 for trial in trial_results])
    x_at_200    = np.array([trial.t200 for trial in trial_results])
    x_at_300    = np.array([trial.t300 for trial in trial_results])
    
    # Calculate Sample Statistics
    monte_carlo_stats = {
        't0'    : {'mean' : np.mean(x_at_0, axis = 0),   'cov'   : np.cov(x_at_0.T)},
        't100'  : {'mean' : np.mean(x_at_100, axis = 0), 'cov'   : np.cov(x_at_100.T)},
        't200'  : {'mean' : np.mean(x_at_200, axis = 0), 'cov'   : np.cov(x_at_200.T)},
        't300'  : {'mean' : np.mean(x_at_300, axis = 0), 'cov'   : np.cov(x_at_300.T)}
    }
    
    return monte_carlo_stats, x_at_0, x_at_100, x_at_200, x_at_300

'''
********************************************************************
Plotting System
********************************************************************
'''
def make_sol_plots(saveplot = False):
    
    '''
    Function to create plots needed for project. Creates x(t) vs t, xdot(t) vs t, xdot(t) vs x(t), measurements over x(t) vs t
    '''
    print('Plotting Solution ...')
    # Plotting Solution State Variables vs Time
    solution_fig, axs   = plt.subplots(2, 1, figsize = (10, 4), sharex='col')
    xvt, xdotvt         = axs

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
    Integrating System Dynamics
    ********************************************************************
    '''

    # Initial State
    phi0    = np.eye(2)
    psi0    = np.zeros((2, 6))
    state0  = np.concatenate([x0, phi0.flatten(), psi0.flatten()])

    print(delim_equals + '\nTask 1: Simulating Motion, Measurements, and Validating Matrices\n' + delim_equals)
    print("\nIntegrating System Dynamics ...")

    # Integrating System
    simulated_motion    = solve_ivp(dynamics, tspan, state0, args = (p,), rtol = 1E-10, atol=1E-10)
    measured_states     = solve_ivp(dynamics, tspan, state0, t_eval = teval, args = (p,), rtol = 1E-10, atol=1E-10)
    
    # Validating State Transition Matrix
    print('\nValidating Variational Matrices ...\n' + delim_dash)
    
    # times at which to test for validation
    test_times = [10, 50, 100, 150, 200, 250, 300]


    """
    ********************************************************************
    Adding Measurement Noise
    ********************************************************************
    """

    # Setting numpy seed
    rng = np.random.default_rng(seed=2025)

    # adding gaussian noise to measurements
    ytilde = measured_states.y[0, :] + rng.normal(loc=0, scale=sigma, size = len(teval))
    
    # Estimating z = [x^T, p^T]^T
    z_true = np.concatenate([x0, p])

    # Adjusting to create an initial guess
    z = scale_factor * z_true

    print(delim_equals + '\nTask 2: GLSDC Algorithm\n'+delim_equals)
    
    '''
    ********************************************************************
    Implementing GLSDC from Tapley, Shultz, and Born 2004
    ********************************************************************
    '''
    
    z, glsdc_traj, Lambda = glsdc(dynamics, z, ytilde, verbose = True)

    print('Final Estimate')
    x0_guess = z[:2]
    p_guess = z[2:]

    print(f'x0 = {np.array2string(x0_guess)}')
    print(f'p  = {np.array2string(p_guess)}') 
    
    '''
    ********************************************************************
    Covariance Propagation
    ********************************************************************
    '''
    print('\n' + delim_equals + '\nProducing Covariance Ellipses\n' + delim_equals)
    print(f'\nCalculating P(t) at t = 0, 100, 200, 300')
    
    xt, Pt = propagate_cov(z, glsdc_traj, Lambda)
    
    '''
    ********************************************************************
    Monte Carlo Simulation
    ********************************************************************
    '''
    
    print('\n' + delim_equals + '\nMonte Carlo Simulation\n' + delim_equals + '\n')
    
    monte_carlo_stats, x_at_0, x_at_100, x_at_200, x_at_300 = monte_carlo_sim(dynamics, z_true, measured_states, niter = args.ntrials)
    
    # Printing Statistics to Console
    print(delim_dash + '\nMonte Carlo Statsitcs')
       
    for key,value in monte_carlo_stats.items():

        print(f'Statistics for t = '+ key)
        print(f'    - Mean: {value['mean']}')
        print(f'    - Cov:\n{np.array2string(value['cov'])}')
        print(np.linalg.eig(value['cov']))
        
    # Collecting sample statistics for plotting
    sample_states       = np.zeros((2, 4))
    sample_states[:, 0] = monte_carlo_stats['t0']['mean']
    sample_states[:, 1] = monte_carlo_stats['t100']['mean']
    sample_states[:, 2] = monte_carlo_stats['t200']['mean']
    sample_states[:, 3] = monte_carlo_stats['t300']['mean']
    
    sample_cov    = np.zeros((4, 2, 2))
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
    cov_plot = plot_cov_ellipse(sample_cov, Pt[:, :2, :2], sample_states, xt, true_states, saveplot=True)
    
    task1 = make_sol_plots(True)
    
    plt.show()
    
        
        