import matplotlib.pyplot as plt, numpy as np
from numpy import pi, sin, cos
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from numba import njit
from collections import namedtuple
from joblib import Parallel, delayed
import multiprocessing
import time
from tqdm import tqdm
'''
EAS 6414 Project 3
Dylan D'Silva
'''

# Settings to change how file runes
sigma           = 0.1 # Given Standard Deviation 
maxiter         = 30 # Max iteration count for GLSDC 
scale_factor    = 0.9 # Scale factor for GLSDC Guess
tol             = 1E-3 # Error Tolerance for GLSDC

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
        y: State vector [x, phi.flat, psi.flat] (length 18)
        p: Parameter vector (length 6)
    
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
    # Phidot = A @ Phi
    # A = [
    #     [0, 1]
    #     [-p2 - 3p3x**2, -p1]
    # ]
    phi             = y[2:6].reshape((2, 2))
    phidot          = np.empty((2, 2))
    phidot[0, 0]    = phi[1, 0]
    phidot[0, 1]    = phi[1, 1]
    phidot[1, 0]    = -(p2 + 3.0 * p3 * x[0]**2) * phi[0, 0] - p1 * phi[1, 0]
    phidot[1, 1]    = -(p2 + 3.0 * p3 * x[0]**2) * phi[0, 1] - p1 * phi[1, 1]
    
    # Psi (2x6)
    # Psidot = A @ Psi + df/dp
    # A = [
    #     [0, 1]
    #     [-p2 - 3p3x**2, -p1]
    # ]
    # df/dp = [
    #     [Zeros]
    #     [-xdot, -x, -x**3, -sin(theta), -p4*t*cos(theta), -p4cos(theta)]
    # ]
    psi = y[6:18].reshape((2, 6))
    psidot = np.empty((2, 6))
    for j in range(6):
        psidot[0, j] = psi[1, j] + dfdp[0, j]
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
    
    phi0    = np.eye(2)
    psi0    = np.zeros((2, 6))
    
    old_cost = np.inf
    new_cost = 1
    
    # Precompute weights for all time steps (vectorized)
    teval_arr   = np.array(teval)
    W_diag      = 1.0 / ((1 + teval_arr) * sigma**2)  # Diagonal weight matrix elements
    
    # H matrix selector (avoid repeated array creation)
    dgdx = np.array([[1.0, 0.0]])
    
    if verbose:
        print(f'\n{"Iter":>5} {"x(0)":>8} {"xdot(0)":>8} {"p1":>8} {"p2":>8} '
              f'{"p3":>8} {"p4":>8} {"p5":>8} {"p6":>8} {"Cost":>12}')
        print('-' * 80)
    
    for i in range(maxiter):
        
        # Initialize for iteration
        x0_guess    = z[:2]
        p_guess     = z[2:]
        
        # Record cost of most recent iteration
        old_cost    = new_cost
        
        # Initializing matrices to solve normal equation
        Lambda  = np.zeros((8, 8))
        N       = np.zeros(8)
        
        # Set up initial state
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
        # function exited integration prematurely
        if glsdc_guess_traj.status == -1:
            print(f"Integration failed: {glsdc_guess_traj.message}")
            return z, glsdc_guess_traj, Lambda
        
        # Vectorized error computation
        err = ytilde -  glsdc_guess_traj.y[0, :]  # Shape: (n_measurements,)
        
        # The performance index is given as J = err^T @ W @ err
        # The matrix multiplication is converted to elementwise multiplication and summation
        new_cost = np.sum(W_diag * err**2)
        
        # Extracting Phi and Psi data from solution
        phi_data = glsdc_guess_traj.y[2:6, :].T # dim = len(teval), 4
        psi_data = glsdc_guess_traj.y[6:18, :].T # dim = len(teval), 12
        
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
        
        # Check convergence
        if abs(new_cost - old_cost) / old_cost <= tol:
            break
        
        # Solve for update step (use solve instead of inv for numerical stability)
        delta_z = np.linalg.solve(Lambda, N)
        
        # Update state using calculated step
        z += delta_z
        
        # Keep p6 in [0, 2pi]
        z[-1] = z[-1] % (2 * pi)
    
    if verbose:
        print('-' * 80)
    
    if abs(new_cost - old_cost) / old_cost > tol:
        print(f'\nGLSDC failed to converge (tol={tol}) in {maxiter} iterations')
    
    return GLSDC_SOL(z, glsdc_guess_traj, Lambda)

 
'''
********************************************************************
Covariance Matrix Calculations
********************************************************************
'''

def propagate_covar(z: np.ndarray, glsdc_traj, Lambda: np.ndarray, saveplot: bool = False, filename: str = 'Images/prop_cov_ellipse.png'):
    
    # Times at which system is sampled
    sample_times = [100, 200, 300]
    # Creating variable to hold covariance matrix at each sample time
    Pt = np.zeros((4, 8, 8))

    # Calculating P(t_0) as inv(Lambda)
    P0 = np.linalg.inv(Lambda)
    Pt[0, :, :] = P0

    # Using solution calculated from GLSDC to construct Phi(t) and Psi(t)
    for k,t in enumerate(sample_times):
        PhiPsiVec = glsdc_traj.y[2:, t*10-1]
        Phi = PhiPsiVec[:4].reshape(2,2)
        Psi = PhiPsiVec[4:].reshape(2, 6)
        dzdz0 = np.block([[Phi, Psi], [np.zeros((6, 2)), np.eye(6)]])
        Pt[k+1, :, :] = dzdz0 @ P0 @ dzdz0.T


    # Creating cos and sin of parameter for ellipse
    t = np.linspace(0, 2*pi, 100)
    ellipse = np.array([cos(t), sin(t)])

    covar_ellipse_plot, covar_ellipse_axs = plt.subplots(2, 2, layout='tight')
    for n in range(2):
        for i in range(2):
            
            k = 2*n + i
            
            # Get state
            if n==0 and i==0:
                state = z[:2]
            else:
                state = glsdc_traj.y[:2, glsdc_traj.t==sample_times[k-1]]
            
            # Extract Px (2x2 matrix in upper left corner) from P(t)
            Px = Pt[k, :2, :2]
            
            # Get  Eigenvalues and Eigen Vectors of Px
            D, V = np.linalg.eig(Px)
            covar_ellipse_axs[n, i].scatter(state[0], state[1], color = 'k')
            for sigma_lvl in range(1, 4):
                
                cov_ellispe = V @ (sigma_lvl * np.sqrt(D) * ellipse.T).T + state.reshape(2, 1)
                covar_ellipse_axs[n, i].plot(cov_ellispe[0], cov_ellispe[1], label = r'$\sigma = $' + f'{sigma_lvl}')
                covar_ellipse_axs[n, i].set_title(f't = {100 * k}')
                covar_ellipse_axs[n, i].set_xlabel(r'$x(t)$')
                covar_ellipse_axs[n, i].set_ylabel(r'$\dot{x}(t)$')
                covar_ellipse_axs[n, i].legend()

    plt.show()
    if saveplot:
        covar_ellipse_plot.savefig(filename, format='png')
    
    return Pt
'''
********************************************************************
Sample Statistics
********************************************************************
'''


def monte_carlo_sim(niter = 1000):

    x_at_0 = np.zeros((len(teval), 2))
    x_at_100 = np.zeros((len(teval), 2))
    x_at_200 = np.zeros((len(teval), 2))
    x_at_300 = np.zeros((len(teval), 2))
    
    num_cores = multiprocessing.cpu_count()
    def extract_relevant_times(n, z, glsdc_sol):
        x_at_0[n, :] = z[:2]
        x_at_100[n, :] = np.squeeze(glsdc_sol.y[:2, 999])
        x_at_200[n, :] = np.squeeze(glsdc_sol.y[:2, 1999])
        x_at_300[n, :] = np.squeeze(glsdc_sol.y[:2, 2999])
        
    
    for n in tqdm(range(niter)):
        ytilde = measured_states.y[0, :] + np.random.normal(loc=0, scale=sigma, size = len(teval))
        z, glsdc_sol, Lambda = glsdc(dynamics, scale_factor*z_true, ytilde)
        extract_relevant_times(n, z, glsdc_sol)

'''
********************************************************************
Plotting System
********************************************************************
'''
def make_plots(saveplot = False):
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

    # Changing Axis Properties
    xvt.set_xlim(tspan)
    xdotvt.set_xlabel('t')

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
    measurement_ax.legend()
    
    plt.show()

    if saveplot:
        solution_fig.savefig('Images/solution_fig.png', format='png', dpi=1440)
        phase_portrait.savefig('Images/phase_portrait.png', format='png', dpi=1440)
        measurement_fig.savefig('Images/measurements_fig.png', format='png', dpi=1440)
        print('Plots Generated and Saved\n')
        
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

    """
    ********************************************************************
    Adding Measurement Noise
    ********************************************************************
    """

    # Setting numpy seed
    rng = np.random.default_rng(seed=2025)

    # adding gaussian noise to measurements
    ytilde = measured_states.y[0, :] + np.random.normal(loc=0, scale=sigma, size = len(teval))
    
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
    print(type(glsdc_traj))
    print('Final Estimate')
    x0_guess = z[:2]
    p_guess = z[2:]

    print(f'x0 = {np.array2string(x0_guess)}')
    print(f'p  = {np.array2string(p_guess)}') 
    
    print('\n' + delim_equals + '\nProducing Covariance Ellipses\n' + delim_equals)
    print(f'\nCalculating P(t) at t = 0, 100, 200, 300')
    
    Pt = propagate_covar(z, glsdc_traj, Lambda, saveplot=False)
    
    print('\n' + delim_equals + '\nMonte Carlo Simulation\n' + delim_equals + '\n')




