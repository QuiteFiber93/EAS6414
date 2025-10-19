import matplotlib.pyplot as plt, numpy as np
from numpy import pi, sin, cos
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from numba import njit
'''
EAS 6414 Project 3
Dylan D'Silva
'''

# Settings to change how file runes
sigma           = 0.1 # Given Standard Deviation 
maxiter         = 30 # Max iteration count for GLSDC 
scale_factor    = 0.9 # Scale factor for GLSDC Guess
doPlot          = False # Used to control whether the generated plots are shown
tol             = 1E-3 # Error Tolerance for GLSDC


# Delimeter strings for prinmt statements
delim_equals = '='*80
delim_dash = '-'*80

'''
********************************************************************
Given Values
********************************************************************
'''
x0              = np.array([2, 0]) # Initial Conditions
p               = np.array([0.05, 4, 0.2, -0.5, 10, pi/2]) # True Parameters
tspan           = [0, 300] # Initial and Final Times of Simulation
teval           = [i/10 for i in range(1, 3001)] # Times at which system is sampled for measurement

print(delim_equals + '\nEAS 6414 Project 3: Initial State and Parameter Estimation\n' + delim_equals)

print('\nGiven Values\n' + delim_dash)
print(f'x0                              = {np.array2string(x0)}')
print(f'p                               = {np.array2string(p)}')
print(f'Measurement Covariance          = {sigma}^2')
print(f'Maximum Iteration for GLSDC     = {maxiter}')
print(f'Scale factor for initial guess  = {scale_factor}\n')

'''
********************************************************************
System Dynamics
********************************************************************
'''
@njit(cache=True)
def dynamics(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Uses Numba Just-In-Time machine code compilation and caching to speed up function calls
    
    
    Args:
        t: Time variable
        y: State vector [x, phi.flat, psi.flat] (length 18)
        p: Parameter vector (length 6)
    
    Returns:
        Time derivative of state vector
    """
    # Extract states
    x0, x1 = y[0], y[1]
    
    # Extract parameters
    p1, p2, p3, p4, p5, p6 = p[0], p[1], p[2], p[3], p[4], p[5]
    
    # Computing a often repeated value
    theta = p5 * t + p6
    
    # A matrix <- df/dx(2x2)
    A00, A01 = 0.0, 1.0
    A10 = -(p2 + 3.0 * p3 * x0**2)
    A11 = -p1
    
    # df/dp (2x6)
    dfdp = np.zeros((2, 6))
    dfdp[1, 0] = -x1
    dfdp[1, 1] = -x0
    dfdp[1, 2] = -x0**3
    dfdp[1, 3] = -sin(theta)
    dfdp[1, 4] = -p4 * t * cos(theta)
    dfdp[1, 5] = -p4 * cos(theta)
    
    # State derivative
    xdot = np.empty(2)
    xdot[0] = x1
    xdot[1] = -(p1 * x1 + p2 * x0 + p3 * x0**3 + p4 * sin(theta))
    
    # Phi matrix operations (2x2)
    phi = y[2:6].reshape((2, 2))
    phidot = np.empty((2, 2))
    phidot[0, 0] = A01 * phi[1, 0]
    phidot[0, 1] = A01 * phi[1, 1]
    phidot[1, 0] = A10 * phi[0, 0] + A11 * phi[1, 0]
    phidot[1, 1] = A10 * phi[0, 1] + A11 * phi[1, 1]
    
    # Psi matrix operations (2x6)
    psi = y[6:18].reshape((2, 6))
    psidot = np.empty((2, 6))
    for j in range(6):
        psidot[0, j] = A01 * psi[1, j] + dfdp[0, j]
        psidot[1, j] = A10 * psi[0, j] + A11 * psi[1, j] + dfdp[1, j]
    
    # Concatenate results
    result = np.empty(18)
    result[0:2] = xdot
    result[2:6] = phidot.ravel()
    result[6:18] = psidot.ravel()
    
    return result

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

'''
********************************************************************
Plotting System
********************************************************************
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

if doPlot:
    solution_fig.savefig('Images/solution_fig.png', format='png', dpi=1440)
    phase_portrait.savefig('Images/phase_portrait.png', format='png', dpi=1440)
    measurement_fig.savefig('Images/measurements_fig.png', format='png', dpi=1440)
    print('Plots Generated and Saved\n')


'''
********************************************************************
Implementing GLSDC from Tapley, Shultz, and Born 2004
********************************************************************
'''

# Estimating z = [x^T, p^T]^T
z_true = np.concatenate([x0, p])

# Adjusting to create an initial guess
z = scale_factor * z_true

# Initializing cost values used for determining convergence
old_cost = np.inf
new_cost = 1


print(delim_equals + '\nTask 2: GLSDC Algorithm\n'+delim_equals)
# Entering GLSDC Loop

def glsdc(dynamics, z: np.ndarray, ytilde: np.ndarray, 
                    teval: list = teval, tspan: list = tspan,
                    sigma: float = sigma, tol: float = 1e-3, 
                    maxiter: int = 30, verbose: bool = False):
    """Optimized GLSDC with vectorized operations and precomputed values.
    
    Key optimizations:
    1. Vectorized weight matrix operations
    2. Precompute time-dependent weights
    3. Use einsum for efficient matrix operations
    4. Reduce memory allocations
    """
    old_cost = np.inf
    new_cost = 1
    
    # Precompute weights for all time steps (vectorized)
    teval_arr = np.array(teval)
    R_diag = (1 + teval_arr) * sigma**2
    W_diag = 1.0 / R_diag  # Diagonal weight matrix elements
    
    # H matrix selector (avoid repeated array creation)
    H_selector_x = np.array([[1.0, 0.0]])
    
    if verbose:
        print(f'\n{"Iter":>5} {"x(0)":>8} {"xdot(0)":>8} {"p1":>8} {"p2":>8} '
              f'{"p3":>8} {"p4":>8} {"p5":>8} {"p6":>8} {"Cost":>12}')
        print('-' * 80)
    
    for iteration in range(maxiter):
        # Extract guess
        x0_guess = z[:2]
        p_guess = z[2:]
        
        old_cost = new_cost
        
        # Initialize accumulator matrices (more efficient than zeros + addition)
        Lambda = np.zeros((8, 8))
        N = np.zeros(8)
        
        # Set up initial state
        initial_state_guess = np.concatenate([x0_guess, phi0.flatten(), psi0.flatten()])
        
        # Integrate trajectory
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
            return z, glsdc_guess_traj, Lambda
        
        # Vectorized error computation
        y_pred = glsdc_guess_traj.y[0, :]
        err = ytilde - y_pred  # Shape: (n_measurements,)
        
        # Vectorized cost computation
        new_cost = np.sum(W_diag * err**2)
        
        # Process all measurements (can't fully vectorize due to matrix shapes)
        # But we optimize the loop body
        phi_data = glsdc_guess_traj.y[2:6, :].T  # (n_times, 4)
        psi_data = glsdc_guess_traj.y[6:18, :].T  # (n_times, 12)
        
        for k in range(len(teval)):
            # Reshape variational matrices
            phi_k = phi_data[k].reshape(2, 2)
            psi_k = psi_data[k].reshape(2, 6)
            
            # Compute H_i more efficiently
            H_phi = H_selector_x @ phi_k  # (1, 2)
            H_psi = H_selector_x @ psi_k  # (1, 6)
            H_i = np.concatenate([H_phi, H_psi], axis=1)  # (1, 8)
            
            # Weighted update (using scalar weight)
            w_k = W_diag[k]
            H_weighted = H_i.T * w_k  # (8, 1)
            
            # Accumulate normal equations
            Lambda += H_weighted @ H_i  # (8, 8)
            N += H_weighted.squeeze() * err[k]  # (8,)
        
        if verbose:
            print(f'{iteration:5d} {z[0]:8.4f} {z[1]:8.4f} {z[2]:8.4f} '
                  f'{z[3]:8.4f} {z[4]:8.4f} {z[5]:8.4f} {z[6]:8.4f} '
                  f'{z[7]:8.4f} {new_cost:12.6e}')
        
        # Check convergence
        if abs(new_cost - old_cost) / old_cost <= tol:
            break
        
        # Solve for update (use solve instead of inv for better numerics)
        delta_z = np.linalg.solve(Lambda, N)
        
        # Update state
        z += delta_z
        
        # Keep phase angle in [0, 2π]
        z[-1] = z[-1] % (2 * np.pi)
    
    if verbose:
        print('-' * 80)
    
    if abs(new_cost - old_cost) / old_cost > tol:
        print(f'\nGLSDC failed to converge (tol={tol}) in {maxiter} iterations')
    
    return z, glsdc_guess_traj, Lambda

z, glsdc_traj, Lambda = glsdc(dynamics, z, ytilde, verbose = True)


print('Final Estimate')
x0_guess = z[:2]
p_guess = z[2:]

print(f'x0 = {np.array2string(x0_guess)}')
print(f'p  = {np.array2string(p_guess)}')   
 
'''
********************************************************************
Covariance Matrix Calculations
********************************************************************
'''

# Times at which system is sampled
sample_times = [100, 200, 300]
print('\n' + delim_equals + '\nProducing Covariance Ellipses\n' + delim_equals)
print(f'\nCalculating P(t) at t = 0, 100, 200, 300')

# Creating variable to hold covariance matrix at each sample time
Pt = np.zeros((4, 8, 8))

# Calculating P(t_0) as inv(Lambda)
P0 = np.linalg.inv(Lambda)
Pt[0, :, :] = P0

# Using solution calculated from GLSDC to construct Phi(t) and Psi(t)
for k,t in enumerate(sample_times):
    PhiPsiVec = glsdc_traj.y[2:, glsdc_traj.t==t]
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
'''
********************************************************************
Sample Statistics
********************************************************************
'''
print('\n' + delim_equals + '\nMonte Carlo Simulation\n' + delim_equals + '\n')
# Iterating 1000  times
for k in  range(1000):
    if k+1 % 50 == 0:
        print(f'{int((k+1)/10)} %')
    ytilde = measured_states.y[0, :] + np.random.normal(loc=0, scale=sigma, size = len(teval))