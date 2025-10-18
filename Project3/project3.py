import matplotlib.pyplot as plt, numpy as np
from numpy import pi, sin, cos
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

'''
EAS 6414 Project 3
Dylan D'Silva
'''

# Settings to change how file runes
sigma           = 0.1 # Given Standard Deviation 
maxiter         = 30 # Max iteration count for GLSDC 
scale_factor    = 0.98 # Scale factor for GLSDC Guess
doPlot          = False # Used to control whether the generated plots are shown
tol             = 1E-3 # Error Tolerance for GLSDC


# Delimeter strings for print statements
delimeter_equals = '='*80
delimeter_dash = '-'*80

'''
********************************************************************
Given Values
********************************************************************
'''
x0              = np.array([2, 0]) # Initial Conditions
p               = np.array([0.05, 4, 0.2, -0.5, 10, pi/2]) # True Parameters
tspan           = [0, 300] # Initial and Final Times of Simulation
teval           = [i/10 for i in range(1, 3001)] # Times at which system is sampled for measurement

print(delimeter_equals + '\nEAS 6414 Project 3: Initial State and Parameter Estimation\n' + delimeter_equals)

print('\nGiven Values\n' + delimeter_dash)
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
def dynamics(t: float, y: NDArray[np.floating], p: NDArray[np.floating]) -> NDArray:
    """Integrates the system dynamics and variational matrices.
    
    Args:
        t: Time variable
        y: State vector [x, phi.flat, psi.flat] (length 18)
        p: Parameter vector (length 6)
    
    Returns:
        Time derivative of state vector
    """
    # Extracting x, phi, psi
    x, phi, psi = y[:2], y[2:6].reshape(2, 2), y[6:].reshape(2, 6)
    p1, p2, p3, p4, p5, p6 = p
    
    # Partial derivative of f wrt x and xdot
    A = np.array([
                    [0, 1],
                    [-(p2 + 3*p3*x[0]**2), -p1]
                ])
    
    theta = p5*t + p6
    # Partial derivative of f wrt p
    dfdp = np.array([
                        [0, 0, 0, 0, 0, 0],
                        [-x[1], -x[0], -x[0]**3, -sin(theta), -p4*t*cos(theta), -p4*cos(theta)]
                    ])
    
    # Equation for xdot
    xdot = np.array([
                        x[1],
                        -( p1*x[1] + p2*x[0] + p3*x[0]**3 + p4*sin(theta) )
                    ])
    
    # Definitions of PhiDot and PsiDot
    phidot = A @ phi
    psidot = A @ psi + dfdp
    
    # Returning stacked column vector of each time rate
    return np.concatenate([xdot, phidot.ravel(), psidot.ravel()])

'''
********************************************************************
Integrating System Dynamics
********************************************************************
'''

# Initial State
phi0    = np.eye(2)
psi0    = np.zeros((2, 6))
state0  = np.concatenate([x0, phi0.flatten(), psi0.flatten()])

print(delimeter_equals + '\nTask 1: Simulating Motion, Measurements, and Validating Matrices\n' + delimeter_equals)
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


print(delimeter_equals + '\nTask 2: GLSDC Algorithm\n'+delimeter_equals)
# Entering GLSDC Loop
print('\nIteration   x(0)   xdot(0)   p1      p2      p3     p4     p5     p6     Cost\n' + delimeter_dash)

def glsdc(dynamics, z: NDArray, ytilde: NDArray = ytilde, tol: float = tol, maxiter: int = maxiter):
    old_cost = np.inf
    new_cost = 1
    
    for i in range(maxiter):
        
        # Initialize for iteration
        x0_guess = z[:2]
        p_guess = z[2:]
        
        old_cost = new_cost
        new_cost = 0
        
        # Values for solving the normal equations Lambda @ deltaZ = N
        Lambda = np.zeros((8, 8))
        N = np.zeros((8, 1)) 
        
        initial_state_guess = np.concatenate([x0_guess, phi0.flatten(), psi0.flatten()])
        
        # Integrate trajectory based off of x0_guess and p_guess

        glsdc_guess_traj = solve_ivp(
                                dynamics,
                                tspan,
                                initial_state_guess,
                                t_eval=teval,
                                method = 'DOP853',
                                rtol = 1E-8,
                                atol = 1E-8,
                                args = (p_guess,)
                            )
        
        # Parsing through data from integration
        if glsdc_guess_traj.status == -1:
            print(glsdc_guess_traj.message)
            
        err = np.zeros((len(teval), 1))
        
        for k, t_k in enumerate(teval):
            err[k, 0] = ytilde[k] - glsdc_guess_traj.y[0, k]
            
            # Creating Weight Matrix
            R = np.diag([(1+t_k)*sigma**2])
            W = np.linalg.inv(R)
            
            # Pulling Variational Matrices
            phi = glsdc_guess_traj.y[2:6, k].reshape(2, 2)
            psi = glsdc_guess_traj.y[6:, k].reshape(2, 6)
            H_i = np.concatenate([ np.squeeze(np.array([[1, 0]]) @ phi), np.squeeze(np.array([[1, 0]]) @ psi)]).reshape(1, 8)
            
            # Adding to matrices for normal equation
            Lambda += H_i.T @ W @ H_i
            N += H_i.T @ W @ err[k].reshape(1, 1)
            
            # Adding error to weighted least squares cost
            new_cost += err[k].T @ W @ err[k]

        # Checking if error tolerance is met
        if abs(new_cost - old_cost) / old_cost <= tol:
            break
        
        # If not converged, calculated next step in z
        delta_z = np.linalg.solve(Lambda, N).reshape(8)
        
        # Stepping z
        z += delta_z
        
        # Keep between zero and 2*pi
        z[-1] = z[-1] % (2*pi)
    return z, glsdc_guess_traj, Lambda

z, glsdc_traj, Lambda = glsdc(dynamics, z, ytilde)

print(delimeter_dash)
if abs(new_cost - old_cost) / old_cost > tol:
    print(delimeter_equals + f'\nGLSDC Failed to converge to a solution which met tol = {tol} within {maxiter} iterations')

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
print('\n' + delimeter_equals + '\nProducing Covariance Ellipses\n' + delimeter_equals)
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
print('\n' + delimeter_equals + '\nMonte Carlo Simulation\n' + delimeter_equals + '\n')
# Iterating 1000  times
for k in  range(1000):
    if k+1 % 50 == 0:
        print(f'{int((k+1)/10)} %')
    ytilde = measured_states.y[0, :] + np.random.normal(loc=0, scale=sigma, size = len(teval))