import matplotlib.pyplot as plt, numpy as np
from numpy import pi, sin, cos
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

'''
EAS 6414 Project 3
Dylan D'Silva
'''


'''
Given Values
'''
x0      = np.array([2, 0])
p       = np.array([0.05, 4, 0.2, -0.5, 10, pi/2])
tspan   = [0, 300]
teval   = [i/10 for i in range(1, 3001)]

# Settings to change
sigma = 0.1 # Given Standard Deviation 
maxiter = 20 # Max iteration count for GLSDC 
scale_factor = 0.9 # Scale factor for GLSDC Guess

'''
System Dynamics
'''
def dynamics(t: float, y: NDArray[np.number], p: NDArray[np.number]) -> NDArray:
    """Integrates the system dynamics and variational matrices

    Args:
        t (float): time variable
        y (NDArray[np.number]): one dimensional array with length 18 (vertical concatenation of xdot, flattened Phi matrix, flattend Psi matrix)
        p (NDArray[np.number]): vector of parameters

    Returns:
        NDArray: _description_
    """
    # Extracting x, phi, psi
    x, phi, psi = y[:2], y[2:6].reshape(2, 2), y[6:].reshape(2, 6)
    p1, p2, p3, p4, p5, p6 = p
    
    # Partial derivative of f wrt x and xdot
    A = np.array([
                    [0, 1],
                    [-(p2 + 3*p3*x[0]**2), -p1]
                ])
    
    # Partial derivative of f wrt p
    dfdp = np.array([
                        [0, 0, 0, 0, 0, 0],
                        [-x[1], -x[0], -x[0]**3, -sin(p5*t + p6), -p4*t*cos(p5*t+p6), -p4*cos(p5*t+p6)]
                    ])
    
    # Equation for xdot
    xdot = np.array([
                        x[1],
                        -( p1*x[1] + p2*x[0] + p3*x[0]**3 + p4*sin(p5*t + p6) )
                    ])
    
    # Definitions of PhiDot and PsiDot
    phidot = A @ phi
    psidot = A @ psi + dfdp
    
    # Returning stacked column vector of each time rate
    return np.concatenate([xdot, phidot.flatten(), psidot.flatten()])

'''
Integrating System Dynamics
'''

# Initial State
phi0    = np.eye(2)
psi0    = np.zeros((2, 6))
state0  = np.concatenate([x0, phi0.flatten(), psi0.flatten()])

print('='*50)
print(f'x0 = {np.array2string(x0)}')
print(f'p  = {np.array2string(p)}')
print('-'*50)
print("Beginning Integration of System Dynamics ...")

# Integrating System
simulated_motion    = solve_ivp(dynamics, tspan, state0, args = (p,), rtol = 1E-10, atol=1E-10)
measured_states     = solve_ivp(dynamics, tspan, state0, t_eval = teval, args = (p,), rtol = 1E-10, atol=1E-10)

print('Integration Complete')

"""
Adding Measurement Noise
"""

print('-'*50)
print('Adding Gaussian Noise to x(t)')
# Given Standard Deviation
print(f'\t- mean  = {0}')
print(f'\t- sigma = {sigma}')

# Setting numpy seed
rng = np.random.default_rng(2025)

# adding gaussian noise to measurements
ytilde = measured_states.y[0, :] + np.random.normal(loc=0, scale=sigma, size = len(teval))

'''
Plotting System
'''
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

# Plotting phase portrait
phase_portrait = plt.figure()
plase_portrait_ax = phase_portrait.gca()
plase_portrait_ax.plot(simulated_motion.y[0, :], simulated_motion.y[1, :])

# Changing plot properties
plase_portrait_ax.set_title('Phase Portrait of System')
plase_portrait_ax.set_xlabel(r'$x(t)$ vs $t$')
plase_portrait_ax.set_ylabel(r'$\dot{x}(t)$ vs $t$')

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


'''
Implementing GLSDC from Tapley, Shultz, and Born 2004
'''

# Estimating z = [x^T, p^T]^T
z_true = np.concatenate([x0, p])

# Adjusting to create an initial guess
z = scale_factor * z_true

# Initializing cost values used for determining convergence
old_cost = np.inf
new_cost = 1

# Error Tolerance for GLSDC
tol = 1E-5

# Entering GLSDC Loop
for i in range(maxiter):
    print("="*50)
    print('Beginning GLSDC Iteration ', i+1)
    
    # Initialize for iteration
    print('-'*50)
    print('Starting guess for parameters')
    x0_guess = z[:2]
    p_guess = z[2:]
    print(f'x0 = {np.array2string(x0_guess)}')
    print(f'p  = {np.array2string(p_guess)}')
    
    
    old_cost = new_cost
    new_cost = 0
    
    # Values for solving the normal equations Lambda @ deltaZ = N
    Lambda = np.zeros((8, 8))
    N = np.zeros((8, 1))
    
    initial_state_guess = np.concatenate([x0_guess, phi0.flatten(), psi0.flatten()])
    
    # Integrate trajectory based of of x0_guess and p_guess
    print('-'*50)
    print('Beginning Integration')
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
    print('Compelted Integration ...')
    if glsdc_guess_traj.status == -1:
        print(glsdc_guess_traj.message)
        
    err = np.zeros((len(teval), 1))
    
    for k, t_k in enumerate(teval):
        err[k, 0] = ytilde[k] - glsdc_guess_traj.y[0, k]
        
        # Creating Weight Matrix
        R = np.diag([(1 + t_k)*sigma**2])
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

    delta_z = np.linalg.solve(Lambda, N).reshape(8)
    
    print('-'*50)
    print(f'Least Squares Error Cost: {new_cost}')
    print(f'Change in Error Cost:\t {new_cost - old_cost}')
    print(f'Relative Change of new_cost: {(new_cost - old_cost) / old_cost}')
    print(f'Delta z: \n{np.array2string(delta_z)}')
    
    # Checking if error tolerance is met
    if abs(new_cost - old_cost) / old_cost <= tol:
        break
    
    z += delta_z
    
    # Keep between zero and 2*pi
    z[-1] = z[-1] % (2*pi)

print('='*50)
print('Exiting GLSDC Loop')
if abs(new_cost - old_cost) / old_cost > tol:
    print('='*50)
    print(f'GLSDC Failed to converge to a solution which met tol = {tol} within {maxiter} iterations')

print('Final Estimate')
x0_guess = z[:2]
p_guess=  z[2:]

print(f'x0 = {np.array2string(x0_guess)}')
print(f'p  = {np.array2string(p_guess)}')   
    
doPlot = False
if doPlot:
    plt.show()