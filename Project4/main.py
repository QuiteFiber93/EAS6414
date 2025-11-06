import numpy as np, matplotlib.pyplot as plt
from numpy import sin, cos, arcsin, arctan2
from scipy.integrate import solve_ivp

# Physical Parameters/ Constants
R_obsv = 6731 # Radius of Spherical Earth
obsv_lat = np.radians(5) # Observer Lattitude
lst = np.radians(10) # Inertial Siderial Time
omega_E = 7.2921159E-5 # Angular Velocity of Earth
mu = 398600.4415 # Earth gravitational parameter

# Delimeters
delim_eq = "="*90
delim_dash = '-'*90
    
# Defining Two Body Dynamics
def twobody(t: float, x: np.ndarray, mu: float) -> np.ndarray:
    """Two Body Equations of Motion relative to primary body

    Args:
        t (float): epoch
        x (np.ndarray): state
        mu (float): Gravitational Parameter of primary body

    Returns:
        np.ndarray: time rate of state
    """
    # Unpacking State 
    r = x[:3]
    v = x[3:]
    
    xdot = np.empty(6)
    xdot[:3] = v
    xdot[3:] = - mu / np.linalg.norm(r)**3 * r
    
    return xdot

def to_intertial_range(r: np.ndarray, R: float = R_obsv, obsv_lat: float = obsv_lat, lst: float = lst) -> np.ndarray:
    """Converts body fixed position to inertial 

    Args:
        r (np.ndarray): body fixed position
        R (float, optional): Radius of primary body. Defaults to R_obsv.
        obsv_lat (float, optional): _description_. Defaults to obsv_lat.
        lst (float, optional): _description_. Defaults to lst.

    Returns:
        np.ndarray: _description_
    """
    translation = R * np.array([
        np.cos(obsv_lat) * np.cos(lst),
        np.cos(obsv_lat) * np.sin(lst),
        np.sin(obsv_lat)
    ])
    
    return r - translation.reshape(3,-1)

def intertial_to_obsv(inertial: np.ndarray, obsv_lat, local_lst):
    rotation1 = np.array([
        [cos(local_lst), sin(local_lst), 0],
        [-sin(local_lst), cos(local_lst), 0],
        [0, 0, 1]
    ])
    rotation2 = np.array([
        [cos(obsv_lat), 0, sin(obsv_lat)],
        [0, 1, 0],
        [-sin(obsv_lat), 0, cos(obsv_lat)]
    ])
    
    return rotation2 @ rotation1 @ inertial
    

    
def main():
    
    # Time values
    tspan = [0, 100]
    teval = [n for n in range(10, 101, 10)]

    # True Initial Conditions of System
    r0 = np.array([7000, 1000, 200])
    v0 = np.array([4, 7, 2])
    x0 = np.concatenate([r0, v0])
    
    print(delim_eq + '\nBeginning Orbit Determination Simulation\n' + delim_eq)
    print('\nTrue Initial Conditions\n' + delim_dash)
    print(f'r0 = {np.array2string(r0)}')
    print(f'v0 = {np.array2string(v0)}')
    
    # Simulating True Motion of problem
    true_motion = solve_ivp(twobody, tspan, x0, method = 'DOP853', rtol=1E-9, atol=1E-10, args = (mu,))
    r_true = true_motion.y[:3, :]
    
    inertial_ranges = to_intertial_range(r_true, obsv_lat, lst)
    observer_relative = intertial_to_obsv()
    

if __name__ == '__main__':
    main()