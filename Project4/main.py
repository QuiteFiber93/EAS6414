import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Physical Parameters/ Constants
R = 6731 # Radius of Spherical Earth
obsv_lat = np.radians(5) # Observer Lattitude
IST = np.radians(10) # Inertial Siderial Time
omega_E = 7.2921159E-5 # Angular Velocity of Earth
mu = 3.986E9 # Earth gravitational parameter
    
# Defining Two Body Dynamics
def twobody(t, x, mu):

    # Unpacking State
    r = x[:3]
    v = x[3:]
    
    xdot = np.empty(6)
    xdot[:3] = v
    xdot[3:] = - mu / np.linalg.norm(r)**3 * r
    
    return xdot

def to_intertial_range(r, obsv_lat, ist):
    translation = R * np.array([
        np.cos(obsv_lat) * np.cos(ist),
        np.cos(obsv_lat) * np.sin(ist),
        np.sin(obsv_lat)
    ])
    
    return r - translation.reshape(3,-1)

    
def main():
    
    # Time values
    tspan = [0, 100]
    teval = [n for n in range(10, 101, 10)]

    # True Initial Conditions of System
    r0 = np.array([7000, 1000, 200])
    v0 = np.array([4, 7, 2])
    x0 = np.concatenate([r0, v0])
    
    # Simulating True Motion of problem
    true_motion = solve_ivp(twobody, tspan, x0, method = 'DOP853', rtol=1E-9, atol=1E-9, args = (398600.4415,))
    r_true = true_motion.y[:3, :]
    
    intertial_ranges = to_intertial_range(r_true, obsv_lat, IST)
    

if __name__ == '__main__':
    main()