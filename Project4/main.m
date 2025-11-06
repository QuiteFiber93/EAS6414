%% Initial Conditions and Constants
clear; clc; close all;
r0 = [7000, 1000, 200]; % km
v0 = [4, 7, 2]; % km/s
y0 = [r0; v0];
R_earth = 6731;
obsv_lat = deg2rad(5);
IST = deg2rad(10);
omega_E = 7.2921159E-5; % rad/s
mu = 398600.4415;

% tspan for trajectory integration
tspan = [0, 100];

% time values for measurements
tmeas = 0:10:100;

% Noise Covariance
sigma_rho = 1; % km
sigma_az = deg2rad(0.01); % rad
sigma_el = deg2rad(0.01); % rad
R = diag([sigma_rho^2, sigma_az^2, sigma_el^2]);

%% Generating Measurements
func = @(t, y) twobody(t, y, mu);
options = odeset('RelTol',1E-8, 'AbsTol',1E-10);
[t, true_motion] = ode45(func, tspan, y0, options);

[tmeas, measured_states] = ode45(func, tmeas, y0, options);
% Converting from current frame to inertial then observer frame
measured_states_obsv = to_inertial_frame(measured_states(:, 1:3)', R_earth, obsv_lat, IST)';

% Converting to range, altitude, azimuth
h_cords = horizontal_cords(measured_states_obsv);

% Generating measurements from horizontal coordiantes
measurements = generate_measurements(h_cords, tmeas, R);

%% Function Definitions

% Function for true twobody motion
function ydot = twobody(t, y, mu)
    % Preallocating size
    ydot = zeros(6, 1);
    r = y(1:3);
    v = y(4:6);

     % rdot = v
    ydot(1:3) = v;
    % vdot = -mu/|r|^3 * r
    ydot(4:6) = -mu/norm(r)^3 * r;
end

% Function for converting to inertial range
function rho_inertial = to_inertial_frame(r, R, obsv_lat, local_ist)
    translation = R * [cos(obsv_lat) * cos(local_ist); ...
                        cos(obsv_lat)*sin(local_ist); ...
                        sin(local_ist)];
    rho_inertial = r - translation;
end

% Function to convert from inertial frame to obsv frame
function rho_obsv = inertial_to_obsv(rho_intertial, obsv_lat, local_ist)
    rotation1 = [cos(local_ist), sin(local_ist), 0; ...
                -sin(local_ist), cos(local_ist), 0; ...
                0, 0, 1];    
    rotation2 = [cos(obsv_lat), 0, sin(obsv_lat); ...
                0, 1, 0; ...
                -sin(obsv_lat), 0, cos(obsv_lat)];
    rho_obsv = rotation2 * rotation1 * rho_intertial;
end

% Function to convert from observer relative position to range, altitude,
% azimuth
function h_cords = horizontal_cords(rho)
    mag = vecnorm(rho, 2, 2);
    az = atan2(rho(:, 2), rho(:, 3));
    el = (rho(:, 1)./mag);
    h_cords = [mag, az, el];
end

% Function to generate measurements from horizontal coordinates
function measurements = generate_measurements(h_cords, tmeas, R)
    measurements = h_cords + mvnrnd([0, 0, 0], R, length(tmeas));
end

% Function for integrating two body equations and the state transition
% matrix
function ydot = twobody_STM(t, y, mu)
    % Preallocating size
    ydot = zeros(42, 1);
    r = y(1:3);
    v = y(4:6);
    Phi = reshape(y(7:42),6, 6);

    % rdot = v
    ydot(1:3) = v;

    % vdot = -mu/|r|^3 * r
    ydot(4:6) = -mu/norm(r)^3 * r;
    % Phidot = A * Phi
    % A =   [dv/dr, dv/dv   ]
    %       [dvdot/dr, dvdot/dv]
    F21 = zeros(3, 3);
    A = [zeros(3), eye(3); F21, zeros(3)];
    Phidot = A * Phi;
    ydot(7:42) = Phidot(:);
end

% Function to implement the glsdc algorithm
function estimate = glsdc(dynamics, measurements)

end