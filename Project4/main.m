%% Initial Conditions and Constants
clear; clc; close all;
r0 = [7000; 1000; 200]; % km
v0 = [4; 7; 2]; % km/s
y0 = [r0; v0]; % Combined initial state

% Constants
R_obsv = 6371;% Radius of Earth in km
obsv_lat = deg2rad(5); % Observer latitude
LST0 = deg2rad(10); % Observer local siderial time
omega_E = 7.2921159E-5; % rad/s
mu = 398600.4415; % Gravitational paramter of earth

% tspan for trajectory integration
tspan = [0, 100];

% time values for measurements
tmeas = 0:10:100;

% Computing LST at tmeas
LST = LST0 + omega_E * tmeas;

% Noise Covariance
sigma_rho = 1; % km
sigma_az = deg2rad(0.01); % rad
sigma_el = deg2rad(0.01); % rad
R = diag([sigma_rho^2, sigma_az^2, sigma_el^2]);


% Delimeter for text/section outputs
delim_eq = repelem('=', 70);
delim_dash = repelem('-', 70);
%% Generating Measurements

% Simulating true motion
func = @(t, y) twobody(t, y, mu);
options = odeset('RelTol',1E-8, 'AbsTol',1E-10);
[~, measured_states] = ode45(func, tmeas, y0, options);

% Generating inertial range vector
rho = inertial_range(measured_states(:, 1:3), R_obsv, obsv_lat, LST);

% Range vector in observer frame
rho_obsv = observer_range(rho, obsv_lat, LST);

% Collecting measurements
h_cords = horizontal_coordinates(rho_obsv);

% Adding Noise
measurements = generate_measurements(h_cords, R, length(tmeas));

%% GLSDC
r0hat = [6990; 1; 1];
v0hat = [1; 1; 1];
x0hat = [r0hat; v0hat];

maxiter = 7;
tol = 1E-5;


%% Function Definitions

% Function for true twobody motion
function ydot = twobody(t, y, mu)
    r = y(1:3);
    v = y(4:6);
    ydot = [v; -mu/norm(r)^3*r];
end

% Function to convert from twobody relative position to inertial range
function rho = inertial_range(r, R, obsv_lat, LST)
    transformation = - R * [cos(obsv_lat)*cos(LST); cos(obsv_lat)*sin(LST); sin(obsv_lat)*ones(size(LST))]';
    rho = r + transformation;
end

% Function translates inertial slant range to observer frame
function rho_obsv = observer_range(rho, obsv_lat, LST)
    rho_obsv = zeros(size(rho));
    rotation2 = [cos(obsv_lat), 0, sin(obsv_lat); 0, 1, 0; -sin(obsv_lat), 0, cos(obsv_lat)];
    for k =1:length(LST)
        LST_k = LST(k);
        rotation1 = [cos(LST_k), sin(LST_k), 0; -sin(LST_k), cos(LST_k), 0; 0, 0, 1];
        rho_obsv(k, :) = (rotation2 * rotation1 * rho(k, :)')';
    end
end

% Function conversts slant range in observer frame to horizontal (alt-az)
% coordinates
function h_cords = horizontal_coordinates(rho_obsv)
    rho_mag = vecnorm(rho_obsv, 2, 2);
    az = atan2(rho_obsv(:, 2), rho_obsv(:, 3));
    el = asin(rho_obsv(:, 1)./rho_mag);
    h_cords = [rho_mag, az, el];
end

% Function adds multi-variate gaussian noise 
function noisy_measurements = generate_measurements(h_cords, R, n)
    noise = randn(n, 3) * sqrt((R));
    noisy_measurements = h_cords + noise;
end

% Function to integrate Two Body mechanics and STM ODE
function ydot = two_body_STM(t, y, mu)
    r = y(1:3);
    v = y(4:6);
    Phi = rehshape(y(7:42));

    % Phidot = df/dx * Phi
    % A = df/dx = [dv/dr, dv/dv; dvdot/dr, dvdot/dv]
    %           = [0, I; A_21, 0]
    A = [zeros(3), eye(3); 3*mu/norm(r)^5*(r*r') - mu/norm(r)^3*eye(3), zeros(3)];
    Phidot = A * Phi;
    ydot = [v; -mu*r/norm(r)^3; Phidot(:)];
end

% Function for sinlge GLSDC Iteration
function [estimate, Lambda] = glsdc(dynamics, guess, measurements, R, tmeas, R_obsv, LST)
    % Initializing relevant values
    W = inv(R);
    Phi0 = zeros(6);

    for n = 1:maxiter

        % Lambda and N for normal equations
        Lambda = zeros(6);
        N = zeros(6, 1);
        
        % Initial state = [guess; Phi0(:)]
        initial_state_guess = [guess; Phi0(:)];

        % Integrate dynamics and STM
        options = odeset('RelTol',1E-8, 'AbsTol',1E-10);
        [~, state_estimates] = ode45(dynamics, tmeas, initial_state_guess, options);

        % Converting to inertial, then obsv, then horizontal coordinates to
        % arrived at measurement estimate
        rho = inertial_range(state_estimates(:, 1:3), R_obsv, obsv_lat, LST);
        rho_obsv = observer_range(rho, obsv_lat, LST);
        estimated_measurments = horizontal_coordinates(rho_obsv);

        % Using estimated measurements to calculate error
        err = measurements - estimated_measurments;
        
        % Going through and calculating Lambda and N
        for k = 1:length(tmeas)
            % Calculate H_k = dh/dx * Phi
            % Expressions for dh/dx
            dhdx = zeros(3, 6);

            % Extracting Phi
            Phi_k = reshape(state_estimates(k, 7:42), 6, 6);

            % H_k
            H_k = dhdx * Phi_k;

        end
        
        % Check for convergence

        % Solve for delta_x
        delta_x = 0;

        
        % Updating guess
        guess = guess + delta_x;
    end

    
end