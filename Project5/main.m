%% Initial Conditions and Constants
clear; clc; close all;
r0 = [7000; 1000; 200]; % km
v0 = [4; 7; 2]; % km/s
y0 = [r0; v0]; % Combined initial state

% Physical
R_obsv = 6371;% Radius of Earth in km
omega_E = 7.2921159E-5; % rad/s
mu = 398600.4415; % Gravitational paramter of earth

% Positional Constants
obsv_lat = deg2rad(5); % Observer latitude
LST0 = deg2rad(10); % Observer local siderial time

% tspan for trajectory integration
tspan = [0, 3000];

% time values for measurements
tmeas = 0:10:3000;

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
% 
% % Collecting measurements
h_cords = horizontal_coordinates(rho_obsv);
% h_cords = measurement_model(measured_states(:, 1:3), obsv_lat, LST, R_obsv);
% Adding Noise
measurements = generate_measurements(h_cords, R, length(tmeas));


%% Implementing Extended Kalman Filter

% Initial guess 
x0_ekf = [6990; 1; 1; 1; 1; 1];

% Process noise covariance
Q = eye(6) * 1E-8;

% Run EKF
[xhat_ekf, P_ekf] = EKF(x0_ekf, measurements, tmeas, Q, R, mu, obsv_lat, LST, R_obsv);

% Calculating error
ekf_error = measured_states - xhat_ekf;

% Extracting 3 sigma error bounds
sigma_bounds = zeros(length(tmeas), 6);
for k=1:length(tmeas)
    sigma_bounds(k, :) = 3 * sqrt(diag(P_ekf(:, :, k)));
end

%% Plotting measurements vs true trajectory in observer frame

[t, true_motion] = ode45(func, tspan, y0, options);
LST_true = LST0 + omega_E*t';
rho_true_motion = inertial_range(true_motion(:, 1:3), R_obsv, obsv_lat, LST_true);
rho_obsv_true_motion = observer_range(rho_true_motion, obsv_lat, LST_true);
h_cords_true_motion = horizontal_coordinates(rho_obsv_true_motion);

% converting true trajectory from spherical (horizontal) frame to cartesian
rho_true = h_cords_true_motion(:, 1);
az_true = h_cords_true_motion(:, 2);
el_true = h_cords_true_motion(:, 3);
[x_obsv, y_obsv, z_obsv] = sph2cart(az_true, el_true, rho_true);

% Decides to actually plot
plot_measurements = false;
if plot_measurements
    plotting_measurements;
end

%% Plotting Error For EKF

plot_ekf = true;
if plot_ekf
    plotting_ekf;
end

%% Function Definitions

% Function for true twobody motion
function ydot = twobody(t, y, mu)
    r = y(1:3);
    v = y(4:6);
    ydot = [v; -mu/norm(r)^3*r];
end

% Function to convert from twobody relative position to inertial range
function rho = inertial_range(r, R, obsv_lat, LST)
    transformation = R * [cos(obsv_lat)*cos(LST); cos(obsv_lat)*sin(LST); sin(obsv_lat)*ones(size(LST))]';
    rho = r - transformation;
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
    az = wrapTo2Pi(atan2(rho_obsv(:, 2), rho_obsv(:, 3)));
    el = asin(rho_obsv(:, 1)./rho_mag);
    h_cords = [rho_mag, az, el];
end

% Function adds multi-variate gaussian noise 
function noisy_measurements = generate_measurements(h_cords, R, n)
    noise = mvnrnd(zeros(1,3), R, n);
    noisy_measurements = h_cords + noise;
end

% Function to integrate Two Body mechanics and Evolution of covariance
function ydot = combined_dynamics(t, y, Q, mu)
    r = y(1:3);
    v = y(4:6);
    xdot = [v; -mu/norm(r)^3*r];
    
    % Computing State Jacobian
    r_mag = norm(r);
    F = zeros(6, 6);
    % dv/dv = I
    F(1:3, 4:6) = eye(3);
    F(4:6, 1:3) = 3*mu/r_mag^5 * (r*r') - mu/r_mag^3 * eye(3);

    % Covariance Dynamics
    P = reshape(y(7:42), 6, 6);
    Pdot = F * P + P * F' + Q;

    % Combined state dynamics
    ydot = [xdot; Pdot(:)];
end

% EKF
function [xhat, P_out] = EKF(x0, ytilde, tmeas, Q, R, mu, obsv_lat, LST, R_obsv)
    
    n_meas = length(tmeas);
    
    % Creating variables to store results
    xhat = zeros(n_meas, 6); 
    P_out = zeros(6, 6, n_meas);
    
    % Initializing
    xhat_minus = x0;
    P_minus = diag([1E6, 1E6, 1E6, 1E2, 1E2, 1E2]);

    % Iterating through measurements
    for k = 1:n_meas
        % Computing H_k and expected measurements
        H_k = compute_H(xhat_minus, obsv_lat, LST(k), R_obsv);
        
        r_inertial = xhat_minus(1:3);
        rho_inertial = inertial_range(r_inertial', R_obsv, obsv_lat, LST(k));
        rho_obsv = observer_range(rho_inertial, obsv_lat, LST(k));
        y_prediction = horizontal_coordinates(rho_obsv)';

        % Computing Kalman Gain
        K_k = P_minus * H_k' / (H_k * P_minus * H_k' + R);
    
        % Updating state and covariance estimates
        xhat_plus = xhat_minus + K_k * (ytilde(k, :)' - y_prediction);
        P_plus = (eye(6) - K_k*H_k) * P_minus * (eye(6) - K_k*H_k)' + K_k * R * K_k'; 

        % Storing results
        xhat(k, :) = xhat_plus';
        P_out(:, :, k) = P_plus;

        % Propagating to the next time step
        if k < n_meas
            % Defining time span and ODE options
            tspan = [tmeas(k), tmeas(k+1)];
            options = odeset('RelTol',1E-8, 'AbsTol',1E-10);

            % Propagating state and covariance
            initial_state = [xhat_plus; P_plus(:)];
            [~, state_prop] = ode45(@(t, y) combined_dynamics(t, y, Q, mu), tspan, initial_state, options);
            next_step = state_prop(end, :)';
            xhat_minus = next_step(1:6);
            P_minus = reshape(next_step(7:42), 6, 6);
            
        end
    end
end

% This function calculates the partials used for the EKF
function H_k = compute_H(x, obsv_lat, LST, R_obsv)

    % H_k = [H_11, 0];
    % H_11 = [drho/dx; daz/dx; del/dx]

    % Defining measurement values
    r_inertial = x(1:3);
    rho_inertial = inertial_range(r_inertial', R_obsv, obsv_lat, LST);
    rho_obsv = observer_range(rho_inertial, obsv_lat, LST);

    % Extract components
    rho_u = rho_obsv(1);
    rho_e = rho_obsv(2);
    rho_n = rho_obsv(3);
    rho_mag = norm(rho_obsv);

    % Preallocating H11
    H11 = zeros(3, 3);
    
    % First Row
    H11(1, 1) = (rho_u * cos(obsv_lat) * cos(LST) - rho_e * sin(LST) - rho_n * sin(obsv_lat) * cos(LST)) / rho_mag;
    H11(1, 2) = (rho_u * cos(obsv_lat) * sin(LST) + rho_e * cos(LST) - rho_n * sin(obsv_lat) * sin(LST)) / rho_mag;
    H11(1, 3) = (rho_u * sin(obsv_lat) + rho_n * cos(obsv_lat)) / rho_mag;
    
    % Second Row
    denom_az = rho_n^2 + rho_e^2;
    H11(2, 1) = (rho_e * sin(obsv_lat) * cos(LST) - rho_n * sin(LST)) / denom_az;
    H11(2, 2) = (rho_e * sin(obsv_lat) * sin(LST) + rho_n * cos(LST)) / denom_az;
    H11(2, 3) = -rho_e * cos(obsv_lat) / denom_az;
    
    % Third Row
    denom_el = rho_mag * sqrt(rho_mag^2 - rho_u^2);
    H11(3, 1) = (rho_mag * cos(obsv_lat) * cos(LST) - rho_u * H11(1, 1)) / denom_el;
    H11(3, 2) = (rho_mag * cos(obsv_lat) * sin(LST) - rho_u * H11(1, 2)) / denom_el;
    H11(3, 3) = (rho_mag * sin(obsv_lat) - rho_u * H11(1, 3)) / denom_el;
    
    H_k = [H11, zeros(3, 3)];
end