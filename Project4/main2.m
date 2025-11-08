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
fprintf('Sample measurements:\n');
fprintf('  Range: %.3f km\n', measurements(1,1));
fprintf('  Azimuth: %.6f rad (%.3f deg)\n', measurements(1,2), rad2deg(measurements(1,2)));
fprintf('  Elevation: %.6f rad (%.3f deg)\n', measurements(1,3), rad2deg(measurements(1,3)));
%% GLSDC - CORRECTED INITIAL GUESS
r0hat = [6990; 1; 1];  % CORRECTED: Must match project specification
v0hat = [1; 1; 1];
x0hat = [r0hat; v0hat];

maxiter = 50;  % Increased max iterations
tol = 1E-5;
dynamics = @(t, y) twobody_STM(t, y, mu);
[estimate, Lambda, iteration_history] = glsdc(dynamics, x0hat, measurements, tol, R, tmeas, maxiter, R_obsv, LST, obsv_lat);

fprintf('\n%s\n', delim_eq);
fprintf('GLSDC CONVERGENCE RESULTS\n');
fprintf('%s\n', delim_eq);
fprintf('Converged in %d iterations\n\n', length(iteration_history));

fprintf('Iteration History:\n');
fprintf('%-4s  %-15s  %-15s  %-15s\n', 'Iter', 'Cost', '||delta_x||', 'Change in Cost');
fprintf('%s\n', delim_dash);
for i = 1:length(iteration_history)
    if i == 1
        fprintf('%-4d  %-15.6e  %-15.6e  %-15s\n', ...
            iteration_history(i).iter, ...
            iteration_history(i).cost, ...
            iteration_history(i).delta_norm, ...
            'N/A');
    else
        cost_change = abs(iteration_history(i).cost - iteration_history(i-1).cost);
        fprintf('%-4d  %-15.6e  %-15.6e  %-15.6e\n', ...
            iteration_history(i).iter, ...
            iteration_history(i).cost, ...
            iteration_history(i).delta_norm, ...
            cost_change);
    end
end
fprintf('%s\n\n', delim_dash);

fprintf('True Initial Conditions:\n');
fprintf('  r0 = [%.3f, %.3f, %.3f] km\n', r0(1), r0(2), r0(3));
fprintf('  v0 = [%.3f, %.3f, %.3f] km/s\n\n', v0(1), v0(2), v0(3));

fprintf('Estimated Initial Conditions:\n');
fprintf('  r0_hat = [%.3f, %.3f, %.3f] km\n', estimate(1), estimate(2), estimate(3));
fprintf('  v0_hat = [%.3f, %.3f, %.3f] km/s\n\n', estimate(4), estimate(5), estimate(6));

fprintf('Estimation Errors:\n');
fprintf('  Position error = [%.3f, %.3f, %.3f] km\n', estimate(1:3)' - r0');
fprintf('  Velocity error = [%.3f, %.3f, %.3f] km/s\n', estimate(4:6)' - v0');
fprintf('  ||Position error|| = %.3f km\n', norm(estimate(1:3) - r0));
fprintf('  ||Velocity error|| = %.3f km/s\n\n', norm(estimate(4:6) - v0));

% Compute covariance
P = inv(Lambda);
sigma = sqrt(diag(P));

fprintf('1-sigma Uncertainties:\n');
fprintf('  Position: [%.3f, %.3f, %.3f] km\n', sigma(1), sigma(2), sigma(3));
fprintf('  Velocity: [%.6f, %.6f, %.6f] km/s\n', sigma(4), sigma(5), sigma(6));
fprintf('%s\n', delim_eq);

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
    az = atan2(rho_obsv(:, 2), rho_obsv(:, 3));
    el = asin(rho_obsv(:, 1)./rho_mag);
    h_cords = [rho_mag, az, el];
end

% Function adds multi-variate gaussian noise 
function noisy_measurements = generate_measurements(h_cords, R, n)
    noise = mvnrnd(zeros(1,3), R, n);
    noisy_measurements = h_cords + noise;
end

% Function to integrate Two Body mechanics and STM ODE
function ydot = twobody_STM(t, y, mu)
    r = y(1:3);
    v = y(4:6);
    Phi = reshape(y(7:42), 6, 6);

    % Phidot = df/dx * Phi
    % A = df/dx = [dv/dr, dv/dv; dvdot/dr, dvdot/dv]
    %           = [0, I; A_21, 0]
    r_norm = norm(r);
    A = [zeros(3), eye(3); 
         3*mu/r_norm^5*(r*r') - mu/r_norm^3*eye(3), zeros(3)];
    Phidot = A * Phi;
    ydot = [v; -mu*r/r_norm^3; Phidot(:)];
end

% Function for GLSDC Iteration - CORRECTED VERSION
function [estimate, Lambda, iteration_history] = glsdc(dynamics, guess, measurements, tol, R, tmeas, maxiter, R_obsv, LST, obsv_lat)

    % Initializing relevant values
    W = inv(R); % 3x3
    Phi0 = eye(6);

    % Cost for iteration convergence
    old_cost = inf;
    new_cost = 0;
    
    % Store iteration history
    iteration_history = struct('iter', {}, 'cost', {}, 'delta_norm', {});

    for n = 1:maxiter

        % Lambda and N for normal equations
        Lambda = zeros(6);
        N = zeros(6, 1);

        % Initial state including r, v, and STM
        initial_state_guess = [guess; Phi0(:)];

        % Integrate dynamics and STM
        options = odeset('RelTol',1E-8, 'AbsTol',1E-10);
        [~, state_estimates] = ode45(dynamics, tmeas, initial_state_guess, options);

        % Converting to inertial, then obsv, then horizontal coordinates to
        % arrived at measurement estimate
        rho = inertial_range(state_estimates(:, 1:3), R_obsv, obsv_lat, LST);
        rho_obsv = observer_range(rho, obsv_lat, LST);
        estimated_measurements = horizontal_coordinates(rho_obsv);

        % Using estimated measurements to calculate error
        % CRITICAL FIX: Handle azimuth wraparound
        err = measurements - estimated_measurements; % 11 x 3
        
        % Wrap azimuth error to [-pi, pi]
        err(:, 2) = wrapToPi(err(:, 2));

        % Going through and calculating Lambda and N
        new_cost = 0;
        for k = 1:length(tmeas)
            % Calculate H_k = dh/dx * Phi
            % Expressions for dh/dx
            % dh/dx = [dh/dr, dh/dv]; dh/dv = 0
            dh_dx = zeros(3, 6);
            rho_u = rho_obsv(k, 1);
            rho_e = rho_obsv(k, 2);
            rho_n = rho_obsv(k, 3);
            rho_mag = estimated_measurements(k, 1);
            LST_k = LST(k);

            % d(rho) / dr
            drho_dx = (rho_u*cos(obsv_lat)*cos(LST_k) - rho_e*sin(LST_k) - rho_n*sin(obsv_lat)*cos(LST_k))/rho_mag;
            drho_dy = (rho_u*cos(obsv_lat)*sin(LST_k) + rho_e*cos(LST_k) - rho_n*sin(obsv_lat)*sin(LST_k))/rho_mag;
            drho_dz = (rho_u*sin(obsv_lat) + rho_n*cos(obsv_lat))/rho_mag;

            % d(az) / dr
            denom_az = rho_n^2 + rho_e^2;
            daz_dx = (rho_e*sin(obsv_lat)*cos(LST_k) - rho_n*sin(LST_k)) / denom_az;
            daz_dy = (rho_e*sin(obsv_lat)*sin(LST_k) + rho_n*cos(LST_k)) / denom_az;
            daz_dz = -rho_e*cos(obsv_lat) / denom_az;

            % d(el) / dr
            denom_el = rho_mag*sqrt(rho_mag^2 - rho_u^2);
            del_dx = (rho_mag*cos(obsv_lat)*cos(LST_k) - rho_u*drho_dx) / denom_el;
            del_dy = (rho_mag*cos(obsv_lat)*sin(LST_k) - rho_u*drho_dy) / denom_el;
            del_dz = (rho_mag*sin(obsv_lat) - rho_u*drho_dz) / denom_el;

            % Combining
            dh_dx(1:3, 1:3) = [drho_dx, drho_dy, drho_dz; 
                               daz_dx, daz_dy, daz_dz; 
                               del_dx, del_dy, del_dz];

            % Extracting Phi
            Phi_k = reshape(state_estimates(k, 7:42), 6, 6);

            % H_k
            H_k = dh_dx * Phi_k;

            % Building Lambda and N
            Lambda = Lambda + H_k' * W * H_k;
            N = N + H_k' * W * err(k, :)';

            % Adding to cost function
            new_cost = new_cost + err(k, :) * W * err(k, :)';
        end

        % Solve for delta_x with regularization to prevent singular matrix
        delta_x = (Lambda + 1e-10*eye(6)) \ N;
        
        % Store iteration info
        iteration_history(n).iter = n;
        iteration_history(n).cost = new_cost;
        iteration_history(n).delta_norm = norm(delta_x);

        % Check for convergence
        if n > 1 && abs(new_cost - old_cost) < tol
            break;
        end

        % Updating guess
        guess = guess + delta_x;
        old_cost = new_cost;

    end

    estimate = guess;
end