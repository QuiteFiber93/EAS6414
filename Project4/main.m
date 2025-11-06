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
disp(delim_eq)
disp('GENERATING MEASUREMENTS')
disp(delim_eq)

func = @(t, y) twobody(t, y, mu);
options = odeset('RelTol',1E-8, 'AbsTol',1E-10);
[~, true_motion] = ode45(func, tspan, y0, options);
[tmeas, measured_states] = ode45(func, tmeas, y0, options);

% Converting from current frame to inertial then observer frame
measured_states_obsv = to_inertial_frame(measured_states(:, 1:3), R_obsv, obsv_lat, LST);
measured_states_obsv = inertial_to_obsv(measured_states_obsv, obsv_lat, LST);
% 
% % Converting to range, altitude, azimuth
h_cords = horizontal_cords(measured_states_obsv);
% 
% % Generating measurements from horizontal coordiantes
measurements = generate_measurements(h_cords, tmeas, R);

%% Single GLSDC Run
disp(delim_eq)
disp('GLSDC Run')
disp(delim_eq)

fprintf('\nInitial Guess\n')
disp(delim_dash)
% Initial Guess
r0hat = [6900; 1; 1];
v0hat = [1; 1; 1];
x0hat = [r0hat; v0hat];
disp(['x0hat = ', num2str(x0hat')])

fprintf('\nGLSDC Estimate\n')
disp(delim_dash)
dynamics = @(t, y) twobody_STM(t, y, mu);
glsdc(dynamics, x0hat, tmeas, measurements, R, 10, 1E-6)
%% Covariance of GLSDC Solution
% P

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
function rho_inertial = to_inertial_frame(r, R, obsv_lat, lst)
    rho_inertial = zeros(size(r));
    for k = 1:length(lst)
        translation = R *  [cos(obsv_lat) * cos(lst(k)), ...
                        cos(obsv_lat)*sin(lst(k)), ...
                        sin(lst(k))];
        rho_inertial(k, :) = r(k, :) - translation;
    end
end

% Function to convert from inertial frame to obsv frame
function rho_obsv = inertial_to_obsv(rho_inertial, obsv_lat, LST)
    rho_obsv = zeros(size(rho_inertial));
    rotation2 = [cos(obsv_lat), 0, sin(obsv_lat); ...
                0, 1, 0; ...
                -sin(obsv_lat), 0, cos(obsv_lat)];
    for k = 1:length(LST)
        rotation1 = [cos(LST(k)), sin(LST(k)), 0; -sin(LST(k)), cos(LST(k)), 0; 0, 0, 1];
        rho_obsv(k, :) = rotation2 * rotation1 * rho_inertial(k, :)';
    end
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
    A = [zeros(3), eye(3); -mu * (1/norm(r)^3 - 3/norm(r)^5*(r*r')), zeros(3)];
    Phidot = A * Phi;
    ydot(7:42) = Phidot(:);
end

% Function to implement the glsdc algorithm
function estimate = glsdc(dynamics, guess, tmeas, measurements, R, maxiter, tol)
    R_obsv = 6371; % Radius of Earth in km
    obsv_lat = deg2rad(5);
    LST0 = deg2rad(10);
    omega_E = 7.2921159E-5; % rad/s
    LST = LST0 + omega_E * tmeas;

    % Initialize for GLSDC
    Phi0 = eye(6); % Psi(t=t0) 
    W = inv(R); % Weight matrix
    
    for n = 1:maxiter
        % Appending Phi(t0) to initial guess
        initial_state = [guess; Phi0(:)];

        % Lambda*xhat = N
        % Lambda is information matrix
        info_matrix = zeros(6);
        N = zeros(6, 1);
        
        % TODO: Integrate twobody_STM, return states at tmeas
        options = odeset('RelTol',1E-8, 'AbsTol',1E-10);
        [~, state_estimates] = ode45(dynamics, tmeas, initial_state, options);

        % Convert states into estimated measurements
        inertial_pos = to_inertial_frame(state_estimates(:, 1:3), R_obsv, obsv_lat, LST);
        obsv_rel_pos = inertial_to_obsv(inertial_pos, obsv_lat, LST);
        h_cords = horizontal_cords(obsv_rel_pos);
        estimated_measurements = generate_measurements(h_cords, tmeas, R);
        
        % Use estimated measurements to calculate error
        err = measurements - estimated_measurements;
        
        % Calculating H_k, Lambda, N
        % Lambda = Lambda + H_k' * W * H_k
        % N = N +H_k' * W * err
        % deltaX = solve(Lambda, N)
        for k = 1:length(tmeas)
            % H_k = dh/dx * Phi
            % Calculating dh/dx
            rho_mag = norm(obsv_rel_pos(k, :));
            rho_u = obsv_rel_pos(k, 1);
            rho_e = obsv_rel_pos(k, 2);
            rho_n = obsv_rel_pos(k, 3);
            
            dpdx = (rho_u * cos(obsv_lat)*cos(LST(k)) - rho_e * sin(LST(k)) - rho_n * sin(obsv_lat)*cos(LST(k)))/rho_mag;
            dpdy = (rho_u * cos(obsv_lat)*sin(LST(k)) + rho_e * cos(LST(k)) - rho_n * sin(obsv_lat)*sin(LST(k)))/rho_mag;
            dpdz = (rho_u * sin(obsv_lat) + rho_n * cos(obsv_lat))/rho_mag;

            dazdx = (rho_e * sin(obsv_lat)*cos(LST(k)) - rho_n * sin(LST(k)))/ (rho_n^2 + rho_e^2);
            dazdy = (rho_e * sin(obsv_lat)*sin(LST(k)) + rho_n * cos(LST(k)))/ (rho_n^2 + rho_e^2);
            dazdz = -(rho_e * cos(obsv_lat))/ (rho_n^2 + rho_e^2);

            deldx = (rho_mag * cos(obsv_lat) * cos(LST(k)) - rho_u * dpdx) / (rho_mag * sqrt(rho_mag^2 - rho_u^2));
            deldy = (rho_mag * cos(obsv_lat) * sin(LST(k)) - rho_u * dpdy) / (rho_mag * sqrt(rho_mag^2 - rho_u^2));
            deldz = (rho_mag * sin(obsv_lat) - rho_u * dpdz) / (rho_mag * sqrt(rho_mag^2 - rho_u^2));

            dhdr = [dpdx, dpdy, dpdz; dazdx, dazdy, dazdz; deldx, deldy, deldz];
            dhdx = [dhdr, zeros(3)];
            % Extracting Phi from state estimates
            Phi_k = reshape(state_estimates(k, 7:42), 6, 6);
            H_k = dhdx * Phi_k;
            info_matrix = info_matrix + H_k' * W * H_k;
            N = N + H_k' * W * err(k, :)';
        end
        
        % Use Lambda and N to solve for deltaX
        deltaX = info_matrix \ N;

        % Check if error is converged to tolerance
        if abs(err) <= tol
            break;
        end

        guess = guess + deltaX;

    end

    estimate = guess;
end