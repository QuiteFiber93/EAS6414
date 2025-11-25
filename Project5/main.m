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

% Figure Plotting true trajectory
% Creating subplots for x, y, and z
fig = tiledlayout(3, 1);
ax1 = nexttile;
plot(ax1, t, true_motion(:, 1))
ylabel('x(t) (km)')

ax2 = nexttile;
plot(ax2, t, true_motion(:, 2))
ylabel('y(t) (km)')

ax3 = nexttile;
plot(ax3, t, true_motion(:, 3))
ylabel('z(t) (km)')
xlabel('t (s)')
linkaxes([ax1, ax2, ax3], 'x')
title(fig, 'Earth Fixed Position');


% Plotting x, y, and z velocities in subplots and saving the figure
fig = tiledlayout(3, 1);
ax1 = nexttile;
plot(ax1, t, true_motion(:, 4))
ylabel('xdot(t) (km/s)')

ax2 = nexttile;
plot(ax2, t, true_motion(:, 5))
ylabel('ydot(t) (km/s)')

ax3 = nexttile;
plot(ax3, t, true_motion(:, 6))
ylabel('zdot(t) (km/s)')
xlabel('t (s)')
title(fig, 'Earth Fixed Velocity');


figure
% converting true trajectory from spherical (horizontal) frame to cartesian
rho_true = h_cords_true_motion(:, 1);
az_true = h_cords_true_motion(:, 2);
el_true = h_cords_true_motion(:, 3);
[x_obsv, y_obsv, z_obsv] = sph2cart(az_true, el_true, rho_true);
plot3(x_obsv, y_obsv, z_obsv, 'DisplayName','True Trajectory');
hold on
[x_meas, y_meas, z_meas] = sph2cart(measurements(:, 2), measurements(:, 3), measurements(:, 1));
scatter3(x_meas, y_meas, z_meas, 5, 'filled', 'DisplayName', 'Measured Positions');
hold off
title('True Trajectory vs Measured States')
grid()
legend('Location','northeast')
xlabel('x (km)')
ylabel('y (km)')
zlabel('z (km)')

% Saving plot

% Plotting x, y, z trajectory relative to the observer vs time and
% overlaying measurements
figure;
ax1 = subplot(3, 1, 1);
hold on
plot(ax1, t, x_obsv, 'DisplayName','True')
scatter(ax1, tmeas, x_meas, 5, 'filled', 'DisplayName','Measured')
hold off
ylabel('x(t) (km)')
legend('Location','southeast')

ax2 = subplot(3, 1, 2);
hold on
plot(ax2, t, y_obsv, 'DisplayName','True')
scatter(ax2, tmeas, y_meas, 5, 'filled', 'DisplayName','Measured')
hold off
ylabel('y(t) (km)')
legend('Location','southeast')

ax3 = subplot(3, 1, 3);
hold on
plot(ax3, t, z_obsv, 'DisplayName','True')
scatter(ax3, tmeas, z_meas, 5, 'filled', 'DisplayName','Measured')
hold off
ylabel('z(t) (km)')
xlabel('t (s)')
legend('Location','southeast')

sgtitle('Position of Satellite Relative to Observer')

% Creating a figure which shows the altitude and azimuth of the satellite
% overhead from frame of observer
figure
polarplot(az_true, 90 - rad2deg(el_true), 'DisplayName','True Trajectory')
hold on
polarscatter(measurements(:, 2), 90 - rad2deg(measurements(:, 3)), 5, 'filled', 'DisplayName','Measurements')
hold off
title('Polar Plot for Altitude and Azimuth')
rlim([0, 90])
ax = gca;
ax.RTick = 0:30:90;
ax.RTickLabel = {'90°', '60°', '30°', '0°'};

% Plotting Range vs Range Measurements
figure
plot(t, rho_true, 'DisplayName','True Range')
hold on
scatter(tmeas, measurements(:, 1), 5, 'filled', 'DisplayName','Range Measurements')
hold off
title('True Range vs Range Measurements')
xlabel('t (s)')
ylabel('Range (km)')
legend()


exportgraphics(gcf, 'Images/body_fixed_pos.png','Resolution',300)
exportgraphics(gcf, 'Images/body_fixed_vel.png','Resolution',300)
exportgraphics(gca, 'Images/measurements.png','Resolution',300)
exportgraphics(gcf, 'Images/cartesian_pos_measurements.png', 'Resolution',300)
exportgraphics(gcf, 'Images/alt_az_plot.png', 'Resolution',300)
exportgraphics(gcf, 'Images/range_plot.png', 'Resolution',300)

%% Plotting Error For EKF

figure
title('Position Error')
subplot(3, 1, 1)
hold on
plot(tmeas, ekf_error(:, 1), 'DisplayName','Error')
plot(tmeas, sigma_bounds(:, 1), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas, -sigma_bounds(:, 1), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
ylabel('x Error (km)')
legend()

subplot(3, 1, 2)
hold on
plot(tmeas, ekf_error(:, 2), 'DisplayName','Error')
plot(tmeas, sigma_bounds(:, 2), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas, -sigma_bounds(:, 2), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
ylabel('y Error (km)')
legend()

subplot(3, 1, 3)
hold on
plot(tmeas, ekf_error(:, 3), 'DisplayName','Error')
plot(tmeas, sigma_bounds(:, 3), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas, -sigma_bounds(:, 3), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
ylabel('z Error (km)')
legend()

figure
title('Velocity Error')
subplot(3, 1, 1)
hold on
plot(tmeas, ekf_error(:, 4), 'DisplayName','Error')
plot(tmeas, sigma_bounds(:, 4), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas, -sigma_bounds(:, 4), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
ylabel('xdot Error (km/s)')
legend()

subplot(3, 1, 2)
hold on
plot(tmeas, ekf_error(:, 5), 'DisplayName','Error')
plot(tmeas, sigma_bounds(:, 5), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas, -sigma_bounds(:, 5), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
ylabel('ydot Error (km/s)')
legend()

subplot(3, 1, 3)
hold on
plot(tmeas, ekf_error(:, 6), 'DisplayName','Error')
plot(tmeas, sigma_bounds(:, 6), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas, -sigma_bounds(:, 6), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
ylabel('zdot Error (km/s)')
legend()


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
    P_minus = eye(6) * 1E6;

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