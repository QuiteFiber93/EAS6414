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

exportgraphics(gcf, 'Images/body_fixed_pos.png','Resolution',300)

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

exportgraphics(gcf, 'Images/body_fixed_vel.png','Resolution',300)

figure
% converting true trajectory from spherical (horizontal) frame to cartesian
rho_true = h_cords_true_motion(:, 1);
az_true = h_cords_true_motion(:, 2);
el_true = h_cords_true_motion(:, 3);
[x_obsv, y_obsv, z_obsv] = sph2cart(az_true, el_true, rho_true);
plot3(x_obsv, y_obsv, z_obsv, 'DisplayName','True Trajectory');
hold on
[x_meas, y_meas, z_meas] = sph2cart(measurements(:, 2), measurements(:, 3), measurements(:, 1));
scatter3(x_meas, y_meas, z_meas, 10, 'filled', 'DisplayName', 'Measured Positions');
hold off
title('True Trajectory vs Measured States')
grid()
legend('Location','northeast')
xlabel('x (km)')
ylabel('y (km)')
zlabel('z (km)')

% Saving plot
exportgraphics(gca, 'Images/measurements.png','Resolution',300)

% Plotting x, y, z trajectory relative to the observer vs time and
% overlaying measurements
figure;
ax1 = subplot(3, 1, 1);
hold on
plot(ax1, t, x_obsv, 'DisplayName','True')
scatter(ax1, tmeas, x_meas, 10, 'filled', 'DisplayName','Measured')
hold off
ylabel('x(t) (km)')
legend('Location','southeast')

ax2 = subplot(3, 1, 2);
hold on
plot(ax2, t, y_obsv, 'DisplayName','True')
scatter(ax2, tmeas, y_meas, 10, 'filled', 'DisplayName','Measured')
hold off
ylabel('y(t) (km)')
legend('Location','southeast')

ax3 = subplot(3, 1, 3);
hold on
plot(ax3, t, z_obsv, 'DisplayName','True')
scatter(ax3, tmeas, z_meas, 10, 'filled', 'DisplayName','Measured')
hold off
ylabel('z(t) (km)')
xlabel('t (s)')
legend('Location','southeast')

sgtitle('Position of Satellite Relative to Observer')
exportgraphics(gcf, 'Images/cartesian_pos_measurements.png', 'Resolution',300)

% Creating a figure which shows the altitude and azimuth of the satellite
% overhead from frame of observer
figure
polarplot(az_true, 90 - rad2deg(el_true), 'DisplayName','True Trajectory')
hold on
polarscatter(measurements(:, 2), 90 - rad2deg(measurements(:, 3)), 7, 'filled', 'DisplayName','Measurements')
hold off
title('Polar Plot for Altitude and Azimuth')
rlim([0, 90])
ax = gca;
ax.RTick = 0:30:90;
ax.RTickLabel = {'90°', '60°', '30°', '0°'};
exportgraphics(gcf, 'Images/alt_az_plot.png', 'Resolution',300)

% Plotting Range vs Range Measurements
figure
plot(t, rho_true, 'DisplayName','True Range')
hold on
scatter(tmeas, measurements(:, 1), 10, 'filled', 'DisplayName','Range Measurements')
hold off
title('True Range vs Range Measurements')
xlabel('t (s)')
ylabel('Range (km)')
legend()
exportgraphics(gcf, 'Images/range_plot.png', 'Resolution',300)


%% GLSDC
r0hat = [6990; 1; 1];
v0hat = [1; 1; 1];
x0hat = [r0hat; v0hat];

maxiter = 10;
tol = 1E-3;
dynamics = @(t, y) twobody_STM(t, y, mu);
[glsdc_estimate, Lambda] = glsdc(dynamics, x0hat, measurements, tol, R, tmeas, maxiter, R_obsv, LST, obsv_lat);

%% Get Covariance of GLSDC Estimate
% P is the inverse of the covariance matrix
P_glsdc = inv(Lambda);

%% Monte Carlo Simulations for GLSDC
nruns = 1000;
mc_results = zeros(nruns, 6);

for n = 1:nruns
    measurements = generate_measurements(h_cords, R, length(tmeas));
    [mc_estimate, Lambda] = glsdc(dynamics, x0hat, measurements, tol, R, tmeas, maxiter, R_obsv, LST, obsv_lat);
    mc_results(n, :) = mc_estimate';
end

% Calculating mean and covariance of estimates from monte carlo runs
monte_carlo_estimate = mean(mc_results)';
P_mc = cov(mc_results);

%% Plotting the Trajectories according to the MC and GLSDC Estimates
% Integrating using estimates
func = @(t, y) twobody(t, y, mu);
options = odeset('RelTol',1E-8, 'AbsTol',1E-10);
[t, true_traj] = ode45(func, tspan, y0, options);
[t, glsdc_traj] = ode45(func, t, glsdc_estimate, options);
[t, monte_carlo_traj] = ode45(func, t, monte_carlo_estimate, options);

% This figure will create a 3d plot of all 3 trajectories, the save it
figure
plot3(true_traj(:, 1), true_traj(:, 2), true_traj(:, 3), 'DisplayName', 'True')
hold on
plot3(glsdc_traj(:, 1), glsdc_traj(:, 2), glsdc_traj(:, 3), 'DisplayName', 'GLSDC')
plot3(monte_carlo_traj(:, 1), monte_carlo_traj(:, 2), monte_carlo_traj(:, 3), 'DisplayName', 'Monte Carlo')
hold off
title('Trajectory Comparison')
xlabel('x (km)')
ylabel('y (km)')
zlabel('z (km)')
legend('Location','northeast')
grid()
exportgraphics(gcf, 'Images/trajectory_comparison.png', 'Resolution', 300)

% This figure will create a 3d plot showing the errors in trajectory for
% the GLSDC and Monte Carlo Estimates, then save it
figure
plot3(true_motion(:, 1) - glsdc_traj(:, 1), true_motion(:, 2) - glsdc_traj(:, 2), true_motion(:, 3) - glsdc_traj(:, 3), 'DisplayName', 'GLSDC')
hold on
plot3(true_motion(:, 1) - monte_carlo_traj(:, 1), true_motion(:, 2) - monte_carlo_traj(:, 2), true_motion(:, 3) - monte_carlo_traj(:, 3), 'DisplayName', 'Monte Carlo')
hold off
title('Estimate Trajectory Error')
xlabel('$\Delta x$ (km)', 'Interpreter','latex')
ylabel('$\Delta y$ (km)', 'Interpreter','latex')
zlabel('$\Delta z$ (km)', 'Interpreter','latex')
legend('Location','northeast')
grid()
exportgraphics(gcf, 'Images/estimate_trajectory_error.png', 'Resolution', 300)

% This figure will create a series of plots which show the estimated
% trajectory errors for x, y, and z, then save it
% It uses the tiledlayout, which is like subplot
fig = tiledlayout(3, 1);
nexttile;
hold on
plot(t, true_motion(:, 1) - glsdc_traj(:, 1), 'DisplayName','GLSDC')
plot(t, true_motion(:, 1) - monte_carlo_traj(:, 1), 'DisplayName','Monte Carlo')
hold off
title('$x(t)$ Error', 'Interpreter','latex')
ylabel('$\Delta x$ (km)', 'Interpreter','latex')
legend()
grid()

nexttile;
hold on
plot(t, true_motion(:, 2) - glsdc_traj(:, 2), 'DisplayName','GLSDC')
plot(t, true_motion(:, 2) - monte_carlo_traj(:, 2), 'DisplayName','Monte Carlo')
hold off
title('$y(t)$ Error', 'Interpreter','latex')
ylabel('$\Delta y$ (km)', 'Interpreter','latex')
legend()
grid()

nexttile;
hold on
plot(t, true_motion(:, 3) - glsdc_traj(:, 3), 'DisplayName','GLSDC')
plot(t, true_motion(:, 3) - monte_carlo_traj(:, 3), 'DisplayName','Monte Carlo')
hold off
title('$z(t)$ Error', 'Interpreter','latex')
xlabel('t (s)')
ylabel('$\Delta z$ (km)', 'Interpreter','latex')
legend()
grid()
sgtitle('Trajectory Estimate Position Error')
exportgraphics(gcf, 'Images/traj_estimate_pos_err.png', 'Resolution',300)

% This figure will create a series of plots which show the estimated
% trajectory errors for x, y, and z velocities, then save it
% It uses the tiledlayout, which is like subplot
fig = tiledlayout(3, 1);
nexttile;
hold on
title('$\dot{x}(t)$ Error', 'Interpreter','latex')
plot(t, true_motion(:, 4) - glsdc_traj(:, 4), 'DisplayName','GLSDC')
plot(t, true_motion(:, 4) - monte_carlo_traj(:, 4), 'DisplayName','Monte Carlo')
hold off
ylabel('$\Delta \dot{z}$ (km)', 'Interpreter','latex')
legend()
grid()

nexttile;
hold on
plot(t, true_motion(:, 5) - glsdc_traj(:, 5), 'DisplayName','GLSDC')
plot(t, true_motion(:, 5) - monte_carlo_traj(:, 5), 'DisplayName','Monte Carlo')
hold off
title('$\dot{y}(t)$ Error', 'Interpreter','latex')
ylabel('$\Delta \dot{y}$ (km)', 'Interpreter','latex')
legend()
grid()

nexttile;
hold on
plot(t, true_motion(:, 6) - glsdc_traj(:, 6), 'DisplayName','GLSDC')
plot(t, true_motion(:, 6) - monte_carlo_traj(:, 6), 'DisplayName','Monte Carlo')
hold off
title('$\dot{z}(t)$ Error', 'Interpreter','latex')
xlabel('t (s)')
ylabel('$\Delta \dot{z}$ (km)', 'Interpreter','latex')
legend()
grid()
sgtitle('Trajectory Estimate Velocity Error')
exportgraphics(gcf, 'Images/traj_estimate_vel_err.png', 'Resolution',300)

%% Plotting Covariance Ellipses and covariance ellipsoid

% Number of points to generate on ellipse
npoints = 60;

% Plotting Covariances for y-z
figure

% Index tells to use x, y, or z [1, 2, or 3]
idx = [2,3];

hold on
% Scale is the number of standard deviations
for scale = 1:3
    glsdc_ellipse = cov_ellipse(glsdc_estimate(idx), P_glsdc(idx, idx), scale, npoints);
    plot(glsdc_ellipse(1, :), glsdc_ellipse(2, :), 'b')

    monte_carlo_ellipse = cov_ellipse(monte_carlo_estimate(idx), P_mc(idx, idx), scale, npoints);
    plot(monte_carlo_ellipse(1, :), monte_carlo_ellipse(2, :), 'r')
end

scatter(glsdc_estimate(idx(1)), glsdc_estimate(idx(2)), 'b', 'filled', 'DisplayName','GLSDC Estimate')
scatter(monte_carlo_estimate(idx(1)), monte_carlo_estimate(idx(2)), [], 'r', 'filled')
scatter(y0(idx(1)), y0(idx(2)), 50, 'k', '+', 'DisplayName', 'True Initial State', 'LineWidth',2)
hold off
title('Initial Position Estimate in y-z Plane')
xlabel('y (km)')
ylabel('z (km)')
legend('GLSDC', '', '', 'Monte Carlo', '', '', '', '', 'True Initial State')
exportgraphics(gca, 'Images/cov_yz.png', 'Resolution',300);

% Plotting covariance for x-z
figure

% Index tells to use x, y, or z [1, 2, or 3]
idx = [1,3];

hold on
% Scale is the number of standard deviations
for scale = 1:3
    glsdc_ellipse = cov_ellipse(glsdc_estimate(idx), P_glsdc(idx, idx), scale, npoints);
    plot(glsdc_ellipse(1, :), glsdc_ellipse(2, :), 'b')

    monte_carlo_ellipse = cov_ellipse(monte_carlo_estimate(idx), P_mc(idx, idx), scale, npoints);
    plot(monte_carlo_ellipse(1, :), monte_carlo_ellipse(2, :), 'r')
end
scatter(glsdc_estimate(idx(1)), glsdc_estimate(idx(2)), 'b', 'filled', 'DisplayName','GLSDC Estimate')
scatter(monte_carlo_estimate(idx(1)), monte_carlo_estimate(idx(2)), [], 'r', 'filled')
scatter(y0(idx(1)), y0(idx(2)), 50, 'k', '+', 'DisplayName', 'True Initial State', 'LineWidth',2)

title('Initial Position Estimate in x-z Plane')
xlabel('x (km)')
ylabel('z (km)')
legend('GLSDC', '', '', 'Monte Carlo', '', '', '', '', 'True Initial State')
exportgraphics(gca, 'Images/cov_xz.png', 'Resolution',300);

% Plotting covariance for x-y
figure

% Index tells to use x, y, or z [1, 2, or 3]
idx = [1,2];

hold on
% Scale is the number of standard deviations
for scale = 1:3
    glsdc_ellipse = cov_ellipse(glsdc_estimate(idx), P_glsdc(idx, idx), scale, npoints);
    plot(glsdc_ellipse(1, :), glsdc_ellipse(2, :), 'b')

    monte_carlo_ellipse = cov_ellipse(monte_carlo_estimate(idx), P_mc(idx, idx), scale, npoints);
    plot(monte_carlo_ellipse(1, :), monte_carlo_ellipse(2, :), 'r')
end
scatter(glsdc_estimate(idx(1)), glsdc_estimate(idx(2)), 'b', 'filled', 'DisplayName','GLSDC Estimate')
scatter(monte_carlo_estimate(idx(1)), monte_carlo_estimate(idx(2)), [], 'r', 'filled')
scatter(y0(idx(1)), y0(idx(2)), 50, 'k', '+', 'DisplayName', 'True Initial State', 'LineWidth',2)

title('Initial Position Estimate in x-y Plane')
xlabel('x (km)')
ylabel('y (km)')
legend('GLSDC', '', '', 'Monte Carlo', '', '', '', '', 'True Initial State')
exportgraphics(gca, 'Images/cov_xy.png', 'Resolution',300);

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

% Function to integrate Two Body mechanics and STM ODE
function ydot = twobody_STM(t, y, mu)
    r = y(1:3);
    v = y(4:6);
    Phi = reshape(y(7:42), 6, 6);

    % Phidot = df/dx * Phi
    % A = df/dx = [dv/dr, dv/dv; dvdot/dr, dvdot/dv]
    %           = [0, I; A_21, 0]
    A = [zeros(3), eye(3); 3*mu/norm(r)^5*(r*r') - mu/norm(r)^3*eye(3), zeros(3)];
    Phidot = A * Phi;
    ydot = [v; -mu*r/norm(r)^3; Phidot(:)];
end

% Function for sinlge GLSDC Iteration
function [estimate, Lambda] = glsdc(dynamics, guess, measurements, tol, R, tmeas, maxiter, R_obsv, LST, obsv_lat)

% Initializing relevant values
W = inv(R); % 3x3
Phi0 = eye(6);

% Cost for iteration convergence
old_cost = inf;

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
    err = measurements - estimated_measurements; % 11 x 3

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
        daz_dx = (rho_e*sin(obsv_lat)*cos(LST_k) - rho_n*sin(LST_k)) / (rho_n^2 + rho_e^2);
        daz_dy = (rho_e*sin(obsv_lat)*sin(LST_k) + rho_n*cos(LST_k)) / (rho_n^2 + rho_e^2);
        daz_dz = -rho_e*cos(obsv_lat) / (rho_n^2 + rho_e^2);

        % d(el) / dr
        del_dx = (rho_mag*cos(obsv_lat)*cos(LST_k) - rho_u*drho_dx) / ( rho_mag*sqrt(rho_mag^2 - rho_u^2) );
        del_dy = (rho_mag*cos(obsv_lat)*sin(LST_k) - rho_u*drho_dy) / ( rho_mag*sqrt(rho_mag^2 - rho_u^2) );
        del_dz = (rho_mag*sin(obsv_lat) - rho_u*drho_dz) / ( rho_mag*sqrt(rho_mag^2 - rho_u^2) );

        % Combining
        dh_dx(1:3, 1:3) = [drho_dx, drho_dy, drho_dz; daz_dx, daz_dy, daz_dz; del_dx, del_dy, del_dz];

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

    % Check for convergence
    if n > 1 && abs(new_cost - old_cost)/new_cost < tol
        break;
    end

    % Solve for delta_x
    delta_x = Lambda \ N;

    % Updating guess
    guess = guess + delta_x;
        old_cost = new_cost;
    
end

estimate = guess;
end

% Function to generate ellipse from given covariance
function ellipse = cov_ellipse(mu, P, scale, n)
    % Parameter used to define ellispe
    t = linspace(0, 2*pi, n);

    % Base circle to be transformed
    ellipse = [cos(t); sin(t)];
    
    % Getting Eigenvalues and Eigenvectors of covariance
    [V, D] = eig(P);

    ellipse = V * scale * sqrt(D) * ellipse + mu;

end
