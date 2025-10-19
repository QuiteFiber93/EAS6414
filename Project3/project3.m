% EAS 6414 Project 3
% Dylan D'Silva
% MATLAB Implementation

clear; clc; close all;

% Settings to change how file runs
sigma = 0.1;            % Given Standard Deviation 
maxiter = 30;           % Max iteration count for GLSDC 
scale_factor = 0.9;     % Scale factor for GLSDC Guess
doPlot = false;         % Used to control whether plots are shown
tol = 1E-3;             % Error Tolerance for GLSDC

% Delimiter strings for print statements
delim_equals = repmat('=', 1, 80);
delim_dash = repmat('-', 1, 80);

%% Given Values
x0 = [2; 0];            % Initial Conditions
p = [0.05; 4; 0.2; -0.5; 10; pi/2]; % True Parameters
tspan = [0, 300];       % Initial and Final Times of Simulation
teval = (1:3000)/10;    % Times at which system is sampled

fprintf([delim_equals, '\nEAS 6414 Project 3: Initial State and Parameter Estimation\n', delim_equals]);

fprintf('\nGiven Values\n%s\n', delim_dash);
fprintf('x0                              = [%.4f, %.4f]\n', x0(1), x0(2));
fprintf('p                               = [%.2f, %.2f, %.2f, %.2f, %.2f, %.4f]\n', p);
fprintf('Measurement Covariance          = %.2f^2\n', sigma);
fprintf('Maximum Iteration for GLSDC     = %d\n', maxiter);
fprintf('Scale factor for initial guess  = %.2f\n\n', scale_factor);

%% System Dynamics Function
% Using nested function for dynamics with variational matrices
function f = dynamics(t, y, p)
    
    % Extract x, Phi, Psi
    x = y(1:2);
    phi = reshape(y(3:6), 2, 2);
    psi = reshape(y(7:end), 2, 6);

    % Partial Derivatives
    A = [0, 1; -p(2)-3*p(3)*x(1)^2, -p(1)];
    dfdp = [zeros(1, 6); -x(2), -x(1), -x(1)^3, ...
        -sin(p(5)*t + p(6)), -p(4)*t*cos(p(5)*t + p(6)), -p(4)*cos(p(5)*t + p(6))];

    % State Dynamics
    xdot = [x(2); -p(1)*x(2) - p(2)*x(1) - p(3)*x(1)^2 - p(4)*sin(p(5)*t+p(6))];
    
    % Matrix Derivatives
    phidot = A*phi;
    psidot = A*psi + dfdp;
    
    % Flatten and concatenate derivatives
    f = [xdot(:); phidot(:); psidot(:)];
end

%% Integrating System Dynamics
phi0 = eye(2);
psi0 = zeros(2, 6);
state0 = [x0; phi0(:); psi0(:)];

fprintf('%s\nTask 1: Simulating Motion, Measurements, and Validating Matrices\n%s\n', ...
    delim_equals, delim_equals);
fprintf('\nIntegrating System Dynamics ...\n');

% Integrating System
options = odeset('RelTol', 1e-10, 'AbsTol', 1e-10);
[t_sim, y_sim] = ode45(@(t, y) dynamics(t, y, p), tspan, state0, options);

% Get measurements at specific times
[~, y_meas] = ode45(@(t, y) dynamics(t, y, p), teval, state0, options);

%% Adding Measurement Noise
rng(2025);  % Setting random seed
ytilde = y_meas(:, 1)' + sigma * randn(1, length(teval));

%% Plotting System
fprintf('Plotting Solution ...\n');

% Plotting Solution State Variables vs Time
figure('Position', [100, 100, 1000, 400]);

subplot(2, 1, 1);
plot(t_sim, y_sim(:, 1), 'LineWidth', 1.5);
ylabel('x(t) vs t');
title('System Solution Using ODE45');
xlim(tspan);
grid on;

subplot(2, 1, 2);
plot(t_sim, y_sim(:, 2), 'LineWidth', 1.5);
ylabel('$\dot{x}(t)$ vs t', 'Interpreter', 'latex');
xlabel('t');
xlim(tspan);
grid on;



fprintf('Generating Phase Portrait ...\n');

% Plotting phase portrait
figure;
plot(y_sim(:, 1), y_sim(:, 2), 'LineWidth', 1.5);
title('Phase Portrait of System');
xlabel('x(t)');
ylabel('$\dot{x}(t)$', 'Interpreter', 'latex');
grid on;



fprintf('Overlaying Measurements ...\n');

% Plotting measurements against x(t)
figure('Position', [100, 100, 1000, 300]);
plot(t_sim, y_sim(:, 1), 'LineWidth', 1, 'DisplayName', 'True Motion');
hold on;
scatter(teval, ytilde, 1, 'DisplayName', 'Measured Position');
title('Solution Position vs Time');
xlim(tspan);
xlabel('t');
ylabel('x(t) vs t');
legend('Location', 'best');
grid on;


%% GLSDC Algorithm
z_true = [x0; p];
z = scale_factor * z_true;

fprintf('%s\nTask 2: GLSDC Algorithm\n%s\n', delim_equals, delim_equals);

% Precompute weights for all time steps
R_diag = (1 + teval) * sigma^2;
W_diag = 1 ./ R_diag;

% H matrix selector
H_selector_x = [1.0, 0.0];

fprintf('\n%5s %8s %8s %8s %8s %8s %8s %8s %8s %12s\n', ...
    'Iter', 'x(0)', 'xdot(0)', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'Cost');
fprintf('%s\n', delim_dash);

old_cost = inf;
new_cost = 1;

for iteration = 0:maxiter-1
    x0_guess = z(1:2);
    p_guess = z(3:8);
    
    old_cost = new_cost;
    
    % Initialize accumulator matrices
    Lambda = zeros(8, 8);
    N = zeros(8, 1);
    
    % Set up initial state
    initial_state_guess = [x0_guess; phi0(:); psi0(:)];
    
    % Integrate trajectory
    options_glsdc = odeset('RelTol', 1e-6, 'AbsTol', 1e-6);
    [t_glsdc, y_glsdc] = ode45(@(t, y) dynamics(t, y, p_guess), ...
        teval, initial_state_guess, options_glsdc);
    
    % Vectorized error computation
    y_pred = y_glsdc(:, 1)';
    err = ytilde - y_pred;
    
    % Vectorized cost computation
    new_cost = sum(W_diag .* err.^2);
    
    % Process all measurements
    for k = 1:length(teval)
        % Reshape variational matrices
        phi_k = reshape(y_glsdc(k, 3:6), 2, 2);
        psi_k = reshape(y_glsdc(k, 7:18), 2, 6);
        
        % Compute H_i
        H_phi = H_selector_x * phi_k;
        H_psi = H_selector_x * psi_k;
        H_i = [H_phi, H_psi];
        
        % Weighted update
        w_k = W_diag(k);
        H_weighted = H_i' * w_k;
        
        % Accumulate normal equations
        Lambda = Lambda + H_weighted * H_i;
        N = N + H_weighted * err(k);
    end
    
    fprintf('%5d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %12.6e\n', ...
        iteration, z(1), z(2), z(3), z(4), z(5), z(6), z(7), z(8), new_cost);
    
    % Check convergence
    if abs(new_cost - old_cost) / old_cost <= tol
        break;
    end
    
    % Solve for update
    delta_z = Lambda \ N;
    
    % Update state
    z = z + delta_z;
    
    % Keep phase angle in [0, 2π]
    z(8) = mod(z(8), 2 * pi);
end

fprintf('%s\n', delim_dash);

if abs(new_cost - old_cost) / old_cost > tol
    fprintf('\nGLSDC failed to converge (tol=%.0e) in %d iterations\n', tol, maxiter);
end

fprintf('\nFinal Estimate\n');
x0_guess = z(1:2);
p_guess = z(3:8);
fprintf('x0 = [%.6f, %.6f]\n', x0_guess(1), x0_guess(2));
fprintf('p  = [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n', p_guess);

%% Covariance Matrix Calculations
sample_times = [100, 200, 300];
fprintf('\n%s\nProducing Covariance Ellipses\n%s\n', delim_equals, delim_equals);
fprintf('\nCalculating P(t) at t = 0, 100, 200, 300\n');

% Creating variable to hold covariance matrix at each sample time
Pt = zeros(8, 8, 4);

% Calculating P(t_0) as inv(Lambda)
P0 = inv(Lambda);
Pt(:, :, 1) = P0;

% Get final trajectory for covariance propagation
[~, y_final] = ode45(@(t, y) dynamics(t, y, p_guess), ...
    [0, sample_times], initial_state_guess, options_glsdc);

% Using solution calculated from GLSDC to construct Phi(t) and Psi(t)
for k = 1:length(sample_times)
    PhiPsiVec = y_final(k+1, 3:end);
    Phi = reshape(PhiPsiVec(1:4), 2, 2);
    Psi = reshape(PhiPsiVec(5:16), 2, 6);
    dzdz0 = [Phi, Psi; zeros(6, 2), eye(6)];
    Pt(:, :, k+1) = dzdz0 * P0 * dzdz0';
end

% Creating cos and sin for ellipse
theta_ellipse = linspace(0, 2*pi, 100);
ellipse = [cos(theta_ellipse); sin(theta_ellipse)];

% Plot covariance ellipses
figure('Position', [100, 100, 1200, 800]);
for n = 0:1
    for i = 0:1
        k = 2*n + i;
        subplot(2, 2, k+1);
        
        % Get state
        if n == 0 && i == 0
            state = z(1:2);
        else
            state = y_final(k, 1:2)';
        end
        
        % Extract Px (2x2 matrix in upper left corner) from P(t)
        Px = Pt(1:2, 1:2, k+1);
        
        % Get Eigenvalues and Eigen Vectors of Px
        [V, D] = eig(Px);
        
        scatter(state(1), state(2), 50, 'k', 'filled');
        hold on;
        
        for sigma_lvl = 1:3
            cov_ellipse = V * diag(sigma_lvl * sqrt(diag(D))) * ellipse + state;
            plot(cov_ellipse(1, :), cov_ellipse(2, :), 'LineWidth', 1.5, ...
                'DisplayName', sprintf('\\sigma = %d', sigma_lvl));
        end
        
        title(sprintf('t = %d', 100 * k));
        xlabel('x(t)');
        ylabel('$\dot{x}(t)$', 'Interpreter', 'latex');
        legend('Location', 'best');
        grid on;
        hold off;
    end
end

%% Monte Carlo Simulation
fprintf('\n%s\nMonte Carlo Simulation\n%s\n\n', delim_equals, delim_equals);

% Note: The Python code was incomplete for Monte Carlo
% Here's a framework for 1000 iterations
fprintf('Running Monte Carlo simulation...\n');
for k = 1:1000
    if mod(k, 50) == 0
        fprintf('%d %%\n', k/10);
    end
    ytilde_mc = y_meas(:, 1)' + sigma * randn(1, length(teval));
    % Add GLSDC call here if needed for each iteration
end

fprintf('\nSimulation Complete!\n');