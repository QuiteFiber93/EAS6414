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

% Given Values
x0 = [2; 0];            % Initial Conditions
p = [0.05; 4; 0.2; -0.5; 10; pi/2]; % True Parameters
tspan = [0, 300];       % Initial and Final Times of Simulation
teval = (1:3000)/10;    % Times at which system is sampled

% Initial conditions
phi0 = eye(2);
psi0 = zeros(2, 6);
y0 = [x0(:); phi0(:); psi0(:)];

% Integrating ODE 
% Creating a "true" simulation of motion
% Then creating integrating to return points at measurements
opts = odeset('RelTol', 1E-10, 'AbsTol',1E-10);
func = @(t, y) xdot(t, y, p);

% "True Motion Solution"
[t, true_state] = ode45(func, tspan, x0, opts);

% Measured States
[~, measured_states] = ode45(func, teval, x0, opts);

% Adding Random Noise 
% ytilde = x(t) + v_i
% Setting seed for repeatability
rng(2025);
ytilde = measured_states(:, 1) + sigma * randn(length(teval), 1);

% Plotting Generated Solution
subplot(2, 1, 1);
plot(t, true_state(:, 1));
ylabel('$x(t)$ vs $t$', 'Interpreter', 'latex');
title('System Solution Using ODE45');
xlim(tspan);

subplot(2, 1, 2);
plot(t, true_state(:, 2));
ylabel('$\dot{x}(t)$ vs t', 'Interpreter', 'latex');
xlabel('$t$', 'Interpreter', 'latex');
xlim(tspan);

figure;
plot(true_state(:, 1), true_state(:, 2));
title('Phase Portrait of System');
xlabel('x(t)');
ylabel('$\dot{x}(t)$', 'Interpreter', 'latex');

figure('Position', [100, 100, 1000, 300]);
plot(t, true_state(:, 1), 'DisplayName', 'True Motion');
hold on;
scatter(teval, ytilde, 1, 'DisplayName', 'Measured Position');
title('Solution Position vs Time');
xlim(tspan);
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$x(t)$', 'Interpreter', 'latex');
legend('Location', 'best');

%% GLSDC Algorithm
z_true = [x0; p];
z = scale_factor * z_true;

glsdc(@(t, y, p) dynamics(t, y, p), z, ytilde, teval, sigma, tol, maxiter)

%% Function Definitions
function f = xdot(t, x, p)
    f = [x(2);
        -p(1)*x(2) - p(2)*x(1) - p(3)*x(1)^2 - p(4)*sin(p(5)*t+p(6))];
end

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

function [z_est, y_traj, Lambda] = ...
            glsdc(dynamics_func, z_init, ytilde, teval, sigma, tol, maxiter)
        % GLSDC - Generalized Least Squares Differential Correction
        %
        % Inputs:
        %   dynamics_func - Function handle for system dynamics
        %   z_init        - Initial guess for [x0; p] (8x1)
        %   ytilde        - Measured data (1xN)
        %   teval         - Measurement times (1xN)
        %   sigma         - Measurement standard deviation
        %   tol           - Convergence tolerance
        %   maxiter       - Maximum iterations
        %
        % Outputs:
        %   z_est       - Estimated state and parameters [x0; p] (8x1)
        %   y_traj      - Final trajectory (Nx18)
        %   Lambda      - Information matrix (8x8)
        %   converged   - Boolean indicating convergence
        %   final_cost  - Final cost value
        %   num_iter    - Number of iterations performed
        
        
        z = z_init;
        phi0 = eye(2);
        psi0 = zeros(2, 6);
        
        % Precompute weights for all time steps
        R_diag = (0 + teval) * sigma^2;
        W_diag = 1 ./ R_diag;
        
        % H matrix selector
        H_selector_x = [1.0, 0.0];
        
  
        
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
            glsdc_opt = odeset('RelTol', 1e-6, 'AbsTol', 1e-6);
            [~, y_glsdc] = ode45(@(t, y) dynamics_func(t, y, p_guess), teval, initial_state_guess, glsdc_opt);
            
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
        
        z_est = z;
        y_traj = y_glsdc;
    end