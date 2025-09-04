%% Problem 1
clear; clc; close all;

% givens and nominal values
x0N = [2; 0.5];
p0N = [0.2; 4; 0.1];
tspan = [0, 30];

% Numerical Integration
func = @(t,x) xdot(t, x, p0N);
[t, xN] = ode45(func, tspan, x0N);


% Plotting
figure
plot(t, xN(:, 1))
title('x_1 vs t')

figure
plot(t, xN(:, 1))
title('x_2 vs t')

figure
plot(xN(:, 1), xN(:, 2))
title('x_1 vs x_2')

% Deviations
deltaX = [0.01; 0.01];
deltaP = [0.001; 0.001; 0.001];

% Initial Conditions of perturbed system
x0 = x0N + deltaX;
p0 = p0N + deltaP;

% Numerical Integration of phi, return values evaled at times returned for
% xN

% Numerical Integration of psi, return values evaled at times returned for
% xN

% Calculated xLP

% Integrate x0, return values evaled at times returned for xN

% Plot Error of x - xLP


%% Problem 2
clear; clc;
% Creating random coefficients
a = rand(7, 1);

% Creating true x values
x = linspace(-5, 5, 200);

% Adding noise to measurements
