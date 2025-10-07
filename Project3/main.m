%% Task 1
clc; clear; close all;

% Getting array of times
t = (1:3000)/10;

% Initializing given variables
x0 = 2; xdot0 = 0;
state0 = [x0; xdot0];

p1 = 0.05; p2 = 4;
p3 = 0.2; p4 = -0.5;
p5 = 10; p6 = pi/2;
p = [p1; p2; p3; p4; p5; p6];

% Integrating Original ODE
nlfunc = @(t, x) xdot(t, x, p);
[t, sol] = ode45(nlfunc, t, state0);

subplot(1, 3, 1)
plot(t, sol(:, 1))
title('$x(t)$ vs $t$', 'Interpreter', 'latex')
xlabel('$t$', 'Interpreter', 'latex')
ylabel("$x(t)$", 'Interpreter', 'latex')

subplot(1, 3, 2)
plot(t, sol(:, 2))
title('$\dot{x}(t)$ vs $t$', 'Interpreter', 'latex')
xlabel('$t$', 'Interpreter', 'latex')
ylabel("$\dot{x}(t)$", 'Interpreter', 'latex')

subplot(1, 3, 3)
plot(sol(:, 1), sol(:, 2))
title('$\dot{x}(t)$ vs $x(t)$', 'Interpreter', 'latex')
xlabel('$x(t)$', 'Interpreter', 'latex')
ylabel("$\dot{x}(t)$", 'Interpreter', 'latex')