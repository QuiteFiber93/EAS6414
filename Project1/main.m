%% Problem 1
clear; clc; close all;

% givens and nominal values
x0N = [2; 0.5];
p0N = [0.2; 4; 0.1];
tspan = [0, 30];

% Initial Conditions
phi0 = eye(2);
psi0 = zeros(2, 6);
state0 = [x0N;phi0(:); psi0(:)];

% Integrating ODE for system and for transition matrices
nlfunc = @(t, state) dynamics_LP(t, state, p0N);
[t, sol] = ode45(nlfunc, tspan, state0);
xN = sol(:, 1:2);
phit = sol(:, 3:6);
psit = sol(:, 7:12);

% Deviations from nominal conditions
deltaX = [0.01; 0.01];
deltaP = [0.001; 0.001; 0.001];

% Initial Conditions of perturbed system
x0 = x0N + deltaX;
p0 = p0N + deltaP;

% Phi and psi need to be resized due to function definitions
phi = zeros(2, 2, length(t));
psi = zeros(2, 3, length(t));

% Creating variable to hold approximation calculations
xLP = zeros(size(xN));

for n = 1:length(t)
    phi(:, :, n) = reshape(sol(n, 3:6), 2, 2);
    psi(:, :, n) = reshape(sol(n, 7:12), 2, 3);
    xLP(n, :) = xN(n, :)' + phi(:, :, n)*deltaX + psi(:, :, n)*deltaP;
end

% Integrating with true initial conditions
func = @(t, y) xdot(t, y, p0);
[t, x] = ode45(func, t, x0);

% Caclculating Error
e = x - xLP;

% Plotting
figure
plot(t, xN(:, 1))
title('x_1 vs t')
xlabel('t')
ylabel('x_1')

figure
plot(t, xN(:, 2))
title('x_2 vs t')
xlabel('t')
ylabel('x_2')

figure
plot(xN(:, 1), xN(:, 2))
title('x_2 vs x_1')
xlabel('x_1')
ylabel('x_2')

figure
plot(t, e(:, 1))
title('x(t) - x_{LP}(t)')
xlabel('t')
ylabel('e(t)')

%% Problem 2
clear; clc; close all;
% Creating random coefficients
scale_coeff = 1;
rng(2025)
a = scale_coeff * rand(7, 1);

% Creating x values
m = 200;
upper = 5;
lower = -5;
x = linspace(lower, upper, m)';

% true measurements
H = [x.^6, x.^5, x.^4, x.^3, x.^2, x, x.^0];
y = H*a;

% adding noise to y measurements
sigma = 0.1;
noise = randn(size(y)) * sigma;
ytilde = y + noise;

% calculating yhat with batch estimation
ahat_batched = pinv(H) * ytilde;

% Caculating measurements with ahat
yhat_batched = H*ahat_batched;

figure 
hold on
plot(x, y, 'DisplayName', 'True');
scatter(x, ytilde, 10, 'filled', 'DisplayName', 'Noisy Data');
hold off
legend()
title('True Measurements and Added Noise')
xlabel('x')
ylabel('Measurement')

figure
hold on
plot(x, yhat_batched, 'DisplayName', 'Batch Estimated');
scatter(x, ytilde, 10, 'filled', 'DisplayName', 'Noisy Data');
hold off
legend()
title('Batch Esimtaed Measurements and Added Noise')
xlabel('x')
ylabel('Measurement')

figure
plot(x, yhat_batched - y)
title('Error in Batch Estimation')
xlabel('x')
ylabel('yhat - y')

figure
scatter(1:7, ahat_batched - a, 25, 'filled')
title('Error in Estimated a')
xlabel('a_i')
ylabel('ahat - a')

%% Problem 2 Sequential Estimation
clear; clc; close all;

% Creating random coefficients
scale_coeff = 1;
rng(2025)
a = scale_coeff * rand(7, 1);

% Creating x values with chebyshev spacing
m = 200;
upper = 5;
lower = -5;
x = linspace(lower, upper, m)';

% true measurements
H = [x.^6, x.^5, x.^4, x.^3, x.^2, x, x.^0];
y = H*a;

% adding noise to y measurements
sigma = 0.1;
noise = randn(size(y)) * sigma;
ytilde = y + noise;

% Initial Guess Based on Batch Estimation
m1 = 45;
W = 1;
P = eye(7) / ( H(1:m1, :)' * W * H(1:m1, :));
ahat_seq = P * ( H(1:m1, :)'*W*ytilde(1:m1));

% Sequential estimation
m2 = m - m1;
W = 1/(sigma^2);
for k = m2:m-1
    K = P * H(k+1, :)' / (H(k+1, :)*P*H(k+1, :)' + inv(W));
    P = (eye(7) - K * H(k+1, :))*P;
    ahat_seq = ahat_seq + K*(ytilde(k+1) - H(k+1, :)*ahat_seq);
end

yhat_seq = H*ahat_seq;

figure 
hold on
plot(x, y, 'DisplayName', 'True');
scatter(x, ytilde, 10, 'filled', 'DisplayName', 'Noisy Data');
hold off
legend()
title('True Measurements and Added Noise')
xlabel('x')
ylabel('Measurement')

figure
hold on
plot(x, yhat_seq, 'DisplayName', 'Sequentially Estimated');
scatter(x, ytilde, 10, 'filled', 'DisplayName', 'Noisy Data');
hold off
legend()
title('Sequential Esimtaed Measurements and Added Noise')
xlabel('x')
ylabel('Measurement')

figure
plot(x, yhat_seq - y)
title('Error in Sequential Estimation')
xlabel('x')
ylabel('yhat - y')

figure
scatter(1:7, ahat_seq - a, 25, 'filled')
title('Error in Estimated a')
xlabel('a_i')
ylabel('ahat - a')

%% Sequential Estimation with Alpha and Beta
clear; clc; close all;

% Creating random coefficients
rng(2025)
scale_coeff = 1;
a = scale_coeff * rand(7, 1);

% Creating x values with chebyshev spacing
m = 200;
upper = 5;
lower = -5;
x = linspace(lower, upper, m)';

% true measurements
H = [x.^6, x.^5, x.^4, x.^3, x.^2, x, x.^0];
y = H*a;

% adding noise to y measurements
sigma = 0.1;
noise = randn(size(y)) * sigma;
ytilde = y + noise;

% Initial guess based off alpha and beta
alpha = 1E1;
beta = 1E-2 * ones(7, 1);
W = 1/sigma^2;
P = eye(7) / (1/alpha^2 * eye(7) + H(1, :)' * W * H(1, :));
ahat_seq = P * (1/alpha * beta + H(1, :)'*W*ytilde(1));

% Sequential estimation for k = 2 - m
for k = 1:m-1
    K = P * H(k+1, :)' / (H(k+1, :)*P*H(k+1, :)' + inv(W));
    P = (eye(7) - K * H(k+1, :))*P;
    ahat_seq = ahat_seq + K*(ytilde(k+1) - H(k+1, :)*ahat_seq);
end

yhat_seq = H*ahat_seq;

figure 
hold on
plot(x, y, 'DisplayName', 'True');
scatter(x, ytilde, 10, 'filled', 'DisplayName', 'Noisy Data');
hold off
legend()
title('True Measurements and Added Noise')
xlabel('x')
ylabel('Measurement')

figure
hold on
plot(x, yhat_seq, 'DisplayName', 'Sequentially Estimated');
scatter(x, ytilde, 10, 'filled', 'DisplayName', 'Noisy Data');
hold off
legend()
title('Sequential Esimtaed Measurements and Added Noise')
xlabel('x')
ylabel('Measurement')

figure
plot(x, yhat_seq - y)
title('Error in Sequential Estimation')
xlabel('x')
ylabel('yhat - y')

figure
scatter(1:7, ahat_seq - a, 25, 'filled')
title('Error in Estimated a')
xlabel('a_i')
ylabel('ahat - a')
