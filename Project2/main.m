%% Central Limit Theorem
clear; clc; close all;

% Setting seed for repeatability
rng(2025)

% Number of samples for each histogram
sample_size = 1000;

% List of values for N, the number of points which will be summed
N_list = [5, 10, 20, 50, 100];

% Pre allocating array for sums
% Each column with correspond to a given number of trials in each sample, N
% 1st column will correspond to the 1st entry in N_list, and so on
uniform_data = zeros(sample_size, length(N_list));
gaussian_data = zeros(sample_size, length(N_list));

% Histogram data

% Creating histogram for each value of N in N_list
for n = 1:length(N_list)
    N = N_list(n);

    % Creating uniformly sampled data array 
    % with dimensions (sample_size by N)
    % Summing along columns to generate sum of each sample
    uniform_data(:, n) = sum(rand(sample_size, N), 2);

    % Generating Normal Distribution plot
    x = linspace(0, N, sample_size);
    gaussian_data(:, n) = normpdf(x, N/2, sqrt(N/12));

    % Creating plots
    subplot(length(N_list), 1, n)
    % Normalizing sample data in histogram to match pdf 
    histogram(uniform_data(:, n), 'Normalization', 'pdf')
    hold on
    plot(x, gaussian_data(:, n), 'k')
    hold off

    title(['Uniform Sampling Compared to Gaussian Distribution, N = ', num2str(N)])
end


%% Covariance Representation
clc; clear; close all;


% Probability Table (rows are scheme 1, cols are scheme 2)
p_x = [0.1, 0.2, 0.3; 0.2, 0.1, 0.1];

% Probability of of scheme 1;
px1 = sum(p_x(1, :));
px2 = sum(p_x(2, :));
p1 = [px1; px2];

% Reward of scheme 1
r1 = [1; 2];

% Expected Value of scheme 1
E1 = dot(p1, r1);

% Probability of of scheme 2;
pxa = sum(p_x(:, 1));
pxb = sum(p_x(:, 2));
pxc = sum(p_x(:, 3));
p2 = [pxa; pxb; pxc];

% Reward of scheme 2
r2 = [3; -2; 3];

% Expected Value of scheme 2
E2 = dot(p2, r2);

% Location of center
mu = [E1; E2];

% covariance

%{
Two different ways of calculating the matrix
The first way is 
(x - mu_x)' * p(x, y) * (y - mu_y)

The second way is 
sum ( p(x,y) * (x; y) * (x, y) ) - (mu_x; mu_y) * (mu_x, mu_y)

Methods yield the same answer
%}

% First method
var_x = p1' * (r1 - E1).^2;
var_y = p2' * (r2 - E2).^2;
covar = (r1-E1)' * p_x * (r2-E2);
sigma_xy = [var_x, covar; covar, var_y];


% Second method
sigma_xy = zeros(2,2);
for state_1 = 1:2
    for state_2 = 1:3
        sigma_xy = sigma_xy + p_x(state_1, state_2) * ([r1(state_1); r2(state_2)])*([r1(state_1), r2(state_2)]);
    end
end
sigma_xy = sigma_xy - mu*mu';

% Parameter for ellipse
t = linspace(0, 2*pi, 100);

% Creating figure for plots
figure
hold on

% Creating covirance ellipses
ellipse = [cos(t); sin(t)];

% Multiples of std to be plotted
sigma_lvls = [1, 2, 3];

for STD = sigma_lvls
    % Creating scale
    scale = STD;

    % Creating rotation and scaling of circle to ellipse
    [P, D] = eig(sigma_xy);
    cov_ellipse = P * scale * sqrt(D) * ellipse + mu;
    
    plot(cov_ellipse(1, :), cov_ellipse(2, :), 'DisplayName',[num2str(STD), '\sigma'])
end

% Plotting Mean
scatter(E1, E2, '+', 'k', 'DisplayName', 'Mean')

% Plotting Eigenvectors of covariance ellipse, scaling to match 3sigma
quiver(E1, E2, P(1, 1), P(2, 1), max(sigma_lvls)*sqrt(D(1,1)), '--k', 'HandleVisibility','off')
quiver(E1, E2, P(1, 2), P(2, 2), max(sigma_lvls)*sqrt(D(2,2)), '--k', 'HandleVisibility','off')
hold off
axis equal
legend()
title('Covariance Ellipses For Lottery Schemes')
xlabel('Scheme 1')
ylabel('Scheme 2')