% Position Error Plot
figure
sgtitle('UKF Position Error')

subplot(3, 1, 1)
hold on
plot(tmeas, ukf_error(:, 1), 'DisplayName', 'Error')
plot(tmeas, ukf_error(:, 1) + sigma_bounds_ukf(:, 1), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas, ukf_error(:, 1) - sigma_bounds_ukf(:, 1), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
xlabel('Time (s)')
ylabel('x Error (km)')
legend()
grid on

subplot(3, 1, 2)
hold on
plot(tmeas, ukf_error(:, 2), 'DisplayName', 'Error')
plot(tmeas, ukf_error(:, 2) + sigma_bounds_ukf(:, 2), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas, ukf_error(:, 2) - sigma_bounds_ukf(:, 2), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
xlabel('Time (s)')
ylabel('y Error (km)')
legend()
grid on

subplot(3, 1, 3)
hold on
plot(tmeas, ukf_error(:, 3), 'DisplayName', 'Error')
plot(tmeas, ukf_error(:, 3) + sigma_bounds_ukf(:, 3), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas, ukf_error(:, 3) - sigma_bounds_ukf(:, 3), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
xlabel('Time (s)')
ylabel('z Error (km)')
legend()
grid on

exportgraphics(gcf, sprintf('Images/ukf_position_error_dt%d.png', delta_t), 'Resolution', 300)

% Velocity Error Plot
figure
sgtitle('UKF Velocity Error')

subplot(3, 1, 1)
hold on
plot(tmeas, ukf_error(:, 4), 'DisplayName', 'Error')
plot(tmeas, ukf_error(:, 4) + sigma_bounds_ukf(:, 4), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas, ukf_error(:, 4) - sigma_bounds_ukf(:, 4), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
xlabel('Time (s)')
ylabel('xdot Error (km/s)')
legend()
grid on

subplot(3, 1, 2)
hold on
plot(tmeas, ukf_error(:, 5), 'DisplayName', 'Error')
plot(tmeas, ukf_error(:, 5) + sigma_bounds_ukf(:, 5), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas, ukf_error(:, 5) - sigma_bounds_ukf(:, 5), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
xlabel('Time (s)')
ylabel('ydot Error (km/s)')
legend()
grid on

subplot(3, 1, 3)
hold on
plot(tmeas, ukf_error(:, 6), 'DisplayName', 'Error')
plot(tmeas, ukf_error(:, 6) + sigma_bounds_ukf(:, 6), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas, ukf_error(:, 6) - sigma_bounds_ukf(:, 6), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
xlabel('Time (s)')
ylabel('zdot Error (km/s)')
legend()
grid on

exportgraphics(gcf, sprintf('Images/ukf_velocity_error_dt%d.png', delta_t), 'Resolution', 300)