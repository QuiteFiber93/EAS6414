figure
sgtitle('EKF Position Error')
subplot(3, 1, 1)
hold on
plot(tmeas, ekf_error(:, 1), 'DisplayName','Error')
plot(tmeas,  ekf_error(:, 1) + sigma_bounds(:, 1), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas,  ekf_error(:, 1) - sigma_bounds(:, 1), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
ylabel('x Error (km)')
grid()
legend()

subplot(3, 1, 2)
hold on
plot(tmeas, ekf_error(:, 2), 'DisplayName','Error')
plot(tmeas,  ekf_error(:, 2) + sigma_bounds(:, 2), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas,  ekf_error(:, 2) - sigma_bounds(:, 2), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
ylabel('y Error (km)')
grid()
legend()

subplot(3, 1, 3)
hold on
plot(tmeas, ekf_error(:, 3), 'DisplayName','Error')
plot(tmeas,  ekf_error(:, 3) + sigma_bounds(:, 3), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas,  ekf_error(:, 3) - sigma_bounds(:, 3), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
ylabel('z Error (km)')
grid()
legend()

exportgraphics(gcf, sprintf('Images/ekf_position_error_dt%d.png', delta_t), 'Resolution',300)
figure
sgtitle('EKF Velocity Error')
subplot(3, 1, 1)
hold on
plot(tmeas, ekf_error(:, 4), 'DisplayName','Error')
plot(tmeas,  ekf_error(:, 4) + sigma_bounds(:, 4), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas,  ekf_error(:, 4) - sigma_bounds(:, 4), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
ylabel('xdot Error (km/s)')
grid()
legend()

subplot(3, 1, 2)
hold on
plot(tmeas, ekf_error(:, 5), 'DisplayName','Error')
plot(tmeas,  ekf_error(:, 5) + sigma_bounds(:, 5), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas,  ekf_error(:, 5) - sigma_bounds(:, 5), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
ylabel('ydot Error (km/s)')
grid()
legend()

subplot(3, 1, 3)
hold on
plot(tmeas, ekf_error(:, 6), 'DisplayName','Error')
plot(tmeas,  ekf_error(:, 6) + sigma_bounds(:, 6), 'r--', 'LineWidth', 1, 'DisplayName', '3\sigma');
plot(tmeas,  ekf_error(:, 6) - sigma_bounds(:, 6), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
hold off
ylabel('zdot Error (km/s)')
grid()
legend()
exportgraphics(gcf, sprintf('Images/ekf_velocity_error_dt%d.png', delta_t), 'Resolution',300)