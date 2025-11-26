% Figure Plotting true trajectory
% Creating subplots for x, y, and z
fig = tiledlayout(3, 1);
ax1 = nexttile;
plot(ax1, t, true_motion(:, 1))
ylabel('x(t) (km)')
grid()

ax2 = nexttile;
plot(ax2, t, true_motion(:, 2))
ylabel('y(t) (km)')
grid()

ax3 = nexttile;
plot(ax3, t, true_motion(:, 3))
ylabel('z(t) (km)')
xlabel('t (s)')
linkaxes([ax1, ax2, ax3], 'x')
title(fig, 'Earth Fixed Position');
grid()
exportgraphics(gcf, sprintf('Images/body_fixed_pos_%d.png', delta_t),'Resolution',300)

% Plotting x, y, and z velocities in subplots and saving the figure
fig = tiledlayout(3, 1);
ax1 = nexttile;
plot(ax1, t, true_motion(:, 4))
ylabel('xdot(t) (km/s)')
grid()

ax2 = nexttile;
plot(ax2, t, true_motion(:, 5))
ylabel('ydot(t) (km/s)')
grid()

ax3 = nexttile;
plot(ax3, t, true_motion(:, 6))
ylabel('zdot(t) (km/s)')
xlabel('t (s)')
title(fig, 'Earth Fixed Velocity');
grid()
exportgraphics(gcf, sprintf('Images/body_fixed_vel_%d.png', delta_t),'Resolution',300)

figure

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
exportgraphics(gca, sprintf('Images/measurements_%d.png', delta_t),'Resolution',300)

% Plotting x, y, z trajectory relative to the observer vs time and
% overlaying measurements
figure;
ax1 = subplot(3, 1, 1);
hold on
plot(ax1, t, x_obsv, 'DisplayName','True')
scatter(ax1, tmeas, x_meas, 5, 'filled', 'DisplayName','Measured')
hold off
ylabel('x(t) (km)')
grid()
legend('Location','southeast')

ax2 = subplot(3, 1, 2);
hold on
plot(ax2, t, y_obsv, 'DisplayName','True')
scatter(ax2, tmeas, y_meas, 5, 'filled', 'DisplayName','Measured')
hold off
ylabel('y(t) (km)')
grid()
legend('Location','southeast')

ax3 = subplot(3, 1, 3);
hold on
plot(ax3, t, z_obsv, 'DisplayName','True')
scatter(ax3, tmeas, z_meas, 5, 'filled', 'DisplayName','Measured')
hold off
ylabel('z(t) (km)')
xlabel('t (s)')
grid()
legend('Location','southeast')

sgtitle('Position of Satellite Relative to Observer')
exportgraphics(gcf, sprintf('Images/cartesian_pos_measurements_%d.png', delta_t), 'Resolution',300)

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
exportgraphics(gcf, sprintf('Images/alt_az_plot_%d.png', delta_t), 'Resolution',300)

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
grid()
exportgraphics(gcf, sprintf('Images/range_plot_%d.png', delta_t), 'Resolution',300)






