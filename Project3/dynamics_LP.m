function f = dynamics_LP(t, state, p)
    x = state(1:2);
    phi = reshape(state(3:6), 2, 2);
    psi = reshape(state(7:18), 2, 6);

    dfdx = [0, 1; -(p(2)+3*p(3)*x(1)^2), -p(1)];
    dfdp = [zeros(1, 6); -x(2), -x(1), -x(1)^3, -sin(p(5)*t + p(6)), -p(4)*p(5)*cos(p(5)*t + p(6)), -p(4)*cos(p(5)*t + p(6))];

    f = zeros(size(state));
    f(1:2) = xdot(t, x, p);
    f(3:6) = reshape(dfdx*phi, 4, 1);
    f(7:18) = reshape(dfdx*psi + dfdp, 12, 1);
end