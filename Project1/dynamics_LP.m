function f = dynamics_LP(t, state, p)
    x = state(1:2);
    phi = reshape(state(3:6), 2, 2);
    psi = reshape(state(7:12), 2, 3);

    dfdx = [0, 1; -(p(2)+3*p(3)*x(1)^2), -p(1)];
    dfdp = [zeros(1, 3); -x(2), -x(1), -x(1)^3];
   
    f = zeros(size(state));
    f(1:2) = xdot(t, x, p);
    f(3:6) = reshape(dfdx*phi, 4, 1);
    f(7:12) = reshape(dfdx*psi + dfdp, 6, 1);
end