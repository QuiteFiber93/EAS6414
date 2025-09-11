function f = phidot(t, x, p)
    % State variables are x(1) and x(2)
    state = [x(1); x(2)];
    % Elements of the STM are x(3)-x(6)
    phi = [x(3); x(4); x(5); x(6)];
    f = [xdot(t, state, p); phi(3); phi(4);...
        -phi(1)*(p(2) + 3*p(3)*state(1)^2)-phi(3)*p(1); ...
        -phi(2)*(p(2) + 3*p(3)*state(1)^2)-phi(4)*p(1)];
end