function f = xdot(t, x, p)
    f = [x(2); -(p(1)*x(2)+p(2)*x(1) + p(3)*x(1)^3)];
end