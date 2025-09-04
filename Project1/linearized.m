function A = linearized(t, x, p)
    A = [0, 1; -p(2) - 3*p(3)*x(1)^2, -p(1)];
end