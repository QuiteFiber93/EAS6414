function linearized = F(t, x, p)
    linearized = [0, 1; -p(2) - 3*p(3)*x(1)^2];
end