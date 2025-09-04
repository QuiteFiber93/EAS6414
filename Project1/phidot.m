function f = phidot(t, phi, x, p)
    A = linearized(t, x, p);
    f = A * phi;
end