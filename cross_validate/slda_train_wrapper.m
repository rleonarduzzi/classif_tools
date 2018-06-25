function [W, b, PX, Y] = slda_train_wrapper (X, Y, sp, delta, maxiter, tol)

if ~exist ('tol', 'var')
    tol = 1e-6;
end
if ~exist ('maxiter', 'var')
    maxiter = 1000;
end

% Convert Y to dummy variables:
values = unique (sort (Y));
for iv = 1 : length (values)
    Y_dummy(:, iv) = Y == values(iv);
end


W = slda (X, Y_dummy, delta, sp, size (Y_dummy, 2) - 1, maxiter, tol, false);
b = NaN;
PX = X * W;

% Transpose weights because of cross-validate's expectations...
W = W';
