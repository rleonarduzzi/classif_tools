function [W, b, PX, Y] = slda_train_wrapper (X, Y, sp, delta, maxiter, tol)
% function [W, b, PX, Y] = slda_train_wrapper (X, Y, sp, delta, maxiter, tol)
%
% Wrapper for training a sparse LDA classifier through tje slda routine in
% the SpaSM toolbox, for use in cross_validate.
%
% SpaSM toolbox:
%   http://www2.imm.dtu.dk/projects/spasm/
%
% INPUTS:
%   - X (n x nfeat): training data
%   - Y (n x 1): labels for training data
%   - sp (scalar): sparsity level. See slda's help for its  meaning.
%   - delta (scalar): L2-norm penalization for the elastic net.
%   - maxiter (scalar): maximum number of iterations
%   - tol (scalar): tolerance
% OUTPUTS:
%  - W (n x 1): weights of the linear classifier
%  - b (scalar): bias of the linear classifier. Useless for SLDA, it is
%  assigned a NaN.
%  - PX: regression parameters: projections of data onto significant
%  directions.
%  - Y (n x 1): training labels. Needed for testing.
%
% Copyright (C) 2018 Roberto Leonarduzzi
% Distributed under MIT license. See LICENSE.txt for details.

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
