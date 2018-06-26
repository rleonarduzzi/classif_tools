function [yhat,d] = sparse_svm_test( w, b, X )
% function [yhat,d] = sparse_svm_test( w, b, X )
%
% Compute labels and scores on testin set.
%
% Input : w, b - Weights and bias output by sparse_svm_train
%         X - [KxN double] test data matrix (features x subjects)
%
% Output : w - [Kx1 double] normal vector
%          b - [double] offset
%          crit - [#iterationsx1 double] value of objective function
%
% J. Frecon, J. Spilka, N. Pustelnik, P. Abry,
% ENS Lyon, 2015

d = w'*X - b;
yhat = sign(d);

if isnan(w)
    yhat = -1*ones (size (X, 2));
    d = zeros (size (X, 2));
end
