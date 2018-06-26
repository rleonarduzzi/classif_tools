function [yhat,d] = sparse_svm_test( w, b, X, ytest )
% function [yhat,d] = sparse_svm_test( w, b, X, ytest )
%
% Compute labels and scores on testin set.
%
% Input : w, b - Weights and bias output by sparse_svm_train
%         X - [KxN double] test data matrix (features x subjects)
%         Y - [1xN int] (test labels)
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
    yhat = -1*ones(size(ytest));
    d = zeros(size(ytest));
end
