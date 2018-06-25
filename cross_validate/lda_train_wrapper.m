function [w, b, lda] = lda_train_wrapper (data, labels, lda_params)
% function [w, b, lda] = lda_train_wrapper (data, labels, lda_params)
%
% Wrapper for training an LDA classifier through matlab's
% fitcdiscr, for use with cross_validate.
%
% INPUTS:
%   - data (n x nfeat): training data
%   - labels (n x 1): labels for training data
%   - lda_params {nparams x 1}: parameters that will be forwarded
%   to fitcsdiscr.
% OUTPUTS:
%  - w (n x nclass): weights of the linear classifier
%  - b (nclass): bias of the linear classifier
%  - lda: classifier object returned by fitcdiscr.
%
% Copyright (C) 2018 Roberto Leonarduzzi
% Distributed under MIT license. See LICENSE.txt for details.

lda = fitcdiscr (data, labels, lda_params{:});
for ic = 2 : size (lda.Coeffs)
    w(:, ic - 1) = lda.Coeffs(1, ic).Linear;
    b(ic - 1) = lda.Coeffs(1, ic).Const;
end