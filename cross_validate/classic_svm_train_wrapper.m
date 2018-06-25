function [w, b, svm] = classic_svm_train_wrapper (data, labels, svm_params)
% function [w, b, svm] = classic_svm_train_wrapper (data, labels, svm_params)
%
% Wrapper for training an SVM classifier through matlab's fitcsvm, for use
% with cross_validate.
%
% INPUTS:
%   - data (n x nfeat): training data
%   - labels (n x 1): labels for training data
%   - svm_params {nparams x 1}: parameters that will be forwarded
%   to fitcsvm.
% OUTPUTS:
%  - w (n x 1): weights of the linear classifier
%  - b (scalar): bias of the linear classifier
%  - svm: classifier object returned by fitcsvm.
%
% Copyright (C) 2018 Roberto Leonarduzzi
% Distributed under MIT license. See LICENSE.txt for details.

    svm = fitcsvm (data, labels, svm_params{:});
    w = svm.Beta(:);
    if isempty (w)
        w = NaN;
    end
    b = svm.Bias;
end