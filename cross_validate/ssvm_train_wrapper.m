function [weights, b] = ssvm_train_wrapper (data, labels, c, varargin)
% function [weights, b] = ssvm_train_wrapper (data, labels, c, varargin)
%
% Wrapper for training a Sparse SVM through SparseRegularizedSVM_train, for
% use with cross_validate.
%
% INPUTS:
%   - data (n x nfeat): training data
%   - labels (n x 1): labels for training data
%   - c (scalar): L1-regularization constant. Controls sparsity.
%   - varargin: all extra parameters are forwarded to
%   SparseRegularizedSVM_train.
% OUTPUTS:
%  - weights (n x 1): weights of the linear classifier
%  - b (scalar): bias of the linear classifier
%
% Copyright (C) 2018 Roberto Leonarduzzi
% Distributed under MIT license. See LICENSE.txt for details.

% Convert labels to -1 , +1
mm = mean (labels);
labels_conv(labels < mm) = -1;
labels_conv(labels > mm) = 1;

[weights, b] = SparseRegularizedSVM_train (data', labels_conv', c, varargin{:});