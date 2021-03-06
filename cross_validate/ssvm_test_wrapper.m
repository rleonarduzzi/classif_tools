function [labels, scores] = ssvm_test_wrapper (data, weights, bias)
% function [labels, scores] = ssvm_test_wrapper (data, weights, bias)
%
% Wrapper for sparse_svm_test for use with cross_validate.
%
% INPUTS:
%  - data (n x nfeat): testing data
%  - Inputs 3 to 4 are the outputs of lda_train_wrapper
%    -- weights (n x 1): weights of the linear classifier
%    -- bias (n x 1): bias of the linear classifier
% OUTPUTS:
%  - labels (n x 1): estimated labels
%  - scores (n x 1): classifier scores
%
% Copyright (C) 2018 Roberto Leonarduzzi
% Distributed under MIT license. See LICENSE.txt for details.

[labels, scores] = sparse_svm_test (weights, bias, data');
labels = (labels + 3) / 2;