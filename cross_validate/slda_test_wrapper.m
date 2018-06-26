function [labels, scores] = slda_test_wrapper (X, B, bias, PX_train, ytrain)
% function [labels, scores] = slda_test_wrapper (X, B, bias, PX_train, ytrain)
%
% Wrapper function for testing a SLDA classifier, for use with cross_validate.
%
% INPUTS:
%  - X (n x nfeat): testing data
%  - Inputs 3 to 6 are the outputs of lda_train_wrapper
%    -- B (nfeat x nclass): weights of the linear classifier
%    -- bias (scalar): unused. Kept for compatibility
%    -- PX_train: projections of training data onto significant directions
%    -- ytrain: training labels
% OUTPUTS:
%  - labels (n x 1): estimated labels
%  - scores (n x 1): classifier scores
%
% Copyright (C) 2018 Roberto Leonarduzzi
% Distributed under MIT license. See LICENSE.txt for details.


%[n, k] = size (ytrain);

% Convert ytrain from dummy variables to 1, 2, n
%ytrain = sum (ytrain .* repmat (1 : k, n, 1), 2);

% Transpose back... (it was transposed in training function for compatibility
% with cross_validate)
B = B';

[labels, ~, scores] = classify (X * B, PX_train, ytrain, 'linear');
scores = scores(:, 2) - scores(:, 1);