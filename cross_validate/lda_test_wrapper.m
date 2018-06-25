function [labels, scores] = lda_test_wrapper (data_test, label_test, w, b, lda)
% function [labels, scores] = lda_test_wrapper (data_test, label_test, w, b, lda)
%
% Wrapper for matlab's lda.predict for use with cross_validate.
%
% INPUTS:
%   - data_test (n x nfeat): testing data
%   - label_test (n x 1): labels for testing data
%   - Inputs 3 to 5 are the outputs of lda_train_wrapper
%     -- w, b: weights and bias terms. Unused.
%     -- lda: classifier object returned by fitcdiscr.
% OUTPUTS:
%  - labels (n x 1): estimated labels
%  - scores (n x 1): classifier scores
%
% Copyright (C) 2018 Roberto Leonarduzzi
% Distributed under MIT license. See LICENSE.txt for details.

    [labels, sc] = lda.predict (data_test);
    scores = diff (sc, 1, 2);  % pos class - neg class
end
