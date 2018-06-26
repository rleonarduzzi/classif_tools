function [labels, scores] = classic_svm_test_wrapper (data_test, w, b, svm)
% function [labels, scores] = classic_svm_test_wrapper (data_test, w, b, svm)
%
% Wrapper for matlab's svm.predict for use with cross_validate.
%
% INPUTS:
%   - data_test (n x nfeat): testing data
%   - Inputs 3 to 5 are the outputs of lda_train_wrapper
%     -- w, b: weights and bias terms. Unused.
%     -- svm: classifier object returned by fitcsvm.
% OUTPUTS:
%  - labels (n x 1): estimated labels
%  - scores (n x 1): classifier scores
%
% Copyright (C) 2018 Roberto Leonarduzzi
% Distributed under MIT license. See LICENSE.txt for details.

    [labels, sc] = svm.predict (data_test);
    scores = diff (sc, 1, 2);  % pos class - neg class
end
