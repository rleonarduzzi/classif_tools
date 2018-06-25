function [weights, b] = ssvm_train_wrapper (data, y, c, varargin)

% Convert labels to -1 , +1
mm = mean (y);
y_conv(y < mm) = -1;
y_conv(y > mm) = 1;

[weights, b] = SparseRegularizedSVM_train (data', y_conv', c, varargin{:});