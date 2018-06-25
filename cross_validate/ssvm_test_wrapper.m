function [labels, scores] = ssvm_test_wrapper (data, label, weights, bias)


[labels, scores] = SparseRegularizedSVM_test (weights, bias, data', label');
labels = (labels + 3) / 2;