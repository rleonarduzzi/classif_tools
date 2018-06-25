function [labels, scores] = slda_test_wrapper (X, ytst, B, bogus, PX_train, ytrain)

%[n, k] = size (ytrain);

% Convert ytrain from dummy variables to 1, 2, n
%ytrain = sum (ytrain .* repmat (1 : k, n, 1), 2);

% Transpose back...
B = B';

[labels, ~, scores] = classify (X * B, PX_train, ytrain, 'linear');
scores = scores(:, 2) - scores(:, 1);