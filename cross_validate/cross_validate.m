function [weight, label_hat, score] = ...
    cross_validate (train_fun, test_fun, data, label, param_grid, varargin)
% function [weight, label_hat, score] = ...
%     cross_validate (train_fun, test_fun, data, label, param_grid, varargin)
%
% Performs cross-validation of a classifier on a grid of parameters.
% It can be used on arbitrary classifiers, which are accessed through abstrat
% functions for testing and training.
%
% INPUTS:
%   - train_fun (function handle): training function with the
%     following signature:
%       [weights, varargout] = train_fun (data, label, param1, param2, ...)
%     where:
%       -- data (n x nfeat): training data
%       -- labels (n x 1): labels for training data
%       -- param1, param2 (scalars): parameters for the classifier,
%          obtained from each point of the grid
%       -- weights (nfeat x nclass)
%       -- varargout: in addition to the weights, train_fun can return any
%          number of extra (matrix) values. The number of output values needs
%          to be indicated by optional parameter 'num_outputs_train'
%   - test_fun (function handle): testing function with the following
%     signature:
%       [labels, scores] = test_fun (data_test, label_test, varargin)
%     where:
%       -- data_test (n x nfeat): testing data
%       -- label_test (n x 1): labels for testing data
%       -- labels (n x 1): estimated labels
%       -- scores (n x 1): classifier scores
%   - data (N x nfeat): training data
%   - label (N x 1): labels for training data
%   - param_grid {nparams x 1}: grid of paramenters for training.
%     Each element of param_grid is an array of arbitrary length. The length
%     of param_grid must be equal to the number of parameters that train_fun
%     receives (i.e. its number of inputs beyond the second one).
%     The grid is defined as the external product of all the elements of
%     param_grid.
%     A cross-validation loop will be performed for each element of the grid.
%     An extra dimension will be added to the outputs for each dimension of
%     the grid.
%
% OPTIONAL INPUTS (NAME-VALUE PAIRS):
%  - num_fold (scalar): number of cross-validation folds (default is 5).
%  - num_rep (scalar): number of times the cross-validation is repeated
%  (default is 11).
%    Results are averages over all repetitions. Use to reduce sampling bias.
%  - verbose (logical): activate output showing process of cross-validation
%  (default is false).
%  - num_outputs_train (scalar): number of outputs of train_fun (default is 1)
%
% OUTPUTS:
%  - weight (nfeat x nparam1 x nparam2 x ...): weights of the classifier
%  - label_hat (N x nparam1 x nparam2 x ...): estimated labels
%  - score (N x nparam1 x nparam2 x ...): classifier scores
%
%  where nparamk = length (param_grid{k}) and the number of dimensions after
%  the first one is equal to the length of param_grid.
%
% Copyright (C) 2018 Roberto Leonarduzzi

fprintf ('\n========== New version!!! ==========\n')

p = inputParser;
p.addParameter ('num_fold', 5);
p.addParameter ('num_rep', 11);
p.addParameter ('verbose', true, @islogical);
p.addParameter ('num_outputs_train', []);
p.parse (varargin{:});

num_fold = p.Results.num_fold;
num_rep = p.Results.num_rep;
be_verbose = p.Results.verbose;
nout_train = p.Results.num_outputs_train;

[n, n_feat] = size (data);
n_class = size (label, 2);  % 2
n_param = length (param_grid);
param_len = cellfun (@length, param_grid);

% Get the number of outputs from train_fun
% Need to know them to collect all of them generically and pass them to test_fun
if isempty (nout_train)
    nout_train = nargout (train_fun);
    if nout_train == -1  % train_fun uses varargout
        error (['Can''t determine number of outputs from training. '...
                'Use NOutputTrain option'])
    end
end

% Get vector version of label in case it is a multiclass matrix
% The exact label for each class is not important, the vector will just be
% used to determine the cross-validation folds.

if isvector (label)
    % In this case label is a vector whose elements can only have two values
    classes = unique (label);
    for ic = 1 : n_class
        class_len(ic) = sum (label == classes(ic));
    end

    label_vec = label;
else
    % In this case label is a binary matrix, one column for each class
    classes = 0 : n_class - 1;
    class_len = sum (label, 1);
    label_vec = label * classes';
end

% Initialize output
weight = nan ([param_len n_feat n_class]);
label_hat   = nan ([param_len n]);
score  = nan ([param_len n n_class]);

% Auxiliary variable to index all parameter dimensions:
idx_all_param = repmat ({':'}, 1, n_param);

% Since some of the parameters might be cell arrays,  use bogus parameters
% which are just indices to make the grid and then index the real parameters
for ip = 1 : length (param_grid)
    param_grid_bogus{ip} = 1 : length (param_grid{ip});
end

% Outer loop: loops all parameters using linear indices and grids of the parameters
[param_grid_clean{1:n_param}] = ndgrid (param_grid_bogus{:});
for iparam = 1 : prod (param_len)

    % Get indices of current parameters
    [idx_curr{1:n_param}] = ind2sub (size (param_grid_clean{1}), iparam);

    if be_verbose
        fprintf ('===== ')
    end

    % Get current parameters
    for i = 1 : n_param

        % Use idx_curr to index each grid to get bogus params.
        % Use bogus params as indices to get 'real' params
        % Conditional allow to choose proper indexing for each parameter
        if iscell (param_grid{i})
            params_curr{i} = param_grid{i}{param_grid_clean{i}(idx_curr{:})};
            print_fmt = [ '[', ...
                          repmat('%g ', 1, length (params_curr{i})), ...
                          ']'];
        elseif isvector (param_grid{i})
            params_curr{i} = param_grid{i}(param_grid_clean{i}(idx_curr{:}));
            print_fmt = '%g';
        else
            error (['Unsupported type of parameter. Must be array or cell ' ...
                    'array'])
        end

        if be_verbose
            fprintf (['param%i: ', print_fmt, ', '], i, params_curr{i})
        end
    end
    if be_verbose
        fprintf (' =====\n')
    end

    label_hat_cv_rep   = nan ([num_rep, n]);
    score_cv_rep  = nan ([num_rep, n, n_class]);
    weight_cv_rep = nan ([num_rep, num_fold, n_feat, n_class]);

    % Loop repetitions
    for irep = 1 : num_rep
        cCV = cvpartition (label_vec, 'kfold', num_fold);

        if be_verbose
            fprintf('## rep CV: %d/%d, ', irep, num_rep);
        end

        tic
        % Loop folds
        for ifold = 1 : num_fold
            % Get indices for current fold
            idx_trn = cCV.training (ifold);
            idx_tst = cCV.test (ifold);

            data_trn = data(idx_trn, :);
            label_trn = label(idx_trn, :);
            data_tst = data(idx_tst, :);
            label_tst = label(idx_tst, :);

            % Second output might be bias. get it only for single class
            % classifiers...
            % FIXME: I should find a way to get automatically all outputs
            % without assuming that single class will always have a bias
            % term. [train_output{:}] would be ideal but it does not work.
            train_output = cell (1, nout_train);
            [train_output{:}] = train_fun (data_trn, label_trn, params_curr{:});
            [label_hat_loc, score_loc] = test_fun (data_tst, label_tst, ...
                                                   train_output{:});

            % Pack output
            label_hat_cv_rep(irep, idx_tst) = label_hat_loc;
            score_cv_rep(irep, idx_tst, :) = score_loc;
            weight_cv_rep(irep, ifold, :, :) = train_output{1}';
        end  % loop folds
        time = toc;
        if be_verbose
            fprintf ('%g seconds\n', time)
        end
    end % loop repetitions

    % Average over repetitions and folds:
    label_hat(idx_curr{:}, :) = squeeze (nanmedian (label_hat_cv_rep, 1));
    score(idx_curr{:}, :, :) = squeeze (nanmean (score_cv_rep, 1));
    weight(idx_curr{:}, :, :) = ...
        squeeze (nanmean (nanmean (weight_cv_rep, 1), 2));

    % Save tmp for eventual backup:
    save ('backup_cross_validate.mat', '-v7.3')

end % loop params
