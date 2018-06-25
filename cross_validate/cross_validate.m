function [weight, label_hat, score] = ...
    cross_validate (train_fun, test_fun, data, label, params, varargin)
%
% Inputs:
%       - train_fun: training function. Receives data (n x nfeat), label
%       (n x 1), and parameters (scalars), and outputs weights (nfeat x nclass)
%       - test_fun: testing function. Receives weights (nfeat x nclass), data (n
%       x nfeat) and label (n x 1), and outputs estimated class(n x 1) and
%       scores (n x 1)
%       - data: (n x nfeat)
%       - label: (n x 1)
%       - params {nparams x 1}: params that will be looped in the cross
%       validation. Each element of the cell array is an array of parameters,
%       and will add an extra dimension to the outputs.


p = inputParser;
p.addParameter ('num_fold', 1);
p.addParameter ('num_rep', 11);
p.addParameter ('verbose', false, @islogical);
p.addParameter ('num_outputs_train', []);
p.parse (varargin{:});

num_fold = p.Results.num_fold;
num_rep = p.Results.num_rep;
be_verbose = p.Results.verbose;
nout_train = p.Results.num_outputs_train;

[n, n_feat] = size (data);
n_class = size (label, 2);  % 2
n_param = length (params);
param_len = cellfun (@length, params);

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
for ip = 1 : length (params)
    params_bogus{ip} = 1 : length (params{ip});
end

% Outer loop: loops all parameters using linear indices and grids of the parameters
[param_grid{1:n_param}] = ndgrid (params_bogus{:});
for iparam = 1 : prod (param_len)

    % Get indices of current parameters
    [idx_curr{1:n_param}] = ind2sub (size (param_grid{1}), iparam);

    if be_verbose
        fprintf ('===== ')
    end

    % Get current parameters
    for i = 1 : n_param

        % Use idx_curr to index each grid to get bogus params.
        % Use bogus params as indices to get 'real' params
        % Conditional allow to choose proper indexing for each parameter
        if iscell (params{i})
            params_curr{i} = params{i}{param_grid{i}(idx_curr{:})};
            print_fmt = [ '[', ...
                          repmat('%g ', 1, length (params_curr{i})), ...
                          ']'];
        elseif isvector (params{i})
            params_curr{i} = params{i}(param_grid{i}(idx_curr{:}));
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

            % Pack params
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
