function [w, b, svm] = classic_svm_train_wrapper (data, y, svm_params)
    svm = fitcsvm (data, y, svm_params{:});
    w = svm.Beta(:);
    if isempty (w)
        w = NaN;
    end
    b = svm.Bias;
end