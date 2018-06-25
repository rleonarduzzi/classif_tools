function [w, b, lda] = lda_train_wrapper (data, y, lda_params)
    lda = fitcdiscr (data, y, lda_params{:});
    for ic = 2 : size (lda.Coeffs)
        w(:, ic - 1) = lda.Coeffs(1, ic).Linear;
        b(ic - 1) = lda.Coeffs(1, ic).Const;
end