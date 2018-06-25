function [labels, scores] = lda_test_wrapper (data, label, w, b, lda)
    [labels, sc] = lda.predict (data);
    scores = diff (sc, 1, 2);  % pos class - neg class
end
