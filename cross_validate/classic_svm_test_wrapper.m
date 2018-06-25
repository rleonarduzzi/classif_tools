function [labels, scores] = classic_svm_test_wrapper (data, label, w, b, svm)
    [labels, sc] = svm.predict (data);
    scores = diff (sc, 1, 2);  % pos class - neg class
end
