function [w, b, crit] = SparseRegularizedSVM_train(X,y,C,varargin)
% Sparse Regularized SVM
% [w,b,crit] = sparseRegularizedSVM (x,z,C) returns the vector 'w' and the real 'b' which
% minimize the sparse regularized SVM, i.e.
%
%       min_(w,b) = ||w||_1 + C \sum_{n=1}^N max(0, 1 - z_n*(x_n'*w + b))^2
%
% Note: the data fidelity term is adjusted in order that both classes (i.e., z=-1 & z=+1) contribute equally.
%
% Input : X - [KxN double] data matrix (features x subjects)
%         Y - [1xN int] (labels)
%         C - [uint] trade-off between data fidelity & sparsity
%         aFeatNames   - [Kx1 cell] - names of the features
%         aNrGroup     - [int vector, optional] for group sparsity - number of features in each group
%         penalization - [string, default 'L1'] - other supported L12
%         dataFidelity - [string, default 'Hinge']
%         bDebug       - [bool, optional] - names of the features
%
% Output : w - [Kx1 double] normal vector
%          b - [double] offset
%          crit - [#iterationsx1 double] value of objective function
%
% Examples:
%  [w, b, crit] = SparseRegularizedSVM_train(X,y,0.1)
%  [w, b, crit] = SparseRegularizedSVM_train(X,y,0.1,'penalization','L1','debug',1)
%  [w, b, crit] = SparseRegularizedSVM_train(X,y,[],[2,3,2]) - for groups
%  sparsity. [2,3,2] = 7 features within 3 groups
%
% J. Frecon, J. Spilka, N. Pustelnik, P. Abry,
% ENS Lyon, 2015

%% handling input parameters
p = inputParser;
p.addRequired('X',@(x) length(x)>1);
p.addRequired('y',@isvector);
p.addRequired('C',@isvector);
p.addOptional('aFeatNames',[]);
p.addParamValue('featgroups',[]);
p.addParamValue('debug',0,@isscalar);
p.addParamValue('data_fidelity','Hinge',@isstr);
expectedPen = {'L1','L12','L1inf'};
p.addParamValue('penalization','L1',@(x) any(validatestring(x,expectedPen)));
p.addParamValue('beta',0.5,@isscalar);
p.addParamValue('eps', 1e-11, @isscalar);
p.addParamValue('maxiter', 1e6, @isscalar);
p.parse(X,y,C,varargin{:});

aNrGroup        = p.Results.featgroups;
aFeatNames      = p.Results.aFeatNames;
bDebug          = p.Results.debug;
dataFidelity    = p.Results.data_fidelity;
penalization    = p.Results.penalization;
beta            = p.Results.beta;
eps             = p.Results.eps;
max_iter        = p.Results.maxiter;

[m,n] = size(X);
if m > n
    X = X';
end

[m,n] = size(y);
if m > n
    y = y';
end

if size(X,2) ~= length(y)
    error('Dimension must agree');
end

K = size(X,1);

if isempty(aFeatNames)
    for i = 1:K, aFeatNames{i} = ['feat ',num2str(i)]; end
end

if strcmpi(penalization,'L12')
    if isempty(aNrGroup), error('Number of features in each group must be specified');end
    if sum(aNrGroup) ~= K, error('Number of features in each group must be equal to size of matrix X');end
end

cl = sort(unique(y));
if cl(1) ~= -1 || cl(2) ~= 1
    error('Labels -1,1 not found in the input labels y\n');
end

%% Common data

% - Parameters
N          = size(X,2);
ind1       = y == -1;
ind2       = y == +1;
N1         = sum(ind1);
N2         = sum(ind2);
X1         = X(:,ind1);
X2         = X(:,ind2);
%C1         = C;
%C2        = (N1/N2) * C;
C1         = beta * C * N / N1;% normal
C2         = (1-beta) * C * N / N2;% pathological

% - Weights
weightp     = 1./sqrt(aNrGroup);
%atemp = weightp;

% - Algorithm Parameters
L           = 2*C1*sum( sum( X1.^2 ) ) + 2*C2*sum( sum( X2.^2 ) ) + 2*C1*N1 + 2*C2*N2;
%eps         = 10^-11;


% - Data Fidelity
if strcmp(dataFidelity,'Hinge')
    g.grad  = @(w, b)  2*C1*X1*max(0, 1 - b + X1'*w ) - 2*C2*X2*max(0, 1 + b - X2'*w );
    h.grad  = @(w, b) - 2*C1*sum(max(0, 1 - b + X1'*w )) + 2*C2*sum(max(0, 1 + b - X2'*w ));
    g.crit  = @(w, b) C1*sum(max(0,1 - b + X1'*w).^2)    + C2*sum(max(0,1+b-X2'*w).^2);
elseif strcmp(dataFidelity,'Logit')

end

% - Regularization
if strcmp(penalization,'L1')
    f.prox  = @(w,gamma) prox_L1(w, gamma);
    f.crit  = @(w) sum(abs(w));
elseif strcmp(penalization,'L12')
    f.prox  = @(w,nr,gamma) prox_L12(w, nr, gamma);
    f.crit  = @(w,nr) sum(nr.*sqrt(sum(w.^2,1)));
    %f.crit  = @(w,nr) sum(nr.*sqrt(sum(w.^2,1)));
    %f.crit  = @(w,nr) sum(sqrt(sum(w.^2,1)));
elseif strcmp(penalization,'L1inf')
    f.prox  = @(w,nr,gamma) prox_L1Linf(w, 1, gamma);
    f.crit  = @(w) max(sum(abs(w),1));
end


% - Initialization
b           = 2;
w           = zeros(K,1);
crit        = zeros(1);
critm       = g.crit(w,b) ;



%% Forward-Backward

flag        = false;
gamma       = 1.99/L;
j           = 1;

while ~flag

    % - Algorithm
    b0      = b - gamma*h.grad(w,b);
    w       = w-gamma*g.grad(w,b);

    if strcmpi(penalization,'L1')
        w       = f.prox(w,gamma);
        crit(j) = g.crit(w,b0) + f.crit(w) ;
        f1(j)   = f.crit(w);
        g1(j)   = g.crit(w,b0);

    elseif strcmpi(penalization,'L12')

        w_mat   = w_vec2mat(w,aNrGroup);

        w_mat   = f.prox(w_mat,weightp,gamma);
        f1(j)   = f.crit(w_mat,weightp);
        w       = w_mat2vec(w_mat,aNrGroup); % from matrix to vector

        g1(j)   = g.crit(w,b0);
        crit(j) = f1(j) + g1(j);

    elseif strcmpi(penalization,'L1inf')

        w_mat	= w_vec2mat(w,aNrGroup);
        w       = f.prox(w_mat,1,gamma);
        f1(j)   = f.crit(w_mat);
        w       = w_mat2vec(w_mat,aNrGroup); % from matrix to vector

        g1(j)   = g.crit(w,b0);
        crit(j) = f1(j) + g1(j);

    end

    % - Objective function
    %crit(j) = g.crit(w,b0) +f.crit(w) ;
    b       = b0;

    % - Stopping criterion
    if (abs(crit(j) - critm)/crit(j) < eps)  && (j < max_iter)
        flag= true;
    end

    if bDebug
        if j > 1
            if crit(j) > (crit(j-1));
                fprintf('t = %d, crit(j) > (crit(j-1): %1.15f\n',j,crit(j) - crit(j-1));
                %fprintf('%1.15f\n',f1(j) - f1(j-1));
                %fprintf('%1.15f\n',g1(j) - g1(j-1));
            end
        end
    end

    critm   = crit(j);
    j       = j + 1;

    if bDebug
        if rem(j,10^2) == 0
            %fprintf('t = %d, crit = %2.6f\n',j,critm);
        end
    end

end

%% Offset estimation
b = b/norm(w);
% b = b - 0.27133;
w = w/norm(w);

%% plotting

if bDebug

    %disp('Normal vector: w')
    %disp(w)
    %disp('Offset: b');
    %disp(b);

    figure(210); clf;
    set(gca,'fontsize',12);
    stem(w,'b','linewidth',2); hold on;
    xlabel('Features','Interpreter','latex');
    ylabel('Normal vector $w$','Interpreter','latex');
    title('Feature weights')
    grid on;

    [~,ind] = sort(abs(w),'descend');

    feat1 = ind(1);
    feat2 = ind(2);
    %feat1 = 1;
    %feat2 = 2;

%     nmin = min(min(X(feat1,:),X(feat2,:)));
%     nmax = max(max(X(feat1,:),X(feat2,:)));
%     test    = [nmin:.1:nmax];
%     alpha   = -w(feat1,:)/w(feat2,:);
%     beta    = sqrt(alpha^2+1)*b;
%     vector  = alpha*test + beta;
%
%     yhat = SparseRegularizedSVM_test( w, b, X);
%
%     figure(212); clf;
%     hold on;
%     set(gca,'fontsize',12);
%     gscatter(X(feat1,:),X(feat2,:),yhat,'rb','..'); hold on;
%     a = axis;
%     xlabel(aFeatNames{feat1});
%     ylabel(aFeatNames{feat2});
%     plot(test,vector,'b','LineWidth',2);
%     axis(a);
%     grid on;

    [yhat,d] = SparseRegularizedSVM_test( w, b, X, y);
    xt = linspace(min(X(feat1,:)),max(X(feat1,:)));
    yt = linspace(min(X(feat2,:)),max(X(feat2,:)));
    [Xorig,Yorig] = meshgrid(xt,yt);

    t = [Xorig(:),Yorig(:)];
    [~,f,~] = SparseRegularizedSVM_test(w([feat1,feat2]), b, t', yhat');
    [~,fm_pos,~] = SparseRegularizedSVM_test(w([feat1,feat2]), b+1, t', yhat');
    [~,fm_neg,~] = SparseRegularizedSVM_test(w([feat1,feat2]), b-1, t', yhat');

    Z = reshape(f,size(Xorig));
    Zm_pos = reshape(fm_pos,size(Xorig));
    Zm_neg = reshape(fm_neg,size(Xorig));

    idxTP = y == 1 & yhat == 1;
    idxFN = y == 1 & yhat == -1;
    idxTN = y == -1 & yhat == -1;
    idxFP = y == -1 & yhat == 1;
    X1 = X(feat1,:);
    X2 = X(feat2,:);

    figure(215); clf;
    hold on;
    %set(gca,'fontsize',12);
    cLegend = {};
    %gscatter(X(feat1,:),X(feat2,:),yhat,'rb','..'); hold on;
    if sum(idxTN+0) ~= 0, plot(X1(idxTN),X2(idxTN),'xb'); cLegend = [cLegend, {'TN'}]; end;
    if sum(idxFP+0) ~= 0, plot(X1(idxFP),X2(idxFP),'ob'); cLegend = [cLegend, {'FP'}]; end;
    if sum(idxFN+0) ~= 0, plot(X1(idxFN),X2(idxFN),'or','LineWidth',2); cLegend = [cLegend, {'FN'}]; end;
    if sum(idxTP+0) ~= 0, plot(X1(idxTP),X2(idxTP),'xr','LineWidth',2); cLegend = [cLegend, {'TP'}]; end;
    %clf; hold on;
    grid on;
    contour(Xorig,Yorig,Z,[0 0],'k','LineWidth',2);
    contour(Xorig,Yorig,Zm_pos,[0 0],'.k','LineWidth',1);
    contour(Xorig,Yorig,Zm_neg,[0 0],'.k','LineWidth',1);

    xlabel(aFeatNames{feat1});
    ylabel(aFeatNames{feat2});
    cLegend = [cLegend,'Boundary - 2 features'];
    legend(cLegend,'Location','Best')
    title('Scatter plot of two features')

    %figure(213)
    %hold on;
    %plot(crit,'r')
    %plot(100*f1,'k')
    %plot(g1,'b')
    %legend('crit','100*f1','g1+1')
    %title('Convergence rate')
    %grid on;
end

function new_w = w_vec2mat(w,nNr)
% size: max(nNrFeatG) x nNrFeatures

n = length(nNr);
d = max(nNr);
new_w = zeros(d,n);

for i = 1:n
    if i == 1
        new_w(1:nNr(i),i) = w(1:nNr(i));
    else
        offset = sum(nNr(1:i-1));
        new_w(1:nNr(i),i) = w(offset+1:offset+nNr(i));
    end
end

function new_p = w_mat2vec(p,nNr)

n = length(nNr);
d = max(nNr);
new_p = zeros(sum(nNr),1);

for i = 1:n
    if i == 1
        new_p(1:nNr(i)) = p(1:nNr(i),i);
    else
        offset = sum(nNr(1:i-1));
        new_p(offset+1:offset+nNr(i)) = p(1:nNr(i),i);
    end
end