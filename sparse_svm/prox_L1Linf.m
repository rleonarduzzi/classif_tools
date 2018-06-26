 function p = prox_L1Linf(x, w, gamma)
%function p = prox_L1Linf(x, w, gamma)
%
%  Created on: 16/04/2012 - Giovanni Chierchia
%
% The procedure computes the proximity operator of the function:
%
%           f(x) = gamma * sum( w(i) * ||x(:,i)||_\inf )
%
% where 'w' is a vector of non-negative values.

% default inputs
if nargin < 2 || isempty(w)
    w = 1;
end
if nargin < 3 || isempty(gamma)
    gamma = 1;
end

% check input
if any( w(:) < 0 )
    error('''w'' must be non negative');
end
[B N] = size(x);
if ~isscalar(w) && N ~= numel(w)
    error('The weights are not compatible')
end
%-----%


% reshape the weights
sz = size(x);
if isscalar(w)
    w = w * ones( [1 sz(2:end)] );
else
    w = reshape(w, [1 sz(2:end)]);
end

% compute the L1 projection (Ref.: Duchi et al - Efficient Projections onto the L1-Ball for Learning in High Dimensions - 2008)
p = sign(x) .* proj_colwise( abs(x), gamma*w );

% compute the prox (via the conjugate property)
p = x - p;





function p = proj_colwise(x, eta)

% 0/4 - column selection
sel = find( sum(x,1) > eta );

% 1/4 - ordering
s = sort( x(:,sel), 1, 'descend' );

% 2/4 - computing c(j,k) = ( sum( s(1:j,k) ) - eta(k) ) / j
[B N] = size(s);
c = ( cumsum(s,1) - repmat(eta(:,sel),[B 1]) ) ./ repmat( (1:B)', [1 N] );

% 3/4 - finding n(k) = max{ j \in {1,...,B} : s(j,k) > c(j,k) }
ndx = sum(s > c, 1);
ndx = sub2ind([B N], ndx, 1:N);

% 4/4 - projection
p = x;
p(:,sel) = max( 0, x(:,sel) - repmat(c(ndx),[B 1]) );