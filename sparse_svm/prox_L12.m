function p = prox_L12(w,nNr,gamma)
% Compute L12 proximal operator.
%
% J. Frecon, J. Spilka, N. Pustelnik, P. Abry,
% ENS Lyon, 2015

[R,N]   = size(w);
p       = zeros(R,N);



tmp         = sqrt(sum(w.^2,1));
ind         = find(tmp>gamma.*nNr);
p(:,ind)    = (ones(R,1)*(1 - (gamma.*nNr(ind))./tmp(ind))).*w(:,ind);