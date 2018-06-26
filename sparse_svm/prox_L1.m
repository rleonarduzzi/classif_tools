function wp = prox_L1(wx,gamma)
% Compute L1 proximal operator.
%
% J. Frecon, J. Spilka, N. Pustelnik, P. Abry,
% ENS Lyon, 2015

wp = max(abs(wx)-gamma,0).*sign(wx);
