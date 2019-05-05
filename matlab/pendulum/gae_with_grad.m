function [A, dA] = gae_with_grad(data, omega, phiV, gamma, lambda, prob_ratio)
% Computes generalized advantage estimates from potentially off-policy data.
% Data have to be ordered by episode: data.r must have first all samples 
% from the first episode, then all samples from the second, and so on.
% So you cannot use samples collected with COLLECT_SAMPLES2.
%
% Do not pass PROB_RATIO if data is on-policy.
% Truncate PROB_RATIO = min(1,PROB_RATIO) to use Retrace.
%
% =========================================================================
% REFERENCE
% J Schulman, P Moritz, S Levine, M Jordan, P Abbeel
% High-Dimensional Continuous Control Using Generalized Advantage Estimation
% ICLR (2017)
%
% R Munos, T Stepleton, Anna Harutyunyan, M G Bellemare
% Safe and efficient off-policy reinforcement learning
% NIPS (2016)

V = omega'*phiV;
A = gae(data, V, gamma, lambda, prob_ratio);

r = [data.r];
t = [data.t];
t(end+1) = 1;
dA = zeros(size(phiV));

if nargin == 4 || isempty(prob_ratio), prob_ratio = ones(size(V)); end

for k = size(phiV,2) : -1 : 1
    if t(k+1) == 1 % Next state is a new episode init state
        dA(k) = prob_ratio(k) * (r(k) - phiV(k));
    else
        dA(k) = prob_ratio(k) * (r(k) + gamma*phiV(k+1) - phiV(k) + gamma*lambda*dA(k+1));
    end
end