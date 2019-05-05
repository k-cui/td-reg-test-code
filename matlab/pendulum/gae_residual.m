function [f, df, ddf] = gae_residual(data, omega, phiV, gamma, lambda, prob_ratio)
% Computes generalized advantage estimate norms from potentially off-policy data.
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

[A, dA] = gae_with_grad(data, omega, phiV, gamma, lambda, prob_ratio);

% Mean squared error for linear functions
% min_w ||Y-T||^2,   Y = w'X
%
% X and T must be [D x N] matrices, where D is the dimensionality of
% the data and N is the number of samples.

E = A; % T are the targets
f = 0.5*mean(mean(E.^2)); % Function, MSE
df = dA*mean(E,1)'/size(A,2); % Gradient
ddf = dA*dA'/size(A,2); % Hessian
