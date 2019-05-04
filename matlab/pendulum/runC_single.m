function runC_single(trial, do_retrace, reg_type)

clear basis_fourier
rng(trial)

mdp = Pendulum;

tmp_policy.drawAction = @(x)mymvnrnd(zeros(mdp.daction,1), 16*eye(mdp.daction), size(x,2));
ds = collect_samples(mdp, 100, 100, tmp_policy);
B = avg_pairwise_dist([ds.s]);
bfs = @(varargin) basis_fourier(100, mdp.dstate, B, 0, varargin{:});

A0 = zeros(mdp.daction,bfs()+1);
Sigma0 = 16*eye(mdp.daction);
policy = GaussianLinearDiag(bfs, mdp.daction, A0, Sigma0);

episodes_eval = 1000;
episodes_learn = 10;
steps_eval = 150;
steps_learn = 50;
maxiter = 1000;


folder = ['data_single/'];
mkdir(folder)
if do_retrace
    RETR = 'R';
else
    RETR = [];
end
if reg_type == 5
    ALG = 'c2a';
elseif reg_type == 4
    ALG = 'c2t';
elseif reg_type == 3
    ALG = 'i';
elseif reg_type == 2
    ALG = 'c';
elseif reg_type == 1
    ALG = 'a';
elseif reg_type == 0
    ALG = 't';
elseif reg_type == -1
    ALG = 'v';
end


% To learn V
options = optimoptions(@fminunc, 'Algorithm', 'trust-region', ...
    'GradObj', 'on', ...
    'Display', 'off', ...
    'MaxFunEvals', 100, ...
    'Hessian', 'on', ...
    'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 100);

mdp.gamma = 0.99;
kl_bound = 0.01;
lambda_trace = 0.95;

bfsV = bfs;
omega = (rand(bfsV(),1)-0.5)*2;
omega_c = zeros(bfs() + mdp.daction + 1, 1);

data = [];
varnames = {'r','s','nexts','a','t','terminal','logprob'};
bfsnames = { {'phiV', bfsV} };
iter = 1;

max_reuse = 5; % Reuse all samples from the past X iterations
max_samples = zeros(1,max_reuse);

max_ratio = 0;

%% Learning
while iter <= maxiter
    
    % Collect data
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    for i = 1 : numel(ds)
        ds(i).logprob = policy.logpdf(ds(i).a, ds(i).s);
    end
    entropy = policy.entropy([ds.s]);
    max_samples(mod(iter-1,max_reuse)+1) = size([ds.s],2);
    data = getdata(data,ds,sum(max_samples),varnames,bfsnames);
    prob_ratio = exp(policy.logpdf(data.a, data.s) - data.logprob);
    if do_retrace
        prob_ratio = min(1,prob_ratio);
    end
    
    % Train V
    V = omega'*data.phiV;
    A = gae(data,V,mdp.gamma,lambda_trace,prob_ratio);
    omega = fminunc(@(omega)mse_linear(omega,data.phiV,V+A), omega, options);
    
    % Estimate A and TD
    V = omega'*data.phiV;
    A = gae(data,V,mdp.gamma,lambda_trace,prob_ratio);
    TD = gae(data,V,mdp.gamma,0,prob_ratio);
    td_history(iter) = mean(TD.^2);
    
    % TODO Project A to compatible function space. Bad.
    % Also tried: Projecting A+V=Q and using projected Q (optionally
    % subtract V baseline again). Same result. The A_unnorm is used in the
    % working method as well, so it is not the normalization.
    dlogpi = policy.dlogPidtheta(data.s,data.a);
    omega_c = fminunc(@(omega_c)mse_linear(omega_c,dlogpi,A), omega_c, options);
    A = omega_c' * dlogpi;
    A_unnorm = A;
    
    % Estimate natural gradient
    A = (A-mean(A))/std(A);
    TD = (TD-mean(TD))/std(TD);
    if reg_type == 5
        REG = A_unnorm.^2;
        ratio = norm(REG) / norm(A);
        if max_ratio < ratio
            max_ratio = ratio;
        end
        REG_scaled = REG / max_ratio;
        REG_clipped = min(REG_scaled, abs(A));
        REG_term = REG_clipped .* sign(A);
        X = A - REG_term;  
    elseif reg_type == 4
        REG = TD.^2;
        ratio = norm(REG) / norm(A);
        if max_ratio < ratio
            max_ratio = ratio;
        end
        REG_scaled = REG / max_ratio;
        REG_clipped = min(REG_scaled, abs(A));
        REG_term = REG_clipped .* sign(A);
        X = A - REG_term;  
    elseif reg_type == 3
        REG = A.^2;
        ratio = norm(REG) / norm(A);
        if max_ratio < ratio
            max_ratio = ratio;
        end
        X = A + 0.8 * REG / max_ratio;  
    elseif reg_type == 2
        REG = A.^2;
        ratio = norm(REG) / norm(A);
        if max_ratio < ratio
            max_ratio = ratio;
        end
        X = A - REG / max_ratio;  
    elseif reg_type == 1
        REG = A.^2;
        REG = (REG-mean(REG))/std(REG);
        l_base = 1;
        X = A - l_base*0.999^iter*REG;  
    elseif reg_type == 0
        REG = TD.^2;
        REG = (REG-mean(REG))/std(REG);
        l_base = 1;
        X = A - l_base*0.999^iter*REG;
    elseif reg_type == -1
        REG = 0;
        l_base = 0;
        X = A - l_base*0.999^iter*REG;
    end
    grad = mean(bsxfun(@times,dlogpi,X),2);
    F = dlogpi * dlogpi' / length(A);
    [grad_nat,~,~,~,~] = pcg(F,grad,1e-10,50); % Use conjugate gradient (~ are to avoid messages)
    
    % Line search
    stepsize = sqrt(kl_bound / (0.5*grad'*grad_nat));
    max_adv = @(theta) mean(policy.update(theta).logpdf([data.a],[data.s]).*X);
    kl = @(theta) kl_mvn2(policy.update(theta), policy, policy.basis(data.s));
    [success, theta, n_back] = linesearch(max_adv, policy.theta, stepsize*grad_nat, grad'*grad_nat*stepsize, kl, kl_bound);
    if ~success, warning('Could not satisfy the KL constraint.'), end % in this case, theta = policy.theta
    
    % Print info
    norm_g1 = norm(grad);
    norm_g2 = norm(grad);
    norm_ng = norm(grad_nat);
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    fprintf('%d) Entropy: %.2f,   mean(norm(A)): %e,   td_history(iter): %.2f,   Norm (NG): %e,   J: %e \n', ...
        iter, entropy, mean(norm(A_unnorm)), td_history(iter), norm_ng, J);
    J_history(iter) = J;
    e_history(iter) = entropy;
    
    % Update pi
    policy = policy.update(theta);
    
    iter = iter + 1;
    
end

t = policy.theta;
save([folder RETR 'C' ALG '_' num2str(trial) '.mat'], 't', 'J_history', 'e_history', 'td_history');
