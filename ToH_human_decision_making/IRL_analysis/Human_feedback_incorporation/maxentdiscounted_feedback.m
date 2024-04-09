% Compute MaxEnt objective and gradient. Discounted infinite-horizon version.
function val = maxentdiscounted_feedback(wts,F,muE,mu_sa,mdp_data,initD,laplace_prior, model_num, H)

if nargin < 7,
    laplace_prior = 0;
end;

% Compute constants.
actions = mdp_data.actions;

% Convert to full reward.
[row,~,~] = find(F); %Finds non zero indices
rew = zeros(size(F,1),1);

if model_num==0
    rew(row) = wts;
    r = F*rew;
    r_prime = repmat(r,1,actions);
    feedback = 0;
elseif model_num==1 || model_num==4
    rew(row) = wts(1:end-1);
    r_init = F*rew;
    r_prime = repmat(r_init, 1, actions);
    feedback = wts(end)*H;
elseif model_num==2
    rew(row) = wts(1:end-1);
    r_init = F*rew;
    r_init = repmat(r_init, 1, actions);
    r_prime = r_init + wts(end)*H;
    feedback = 0;
elseif model_num==3
    r_prime = 0;
    feedback = wts*H;
end
[~,~,policy,logpolicy] = linearvalueiteration_models(mdp_data, r_prime, model_num, feedback); 
%[~,~,policy,logpolicy] = td_learning_models(mdp_data,r_prime, model_num, feedback);

% Compute value by adding up log example probabilities.
val = sum(sum(logpolicy.*mu_sa));   % This is computing a mean Q value by averaging with the probability of observing (s,a) 

% Add laplace prior.
if laplace_prior,
    val = val - laplace_prior*sum(abs(wts));
end;

% Invert for descent.
val = -val;