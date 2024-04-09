% Run value iteration to solve a linear MDP.
function [v,q,p,logp] = linearvalueiteration_models(mdp_data,r, model_num, feedback)

states = mdp_data.states;
actions = mdp_data.actions;
VITR_THRESH = 1e-4;
%VITR_THRESH = 1e-10;

% Compute log state partition function V.
v = zeros(states,1);
diff = 1.0;
while diff >= VITR_THRESH,
    % Initialize new v.
    vp = v;
    % Compute q function.
    q = r + mdp_data.discount*sum(mdp_data.sa_p.*vp(mdp_data.sa_s),3);
    % Compute softmax.
    v = maxentsoftmax(q);
    
    diff = max(abs(v-vp));
end;

if model_num==1
    q=q+feedback;
    v=maxentsoftmax(q);
elseif model_num==3
    q = feedback;
    v=maxentsoftmax(q);
end

if model_num==4
    logits = exp(q+feedback);
    denom = sum(logits,2)*ones(1,size(logits,2));
    p = logits./denom;
    logp = log(p);

else
    % Compute policy.
    logp = q - repmat(v,1,actions);
    p = exp(logp);
end
