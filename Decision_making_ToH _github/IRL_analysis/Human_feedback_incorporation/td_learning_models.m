% Run value iteration to solve a linear MDP.
function [v,q,p,logp] = td_learning_models(mdp_data,r, model_num, feedback)

states = mdp_data.states;
actions = mdp_data.actions;
CONV_THRESH = 1e-4;

% Compute log state partition function V.
q = zeros(states,actions);
alpha = 0.1;
diff = 1.0;

while diff >= CONV_THRESH,
    qp=q;        
    % Compute q function.
    q = qp+ alpha*(r+ mdp_data.discount*sum(mdp_data.sa_p.*qp(mdp_data.sa_s),3) - qp);                      
    % Compute softmax.
    v = maxentsoftmax(q);
    
    diff = max(max(abs(q-qp)));
end;
if model_num==1
    q=q+feedback;
    v=maxentsoftmax(q);
end
% Compute policy.
logp = q - repmat(v,1,actions);
p = exp(logp);