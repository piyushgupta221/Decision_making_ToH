% Fill in default parameters for the MaxEnt algorithm.
function algorithm_params = maxentdefaultparams(algorithm_params, laplace_prior)

% Create default parameters.
default_params = struct(...
    'seed',0,...
    'laplace_prior',laplace_prior,...
    'all_features',0,...
    'true_features',0);

% Set parameters.
algorithm_params = filldefaultparams(algorithm_params,default_params);


% VS note: changed 'all_features' to 0 and 'laplace_prior to 1'