close all
clearvars
clc
load("exp_2_filter_successful_0_subgoals_0.mat")
%load("exp_2_filter_successful_0_subgoals_1.mat")
%load("exp_2_filter_successful_0_subgoals_2.mat")

num_obs_train_categories = [41,34,27,31,32,35];
num_obs_eval_categories = [18,15,14,16,19,18];

num_categories = size(irl_results_train_category,1);
num_models = size(irl_results_train_category,2);

neg_log_likelihood = zeros(num_categories, num_models);
aic = zeros(num_categories, num_models);
bic = zeros(num_categories, num_models);
for i=1: num_categories
    numObs = num_obs_train_categories(i);
    %fprintf("Results for Category: %d \n", i);
    for j=1:num_models
        if iscell(irl_results_train_category)
            irl_result_train = irl_results_train_category{i,j};
        else
            irl_result_train = irl_results_train_category(i,j);
        end
        %fprintf("Model: %d  Negative Log Likelihood: %f \n", j, irl_result_train.log_like);
        neg_log_likelihood(i,j) = irl_result_train.log_like;
        numParam = length(irl_result_train.wts);
        [aic(i,j),bic(i,j)] = aicbic(-neg_log_likelihood(i,j),numParam,numObs,Normalize=true);
    end
    fprintf("\n");
end
%disp(neg_log_likelihood)
[min_neg_log_like,model_num_neg_log_like] = min(neg_log_likelihood, [], 2);
[min_aic,model_num_aic] = min(aic, [], 2);
[min_bic,model_num_bic] = min(bic, [], 2);

disp("Negative Log Likelihood")
disp(neg_log_likelihood)
disp(model_num_neg_log_like)

disp("AIC")
disp(aic)
disp(model_num_aic)

disp("BIC")
disp(bic)
disp(model_num_bic)
