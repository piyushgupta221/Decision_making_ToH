close all
clearvars
clc
addpaths;

%{
Model_num = 0 ; No feedback incorporation
Model_num = 1 ; Q'(s,a) = Q(s,a) + w*H(s,a)
Model_num = 2 ; R'(s,a) = R(s,a) + w*H(s,a)
Model_num = 3 ; Q'(s,a) = w*H(s,a)
Model_num = 4 ; P(a|s) = exp(Q(s,a) + w*H(s,a))
%}
all_models = [0,1,2,3,4];
exp=2;
filter_only_successful = false;
set_subgoals = 2;   % set = 0,1,2   : 0: all states, 1: 8 features, 2: 16 features

consider_eval = false;

cross_validation=true;  %Keep this to true for cross validation
k_fold = 5;

if isempty(gcp('nocreate'))
    parpool;
end

% No need to set these, only used if no cross-validation
default_laplace_prior = 0.0;
abs_state = 0; %abs_state=0 means we are not using abs_state
plot_figures = false;


d_train = load("states_and_adjacency_4_3.mat");
states_list_train = squeeze(d_train.state_string);
adjacency_train = double(d_train.adjacency);
nodes_pos_train = double(d_train.nodes_pos);

d_eval = load("states_and_adjacency_5_3.mat");
states_list_eval = squeeze(d_eval.state_string);
adjacency_eval = double(d_eval.adjacency);
nodes_pos_eval = double(d_eval.nodes_pos);


G_train = graph(adjacency_train);
G_eval = graph(adjacency_eval);
distance_nodes_train= distances(G_train);
distance_nodes_eval = distances(G_eval);

load("toh_participants_data.mat");


n_train=length(states_list_train);
n_eval = length(states_list_eval);
determinism=1;
discount=0.95;

[sa_s_train, sa_p_train] = set_transtitions(adjacency_train, determinism);
[sa_s_eval, sa_p_eval] = set_transtitions(adjacency_eval, determinism);

[triangle_vertex_train_string, triangle_vertex_eval_string, triangle_vertex_train_num, triangle_vertex_eval_num] = get_triangle_vertices(states_list_train, states_list_eval);

% Create a map
H_map_train_exp2 = containers.Map('KeyType', 'double', 'ValueType', 'any');
H_map_train_exp5 = containers.Map('KeyType', 'double', 'ValueType', 'any');

% Loop through the cell array of target states
for i = 1:length(triangle_vertex_train_num)
    target_state = triangle_vertex_train_num(1,i);
    target_state_string = triangle_vertex_train_string(1,i);
    if target_state_string{1,1}(end)=='2'
        sub_goal = '1112';
    elseif target_state_string{1,1}(end)=='1'
        sub_goal = '2221';
    end
    sub_goal_num = find(ismember(states_list_train, sub_goal, 'rows'));
    H_numeric_feedback = get_feedback(distance_nodes_train, sa_s_train, sa_p_train, target_state);
    H_subgoal_with_numeric_feedback = get_feedback_subgoal(distance_nodes_train, sa_s_train, sa_p_train, target_state, sub_goal_num);
    % Add H to the map with target_state as the key
    H_map_train_exp2(target_state) = H_numeric_feedback;
    H_map_train_exp5(target_state) = H_subgoal_with_numeric_feedback;
end

if consider_eval==true
    % Create a map
    H_map_eval_exp2 = containers.Map('KeyType', 'double', 'ValueType', 'any');
    H_map_eval_exp5 = containers.Map('KeyType', 'double', 'ValueType', 'any');
    
    % Loop through the cell array of target states
    for i = 1:length(triangle_vertex_eval_num)
        target_state = triangle_vertex_eval_num(1,i);
        target_state_string = triangle_vertex_eval_string(1,i);
        if target_state_string{1,1}(end)=='2'
            sub_goal = '11112';
        elseif target_state_string{1,1}(end)=='1'
            sub_goal = '22221';
        end
        sub_goal_num = find(ismember(states_list_eval, sub_goal, 'rows'));
        H_numeric_feedback = get_feedback(distance_nodes_eval, sa_s_eval, sa_p_eval, target_state);
        H_subgoal_with_numeric_feedback = get_feedback_subgoal(distance_nodes_eval, sa_s_eval, sa_p_eval, target_state, sub_goal_num);
        % Add H to the map with target_state as the key
        H_map_eval_exp2(target_state) = H_numeric_feedback;
        H_map_eval_exp5(target_state) = H_subgoal_with_numeric_feedback;
    end
end

if set_subgoals~=0
    [subgoal_states_train_num, subgoal_states_eval_num] = subgoals(states_list_train, states_list_eval, set_subgoals);
else
    subgoal_states_train_num = 1:size(states_list_train,1);
    subgoal_states_eval_num = 1:size(states_list_eval,1);
end

laplace_train_category = zeros(6,length(all_models));
laplace_eval_category = zeros(6,length(all_models));

if exp==1   
    [standard_percentage_score_examples_train, standard_percentage_score_examples_eval, target_states_examples_train, target_states_examples_eval, exp_new_example_samples_train, exp_new_example_samples_eval] = get_state_trajectories(exp1_participants_new, states_list_train, states_list_eval, sa_s_train, sa_s_eval);  
elseif exp==2
     [standard_percentage_score_examples_train, standard_percentage_score_examples_eval, target_states_examples_train, target_states_examples_eval, exp_new_example_samples_train, exp_new_example_samples_eval] = get_state_trajectories(exp2_participants_new, states_list_train, states_list_eval, sa_s_train, sa_s_eval);  
elseif exp==3
     [standard_percentage_score_examples_train, standard_percentage_score_examples_eval, target_states_examples_train, target_states_examples_eval, exp_new_example_samples_train, exp_new_example_samples_eval] = get_state_trajectories(exp5_participants_new, states_list_train, states_list_eval, sa_s_train, sa_s_eval);  
end

irl_results_train_category = [];
irl_results_eval_category = [];
end_states = {'12','22','02','21','01','11'};
target_states_end_train = [];
target_states_end_eval = [];
for k = 1:size(end_states,2)  % Corrected from 'en_states' to 'end_states'
    fprintf('Processing category: %d\n', k);
    category = end_states{1,k};
    eval(sprintf('example_train_%s = categorize_examples(standard_percentage_score_examples_train, target_states_examples_train, exp_new_example_samples_train, category, filter_only_successful, states_list_train);', category));
    examples_train_category = eval(sprintf('example_train_%s', category));
    target_state_train_num = triangle_vertex_train_num(cellfun(@(x) ~isempty(regexp(x, [category '$'], 'once')), triangle_vertex_train_string));
    target_states_end_train = [target_states_end_train; target_state_train_num];
    


    if consider_eval==true
        eval(sprintf('example_eval_%s = categorize_examples(standard_percentage_score_examples_eval, target_states_examples_eval, exp_new_example_samples_eval, category, filter_only_successful, states_list_eval);', category));      
        examples_eval_category = eval(sprintf('example_eval_%s', category));
        target_state_eval_num = triangle_vertex_eval_num(cellfun(@(x) ~isempty(regexp(x, [category '$'], 'once')), triangle_vertex_eval_string));   
        target_states_end_eval = [target_states_end_eval; target_state_eval_num];
    end

    if exp==2
        H_train = H_map_train_exp2(target_state_train_num);
        if consider_eval==true
            H_eval = H_map_eval_exp2(target_state_eval_num);
        end
    elseif exp==3
        H_train = H_map_train_exp5(target_state_train_num);
        if consider_eval==true
            H_eval = H_map_eval_exp5(target_state_eval_num);
        end
    end
    % Update sa_s_train
    updated_sa_s_train = sa_s_train;
    updated_sa_p_train = sa_p_train;
    successors = zeros(1,1,4);
    for s=target_state_train_num
        successors(1,1,1) = s;
        successors(1,1,2) = s;
        successors(1,1,3) = s;
        successors(1,1,4) = s;        
        updated_sa_s_train(s,:,:) = repmat(successors,[1,4,1]);
        updated_sa_p_train(s,:,:) = reshape([zeros(4,3) ones(4,1)],...
            1,4,4);
    end
    if consider_eval==true
        % Update sa_s_eval
        updated_sa_s_eval = sa_s_eval;
        updated_sa_p_eval = sa_p_eval;
        successors = zeros(1,1,4);
        for s=target_state_eval_num
            successors(1,1,1) = s;
            successors(1,1,2) = s;
            successors(1,1,3) = s;
            successors(1,1,4) = s;        
            updated_sa_s_eval(s,:,:) = repmat(successors,[1,4,1]);
            updated_sa_p_eval(s,:,:) = reshape([zeros(4,3) ones(4,1)],...
                1,4,4);
        end
    end
    numModels = length(all_models);
    irl_results_train_models = cell(1, numModels);
    irl_results_eval_models = cell(1, numModels);
    parfor alpha=1:length(all_models)
        fprintf('Processing Model Number: %d\n', alpha);
        model_number = all_models(alpha);
        if cross_validation==true
            laplace_train_category(k,alpha)= run_cross_validation(n_train,discount,determinism, updated_sa_s_train, updated_sa_p_train, adjacency_train, examples_train_category, target_state_train_num, k_fold, subgoal_states_train_num, H_train, model_number);
            if consider_eval==true
                laplace_eval_category(k,alpha)= run_cross_validation(n_eval,discount,determinism, updated_sa_s_eval, updated_sa_p_eval, adjacency_eval, examples_eval_category, target_state_eval_num, k_fold, subgoal_states_eval_num, H_eval, model_number);
            end
        else
            laplace_train_category(k,alpha) = default_laplace_prior;
            if consider_eval==true
                laplace_eval_category(k,alpha)= default_laplace_prior;
            end
        end
        fprintf('Finished cross validation');
        [irl_result_train, perf_met_train]= run_irl_helper(n_train,discount,determinism, updated_sa_s_train, updated_sa_p_train, adjacency_train, examples_train_category, laplace_train_category(k,alpha), plot_figures, target_state_train_num, subgoal_states_train_num, nodes_pos_train, H_train, model_number);
        irl_results_train_models{1,alpha} =irl_result_train;
                  
        if consider_eval==true
            [irl_result_eval, perf_met_eval]= run_irl_helper(n_eval,discount,determinism, updated_sa_s_eval, updated_sa_p_eval, adjacency_eval, examples_eval_category, laplace_eval_category(k,alpha), plot_figures, target_state_eval_num, subgoal_states_eval_num, nodes_pos_eval, H_eval, model_number);
            irl_results_eval_models{1,alpha} = irl_result_eval;   
        end
    end
    irl_results_train_category = [irl_results_train_category;irl_results_train_models];
    if consider_eval==true
        irl_results_eval_category = [irl_results_eval_category;irl_results_eval_models];
    end
end

%{
create_plots(irl_results_train_experiments, adjacency_train, nodes_pos_train, false, target_states_end_train)
if consider_eval==true
    create_plots(irl_results_eval_experiments, adjacency_eval, nodes_pos_eval, true, target_states_end_eval)
end
%}

base_name = "Human_feedback_results/";
targeted_file = sprintf('exp_%d_filter_successful_%d_subgoals_%d.mat', exp, filter_only_successful, set_subgoals);
filename = base_name+targeted_file; 

if consider_eval==true
    save(filename, 'irl_results_train_category', 'adjacency_train', 'nodes_pos_train', 'target_states_end_train', 'laplace_train_category', 'irl_results_eval_category', 'adjacency_eval', 'nodes_pos_eval', 'target_states_end_eval', 'laplace_eval_category');
else
    save(filename, 'irl_results_train_category', 'adjacency_train', 'nodes_pos_train', 'target_states_end_train', 'laplace_train_category');
end

function laplace_cross_validation = run_cross_validation(n, discount, determinism, sa_s, sa_p, adjacency, example_samples, abs_state, k_fold, subgoal_states_num, H, model_number)
    nodes_pos = 0; %Set randomly, not needed
    example_samples = example_samples(randperm(size(example_samples,2)));
    num_trajectories = size(example_samples,2);
    laplace_prior_min=0;
    laplace_prior_max=1;
    laplace_prior_values = linspace(laplace_prior_min, laplace_prior_max, 6);
    log_likelihood_train = zeros(length(laplace_prior_values), k_fold);
    log_likelihood_val = zeros(length(laplace_prior_values), k_fold);
    if (num_trajectories>=2*k_fold)
        % Initialize cell arrays to store training and validation sets
        trainingSets = cell(1, k_fold);
        validationSets = cell(1, k_fold);
        plot = false;
        % Create k-fold cross-validation partitions
        for fold = 1:k_fold
            fprintf('Cross validation fold: %d\n', fold);
            % Calculate the indices for the current fold
            startIdx = 1 + (fold - 1) * (num_trajectories / k_fold);
            endIdx = fold * (num_trajectories / k_fold);
            % Initialize training and validation sets for the current fold
            trainingSet = {};
            validationSet = {};
            % Populate the training and validation sets
            for i = 1:num_trajectories
                if i >= startIdx && i <= endIdx
                    validationSet{end+1} = example_samples{i};
                else
                    trainingSet{end+1} = example_samples{i};
                end
            end
            % Store the sets in the cell arrays
            trainingSets{fold} = trainingSet;
            validationSets{fold} = validationSet;
            for l=1:length(laplace_prior_values)
                lap = laplace_prior_values(l);
                fprintf('Trying laplce value: %d\n', lap);
                train = true;
                learned_wts = 0; %Random Value
                [irl_result_train, ~, F] = run_irl(n, discount, determinism, sa_s, sa_p, adjacency, trainingSet, lap, plot, abs_state, train, learned_wts, subgoal_states_num, nodes_pos, H, model_number);
                learned_wts = irl_result_train.wts; %Learned reward
                % Evaluate negative log likelihood by setting 0 laplace prior
                train = false;
                [irl_result_train, ~, F] = run_irl(n, discount, determinism, sa_s, sa_p, adjacency, trainingSet, lap, plot, abs_state, train, learned_wts, subgoal_states_num, nodes_pos, H, model_number);
                log_likelihood_train(l,fold) = irl_result_train.log_like;
                [irl_result_val, perf_met_val, F] = run_irl(n, discount, determinism, sa_s, sa_p, adjacency, validationSet, lap, plot, abs_state, train, learned_wts, subgoal_states_num, nodes_pos, H, model_number);                    
                log_likelihood_val(l,fold) = irl_result_val.log_like;
            end

        end
        mean_log_like_train = mean(log_likelihood_train,2)';
        mean_log_like_val = mean(log_likelihood_val,2)';
        
        [~,I] = min(mean_log_like_val);
        laplace_cross_validation = laplace_prior_values(I);
    else
        disp("Number of trajectories not enough to perform cross validation");
        laplace_cross_validation = 0.0;
    end
end


function  [irl_result, perf_met] = run_irl_helper(n, discount, determinism, sa_s, sa_p, adjacency, example_samples, laplace_prior, plot_figures, abs_state, subgoal_states_num, nodes_pos, H, model_number)
    example_samples = example_samples(randperm(size(example_samples,2)));
    train = true;
    learned_wts = 0; %Random Value
    [irl_result, perf_met, F] = run_irl(n, discount, determinism, sa_s, sa_p, adjacency, example_samples, laplace_prior, plot_figures, abs_state, train, learned_wts, subgoal_states_num, nodes_pos, H, model_number);
end

function [standard_percentage_score_examples_train, standard_percentage_score_examples_eval, target_states_examples_train, target_states_examples_eval, exp_new_example_samples_train, exp_new_example_samples_eval] = get_state_trajectories(experiment, states_list_train, states_list_eval, sa_s_train, sa_s_eval)    
    
    standard_percentage_score_examples_train=[];
    standard_percentage_score_examples_eval=[];
    target_states_examples_train = [];
    target_states_examples_eval = [];
    exp_new_example_samples_train = {};
    exp_new_example_samples_eval = {};
    for participant=1:length(experiment)
        moves_train = experiment(1,participant).moves_train;
        moves_eval = experiment(1,participant).moves_eval;
        standard_percentage_score_train = experiment(1,participant).standard_percentage_score_train;
        standard_percentage_score_eval = experiment(1,participant).standard_percentage_score_eval;
        reaction_time_train = experiment(1,participant).reaction_time_train;
        reaction_time_eval = experiment(1,participant).reaction_time_eval;
        target_state_train = strsplit(experiment(1,participant).target_state_train, ',');
        target_state_eval = strsplit(experiment(1,participant).target_state_eval, ',');
        state_transitions_train = experiment(1,participant).state_transitions_train;
        state_transitions_eval = experiment(1,participant).state_transitions_eval;
        target_states_examples_train = [target_states_examples_train, target_state_train];
        target_states_examples_eval = [target_states_examples_eval, target_state_eval];
        standard_percentage_score_examples_train = [standard_percentage_score_examples_train, standard_percentage_score_train];
        standard_percentage_score_examples_eval = [standard_percentage_score_examples_eval, standard_percentage_score_eval];
         
        cum_sum = 0;
        for trial=1:length(moves_train)
            state_trajectory = string(state_transitions_train(cum_sum+1:cum_sum+1+moves_train(trial)))';
            traj_s_a = ones(length(state_trajectory),3);
            desired_length = size(states_list_train,2);
            for state=1:length(state_trajectory)
               if (length(state_trajectory(state))<desired_length) %PAD with leading zeros 
                   state_trajectory(state) =sprintf(['%0', num2str(desired_length), 's'], state_trajectory(state));
               end
               traj_s_a(state,2)= find(ismember(states_list_train, state_trajectory(state), 'rows'));
               if (state ~=1)
                   previous_state = traj_s_a(state-1,2);
                   current_state = traj_s_a(state,2);
                   if sa_s_train(previous_state, 1,1)==current_state
                       traj_s_a(state-1,3)=1;
                   elseif sa_s_train(previous_state, 1,2)==current_state
                       traj_s_a(state-1,3)=2;
                   elseif sa_s_train(previous_state, 1,3)==current_state
                       traj_s_a(state-1,3)=3;
                   else
                       error('Invalid state transition. Check adjacency matrix');
                   end
               end
            end
            traj_s_a(length(state_trajectory),3)=4;
            cum_sum = cum_sum+moves_train(trial)+1;
            exp_new_example_samples_train{end+1} = traj_s_a;
        end

        cum_sum = 0;
        for trial=1:length(moves_eval)
            state_trajectory = string(state_transitions_eval(cum_sum+1:cum_sum+1+moves_eval(trial)))';
            traj_s_a = ones(length(state_trajectory),3);
            desired_length = size(states_list_eval,2);
            for state=1:length(state_trajectory)
               if (length(state_trajectory(state))<desired_length) %PAD with leading zeros 
                   state_trajectory(state) =sprintf(['%0', num2str(desired_length), 's'], state_trajectory(state));
               end
               traj_s_a(state,2)= find(ismember(states_list_eval, state_trajectory(state), 'rows'));
               if (state ~=1)
                   previous_state = traj_s_a(state-1,2);
                   current_state = traj_s_a(state,2);
                   if sa_s_eval(previous_state, 1,1)==current_state
                       traj_s_a(state-1,3)=1;
                   elseif sa_s_eval(previous_state, 1,2)==current_state
                       traj_s_a(state-1,3)=2;
                   elseif sa_s_eval(previous_state, 1,3)==current_state
                       traj_s_a(state-1,3)=3;
                   else
                       error('Invalid state transition. Check adjacency matrix');
                   end
               end
            end
            traj_s_a(length(state_trajectory),3)=4;
            cum_sum = cum_sum+moves_eval(trial)+1;
            exp_new_example_samples_eval{end+1} = traj_s_a;
        end
    end

end

function plotter(x,y1, y2)
    figure()
    plot(x,y1, 'b', 'LineWidth',4);
    hold on
    xlabel('laplace prior');
    ylabel('Negative Log Likelihood train')
    title('Training')

    figure()
    plot(x,y2, 'r', 'LineWidth',4);
    hold on
    xlabel('laplace prior')
    ylabel('Negative Log Likelihood train')
    title('Validation')
end


function [sa_s, sa_p] = set_transtitions(adjacency, determinism)
    n=size(adjacency,1);
    sa_s = zeros(n,4,4);
    sa_p = zeros(n,4,4);
    for s=1:size(adjacency,1)
        I=find(adjacency(s,:)==1);
        successors = zeros(1,1,4);
        successors(1,1,1) = I(1);
        successors(1,1,2) = I(2);
        if length(I)==3
            successors(1,1,3) = I(3);
        else
            successors(1,1,3) = s;
        end
        successors(1,1,4) = s;
        sa_s(s,:,:) = repmat(successors,[1,4,1]);
        sa_p(s,:,:) = reshape(eye(4,4)*determinism + ...
            (ones(4,4)-eye(4,4))*((1.0-determinism)/3.0),...
            1,4,4);
    end
    
    %{
    for s=abs_state
        successors(1,1,1) = s;
        successors(1,1,2) = s;
        successors(1,1,3) = s;
        successors(1,1,4) = s;        
        sa_s(s,:,:) = repmat(successors,[1,4,1]);
        sa_p(s,:,:) = reshape([zeros(4,3) ones(4,1)],...
            1,4,4);
    end
    %}

end


function create_plots(irl_results, adjacency, nodes_pos, isEval, target_states)
    %if (isEval==false) % if plotting evaluation results, then swap rows so plots are in correct order
    %    irl_results([2 1], :) = irl_results([1 2], :);
    %end
    fig = figure;
    m = size(irl_results, 1);
    n = size(irl_results, 2);
    mincolor = 100000;
    maxcolor =0;
    tiledlayout(n,m, 'TileSpacing','tight', 'Padding','tight');
    %title("IRL results (train)") 
    hold on
    for j = 1:m
        target_state = target_states(j);
        for i = 1:n
            nexttile
            %subplot(m, n, n * (j - 1) + i);
            irl_result = irl_results(j, i);
            G = graph(adjacency);
            p = plot(G, 'Layout', 'force', 'EdgeLabel', '', 'NodeLabel', '', 'LineWidth', 4, 'MarkerSize', 14);
            p.XData = nodes_pos(:, 1);
            p.YData = nodes_pos(:, 2);            
            colormap(jet)
            G.Nodes.NodeColors = irl_result.r(:, 1);
            mincolor = min(mincolor, min(irl_result.r(:,1)));
            maxcolor = max(maxcolor, max(irl_result.r(:,1)));
            p.NodeCData = G.Nodes.NodeColors;
            hold on
            scatter(p.XData(target_state), p.YData(target_state), 250, 'magenta', 'filled');  % Overlay with a black node of larger size
            hold on
        end
    end   
    h = axes(fig,'visible','off');
    c = colorbar(h,'Position',[0.94 0.1 0.022 0.8]);  % attach colorbar to h
    colormap(c,'jet')
    if mincolor==maxcolor
        maxcolor = mincolor+1;
    end
    clim(h,[mincolor,maxcolor]);             % set colorbar limits
end


function    [irl_result, perf_met, true_feature_map]= run_irl(n,discount,determinism, sa_s, sa_p, adjacency, example_samples, laplace_prior, plot_figures, abs_state, train, learned_wts, subgoal_states_num, nodes_pos, H, model_number)
     % Create MDP data structure.
    mdp_data = struct(...
        'states',n,...
        'actions',4,...
        'discount',discount,...
        'determinism',determinism,...
        'sa_s',sa_s,...
        'sa_p',sa_p);
    
    %Choose appropriate mdp_model
    mdp_model='linearmdp'; 
    mdp_model='standardmdp';
    %%%%%%%%%%%%%%%%%%%%% Following can be commented out
    %{
    % Fill in the reward function.
    r1= zeros(n,1);
    r1(abs_state,1)=1; 
    r=repmat(r1,1,4);

    % Solve example.
    mdp_solution = feval(strcat(mdp_model,'solve'),mdp_data,r);
    %}
    %%%%%%%%%%%%%%%%%%%%% Above can be commented out

    algorithm_params= struct();
    stateadjacency = adjacency + eye(n); 
    splittable=[];
    feature_data = struct('stateadjacency',stateadjacency,'splittable',splittable);
    
    %true_feature_map = eye(n);
    true_feature_map = zeros(n);
    select_states_for_features = subgoal_states_num(:);
    for k=1:length(select_states_for_features)
        true_feature_map(select_states_for_features(k), select_states_for_features(k))=1;
    end
    % Run IRL algorithm.
    %TO DO: Choose between maxentrun or maxentrun1
    irl_result = feval('maxentrun_feedback',algorithm_params,mdp_data,mdp_model,...
        feature_data,example_samples(1,:),true_feature_map,0, laplace_prior,  train, learned_wts, model_number, H);
    
    if (plot_figures==true)
        %Fig1
        %figure
        %plot(irl_result.r(:,1), 'o', 'Markersize',10, 'MarkerFaceColor','b','MarkerEdgeColor','b');
        %xlabel('State', 'Interpreter','latex'); ylabel('Estimated Reward Function','Interpreter','latex');
        
        %{
        %Fig2
        figure
        plot(irl_result.v(:,1), '.','Markersize',10, 'MarkerEdgeColor','r');
        xlabel('State', 'Interpreter','latex'); ylabel('Estimated Value Function','Interpreter','latex');  
        %}
      
        G=graph(adjacency);
        %Fig3
        figure
        p = plot(G,'Layout','force','EdgeLabel','','NodeLabel','', 'LineWidth',4, 'MarkerSize',14);
        p.XData = nodes_pos(:,1);
        p.YData = nodes_pos(:,2);
        %for i=1:length(p.XData)
        %    text(p.XData(i)+0.1,p.YData(i),num2str(i),'fontsize',12);
        %end
        %p.NodeLabel = {};
        colormap(jet)
        G.Nodes.NodeColors = irl_result.r(:,1);
        p.NodeCData = G.Nodes.NodeColors;
        colorbar
        title('Reward Function', 'Interpreter','latex')
        axis off
        %Fig4
        %{
        figure
        p = plot(G,'Layout','force','EdgeLabel','','NodeLabel','', 'LineWidth',4, 'MarkerSize',14);  
        p.XData = nodes_pos(:,1);
        p.YData = nodes_pos(:,2);
        for i=1:length(p.XData)
            text(p.XData(i)+0.1,p.YData(i),num2str(i),'fontsize',12);
        end
        p.NodeLabel = {};
        colormap(jet)
        G.Nodes.NodeColors = irl_result.v(:,1);
        p.NodeCData = G.Nodes.NodeColors;   
        colorbar
        axis off
        %}
    end
    perf_met = 0;
    if (abs_state ~=0)
        %{  
        % Following code is not used
        algorithm_params = maxentdefaultparams(algorithm_params);
        rand('seed',algorithm_params.seed);
        [states,actions,transitions] = size(mdp_data.sa_p);
        N=length(example_samples);
        % Build feature membership matrix.
        F = eye(states);
        % Count features.
        features = size(F,2);
        
        % Compute feature expectations.
        muE = zeros(features,1);
        mu_sa = zeros(states,actions);
        ex_s=[];
        for i=1:N
            T=length(example_samples{i}(:,1));
            for t=1:T
                ex_s=[ex_s; example_samples{i}(t,2)];
                mu_sa(example_samples{i}(t,2),example_samples{i}(t,3)) = mu_sa(example_samples{i}(t,2),example_samples{i}(t,3)) + 1;
                state_vec = zeros(states,1);
                state_vec(example_samples{i}(t,2)) = 1;
                muE = muE + F'*state_vec;
            end
        end
        no_sa=0;
        for i=1:N
            [a,~]=size(example_samples{i});
            no_sa=no_sa+a;
        end
        % Generate initial state distribution for infinite horizon.
        initD = sum(sparse(ex_s,1:length(ex_s),ones(length(ex_s),1),states,length(ex_s))*ones(length(ex_s),1),2);
        for i=1:N
            T=length(example_samples{i}(:,1));
            for t=1:T
                s = example_samples{i}(t,2);
                a = example_samples{i}(t,3);
                for k=1:transitions
                    sp = mdp_data.sa_s(s,a,k);
                    initD(sp) = initD(sp) - mdp_data.discount*mdp_data.sa_p(s,a,k);
                end
            end
        end
        
        % fun = @(r)maxentdiscounted(r,F,muE,mu_sa,mdp_data,initD,algorithm_params.laplace_prior);
        [val,~] = maxentdiscounted(irl_result.r(:,1),eye(states),muE,mu_sa,mdp_data,initD,0);
        val/no_sa;
        %}
        ideal_reward=zeros(n,1);
        ideal_reward(abs_state)=1; 
        
        perf_met = norm(ideal_reward-irl_result.r(:,1),1);
    end

end

function [subgoal_states_train_num, subgoal_states_eval_num] = subgoals(states_list_train, states_list_eval, set_subgoals)
    
    if (set_subgoals==1)
        subgoal_states_train = {'2200', '1110', '0012', '2212'; '1100', '2220', '1121', '0021'};
        subgoal_states_eval = {'11100', '22220', '00021', '11121'; '22200', '11110', '22212', '00012'} ;
    else
        subgoal_states_train = {'2200', '1110', '0012', '2212', '2222', '1122', '1102', '0002'; '1100', '2220', '1121', '0021', '0001', '2201', '2211', '1111'};
        subgoal_states_eval = {'11100', '22220', '00021', '11121', '11111', '22211', '22201', '00001'; '22200', '11110', '22212', '00012', '00002', '11102', '11122', '22222'};
    end
     

    subgoal_states_train_num = zeros(size(subgoal_states_train));
    subgoal_states_eval_num = zeros(size(subgoal_states_eval));
    for i=1:size(subgoal_states_train, 1)
        for j=1:size(subgoal_states_train, 2)
            subgoal_states_train_num(i,j) = find(ismember(states_list_train, subgoal_states_train(i,j))); 
        end
    end
    for i=1:size(subgoal_states_eval, 1)
        for j=1:size(subgoal_states_eval, 2)
            subgoal_states_eval_num(i,j) = find(ismember(states_list_eval, subgoal_states_eval(i,j))); 
        end
    end

end

function [triangle_vertex_train_string, triangle_vertex_eval_string, triangle_vertex_train_num, triangle_vertex_eval_num] = get_triangle_vertices(states_list_train, states_list_eval)
    triangle_vertex_train_string = {'1112', '0022', '2202', '2221', '1101', '0011'};
    triangle_vertex_eval_string = {'22221', '00011', '11101', '11112', '22202', '00022'};

    triangle_vertex_train_num = zeros(size(triangle_vertex_train_string));
    triangle_vertex_eval_num = zeros(size(triangle_vertex_eval_string));
    for i=1:size(triangle_vertex_train_string, 1)
        for j=1:size(triangle_vertex_train_string, 2)
            triangle_vertex_train_num(i,j) = find(ismember(states_list_train, triangle_vertex_train_string(i,j))); 
        end
    end
    for i=1:size(triangle_vertex_eval_string, 1)
        for j=1:size(triangle_vertex_eval_string, 2)
            triangle_vertex_eval_num(i,j) = find(ismember(states_list_eval, triangle_vertex_eval_string(i,j))); 
        end
    end

end

function print_H_map(H_map)
    % Assuming H_map has already been populated
    keys = H_map.keys();
    values = H_map.values();
    
    % Print each key-value pair
    fprintf('Printing H_map content:\n');
    for i = 1:length(keys)
        key = keys{i};
        value = values{i}; % assuming the value is a matrix, adjust if it is not
    
        fprintf('Key: %d\n', key);
        disp('Value (H matrix): ');
        disp(value);
        fprintf('\n'); % Add a newline for readability
    end
end


function examples_category = categorize_examples(score, target_states, examples, str, filter_successfull, states_list)
    examples_category = {};
    for i=1:size(target_states,2)
        t_state = target_states{1,i};
        if filter_successfull==true
            if score(1,i)>0 && strcmp(t_state(end-1:end),str)
                examples_category{1,end+1} = examples{1,i};
            end
        else
            if strcmp(t_state(end-1:end),str)
                examples_category{1,end+1} = examples{1,i};
            end
        end
    end

    for k=1:size(examples_category,2)
        traj = examples_category{1,k};
        index = 0;
        for m=1:size(traj,1)
            current_state = states_list(traj(m,2),:);
            if strcmp(current_state(end-1:end),str)
                index = m;
                break;
            end
        end
        if (index>0)
            examples_category{1,k} = traj(1:index,:);        
        end

    end
end