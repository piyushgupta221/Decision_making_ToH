close all
clearvars
clc
addpaths;

filter_with_target_state = false; % Keep this to false; For a single target state, we do not have enough data in all the experiments
filter_by_triangle = true;  % Keep this to true;
filter_only_successful = false;
set_subgoals = 1;   % set = 0,1,2   : 0: all states, 1: 8 features, 2: 16 features
cross_validation=false;  %Keep this to true for cross validation
k_fold = 5;



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

load("toh_participants_data.mat");


n_train=length(states_list_train);
n_eval = length(states_list_eval);
determinism=1;
discount=0.95;

[sa_s_train, sa_p_train] = set_transtitions(adjacency_train, determinism);
[sa_s_eval, sa_p_eval] = set_transtitions(adjacency_eval, determinism);

if set_subgoals~=0
    [subgoal_states_train_num, subgoal_states_eval_num] = subgoals(states_list_train, states_list_eval, set_subgoals);
else
    subgoal_states_train_num = 1:size(states_list_train,1);
    subgoal_states_eval_num = 1:size(states_list_eval,1);
end


if cross_validation==true
    exp_data_all = exp1_participants_new;
    exp_data_all = [exp_data_all; exp2_participants_new];
    exp_data_all = [exp_data_all; exp5_participants_new];
    [standard_percentage_score_examples_train, standard_percentage_score_examples_eval, target_states_examples_train, target_states_examples_eval, exp_new_example_samples_train, exp_new_example_samples_eval] = get_state_trajectories(exp_data_all, states_list_train, states_list_eval, sa_s_train, sa_s_eval);  
    if (filter_by_triangle==true)
        [~,example_samples_train_1,example_samples_train_2] = filter_samples(standard_percentage_score_examples_train, target_states_examples_train, exp_new_example_samples_train, filter_with_target_state, filter_only_successful, '', filter_by_triangle);
        [~,example_samples_eval_1,example_samples_eval_2] = filter_samples(standard_percentage_score_examples_eval, target_states_examples_eval, exp_new_example_samples_eval, filter_with_target_state, filter_only_successful, '', filter_by_triangle);       
        %laplace_1_train= run_cross_validation(n_train,discount,determinism, sa_s_train, sa_p_train, adjacency_train, example_samples_train_1, abs_state, k_fold, subgoal_states_train_num)
        %laplace_2_train= run_cross_validation(n_train,discount,determinism, sa_s_train, sa_p_train, adjacency_train, example_samples_train_2, abs_state, k_fold, subgoal_states_train_num)
        %laplace_1_eval= run_cross_validation(n_eval,discount,determinism, sa_s_eval, sa_p_eval, adjacency_eval, example_samples_eval_1, abs_state, k_fold, subgoal_states_eval_num)
        %laplace_2_eval= run_cross_validation(n_eval,discount,determinism, sa_s_eval, sa_p_eval, adjacency_eval, example_samples_eval_2, abs_state, k_fold, subgoal_states_eval_num)
        
        %laplace_1_train = 0.3;
        %laplace_2_train = 0.0;
        %laplace_1_eval = 1.6;
        %laplace_2_eval = 0.0;
        laplace_1_train = 0.0;
        laplace_2_train = 0.0;
        laplace_1_eval = 0.0;
        laplace_2_eval = 0.0;
    else
        [example_samples_train,~,~] = filter_samples(standard_percentage_score_examples_train, target_states_examples_train, exp_new_example_samples_train, filter_with_target_state, filter_only_successful, '', filter_by_triangle);
        [example_samples_eval,~,~] = filter_samples(standard_percentage_score_examples_eval, target_states_examples_eval, exp_new_example_samples_eval, filter_with_target_state, filter_only_successful, '', filter_by_triangle);
        laplace_1_train= run_cross_validation(n_train,discount,determinism, sa_s_train, sa_p_train, adjacency_train, example_samples_train, abs_state, k_fold, subgoal_states_train_num);
        laplace_1_eval= run_cross_validation(n_eval,discount,determinism, sa_s_eval, sa_p_eval, adjacency_eval, example_samples_eval, abs_state, k_fold, subgoal_states_eval_num);
    end
end


irl_results_train=[];
irl_results_eval=[];

if (filter_with_target_state==true)
    train_exp = [];
    eval_exp = [];
    for exp=1:3
        if exp==1   
            [standard_percentage_score_examples_train, standard_percentage_score_examples_eval, target_states_examples_train, target_states_examples_eval, exp_new_example_samples_train, exp_new_example_samples_eval] = get_state_trajectories(exp1_participants_new, states_list_train, states_list_eval, sa_s_train, sa_s_eval);
        elseif exp==2
             [standard_percentage_score_examples_train, standard_percentage_score_examples_eval, target_states_examples_train, target_states_examples_eval, exp_new_example_samples_train, exp_new_example_samples_eval] = get_state_trajectories(exp2_participants_new, states_list_train, states_list_eval, sa_s_train, sa_s_eval);  
        elseif exp==3
             [standard_percentage_score_examples_train, standard_percentage_score_examples_eval, target_states_examples_train, target_states_examples_eval, exp_new_example_samples_train, exp_new_example_samples_eval] = get_state_trajectories(exp5_participants_new, states_list_train, states_list_eval, sa_s_train, sa_s_eval);  
        end
        map_count = struct();
        [map_count.elements, map_count.count] = create_map(target_states_examples_train);
        train_exp = [train_exp, map_count];
        [map_count.elements, map_count.count] = create_map(target_states_examples_eval);
        eval_exp = [eval_exp, map_count];
    end
    
    uniq_train = train_exp(1).elements;
    uniq_eval = eval_exp(1).elements;
    for k=2:length(train_exp)
        uniq_train = intersect(uniq_train, train_exp(k).elements);
        uniq_eval = intersect(uniq_eval, eval_exp(k).elements);
    end
    
    count= zeros(size(uniq_train));
    for i=1:length(uniq_train)
        counts = [];
        for exp=1:3
            counts = [counts, train_exp(exp).count];
        end
        uniq_train(i);
        count(i) = min(counts);

    end
    unique_target_states_train = uniq_train;
    unique_target_states_eval = uniq_eval;
end


for exp=1:3
    if exp==1   
        [standard_percentage_score_examples_train, standard_percentage_score_examples_eval, target_states_examples_train, target_states_examples_eval, exp_new_example_samples_train, exp_new_example_samples_eval] = get_state_trajectories(exp1_participants_new, states_list_train, states_list_eval, sa_s_train, sa_s_eval);  
    elseif exp==2
         [standard_percentage_score_examples_train, standard_percentage_score_examples_eval, target_states_examples_train, target_states_examples_eval, exp_new_example_samples_train, exp_new_example_samples_eval] = get_state_trajectories(exp2_participants_new, states_list_train, states_list_eval, sa_s_train, sa_s_eval);  
    elseif exp==3
         [standard_percentage_score_examples_train, standard_percentage_score_examples_eval, target_states_examples_train, target_states_examples_eval, exp_new_example_samples_train, exp_new_example_samples_eval] = get_state_trajectories(exp5_participants_new, states_list_train, states_list_eval, sa_s_train, sa_s_eval);  
    end


    exp_new_example_samples_train_learned = {};
    target_states_examples_train_learned = {};
    standard_percentage_score_examples_train_learned = [];

    for h=1:length(exp_new_example_samples_train)
        if rem(h,10)==0 || rem(h,10)>=6
            exp_new_example_samples_train_learned{end+1} = exp_new_example_samples_train{1,h};
            target_states_examples_train_learned{end+1} = target_states_examples_train{1,h};
            standard_percentage_score_examples_train_learned = [standard_percentage_score_examples_train_learned, standard_percentage_score_examples_train(1,h)];
        end
    end

    %exp_new_example_samples_train
    exp_new_example_samples_train = exp_new_example_samples_train_learned;
    standard_percentage_score_examples_train = standard_percentage_score_examples_train_learned;
    target_states_examples_train = target_states_examples_train_learned;

    %exp_new_example_samples_train
    %size(target_states_examples_train_learned)
    %size(standard_percentage_score_examples_train_learned)



    if filter_with_target_state==true
        for k=1:1  %length(unique_target_states_train)
            abs_state_string = unique_target_states_train(k);
            abs_state_string_train = abs_state_string{1};
            abs_state_train = find(ismember(states_list_train, abs_state_string_train, 'rows'));
            [example_samples_train, ~,~] = filter_samples(standard_percentage_score_examples_train, target_states_examples_train, exp_new_example_samples_train, filter_with_target_state, filter_only_successful, abs_state_string_train, filter_by_triangle);        
            if isempty(example_samples_train)
                disp("No experiment data for this target state")
                disp(abs_state_train)
            else
                sa_s = sa_s_train;
                sa_p = sa_p_train;
                successors = zeros(1,1,4);
                for s=abs_state_train
                    successors(1,1,1) = s;
                    successors(1,1,2) = s;
                    successors(1,1,3) = s;
                    successors(1,1,4) = s;        
                    sa_s(s,:,:) = repmat(successors,[1,4,1]);
                    sa_p(s,:,:) = reshape([zeros(4,3) ones(4,1)],...
                        1,4,4);
                end
                laplace_prior = default_laplace_prior;
                [irl_result_train, perf_met_train]= run_irl_helper(n_train,discount,determinism, sa_s, sa_p, adjacency_train, example_samples_train, laplace_prior, plot_figures, abs_state_train, subgoal_states_train_num, nodes_pos_train);
                irl_results_train = [irl_results_train, irl_result_train];
            end
        end

        for k=4:4 %length(unique_target_states_eval)
            abs_state_string = unique_target_states_eval(k);
            abs_state_string_eval = abs_state_string{1};
            abs_state_eval = find(ismember(states_list_eval, abs_state_string_eval, 'rows'));
            [example_samples_eval,~,~] = filter_samples(standard_percentage_score_examples_eval, target_states_examples_eval, exp_new_example_samples_eval, filter_with_target_state, filter_only_successful, abs_state_string_eval, filter_by_triangle);
            if isempty(example_samples_eval)
                disp("No experiment data for this target state");
            else
                sa_s = sa_s_eval;
                sa_p = sa_p_eval;
                successors = zeros(1,1,4);
                for s=abs_state_eval
                    successors(1,1,1) = s;
                    successors(1,1,2) = s;
                    successors(1,1,3) = s;
                    successors(1,1,4) = s;        
                    sa_s(s,:,:) = repmat(successors,[1,4,1]);
                    sa_p(s,:,:) = reshape([zeros(4,3) ones(4,1)],...
                        1,4,4);
                end
                laplace_prior = default_laplace_prior;
                [irl_result_eval, perf_met_eval]= run_irl_helper(n_eval,discount,determinism, sa_s, sa_p, adjacency_eval, example_samples_eval, laplace_prior, plot_figures, abs_state_eval, subgoal_states_eval_num, nodes_pos_eval);
                irl_results_eval = [irl_results_eval, irl_result_eval];
            end
        end
    elseif filter_by_triangle==true
        [~,example_samples_train_1,example_samples_train_2] = filter_samples(standard_percentage_score_examples_train, target_states_examples_train, exp_new_example_samples_train_learned, filter_with_target_state, filter_only_successful, '', filter_by_triangle);
        [~,example_samples_eval_1,example_samples_eval_2] = filter_samples(standard_percentage_score_examples_eval, target_states_examples_eval, exp_new_example_samples_eval, filter_with_target_state, filter_only_successful, '', filter_by_triangle);
        if (cross_validation==true)
            laplace_prior_train_1 = laplace_1_train;
            laplace_prior_train_2 = laplace_2_train;
            laplace_prior_eval_1 = laplace_1_eval;
            laplace_prior_eval_2 = laplace_2_eval;
        else
            laplace_prior_train_1 = default_laplace_prior;
            laplace_prior_train_2 = default_laplace_prior;
            laplace_prior_eval_1 = default_laplace_prior;
            laplace_prior_eval_2 = default_laplace_prior;
        end
        [irl_result_train1, perf_met_train]= run_irl_helper(n_train,discount,determinism, sa_s_train, sa_p_train, adjacency_train, example_samples_train_1, laplace_prior_train_1, plot_figures, abs_state, subgoal_states_train_num, nodes_pos_train);
        [irl_result_train2, perf_met_train]= run_irl_helper(n_train,discount,determinism, sa_s_train, sa_p_train, adjacency_train, example_samples_train_2, laplace_prior_train_2, plot_figures, abs_state, subgoal_states_train_num, nodes_pos_train);
        [irl_result_eval1, perf_met_eval]= run_irl_helper(n_eval,discount,determinism, sa_s_eval, sa_p_eval, adjacency_eval, example_samples_eval_1, laplace_prior_eval_1, plot_figures, abs_state, subgoal_states_eval_num, nodes_pos_eval);
        [irl_result_eval2, perf_met_eval]= run_irl_helper(n_eval,discount,determinism, sa_s_eval, sa_p_eval, adjacency_eval, example_samples_eval_2, laplace_prior_eval_2, plot_figures, abs_state, subgoal_states_eval_num, nodes_pos_eval);
        
        irl_train = [irl_result_train1; irl_result_train2];
        irl_eval = [irl_result_eval1; irl_result_eval2];
        irl_results_train = [irl_results_train, irl_train];
        irl_results_eval = [irl_results_eval, irl_eval];
    else
         if (cross_validation==true)
            laplace_prior_train_1 = laplace_1_train;
            laplace_prior_eval_1 = laplace_1_eval;
        else
            laplace_prior_train_1 = default_laplace_prior;
            laplace_prior_eval_1 = default_laplace_prior;
        end
        [example_samples_train,~,~] = filter_samples(standard_percentage_score_examples_train, target_states_examples_train, exp_new_example_samples_train, filter_with_target_state, filter_only_successful, '', filter_by_triangle);
        [example_samples_eval,~,~] = filter_samples(standard_percentage_score_examples_eval, target_states_examples_eval, exp_new_example_samples_eval, filter_with_target_state, filter_only_successful, '', filter_by_triangle);
        [irl_result_train, perf_met_train]= run_irl_helper(n_train,discount,determinism, sa_s_train, sa_p_train, adjacency_train, example_samples_train, laplace_prior_train_1, plot_figures, abs_state, subgoal_states_train_num, nodes_pos_train);
        [irl_result_eval, perf_met_eval]= run_irl_helper(n_eval,discount,determinism, sa_s_eval, sa_p_eval, adjacency_eval, example_samples_eval, laplace_prior_eval_1, plot_figures, abs_state, subgoal_states_eval_num, nodes_pos_eval);
        irl_results_train = [irl_results_train, irl_result_train];
        irl_results_eval = [irl_results_eval, irl_result_train];
    end

end


create_plots(irl_results_train, adjacency_train, nodes_pos_train, false)

create_plots(irl_results_eval, adjacency_eval, nodes_pos_eval, true)


function laplace_cross_validation = run_cross_validation(n, discount, determinism, sa_s, sa_p, adjacency, example_samples, abs_state, k_fold, subgoal_states_num)
    nodes_pos = 0; %Set randomly, not needed
    example_samples = example_samples(randperm(size(example_samples,2)));
    num_trajectories = size(example_samples,2);
    laplace_prior_min=0;
    laplace_prior_max=2;
    laplace_prior_values = linspace(laplace_prior_min, laplace_prior_max, 21);
    log_likelihood_train = zeros(length(laplace_prior_values), k_fold);
    log_likelihood_val = zeros(length(laplace_prior_values), k_fold);
    learned_rewards =[];
    if (num_trajectories>=2*k_fold)
        % Initialize cell arrays to store training and validation sets
        trainingSets = cell(1, k_fold);
        validationSets = cell(1, k_fold);
        plot = false;
        % Create k-fold cross-validation partitions
        for fold = 1:k_fold
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
            learned_rewards_fold = [];
            for l=1:length(laplace_prior_values)
                lap = laplace_prior_values(l);
                train = true;
                reward_value = 0; %Random Value
                [irl_result_train, ~, F] = run_irl(n, discount, determinism, sa_s, sa_p, adjacency, trainingSet, lap, plot, abs_state, train, reward_value, subgoal_states_num, nodes_pos);
                reward_value = irl_result_train.r(:,1); %Learned reward
                [row,~,~] = find(F);
                wts = reward_value(row);
                learned_rewards_fold = [learned_rewards_fold, wts];

                % Evaluate negative log likelihood by setting 0 laplace prior
                train = false;
                [irl_result_train, ~, F] = run_irl(n, discount, determinism, sa_s, sa_p, adjacency, trainingSet, lap, plot, abs_state, train, reward_value, subgoal_states_num, nodes_pos);
                log_likelihood_train(l,fold) = irl_result_train.log_like;
                [irl_result_val, perf_met_val, F] = run_irl(n, discount, determinism, sa_s, sa_p, adjacency, validationSet, lap, plot, abs_state, train, reward_value, subgoal_states_num, nodes_pos);                    
                log_likelihood_val(l,fold) = irl_result_val.log_like;
            end

            if (size(learned_rewards,1)==0)
                learned_rewards = learned_rewards_fold;
            else
                learned_rewards=learned_rewards+learned_rewards_fold;
            end

        end
        mean_log_like_train = mean(log_likelihood_train,2)';
        mean_log_like_val = mean(log_likelihood_val,2)';

        learned_rewards = learned_rewards/k_fold;
        %plotter(laplace_prior_values, mean_log_like_train, mean_log_like_val)

        irl_result = irl_result_val;
        perf_met = perf_met_val;
        
        [~,I] = min(mean_log_like_val);
        laplace_cross_validation = laplace_prior_values(I);
    else
        disp("Number of trajectories not enough to perform cross validation");
    end
end


function  [irl_result, perf_met] = run_irl_helper(n, discount, determinism, sa_s, sa_p, adjacency, example_samples, laplace_prior, plot_figures, abs_state, subgoal_states_num, nodes_pos)
    example_samples = example_samples(randperm(size(example_samples,2)));
    train = true;
    reward_value = 0; %Random Value
    [irl_result, perf_met, F] = run_irl(n, discount, determinism, sa_s, sa_p, adjacency, example_samples, laplace_prior, plot_figures, abs_state, train, reward_value, subgoal_states_num, nodes_pos);
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


function create_plots(irl_results, adjacency, nodes_pos, isEval)
    if (isEval==false) % if plotting evaluation results, then swap rows so plots are in correct order
        irl_results([2 1], :) = irl_results([1 2], :);
    end
    fig = figure;
    m = size(irl_results, 1);
    n = size(irl_results, 2);
    mincolor = 100000;
    maxcolor =0;
    tiledlayout(m,n, 'TileSpacing','tight', 'Padding','tight');
    %title("IRL results (train)") 
    hold on
    for j = 1:m
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
            axis on
            if (j==1) && (i==1)
                title("No feedback", "FontSize", 24)
                ylabel({'Target state in'; 'left triangle T_2'}, "FontSize", 24)
            elseif j==1 && i==2
                title("Numeric feedback", "FontSize", 24)
            elseif j==1 && i==3
                title("Sub-goal with Numeric feedback", "FontSize", 24)
            elseif j==2 && i==1
                ylabel({'Target state in'; 'right triangle T_3'}, "FontSize", 24)
            end
            hold on
        end
    end   
    h = axes(fig,'visible','off');
    c = colorbar(h,'Position',[0.94 0.1 0.022 0.8]);  % attach colorbar to h
    colormap(c,'jet')
    clim(h,[mincolor,maxcolor]);             % set colorbar limits
end


function    [irl_result, perf_met, true_feature_map]= run_irl(n,discount,determinism, sa_s, sa_p, adjacency, example_samples, laplace_prior, plot_figures, abs_state, train, reward_value, subgoal_states_num, nodes_pos)
     % Create MDP data structure.
    mdp_data = struct(...
        'states',n,...
        'actions',4,...
        'discount',discount,...
        'determinism',determinism,...
        'sa_s',sa_s,...
        'sa_p',sa_p);
    
    %Choose appropriate mdp_model
    %mdp_model='linearmdp'; 
    mdp_model='standardmdp';

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
    irl_result = feval('maxentrun1',algorithm_params,mdp_data,mdp_model,...
        feature_data,example_samples(1,:),true_feature_map,0, laplace_prior,  train, reward_value);
    
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

function [example_samples,example_samples_1, example_samples_2]  = filter_samples(standard_percentage_score, target_states, exp_new_example_samples, filter_with_target_state, filter_only_successful, abs_state, filter_by_triangle)
    example_samples={};
    example_samples_1={};
    example_samples_2={};
    if filter_with_target_state==false && filter_only_successful==false && filter_by_triangle==false
        example_samples=exp_new_example_samples;
        return
    else         
        if filter_by_triangle==true
             for k=1:length(exp_new_example_samples)
                target = target_states(k);
                target = target{1};
                if (filter_only_successful==true)
                    if (standard_percentage_score(k)>0) && target(end)=='1'
                        example_samples_1{1,end+1} = exp_new_example_samples{1,k};
                    elseif (standard_percentage_score(k)>0) && target(end)=='2'
                        example_samples_2{1,end+1} = exp_new_example_samples{1,k};
                    end
                else
                    if target(end)=='1'
                        example_samples_1{1,end+1} = exp_new_example_samples{1,k};
                    elseif target(end)=='2'
                        example_samples_2{1,end+1} = exp_new_example_samples{1,k};
                    end
                end
             end
        else
            for k=1:length(exp_new_example_samples)
                target = target_states(k);
                target = target{1};
                if (filter_only_successful==true) && (filter_with_target_state==true)              
                    if (standard_percentage_score(k)>0) && strcmp(target,abs_state)
                        example_samples{1,end+1} = exp_new_example_samples{1,k};
                    end
                elseif (filter_only_successful==true) && (filter_with_target_state==false)
                    if (standard_percentage_score(k)>0)
                        example_samples{1,end+1} = exp_new_example_samples{1,k};
                    end
                elseif (filter_only_successful==false) && (filter_with_target_state==true)
                    if strcmp(target,abs_state)
                        example_samples{1,end+1} = exp_new_example_samples{1,k};
                    end
    
                end
            end
        
        end

        
    end

end


function [sortedElements, sortedCounts] = create_map(target_states_examples_train)
    % Initialize a containers.Map to store the count map
    countMap = containers.Map;
    
    % Iterate through the cell array
    for i = 1:numel(target_states_examples_train)
        element = target_states_examples_train{i};
        
        % Check if the element exists in the map
        if isKey(countMap, element)
            % Increment the count if the element is already in the map
            countMap(element) = countMap(element) + 1;
        else
            % Add the element to the map with a count of 1 if it's not in the map
            countMap(element) = 1;
        end
    end
    
    % Sort the map by values in descending order
    [sortedCounts, sortedIndices] = sort(cell2mat(values(countMap)), 'descend');
    sortedElements = keys(countMap);
    sortedElements = sortedElements(sortedIndices);
   



end