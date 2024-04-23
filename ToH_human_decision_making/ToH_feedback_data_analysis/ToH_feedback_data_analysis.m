clc
close all
addpath ..\
% Save variables to a .mat file
load('toh_participants_data.mat', 'exp1_participants_old', 'exp2_participants_old', 'exp3_participants_old', ...
    'exp4_participants_old', 'exp5_participants_old');

experiment1 = setup_experiment(exp1_participants_old);
experiment2 = setup_experiment(exp2_participants_old);
experiment3 = setup_experiment(exp3_participants_old);
experiment4 = setup_experiment(exp4_participants_old);
experiment5 = setup_experiment(exp5_participants_old);

a = [experiment1.standard_percentage_score_train_old',experiment2.standard_percentage_score_train_old', experiment3.standard_percentage_score_train_old', experiment4.standard_percentage_score_train_old', experiment5.standard_percentage_score_train_old'];
box_plot(1, a, 'Percentage Scores: Training');
box_plot_positive(2,a,'Positive Percentage Scores: Training');

a = [experiment1.standard_percentage_score_eval_old',experiment2.standard_percentage_score_eval_old', experiment3.standard_percentage_score_eval_old', experiment4.standard_percentage_score_eval_old', experiment5.standard_percentage_score_eval_old'];
box_plot(3, a, 'Percentage Scores: Transfer');
box_plot_positive(4,a,'Positive Percentage Scores: Transfer');


%% Generate Table for percentage of successful tasks in each experiment
compare_len_positive(experiment1, experiment2, experiment3, experiment4, experiment5);


train=true;
plot_trajectories(5, [experiment1, experiment2, experiment5], "Performance over trails: Training", train);

train=false;
plot_trajectories(6, [experiment1, experiment2, experiment5], "Performance over trails: Transfer", train);

% Assuming you have experiment1, experiment2, and experiment5 defined
[resultsTable1,resultsTable2] = conduct_ttest(experiment1, experiment2, experiment5);

% Display the table of t-test results
disp(resultsTable1);

disp(resultsTable2);



function [resultsTable1,resultsTable2]  = conduct_ttest(experiment1, experiment2, experiment5)
    
% Create cell arrays to store the results for Experiment1 vs. Experiment2 and Experiment1 vs. Experiment5
resultsTable1 = cell(2, 3); % For comparison of Experiment1 vs Experiment2
resultsTable2 = cell(2, 3); % For comparison of Experiment1 vs Experiment5
% Perform t-tests and store results for Experiment1 vs Experiment2
[resultsTable1{1, 2}, resultsTable1{1, 3}] = ttest2(experiment1.standard_percentage_score_train_old, experiment2.standard_percentage_score_train_old);
resultsTable1{1, 1} = 'percentage_score_train_old';
[resultsTable1{2, 2}, resultsTable1{2, 3}] = ttest2(experiment1.standard_percentage_score_eval_old, experiment2.standard_percentage_score_eval_old);
resultsTable1{2, 1} = 'percentage_score_eval_old';

% Perform t-tests and store results for Experiment1 vs Experiment5
[resultsTable2{1, 2}, resultsTable2{1, 3}] = ttest2(experiment1.standard_percentage_score_train_old, experiment5.standard_percentage_score_train_old);
resultsTable2{1, 1} = 'percentage_score_train_old';
[resultsTable2{2, 2}, resultsTable2{2, 3}] = ttest2(experiment1.standard_percentage_score_eval_old, experiment5.standard_percentage_score_eval_old);
resultsTable2{2, 1} = 'percentage_score_eval_old';

% Convert to tables
resultsTable1 = cell2table(resultsTable1, 'VariableNames', {'Comparison', 'h value', 'p-Value'});
resultsTable2 = cell2table(resultsTable2, 'VariableNames', {'Comparison', 'h value', 'p-Value'});

end


function plot_trajectories(fig_num,data,titl, train)    
    figure(fig_num)    
    All_trajectories=[];
    for i=1:size(data,2)
        if train==true
            trajectories = mean(data(i).standard_percentage_score_train_old_participant,1);
        else
            trajectories = mean(data(i).standard_percentage_score_eval_old_participant,1);
        end
        All_trajectories = [All_trajectories, trajectories'];
    end
    h = bar(All_trajectories);
    colormap(summer(size(All_trajectories,2)));
    l = cell(1,3);
    l{1}='No feedback'; l{2}='Numeric feedback'; l{3}='Sub-goal & numeric feedback';
    legend(h,l);
    % Adding title and labels
    title(titl);
    xlabel('Trials');
    ylabel('Mean Percentage Scores');
    % Adding legend
    legend('Location', 'best'); % You can specify the desired location

    % Customize axis properties
    set(gca, 'FontName', 'Arial', 'FontSize', 24);
    set(gca, 'LineWidth', 3);
    set(gca, 'Box', 'off');
    set(gca, 'TickDir', 'out');
    set(gca, 'TickLength', [0.02, 0.02]);
    set(gca, 'XMinorTick', 'off');
    set(gca, 'YMinorTick', 'on');
    %grid on;

    % Adding a legend
    % legend(h, {'Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5'}, 'Location', 'Best');

    % Adjust figure properties
    fig = gcf;
    fig.Color = 'w';  % Background color

    % Adjust position and size of the legend
    % lgd = findobj(gcf, 'Type', 'Legend');
    % lgd.Position(1) = 0.65;  % Adjust horizontal position

    % Manually adjust the plot layout
    set(gcf, 'Position', [100, 100, 1600, 800]);  % Adjust figure size and position
end









function box_plot_positive(fig_num, data, titl)
    figure(fig_num);
    % Initialize an empty cell array to store the filtered data
    filtered_data = cell(1, size(data, 2)); % Assuming data has 5 columns

    % Filter the positive values from each column
    for col = 1:size(data, 2)
        column_data = data(:, col);
        positive_values = column_data(column_data > 0);
        filtered_data{col} = positive_values;
    end

    if (size(data, 2) == 5)
        % Create a cell array to store experiment labels (as strings)
        experiments = {'No feedback','Numeric feedback','Optional feedcack', 'Sub-goal', 'Sub-goal & numeric feedback'};
        
        % Create a vector of x-positions for the box plots
        x_positions = 1:numel(experiments);
        for col = 1:size(data, 2)
            % subplot(1, size(data, 2), col); % Create subplots for each column
            boxplot(filtered_data{col}, 'Labels', {experiments{col}}, 'Positions', x_positions(col), 'Widths', 0.5, 'BoxStyle','outline', 'OutlierSize',10); % Adjust the width as needed
            hold on;
            % title(['Exp', num2str(col)]);
        end
        % Customize the x-axis labels
        xticks(x_positions);
        % Create cell arrays for two rows of labels
        row1 = {'     No', ' Numeric', ' Optional', 'Sub-goal', '      Sub-goal &'};
        row2 = {'feedback', 'feedback', 'feedback', '', 'numeric feedback'};   
        % Combine the labels with line breaks and spaces
        tickLabels = cellfun(@(r1, r2) [r1, '\newline', r2], row1, row2, 'UniformOutput', false);
        % Create x-positions for labels
        xPositions = 1:5;
        % Set custom xtick positions and labels
        ax = gca();
        ax.XTick = xPositions;
        ax.XLim = [0, 6];
        ax.XTickLabel = tickLabels;
        ax.TickLabelInterpreter = 'tex';
        % Adjust the x-axis label rotation if needed
        xtickangle(0); % 0 degrees for horizontal labels
        % boxplot(data,'Labels',{'Exp1','Exp2','Exp3','Exp4','Exp5'});
        % set(gca, 'xticklabel', {'Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5'});

        % Customizing box colors and styles
        boxColors = ['b', 'g', 'r', 'c', 'm']; % Make sure you have enough colors for your data
        
        h = findobj(gca, 'Tag', 'Box');
        for j = 1:length(h)
            patch(get(h(j), 'XData'), get(h(j), 'YData'), boxColors(j), 'FaceAlpha', 0.5);
        end

        % Increase the linewidth of the median
        h2 = findobj(gca, 'Tag', 'Median');
        set(h2, 'LineWidth', 4); % Adjust the linewidth as needed
        % Set the linewidth of the whiskers to 3
        whiskerHandles = findobj(gca, 'Tag', 'Whisker');
        for i = 1:numel(whiskerHandles)
            set(whiskerHandles(i), {'LineWidth'}, {4});
        end

    else
        % Create a cell array to store experiment labels (as strings)
        experiments = {'No feedback','Numeric feedback','Sub-goal & numeric feedback'};
        % Create a vector of x-positions for the box plots
        x_positions = 1:numel(experiments);
        for col = 1:size(data, 2)
            % subplot(1, size(data, 2), col); % Create subplots for each column
            boxplot(filtered_data{col}, 'Labels', {experiments{col}}, 'Positions', x_positions(col), 'Widths', 0.5, 'BoxStyle','outline', 'OutlierSize',10); % Adjust the width as needed
            hold on;
            % title(['Exp', num2str(col)]);
        end
        % Customize the x-axis labels
        xticks(x_positions);
        
        % Create cell arrays for two rows of labels
        row1 = {'     No', ' Numeric', '      Sub-goal &'};
        row2 = {'feedback', 'feedback', 'numeric feedback'};   
        % Combine the labels with line breaks and spaces
        tickLabels = cellfun(@(r1, r2) [r1, '\newline', r2], row1, row2, 'UniformOutput', false);
        % Create x-positions for labels
        xPositions = 1:3;
        % Set custom xtick positions and labels
        ax = gca();
        ax.XTick = xPositions;
        ax.XLim = [0, 4];
        ax.XTickLabel = tickLabels;
        ax.TickLabelInterpreter = 'tex';
        % Adjust the x-axis label rotation if needed
        xtickangle(0); % 0 degrees for horizontal labels

        % boxplot(data,'Labels',{'Exp1','Exp2','Exp3','Exp4','Exp5'});
        % set(gca, 'xticklabel', {'Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5'});

        % Customizing box colors and styles
        boxColors = ['b', 'g', 'r']; % Make sure you have enough colors for your data
        
        h = findobj(gca, 'Tag', 'Box');
        for j = 1:length(h)
            patch(get(h(j), 'XData'), get(h(j), 'YData'), boxColors(j), 'FaceAlpha', 0.5);
        end

        % Increase the linewidth of the median
        h2 = findobj(gca, 'Tag', 'Median');
        set(h2, 'LineWidth', 4); % Adjust the linewidth as needed
        % Set the linewidth of the whiskers to 3
        whiskerHandles = findobj(gca, 'Tag', 'Whisker');
        for i = 1:numel(whiskerHandles)
            set(whiskerHandles(i), {'LineWidth'}, {4});
        end
    end
    hold on;

    % Adding title and labels
    title(titl);
    % xlabel('Experiments');
    ylabel('Percentage Scores');

    % Customize axis properties
    set(gca, 'FontName', 'Arial', 'FontSize', 24);
    set(gca, 'LineWidth', 3);
    set(gca, 'Box', 'off');
    set(gca, 'TickDir', 'out');
    set(gca, 'TickLength', [0.02, 0.02]);
    set(gca, 'XMinorTick', 'off');
    set(gca, 'YMinorTick', 'on');

    % Adjust figure properties
    fig = gcf;
    fig.Color = 'w';  % Background color

    % Manually adjust the plot layout
    set(gcf, 'Position', [100, 100, 1600, 800]);  % Adjust figure size and position

end














function box_plot(fig_num,data,titl)    
    figure(fig_num);
    if (size(data,2)==5)
        boxplot(data,'Labels',{'No feedback','Numeric feedback','Optional feedcack', 'Sub-goal', 'Sub-goal & numeric feedback'});
        %set(gca(), 'xticklabel', {sprintf('No\nfeedback'), sprintf('Numeric\nfeedback'), sprintf('Optional\nfeedback'), 'Sub-goal', sprintf('Sub-goal with\nnumeric feedback')});


        % Create cell arrays for two rows of labels
        row1 = {'     No', ' Numeric', ' Optional', 'Sub-goal', '      Sub-goal &'};
        row2 = {'feedback', 'feedback', 'feedback', '', 'numeric feedback'};   
        % Combine the labels with line breaks and spaces
        tickLabels = cellfun(@(r1, r2) [r1, '\newline', r2], row1, row2, 'UniformOutput', false);
        % Create x-positions for labels
        xPositions = 1:5;
        % Set custom xtick positions and labels
        ax = gca();
        ax.XTick = xPositions;
        ax.XLim = [0, 6];
        ax.XTickLabel = tickLabels;
        ax.TickLabelInterpreter = 'tex';
        % Adjust the x-axis label rotation if needed
        xtickangle(0); % 0 degrees for horizontal labels
                
        % Customizing box colors and styles
        boxColors = ['b', 'g', 'r', 'c', 'm'];
        h = findobj(gca,'Tag','Box');
        for j = 1:length(h)
           patch(get(h(j),'XData'),get(h(j),'YData'),boxColors(j),'FaceAlpha',0.5);
        end
        % Increase the linewidth of the median
        h2 = findobj(gca, 'Tag', 'Median');
        set(h2, 'LineWidth', 4); % Adjust the linewidth as needed
        % Set the linewidth of the whiskers to 3
        whiskerHandles = findobj(gca, 'Tag', 'Whisker');
        for i = 1:numel(whiskerHandles)
            set(whiskerHandles(i), {'LineWidth'}, {4});
        end
    else
        boxplot(data,'Labels',{'No feedback','Numeric feedback', 'Sub-goal & numeric feedback'});
        %set(gca, 'xticklabel', {'No feedback','Numeric feedback', 'Sub-goal with numeric feedback'});
        % Create cell arrays for two rows of labels
        row1 = {'     No', ' Numeric', '      Sub-goal &'};
        row2 = {'feedback', 'feedback','numeric feedback'};   
        % Combine the labels with line breaks and spaces
        tickLabels = cellfun(@(r1, r2) [r1, '\newline', r2], row1, row2, 'UniformOutput', false);
        % Create x-positions for labels
        xPositions = 1:3;
        % Set custom xtick positions and labels
        ax = gca();
        ax.XTick = xPositions;
        ax.XLim = [0, 4];
        ax.XTickLabel = tickLabels;
        ax.TickLabelInterpreter = 'tex';
        % Adjust the x-axis label rotation if needed
        xtickangle(0); % 0 degrees for horizontal labels






        % Customizing box colors and styles
        boxColors = ['b', 'g', 'r'];
        h = findobj(gca,'Tag','Box');
        for j = 1:length(h)
           patch(get(h(j),'XData'),get(h(j),'YData'),boxColors(j),'FaceAlpha',0.5);
        end
                % Increase the linewidth of the median
        h2 = findobj(gca, 'Tag', 'Median');
        set(h2, 'LineWidth', 4); % Adjust the linewidth as needed
        % Set the linewidth of the whiskers to 3
        whiskerHandles = findobj(gca, 'Tag', 'Whisker');
        for i = 1:numel(whiskerHandles)
            set(whiskerHandles(i), {'LineWidth'}, {4});
        end
    end
    hold on

% Adding title and labels
title(titl);
%xlabel('Experiments');
ylabel('Percentage Scores');



% Customize axis properties
set(gca, 'FontName', 'Arial', 'FontSize', 24);
set(gca, 'LineWidth', 3);
set(gca, 'Box', 'off');
set(gca, 'TickDir', 'out');
set(gca, 'TickLength', [0.02, 0.02]);
set(gca, 'XMinorTick', 'off');
set(gca, 'YMinorTick', 'on');
%grid on;

% Adding a legend
%legend(h, {'Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5'}, 'Location', 'Best');

% Adjust figure properties
fig = gcf;
fig.Color = 'w';  % Background color

% Adjust position and size of the legend
%lgd = findobj(gcf, 'Type', 'Legend');
%lgd.Position(1) = 0.65;  % Adjust horizontal position

% Manually adjust the plot layout
set(gcf, 'Position', [100, 100, 1600, 800]);  % Adjust figure size and position

% Save the plot as an image (optional)
% saveas(gcf, 'boxplot_customized.png');

% In this alternative approach, the manual adjustment of the figure size and position helps ensure that the plot fits well within the figure window. You can modify the position and size values as needed to achieve the desired appearance of the plot.

end

function compare_len_positive(experiment1, experiment2, experiment3, experiment4, experiment5)
    % Create a table to compare len_positive values for all experiments
    len_positive_data = [
        experiment1.standard_len_positive_train_old, experiment1.standard_len_positive_eval_old;
        experiment2.standard_len_positive_train_old, experiment2.standard_len_positive_eval_old;
        experiment3.standard_len_positive_train_old, experiment3.standard_len_positive_eval_old;
        experiment4.standard_len_positive_train_old, experiment4.standard_len_positive_eval_old;
        experiment5.standard_len_positive_train_old, experiment5.standard_len_positive_eval_old]';
    % Create a table with row and column names
    row_names = {'% positive train old', '% positive eval old'};
    column_names = {'Experiment 1', 'Experiment 2', 'Experiment 3', 'Experiment 4', 'Experiment 5'};
    len_positive_table = array2table(len_positive_data, 'RowNames', row_names, 'VariableNames', column_names);
    
    % Display the table
    disp('Comparison of len_positive values for all experiments:');
    disp(len_positive_table);
end


function experiment = setup_experiment(exp_participants_old)
    experiment = struct();
    experiment.old_participants = exp_participants_old;

    standard_percentage_score_train_old = [];
    standard_percentage_score_eval_old = [];

    percentage_score_train_old = [];
    percentage_score_eval_old = [];

    standard_percentage_score_train_old_participant = [];
    standard_percentage_score_eval_old_participant = [];

    percentage_score_train_old_participant = [];
    percentage_score_eval_old_participant = [];

    for i=1:size(experiment.old_participants,2)
        standard_percentage_score_train_old = [standard_percentage_score_train_old, experiment.old_participants(i).standard_percentage_score_train];
        standard_percentage_score_eval_old = [standard_percentage_score_eval_old, experiment.old_participants(i).standard_percentage_score_eval];
        standard_percentage_score_train_old_participant = [standard_percentage_score_train_old_participant; experiment.old_participants(i).standard_percentage_score_train];
        standard_percentage_score_eval_old_participant = [standard_percentage_score_eval_old_participant; experiment.old_participants(i).standard_percentage_score_eval];


        percentage_score_train_old = [percentage_score_train_old, experiment.old_participants(i).percentage_score_train];
        percentage_score_eval_old = [percentage_score_eval_old, experiment.old_participants(i).percentage_score_eval];
        percentage_score_train_old_participant = [percentage_score_train_old_participant; experiment.old_participants(i).percentage_score_train];
        percentage_score_eval_old_participant = [percentage_score_eval_old_participant; experiment.old_participants(i).percentage_score_eval];
    end
    
    

    experiment.standard_percentage_score_train_old = standard_percentage_score_train_old;
    experiment.standard_percentage_score_eval_old = standard_percentage_score_eval_old;

    experiment.standard_percentage_score_train_old_participant = standard_percentage_score_train_old_participant;
    experiment.standard_percentage_score_eval_old_participant = standard_percentage_score_eval_old_participant;


    experiment.percentage_score_train_old = percentage_score_train_old;
    experiment.percentage_score_eval_old = percentage_score_eval_old;

    experiment.percentage_score_train_old_participant = percentage_score_train_old_participant;
    experiment.percentage_score_eval_old_participant = percentage_score_eval_old_participant;
    
    experiment.standard_positive_score_train_old = experiment.standard_percentage_score_train_old(experiment.standard_percentage_score_train_old > 0);
    experiment.standard_positive_score_eval_old = experiment.standard_percentage_score_eval_old(experiment.standard_percentage_score_eval_old > 0);
    

    experiment.standard_len_positive_train_old = 100*length(experiment.standard_positive_score_train_old)/length(experiment.standard_percentage_score_train_old);
    experiment.standard_len_positive_eval_old = 100*length(experiment.standard_positive_score_eval_old)/length(experiment.standard_percentage_score_eval_old);

    experiment.positive_score_train_old = experiment.percentage_score_train_old(experiment.percentage_score_train_old > 0);
    experiment.positive_score_eval_old = experiment.percentage_score_eval_old(experiment.percentage_score_eval_old > 0);

    experiment.len_positive_train_old = 100*length(experiment.positive_score_train_old)/length(experiment.percentage_score_train_old);
    experiment.len_positive_eval_old = 100*length(experiment.positive_score_eval_old)/length(experiment.percentage_score_eval_old);
    
end

