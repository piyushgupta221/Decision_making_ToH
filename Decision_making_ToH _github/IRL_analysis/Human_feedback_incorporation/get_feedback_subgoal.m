function H=get_feedback_subgoal(dist, sa_s, sa_p, target_state, subgoal)
sa_s_prime = sum(sa_s.*sa_p,3);
H = zeros(size(sa_s_prime));

for i=1:size(sa_s_prime,1)
    for j=1:size(sa_s_prime,2)
        current_state_dist = dist(i,target_state);
        next_state_distance = dist(sa_s_prime(i,j), target_state);
        if (current_state_dist - next_state_distance)<0
            H(i,j) = -2;
        elseif (current_state_dist - next_state_distance)>0
            H(i,j) = 2;
        else
            H(i,j) = 0;
        end
        if (sa_s_prime(i,j)==subgoal && i~=sa_s_prime(i,j))
            H(i,j)=H(i,j)+5;
        end
        if (i==subgoal && H(i,j)<0)
            H(i,j)=H(i,j)-5;
        end
    end
end