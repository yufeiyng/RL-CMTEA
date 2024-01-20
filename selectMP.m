function [population, rank, bestobj, bestCV, bestX, Flag] = selectMP(population, offspring, bestobj, bestCV, bestX, ep)
    %% Constrained Multifactorial - Elite selection based on scalar fitness
    % Input: population (old), offspring, Tasks, pop_size, bestobj, bestCV, bestX, type
    % Output: population (new), bestobj, bestCV, bestX

    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2022 Yanchi Li. You are free to use the MTO-Platform for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "MTO-Platform" and cite
    % or footnote "https://github.com/intLyc/MTO-Platform"
    %--------------------------------------------------------------------------

    pop_temp = [population, offspring];

    for i = 1:length(pop_temp)
        obj(i) = pop_temp(i).factorial_costs;
        cv(i) = pop_temp(i).constraint_violation;
    end

    rank = sort_EC(obj, cv, ep);
    population = pop_temp(rank(1:length(population)));

    Flag = false;
    bestobj_now = pop_temp(rank(1)).factorial_costs;
    bestCV_now = pop_temp(rank(1)).constraint_violation;
    bestX_now = pop_temp(rank(1)).rnvec;

    if bestCV_now < bestCV || (bestCV_now == bestCV && bestobj_now <= bestobj)
        bestobj = bestobj_now;
        bestCV = bestCV_now;
        bestX = bestX_now;
        Flag = true;
    end

end
