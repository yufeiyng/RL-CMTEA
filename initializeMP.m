function [population, fnceval_calls, bestobj, bestCV, bestX] = initializeMP(Individual_class, pop_size, Tasks, dims, varargin)
    %% Constrained Multifactorial - Initialize and evaluate the population
    % Input: Individual_class, pop_size, Tasks, dim
    % Output: population, calls (function calls number), bestobj, bestCV, bestX, type

    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2022 Yanchi Li. You are free to use the MTO-Platform for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "MTO-Platform" and cite
    % or footnote "https://github.com/intLyc/MTO-Platform"
    %--------------------------------------------------------------------------
    n = numel(varargin);

    if n == 0
        type = 'Feasible_Priority'; % unified [0, 1]
    elseif n == 2
        type = varargin{1};
        var = varargin{2};
    end

    fnceval_calls = 0;

    for t = 1:length(Tasks)

        for i = 1:pop_size
            population{t}(i) = Individual_class();
            population{t}(i).rnvec = rand(1, dims);
        end

        [population{t}, calls] = evaluate(population{t}, Tasks(t), 1);
        fnceval_calls = fnceval_calls + calls;

        switch type
            case 'Feasible_Priority'
                rank = sort_FP(population{t}.factorial_costs, population{t}.constraint_violation);
            case 'Stochastic_Ranking'
                rank = sort_SR(population{t}.factorial_costs, population{t}.constraint_violation, var{t});
            case 'Epsilon_Constraint'
                rank = sort_EC(population{t}.factorial_costs, population{t}.constraint_violation, var{t});
        end

        bestobj(t) = population{t}(rank(1)).factorial_costs;
        bestCV(t) = population{t}(rank(1)).constraint_violation;
        bestX{t} = population{t}(rank(1)).rnvec;

    end

end
