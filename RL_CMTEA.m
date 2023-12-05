classdef RL_CMTEA < Algorithm
    % <Multi-task> <Single-objective> <Constrained>

    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2022 Yanchi Li. You are free to use the MTO-Platform for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "MTO-Platform" and cite
    % or footnote "https://github.com/intLyc/MTO-Platform"
    %--------------------------------------------------------------------------

    properties (SetAccess = private)
        P = 0.1
        RMP0 = 0.3
        GA_MuC = 2
        GA_MuM = 5
        DE_F = 0.5
        DE_CR = 0.7
    end

    methods

        function Parameter = getParameter(Algo)
            Parameter = {'P: 100p% top as pbest', num2str(Algo.P), ...
                             'RMP0: Random Mating Probability', num2str(Algo.RMP0), ...
                             'MuC: GA Simulated Binary Crossover', num2str(Algo.GA_MuC), ...
                             'MuM: GA Polynomial Mutation', num2str(Algo.GA_MuM), ...
                             'DE_F: DE Mutation Factor', num2str(Algo.DE_F), ...
                             'DE_CR: DE Crossover Probability', num2str(Algo.DE_CR)};
        end

        function Algo = setParameter(Algo, Parameter)
            i = 1;
            Algo.P = str2double(Parameter{i}); i = i + 1;
            Algo.RMP0 = str2double(Parameter{i}); i = i + 1;
            Algo.GA_MuC = str2double(Parameter{i}); i = i + 1;
            Algo.GA_MuM = str2double(Parameter{i}); i = i + 1;
            Algo.DE_F = str2double(Parameter{i}); i = i + 1;
            Algo.DE_CR = str2double(Parameter{i}); i = i + 1;
        end

        function run(Algo, Prob)
            % Initialization
            population = Initialization(Algo, Prob, Individual);

            % For knowledge transfer
            maxD = min(Prob.D);
            main_divD = randi([1, maxD]);
            aux_divD = randi([1, maxD]);
            minK = 2;
            maxK = Prob.N / 2;
            main_divK = randi([minK, maxK]);
            aux_divK = randi([minK, maxK]);

            %% For the improve epsilon cht
            tao = 0.05;
            cp = 2;
            alpha = 0.95;
            Tc = 0.9 * ceil(Prob.maxFE / (4 * Prob.N));
            gen = 1;

            EC_Top = 0.2;
            EC_Alpha = 0.8;
            EC_Cp = 2;
            EC_Tc = 0.8;

            %% For QL
            num_pop_each_task = 2;
            num_pop = num_pop_each_task * Prob.T;
            num_operator = 4;
            Q_Table = zeros(num_pop, num_operator);
            alpha_ql = 0.01;
            gamma_ql = 0.9;

            varepsilon_ucb = 1e-6;
            action_counts = zeros(num_pop, num_operator);
            UCB_values = zeros(num_pop, num_operator);
            UCB_T = ceil(Prob.maxFE / (4 * Prob.N));

            for t = 1:Prob.T
                % Epsilon
                n = ceil(EC_Top * length(population{t}));
                cv_temp = [population{t}.CV];
                [~, idx] = sort(cv_temp);
                Ep{t} = cv_temp(idx(n));
                main_pop{t} = Selection_Elit(population{t}(1:Prob.N / 2), population{t}(Prob.N / 2 + 1:Prob.N));
                aux_pop{t} = Selection_Elit(population{t}(1:Prob.N / 2), population{t}(Prob.N / 2 + 1:Prob.N), Ep{t});
            end

            while Algo.notTerminated(Prob)

                main_off1 = KT(Algo, Prob, main_pop, main_divK, main_divD);
                aux_off1 = KT(Algo, Prob, aux_pop, aux_divK, aux_divD);

                % RMP
                if gen < 4
                    RMP(1:Prob.T) = Algo.RMP0;
                else
                    x1 = [Algo.Result(:, gen - 1).Obj];
                    x2 = [Algo.Result(:, gen - 2).Obj];
                    x3 = [Algo.Result(:, gen - 3).Obj];
                    temp1 = x2 - x1;
                    temp2 = x3 - x2;
                    RMP = temp1 ./ (temp1 + temp2);
                    RMP(isnan(RMP)) = Algo.RMP0;
                end

                % Update Epsilon
                fea_percent = sum([aux_pop{t}.CV] <= 0) / length(aux_pop{t});

                if fea_percent < 1
                    Ep{t} = max([aux_pop{t}.CV]);
                end

                if Algo.FE / Prob.maxFE < EC_Tc

                    if fea_percent < EC_Alpha
                        Ep{t} = Ep{t} * (1 - Algo.FE / (Prob.maxFE * EC_Tc)) ^ EC_Cp;
                    else
                        Ep{t} = 1.1 * max([aux_pop{t}.CV]);
                    end

                else
                    Ep{t} = 0;
                end

                for t = 1:Prob.T

                    t_1 = num_pop_each_task * t - 1;
                    t_2 = num_pop_each_task * t;

                    UCB_values(t_1, :) = Q_Table(t_1, :) + sqrt(2 * log(UCB_T) ./ (action_counts(t_1, :) + varepsilon_ucb));
                    UCB_values(t_2, :) = Q_Table(t_2, :) + sqrt(2 * log(UCB_T) ./ (action_counts(t_2, :) + varepsilon_ucb));

                    % Choose actions based on UCB values for each population of task t
                    [~, action(t_1)] = max(UCB_values(t_1, :));
                    [~, action(t_2)] = max(UCB_values(t_2, :));

                    % Increment the count for the chosen actions
                    action_counts(t_1, action(t_1)) = action_counts(t_1, action(t_1)) + 1;
                    action_counts(t_2, action(t_2)) = action_counts(t_2, action(t_2)) + 1;

                    % Update the epsilon for auxiliary population
                    aux_cv{t} = overall_cv(aux_pop{t}.CVs);
                    max_epsilon{t} = max(aux_cv{t});
                    rf{t} = sum(aux_cv{t} <= 1e-6) / length(aux_pop{t});

                    another_t = randi(Prob.T);

                    while another_t == t
                        another_t = randi(Prob.T);
                    end

                    % Generation
                    if action(t_1) == 1
                        main_off2 = SBX(Algo, main_pop{t}, RMP(t), main_pop{another_t});
                    elseif action(t_1) == 2
                        main_off2 = DE_rand_1(Algo, main_pop{t}, RMP(t), main_pop{another_t});
                    elseif action(t_1) == 3
                        main_off2 = DE_rand_2(Algo, main_pop{t}, RMP(t), main_pop{another_t});
                    else
                        main_off2 = DE_best_1(Algo, main_pop{t}, RMP(t), main_pop{another_t});
                    end

                    if action(t_2) == 1
                        aux_off2 = SBX(Algo, aux_pop{t}, RMP(t), aux_pop{another_t});
                    elseif action(t_2) == 2
                        aux_off2 = DE_rand_1(Algo, aux_pop{t}, RMP(t), aux_pop{another_t});
                    elseif action(t_2) == 3
                        aux_off2 = DE_rand_2(Algo, aux_pop{t}, RMP(t), aux_pop{another_t});
                    else
                        aux_off2 = DE_best_1(Algo, aux_pop{t}, RMP(t), aux_pop{another_t});
                    end

                    % Evaluation
                    main_off = [main_off1{t}, main_off2];
                    aux_off = [aux_off1{t}, aux_off2];
                    [main_off, main_flag(t)] = Algo.Evaluation(main_off, Prob, t);
                    [aux_off, aux_flag(t)] = Algo.Evaluation(aux_off, Prob, t);
                    % Selection
                    [main_pop{t}, main_rank{t}] = Selection_Elit(main_pop{t}, [main_off, aux_off]);
                    [aux_pop{t}, aux_rank{t}] = Selection_Elit(aux_pop{t}, aux_off, Ep{t});

                    % determine the transfer rate to update Q table
                    main_next = zeros(length(main_rank{t}), 1);
                    aux_next = zeros(length(aux_rank{t}), 1);
                    main_next(main_rank{t}(1:length(main_pop{t}))) = true;
                    aux_next(aux_rank{t}(1:length(aux_pop{t}))) = true;
                    main_off2_start = length(main_pop{t}) + length(main_off1{t}) + 1;
                    main_off2_end = length(main_pop{t}) + length(main_off1{t}) + length(main_off2);
                    main_succ_rate = (sum(main_next(main_off2_start:main_off2_end))) / (length(main_pop{t}) + length(main_off2));
                    aux_succ_rate = (sum(aux_next(length(aux_pop{t}) + length(aux_off1{t}) + 1:end))) / (length(aux_pop{t}) + length(aux_off2));
                    Q_Table(t_1, action(t_1)) = Q_Table(t_1, action(t_1)) + alpha_ql * (main_succ_rate + gamma_ql * (max(Q_Table(t_1, :))) - Q_Table(t_1, action(t_1)));
                    Q_Table(t_2, action(t_2)) = Q_Table(t_2, action(t_2)) + alpha_ql * (aux_succ_rate + gamma_ql * (max(Q_Table(t_2, :))) - Q_Table(t_2, action(t_2)));
                end

                [main_divD, main_divK] = update_divd_divk(main_flag, main_divD, main_divK, maxD, minK, maxK);
                [aux_divD, aux_divK] = update_divd_divk(aux_flag, aux_divD, aux_divK, maxD, minK, maxK);

                gen = gen + 1;
            end

        end

    end

end

function result = overall_cv(cv)
    cv(cv <= 0) = 0; cv = abs(cv);
    result = sum(cv, 2);
end

function [divD, divK] = update_divd_divk(succ_flag, divD, divK, maxD, minK, maxK)

    if all(~succ_flag)
        divD = randi([1, maxD]);
        divK = randi([minK, maxK]);
    elseif any(~succ_flag)
        divD = min(maxD, max(1, randi([divD - 1, divD + 1])));
        divK = min(maxK, max(minK, randi([divK - 1, divK + 1])));
    end

end
