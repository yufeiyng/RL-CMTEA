classdef RL_CMTEA < Algorithm
    % <Multi><Constrained>
    % alpha = 0.01, gamma = 0.9, F = 0.5, CR = 0.5

    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2022 Yanchi Li. You are free to use the MTO-Platform for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "MTO-Platform" and cite
    % or footnote "https://github.com/intLyc/MTO-Platform"
    %--------------------------------------------------------------------------

    properties (SetAccess = private)
        GA_MuC = 2
        GA_MuM = 5
        DE_F = 0.5
        DE_CR = 0.5
    end

    methods

        function parameter = getParameter(obj)
            parameter = {'MuC: GA Simulated Binary Crossover', num2str(obj.GA_MuC), ...
                             'MuM: GA Polynomial Mutation', num2str(obj.GA_MuM), ...
                             'DE_F: DE Mutation Factor', num2str(obj.DE_F), ...
                             'DE_CR: DE Crossover Probability', num2str(obj.DE_CR)};
        end

        function obj = setParameter(obj, parameter_cell)
            i = 1;
            obj.GA_MuC = str2double(parameter_cell{i}); i = i + 1;
            obj.GA_MuM = str2double(parameter_cell{i}); i = i + 1;
            obj.DE_F = str2double(parameter_cell{i}); i = i + 1;
            obj.DE_CR = str2double(parameter_cell{i}); i = i + 1;
        end

        function data = run(obj, Tasks, RunPara)
            sub_pop = RunPara(1); sub_eva = RunPara(2);
            eva_num = sub_eva * length(Tasks);

            % Initialization
            [population, fnceval_calls, bestobj, bestCV, bestX] = initializeMP(Individual, sub_pop, Tasks, max(Tasks.dims));

            convergence(:, 1) = bestobj;
            convergence_cv(:, 1) = bestCV;
            data.bestX = bestX;

            % For knowledge transfer
            maxD = min(max(Tasks.dims));
            main_divD = randi([1, maxD]);
            aux_divD = randi([1, maxD]);
            minK = 2; maxK = sub_pop / 2;
            main_divK = randi([minK, maxK]);
            aux_divK = randi([minK, maxK]);

            EC_Top = 0.2; EC_Alpha = 0.8; EC_Cp = 2; EC_Tc = 0.8;

            %% For QL
            num_pop_each_task = 2;
            num_pop = num_pop_each_task * length(Tasks);
            num_operator = 4;
            Q_Table = zeros(num_pop, num_operator);
            gamma_ql = 0.9;
            alpha_ql = 0.01;

            varepsilon_ucb = 1e-6;
            action_counts = zeros(num_pop, num_operator);
            UCB_values = zeros(num_pop, num_operator);
            UCB_T = ceil(eva_num / (4 * sub_pop));

            for t = 1:length(Tasks)
                % Epsilon
                n = ceil(EC_Top * length(population{t}));
                cv_temp = [population{t}.constraint_violation];
                [~, idx] = sort(cv_temp);
                Ep{t} = cv_temp(idx(n));
                sub_pop1 = population{t}(1:sub_pop / 2);
                sub_pop2 = population{t}(sub_pop / 2 + 1:sub_pop);
                [main_pop{t}] = selectMP(sub_pop1, sub_pop2, bestobj(t), bestCV(t), bestX{t}, 0);
                [aux_pop{t}] = selectMP(sub_pop1, sub_pop2, bestobj(t), bestCV(t), bestX{t}, Ep{t});
            end

            gen = 1;

            while fnceval_calls < eva_num

                main_off1 = KT(obj, Tasks, main_pop, main_divK, main_divD);
                aux_off1 = KT(obj, Tasks, aux_pop, aux_divK, aux_divD);

                % Update Epsilon
                fea_percent = sum([aux_pop{t}.constraint_violation] <= 0) / length(aux_pop{t});

                if fea_percent < 1
                    Ep{t} = max([aux_pop{t}.constraint_violation]);
                end

                if fnceval_calls / eva_num < EC_Tc

                    if fea_percent < EC_Alpha
                        Ep{t} = Ep{t} * (1 - fnceval_calls / (eva_num * EC_Tc)) ^ EC_Cp;
                    else
                        Ep{t} = 1.1 * max([aux_pop{t}.constraint_violation]);
                    end

                else
                    Ep{t} = 0;
                end

                for t = 1:length(Tasks)

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

                    % Generation
                    if action(t_1) == 1
                        main_off2 = SBX(obj, main_pop{t});
                    elseif action(t_1) == 2
                        main_off2 = DE_rand_1(obj, main_pop{t});
                    elseif action(t_1) == 3
                        main_off2 = DE_rand_2(obj, main_pop{t});
                    else
                        main_off2 = DE_best_1(obj, main_pop{t});
                    end

                    if action(t_2) == 1
                        aux_off2 = SBX(obj, aux_pop{t});
                    elseif action(t_2) == 2
                        aux_off2 = DE_rand_1(obj, aux_pop{t});
                    elseif action(t_2) == 3
                        aux_off2 = DE_rand_2(obj, aux_pop{t});
                    else
                        aux_off2 = DE_best_1(obj, aux_pop{t});
                    end

                    % Evaluation
                    main_off = [main_off1{t}, main_off2];
                    aux_off = [aux_off1{t}, aux_off2];
                    [main_off, calls] = evaluate(main_off, Tasks(t), 1);
                    fnceval_calls = fnceval_calls + calls;
                    [aux_off, calls] = evaluate(aux_off, Tasks(t), 1);
                    fnceval_calls = fnceval_calls + calls;

                    % Selection
                    [main_pop{t}, main_rank{t}, bestobj(t), bestCV(t), bestX{t}, main_flag(t)] = selectMP(main_pop{t}, main_off, bestobj(t), bestCV(t), bestX{t}, 0);
                    [main_pop{t}, ~, bestobj(t), bestCV(t), bestX{t}, ~] = selectMP(main_pop{t}, [main_off, aux_off], bestobj(t), bestCV(t), bestX{t}, 0);
                    [aux_pop{t}, aux_rank{t}, ~, ~, ~, aux_flag(t)] = selectMP(aux_pop{t}, aux_off, bestobj(t), bestCV(t), bestX{t}, Ep{t});

                    % determine the transfer rate to update Q table
                    main_next = zeros(length(main_rank{t}), 1);
                    aux_next = zeros(length(aux_rank{t}), 1);
                    main_next(main_rank{t}(1:length(main_pop{t}))) = true;
                    aux_next(aux_rank{t}(1:length(aux_pop{t}))) = true;
                    main_succ_rate = (sum(main_next(length(main_pop{t}) + length(main_off1{t}) + 1:end))) / (length(main_pop{t}) + length(main_off2));
                    aux_succ_rate = (sum(aux_next(length(aux_pop{t}) + length(aux_off1{t}) + 1:end))) / (length(aux_pop{t}) + length(aux_off2));
                    Q_Table(t_1, action(t_1)) = Q_Table(t_1, action(t_1)) + alpha_ql * (main_succ_rate + gamma_ql * (max(Q_Table(t_1, :))) - Q_Table(t_1, action(t_1)));
                    Q_Table(t_2, action(t_2)) = Q_Table(t_2, action(t_2)) + alpha_ql * (aux_succ_rate + gamma_ql * (max(Q_Table(t_2, :))) - Q_Table(t_2, action(t_2)));
                end

                [main_divD, main_divK] = update_divd_divk(main_flag, main_divD, main_divK, maxD, minK, maxK);
                [aux_divD, aux_divK] = update_divd_divk(aux_flag, aux_divD, aux_divK, maxD, minK, maxK);

                gen = gen + 1;
                convergence(:, gen) = bestobj;
                convergence_cv(:, gen) = bestCV;

            end

            data.convergence = gen2eva(convergence);
            data.convergence_cv = gen2eva(convergence_cv);
            data.bestX = uni2real(bestX, Tasks);
        end

    end

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
