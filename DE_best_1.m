function offspring = DE_best_1(Algo, population, RMP, transfer_pop)

    for i = 1:length(population)
        offspring(i) = population(i);

        X = randperm(length(population));
        X(X == i) = []; x1 = X(1); x2 = X(2);

        if rand < RMP
            idx = randi(ceil(Algo.P * length(transfer_pop)));

            % DE/current-to-best/1
            offspring(i).Dec = population(i).Dec + ...
                Algo.DE_F * (transfer_pop(idx).Dec - population(i).Dec) + ...
                Algo.DE_F * (population(x1).Dec - population(x2).Dec);
        else
            % DE/best/1
            offspring(i).Dec = population(1).Dec + Algo.DE_F * (population(x1).Dec - population(x2).Dec);
        end

        offspring(i).Dec = DE_Crossover(offspring(i).Dec, population(i).Dec, Algo.DE_CR);

        offspring(i).Dec(offspring(i).Dec > 1) = 1;
        offspring(i).Dec(offspring(i).Dec < 0) = 0;

    end

end
