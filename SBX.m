function offspring = SBX(Algo, population, RMP, transfer_pop)

    indorder = randperm(length(population));
    count = 1;

    for i = 1:ceil(length(population) / 2)

        if rand < RMP
            p1 = randi(ceil(Algo.P * length(transfer_pop)));
            p2 = randi(ceil(Algo.P * length(transfer_pop)));

            while p1 == p2
                p2 = randi(ceil(Algo.P * length(transfer_pop)));
            end

            offspring(count) = transfer_pop(p1);
            offspring(count + 1) = transfer_pop(p2);

            [offspring(count).Dec, offspring(count + 1).Dec] = GA_Crossover(transfer_pop(p1).Dec, transfer_pop(p2).Dec, Algo.GA_MuC);

            offspring(count).Dec = GA_Mutation(offspring(count).Dec, Algo.GA_MuM);
            offspring(count + 1).Dec = GA_Mutation(offspring(count + 1).Dec, Algo.GA_MuM);
        else
            p1 = indorder(i);
            p2 = indorder(i + fix(length(population) / 2));
            offspring(count) = population(p1);
            offspring(count + 1) = population(p2);

            [offspring(count).Dec, offspring(count + 1).Dec] = GA_Crossover(population(p1).Dec, population(p2).Dec, Algo.GA_MuC);

            offspring(count).Dec = GA_Mutation(offspring(count).Dec, Algo.GA_MuM);
            offspring(count + 1).Dec = GA_Mutation(offspring(count + 1).Dec, Algo.GA_MuM);
        end

        for x = count:count + 1
            offspring(x).Dec(offspring(x).Dec > 1) = 1;
            offspring(x).Dec(offspring(x).Dec < 0) = 0;
        end

        count = count + 2;

    end

end
