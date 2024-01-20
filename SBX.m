function offspring = SBX(obj, population)

    indorder = randperm(length(population));
    count = 1;

    for i = 1:ceil(length(population) / 2)

        p1 = indorder(i);
        p2 = indorder(i + fix(length(population) / 2));
        offspring(count) = population(p1);
        offspring(count + 1) = population(p2);

        [offspring(count).rnvec, offspring(count + 1).rnvec] = GA_Crossover(population(p1).rnvec, population(p2).rnvec, obj.GA_MuC);

        offspring(count).rnvec = GA_Mutation(offspring(count).rnvec, obj.GA_MuM);
        offspring(count + 1).rnvec = GA_Mutation(offspring(count + 1).rnvec, obj.GA_MuM);

        for x = count:count + 1
            offspring(x).rnvec(offspring(x).rnvec > 1) = 1;
            offspring(x).rnvec(offspring(x).rnvec < 0) = 0;
        end

        count = count + 2;

    end

end
