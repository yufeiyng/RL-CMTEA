function offspring = DE_rand_2(obj, population)

    for i = 1:length(population)
        offspring(i) = population(i);

        X = randperm(length(population));
        X(X == i) = []; x1 = X(1); x2 = X(2); x3 = X(3); x4 = X(4);

        % DE/rand/2
        offspring(i).rnvec = population(i).rnvec + obj.DE_F * (population(x1).rnvec - population(x2).rnvec) + ...
            obj.DE_F * (population(x3).rnvec - population(x4).rnvec);

        offspring(i).rnvec = DE_Crossover(offspring(i).rnvec, population(i).rnvec, obj.DE_CR);

        offspring(i).rnvec(offspring(i).rnvec > 1) = 1;
        offspring(i).rnvec(offspring(i).rnvec < 0) = 0;

    end

end
