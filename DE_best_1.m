function offspring = DE_best_1(obj, population)
    % The DE/best/1 operator

    for i = 1:length(population)
        offspring(i) = population(i);

        X = randperm(length(population));
        X(X == i) = []; x1 = X(1); x2 = X(2);

        offspring(i).rnvec = population(1).rnvec + obj.DE_F * (population(x1).rnvec - population(x2).rnvec);

        offspring(i).rnvec = DE_Crossover(offspring(i).rnvec, population(i).rnvec, obj.DE_CR);

        offspring(i).rnvec(offspring(i).rnvec > 1) = 1;
        offspring(i).rnvec(offspring(i).rnvec < 0) = 0;

    end

end
