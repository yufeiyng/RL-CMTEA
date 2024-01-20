function offspring = KT(Algo, Tasks, population, divK, divD)
    corre = [];

    for t = 1:length(Tasks)
        dim = max(Tasks.dims);

        for i = 1:length(population{t})

            for j = 1:ceil(dim / divD)

                if j * divD > dim
                    corre = [corre; [t, i, 1 + (j - 1) * divD, dim]];
                else
                    corre = [corre; [t, i, 1 + (j - 1) * divD, j * divD]];
                end

            end

        end

    end

    dimVal = [];

    for i = 1:size(corre, 1)
        dimVal = [dimVal; correDecode(population, corre(i, :), divD)];
    end

    idx = kmeans(dimVal, divK);
    subpop = cell(1, divK);

    for i = 1:divK
        subpop{i} = [];
    end

    for i = 1:length(idx)
        subpop{idx(i)} = [subpop{idx(i)}; corre(i, :)];
    end

    offspring_temp = [];
    off_corre = [];

    for k = 1:divK

        for i = 1:size(subpop{k}, 1)

            if size(subpop{k}, 1) < 4
                continue
            end

            A = randperm(size(subpop{k}, 1), 4);
            A(A == i) = []; r1 = A(1); r2 = A(2); r3 = A(3);
            dp1 = correDecode(population, subpop{k}(r1, :), divD);
            dp2 = correDecode(population, subpop{k}(r2, :), divD);
            dp3 = correDecode(population, subpop{k}(r3, :), divD);
            v = dp1 + Algo.DE_F * (dp2 - dp3);
            v = min(1, max(0, v));
            u = correDecode(population, subpop{k}(i, :), divD);
            u = DE_Crossover(v, u, Algo.DE_CR);
            offspring_temp = [offspring_temp; u];
            off_corre = [off_corre; subpop{k}(i, :)];
        end

    end

    offspring = population;

    for i = 1:size(off_corre, 1)
        data_seq = off_corre(i, :);
        offspring{data_seq(1)}(data_seq(2)).rnvec(data_seq(3):data_seq(4)) = offspring_temp(i, 1:data_seq(4) - data_seq(3) + 1);
    end

end

function result = correDecode(pop, correspond_vector, dim_div)
    task_index = correspond_vector(1);
    indv_index = correspond_vector(2);
    dim_start = correspond_vector(3);
    dim_end = correspond_vector(4);

    if dim_end - dim_start + 1 == dim_div
        result = pop{task_index}(indv_index).rnvec(dim_start:dim_end);
    else
        result = zeros(1, dim_div);
        result(1, 1:dim_end - dim_start + 1) = pop{task_index}(indv_index).rnvec(dim_start:dim_end);
    end

end
