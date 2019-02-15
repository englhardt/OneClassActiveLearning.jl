
@testset "DataSplits" begin
    @testset "calc_mask" begin
        pools = fill(:U, 5)
        pools[1] = :Lin
        pools[2] = :Lout
        init_mask = trues(5)
        init_mask[5] = false
        @test calc_mask(FullSplitStrat(), init_mask, pools) == init_mask
        @test calc_mask(UnlabeledSplitStrat(), init_mask, pools) == [false, false, true, true, false]
        @test calc_mask(UnlabeledAndLabeledInlierSplitStrat(), init_mask, pools) == [true, false, true, true, false]
        @test calc_mask(LabeledSplitStrat(), init_mask, pools) == [true, true, false, false, false]
        @test calc_mask(LabeledInlierSplitStrat(), init_mask, pools) == [true, false, false, false, false]
        @test calc_mask(LabeledOutlierSplitStrat(), init_mask, pools) == [false, true, false, false, false]
    end

    @testset "get_local_idx" begin
        pools = fill(:U, 6)
        train = BitArray([true, true, false, false, true, true])
        ds = DataSplits(train, .~train, FullSplitStrat())

        @test OneClassActiveLearning.get_local_idx(1, ds, pools, Val(:train)) == 1
        @test OneClassActiveLearning.get_local_idx(5, ds, pools, Val(:train)) == 3
        @test OneClassActiveLearning.get_local_idx(6, ds, pools, Val(:train)) == 4
        @test_throws ErrorException OneClassActiveLearning.get_local_idx(3, ds, pools, Val(:train))

        @test OneClassActiveLearning.get_local_idx(3, ds, pools, Val(:test)) == 1
        @test_throws ErrorException OneClassActiveLearning.get_local_idx(5, ds, pools, Val(:test))

        pools[3:6] .= :Lin
        ds = DataSplits(train, .~train, LabeledInlierSplitStrat())

        @test_throws ErrorException OneClassActiveLearning.get_local_idx(1, ds, pools, Val(:train))
        @test OneClassActiveLearning.get_local_idx(5, ds, pools, Val(:train)) == 1
    end

    @testset "filter_query_id" begin
        train = vcat(trues(5), falses(5))
        query_ids = [1,3,5]
        query_labels = [:Lin, :Lout, :U]
        @test OneClassActiveLearning.filter_query_id(query_ids, DataSplits(train, .~train, FullSplitStrat()), query_labels, Val(:train)) == [1,3,5]
        @test OneClassActiveLearning.filter_query_id(query_ids, DataSplits(train, .~train, LabeledInlierSplitStrat()), query_labels, Val(:train)) == [1]
        @test OneClassActiveLearning.filter_query_id(query_ids, DataSplits(train, .~train, LabeledOutlierSplitStrat()), query_labels, Val(:train)) == [3]
        @test OneClassActiveLearning.filter_query_id(query_ids, DataSplits(train, .~train, UnlabeledAndLabeledInlierSplitStrat()), query_labels, Val(:train)) == [1,5]
    end

    @testset "no holdout" begin
        data = reshape(collect(1:10), (2, 5))
        pools = fill(:U, 5)
        pools[1] = :Lin
        pools[2] = :Lout
        train = trues(5)
        ds = DataSplits(train, UnlabeledSplitStrat())
        train_data, train_pools, train_idx = get_train(ds, data, pools)
        @test size(train_data, 2) == length(train_pools) == length(train_idx)
        @test train_data == data[:, pools .== :U]
        @test all(train_pools .== :U)
        @test train_idx == [3, 4, 5]
    end

    @testset "with holdout" begin
        data = zeros(2, 5)
        pools = fill(:U, 5)
        train = BitArray([true, true, true, false, false])
        test = BitArray([false, false, false, true, true])
        @assert all(train .⊻ test)
        ds = DataSplits(train, test, FullSplitStrat())
        _, _, train_idx = get_train(ds, data, pools)
        _, _, test_idx = get_test(ds, data, pools)
        _, _, query_idx = get_query(ds, data, pools)
        @test train_idx == [1, 2, 3]
        @test test_idx == [4, 5]
        @test query_idx == [1, 2, 3]
    end
end

@testset "splits and init pools" begin
    data = zeros(2, 100)
    labels = fill(:inlier, 100)
    labels[1:10] .= :outlier
    @test_throws ArgumentError OneClassActiveLearning.get_splits_and_init_pools(data, labels, "foo", "Pu")
    @test_throws ArgumentError OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sf", "foo")
    @testset "full split strategy" begin
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sf", "Pu")
        @test ds.train == ds.test == trues(size(data, 2))
        @test ip == fill(:U, size(data, 2))
        for p in [0.1, 0.2]
            ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sf", "Pp"; p=p)
            @test ds.train == ds.test == trues(size(data, 2))
            @test length(findall(x -> (x == :Lin) || (x == :Lout), ip)) == round(Int, size(data, 2) * p)
        end
        for n in [10, 20, 30]
            ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sf", "Pn"; n=n)
            @test ds.train == ds.test == trues(size(data, 2))
            @test length(findall(x -> (x == :Lin) || (x == :Lout), ip)) == n
        end
        for n in [11, 22, 33]
            ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sf", "Pnin"; n=n)
            @test ds.train == ds.test == trues(size(data, 2))
            @test length(findall(x -> x == :Lin, ip)) == n
        end
        for x in [5, 10, 20]
            ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sf", "Pa"; x=x)
            @test ds.train == ds.test == trues(size(data, 2))
            @test length(findall(x -> x == :Lin, ip)) == size(data, 1) + x
            ds, ip = OneClassActiveLearning.get_splits_and_init_pools(zeros(150, 100), labels, "Sf", "Pa"; x=x)
            @test ds.train == ds.test == trues(size(data, 2))
            @test length(findall(x -> x == :Lin, ip)) == length(findall(x -> x == :inlier, labels))
        end
    end
    @testset "holdout" begin
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sh", "Pu")
        @test (sum(ds.train) == 80) && (sum(ds.test) == 20) && all(ds.train .⊻ ds.test)
        @test ip == fill(:U, size(data, 2))
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sh", "Pu"; holdout_p=0.1)
        @test (sum(ds.train) == 90) && (sum(ds.test) == 10) && all(ds.train .⊻ ds.test)
        @test ip == fill(:U, size(data, 2))
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sh", "Pp")
        @test (sum(ds.train) == 80) && (sum(ds.test) == 20) && all(ds.train .⊻ ds.test)
        @test length(findall(x -> (x == :Lin) || (x == :Lout), ip[ds.train])) == sum(ds.train) * 0.1
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sh", "Pn")
        @test (sum(ds.train) == 80) && (sum(ds.test) == 20) && all(ds.train .⊻ ds.test)
        @test length(findall(x -> (x == :Lin) || (x == :Lout), ip[ds.train])) == 20
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sh", "Pnin")
        @test (sum(ds.train) == 80) && (sum(ds.test) == 20) && all(ds.train .⊻ ds.test)
        @test length(findall(x -> x == :Lin, ip)) == 20
        @test length(findall(x -> x == :Lin, ip[ds.train])) == 20
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sh", "Pa")
        @test (sum(ds.train) == 80) && (sum(ds.test) == 20) && all(ds.train .⊻ ds.test)
        @test length(findall(x -> x == :Lin, ip[ds.train])) == size(data, 1) + 10
    end
    @testset "inff" begin
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Si", "Pu")
        @test ds.train == ds.test == trues(size(data, 2))
        @test ip == fill(:U, size(data, 2))
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Si", "Pp")
        @test ds.train == ds.test == trues(size(data, 2))
        @test length(findall(x -> (x == :Lin) || (x == :Lout), ip)) == sum(ds.train) * 0.1
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Si", "Pn")
        @test ds.train == ds.test == trues(size(data, 2))
        @test length(findall(x -> (x == :Lin) || (x == :Lout), ip)) == 20
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Si", "Pnin")
        @test ds.train == ds.test == trues(size(data, 2))
        @test length(findall(x -> x == :Lin, ip)) == 20
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Si", "Pa")
        @test ds.train == ds.test == trues(size(data, 2))
        @test length(findall(x -> x == :Lin, ip)) == size(data, 1) + 10
    end
    @testset "lff" begin
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sl", "Pu")
        @test ds.train == ds.test == trues(size(data, 2))
        @test ip == fill(:U, size(data, 2))
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sl", "Pp")
        @test ds.train == ds.test == trues(size(data, 2))
        @test length(findall(x -> (x == :Lin) || (x == :Lout), ip)) == sum(ds.train) * 0.1
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sl", "Pn")
        @test ds.train == ds.test == trues(size(data, 2))
        @test length(findall(x -> (x == :Lin) || (x == :Lout), ip)) == 20
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sl", "Pnin")
        @test ds.train == ds.test == trues(size(data, 2))
        @test length(findall(x -> x == :Lin, ip)) == 20
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sl", "Pa")
        @test ds.train == ds.test == trues(size(data, 2))
        @test length(findall(x -> x == :Lin, ip)) == size(data, 1) + 10
    end
end
