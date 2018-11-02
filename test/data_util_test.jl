
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
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sf", "Pp")
        @test ds.train == ds.test == trues(size(data, 2))
        @test length(findall(x -> (x == :Lin) || (x == :Lout), ip)) == size(data, 2) * 0.1
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sf", "Pn")
        @test ds.train == ds.test == trues(size(data, 2))
        @test length(findall(x -> (x == :Lin) || (x == :Lout), ip)) == 20
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sf", "Pa", x=10)
        @test ds.train == ds.test == trues(size(data, 2))
        @test length(findall(x -> x == :Lin, ip)) == size(data, 1) + 10
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sf", "Pa", x=5)
        @test ds.train == ds.test == trues(size(data, 2))
        @test length(findall(x -> x == :Lin, ip)) == size(data, 1) + 5
    end
    @testset "80 20 holdout" begin
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sh", "Pu")
        @test (sum(ds.train) == 80) && (sum(ds.test) == 20) && all(ds.train .⊻ ds.test)
        @test ip == fill(:U, size(data, 2))
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sh", "Pp")
        @test (sum(ds.train) == 80) && (sum(ds.test) == 20) && all(ds.train .⊻ ds.test)
        @test length(findall(x -> (x == :Lin) || (x == :Lout), ip[ds.train])) == sum(ds.train) * 0.1
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sh", "Pn")
        @test (sum(ds.train) == 80) && (sum(ds.test) == 20) && all(ds.train .⊻ ds.test)
        @test length(findall(x -> (x == :Lin) || (x == :Lout), ip[ds.train])) == 20
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Sh", "Pa", x=10)
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
        ds, ip = OneClassActiveLearning.get_splits_and_init_pools(data, labels, "Si", "Pa", x=10)
        @test ds.train == ds.test == trues(size(data, 2))
        @test length(findall(x -> x == :Lin, ip)) == size(data, 1) + 10
    end
end
