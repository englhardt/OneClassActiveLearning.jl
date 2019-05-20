
@testset "SubspaceQs" begin

    dummy_data = rand(4,10)
    subspaces = [[1,2], [3,4]]
    model = SVDD.SubRandomOCClassifier(dummy_data, subspaces)

    params = Dict{Symbol, Any}()
    params[:subspaces] = subspaces
    params[:scale_fct] = min_max_normalize
    params[:combination_fct] = max

    @testset "initialize_qs" begin
        @testset "TestPQs" begin
            p = copy(params)
            p[:x] = 5
            sub_qs = @inferred initialize_qs(SubspaceQs{TestPQs}, model, dummy_data, p)
            @test sub_qs.single_space_strategy.x == 5
            @test sub_qs.scale_fct == min_max_normalize
            @test sub_qs.combination_fct == max
            for labels in [labelmap(fill(:U, 10)), labelmap(fill(:Lin, 10)), labelmap(fill(:Lout, 10))]
                scores = qs_score(sub_qs, dummy_data, labels)
                @test length(scores) == size(dummy_data, 2)
            end
        end

        @testset "scale-and-combine" begin
            labels = labelmap(fill(:U, 10))
            p = copy(params)
            p[:x] = 5

            sub_qs = @inferred initialize_qs(SubspaceQs{TestPQs}, model, dummy_data, p)
            scores = qs_score(sub_qs, dummy_data, labels)
            @test scores ≈ collect(range(0,1,length=10))

            p[:combination_fct] = +
            sub_qs = @inferred initialize_qs(SubspaceQs{TestPQs}, model, dummy_data, p)
            scores = qs_score(sub_qs, dummy_data, labels)
            @test scores ≈ collect(range(0,1,length=10)) .* 2

            p[:scale_fct] = identity
            sub_qs = @inferred initialize_qs(SubspaceQs{TestPQs}, model, dummy_data, p)
            scores = qs_score(sub_qs, dummy_data, labels)
            @test scores ≈ collect(range(1,10,length=10)) .* 2

            scores = qs_score(sub_qs, rand(4,5), labelmap(fill(:U, 5)))
            @test length(scores) == 5

            scores = qs_score(sub_qs, rand(4,25), labelmap(fill(:U, 25)))
            @test length(scores) == 25

            @test_throws DimensionMismatch qs_score(sub_qs, rand(3, 10), labelmap(fill(:U, 10)))
        end

        labels = labelmap(fill(:U, 10))
        for q in [RandomPQs, RandomOutlierPQs]
            @testset "$q" begin
                sub_qs = @inferred initialize_qs(SubspaceQs{q}, model, dummy_data, params)
            end
        end
        @testset "BoundaryNeighborCombinationPQs" begin
            p = copy(params)
            p[:η] = 0.8
            p[:p] = 0.2
            sub_qs = @inferred initialize_qs(SubspaceQs{BoundaryNeighborCombinationPQs{typeof(model)}}, model, dummy_data, p)
            @test_throws DimensionMismatch qs_score(sub_qs, rand(4,5), labels)
            @test sub_qs.single_space_strategy.p == 0.2
            @test sub_qs.single_space_strategy.η == 0.8
            query_data = rand(4,5)
            sub_qs = @inferred initialize_qs(SubspaceQs{BoundaryNeighborCombinationPQs{typeof(model)}}, model, query_data, p)
            scores = qs_score(sub_qs, query_data, labelmap(fill(:U, 5)))
            @test length(scores) == size(query_data, 2)
        end

        for q in [HighConfidencePQs, DecisionBoundaryPQs]
            p = copy(params)
            p[:subspaces] = [[1,2], [3,4], [1,4]]
            @testset "$q" begin
                sub_qs = @inferred initialize_qs(SubspaceQs{q}, model, dummy_data, p)
                @test_throws DimensionMismatch qs_score(sub_qs, rand(3,10), labels)
                @test_throws DimensionMismatch qs_score(sub_qs, rand(5,10), labels)
                scores = qs_score(sub_qs, rand(4, 20), labelmap(fill(:U, 20)))
                @test length(scores) == 20
                scores = qs_score(sub_qs, rand(4, 5), labelmap(fill(:U, 5)))
                @test length(scores) == 5
            end
        end

        @testset "Not implemented Error" begin
            for q in [MinimumLossPQs, MinimumMarginPQs]
                p = copy(params)
                p[:p_inlier] = 0.1
                sub_qs = initialize_qs(SubspaceQs{q}, model, dummy_data, p)
                @test_throws ErrorException qs_score(sub_qs, rand(4, 10), labelmap(fill(:U, 10)))
            end

            for q in [ExpectedMinimumMarginPQs, ExpectedMaximumEntropyPQs, NeighborhoodBasedPQs]
                sub_qs = initialize_qs(SubspaceQs{q}, model, dummy_data, params)
                @test_throws ErrorException qs_score(sub_qs, rand(4, 10), labelmap(fill(:U, 10)))
            end
        end

        @testset "process_queries!" begin
            data = rand(4,10)
            subspaces = [[1,2], [3,4]]
            pools = fill(:U, size(data,2))

            train = BitArray(vcat(false, trues(8), false))
            gamma_strategy = FixedGammaStrategy(MLKernels.GaussianKernel(2))
            C_strategy = FixedCStrategy(0.1)
            init_strategy = SVDD.SimpleSubspaceStrategy(gamma_strategy, C_strategy, gamma_scope=Val(:Global))
            update_strategy = SVDD.FixedWeightStrategy(42.0, 5.0)

            global_query_ids = [2,4]
            query_labels = [:outlier, :inlier]

            @testset "FullSplit" begin
                split_strategy = DataSplits(train, .~train, FullSplitStrat())
                train_data, train_pools, _ = get_train(split_strategy, data, pools)
                model = SVDD.SubSVDD(train_data, subspaces, train_pools)
                set_param!(model, Dict(:weight_update_strategy => update_strategy))
                initialize!(model, init_strategy)
                data_updated, pools_updated, labels_updated = OneClassActiveLearning.process_queries!(global_query_ids, query_labels, model, split_strategy, data, pools, labels)
                # FullSplit -> first train index and third train index are updated
                @test all(model.v .== [5.0, 1.0, 42.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            end

            @testset "UnlabeledAndLabeledInlierSplit" begin
                split_strategy = DataSplits(train, .~train, UnlabeledAndLabeledInlierSplitStrat())
                train_data, train_pools, _ = get_train(split_strategy, data, pools)
                model = SVDD.SubSVDD(train_data, subspaces, train_pools)
                set_param!(model, Dict(:weight_update_strategy => update_strategy))
                initialize!(model, init_strategy)

                data_updated, pools_updated, labels_updated = OneClassActiveLearning.process_queries!(global_query_ids, query_labels, model, split_strategy, data, pools, labels)
                # FullSplit -> first train observation is removed because it is labled :Lout, second train observation in updated data set is updated
                @test all(model.v .== [1.0, 42.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            end
        end
    end
end
