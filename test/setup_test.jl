
@testset "setup" begin
    @testset "simple random example" begin
        experiment = Dict{Symbol, Any}(
            :hash => 0,
            :data_file => TEST_DATA_FILE,
            :output_file => TEST_OUTPUT_FILE,
            :model => Dict(:type => :(SVDD.RandomOCClassifier),
                           :param => Dict{Symbol, Any}(),
                           :init_strategy => SVDD.FixedParameterInitialization(GaussianKernel(2), 0.5)),
            :split_strategy => OneClassActiveLearning.DataSplits(trues(TEST_DATA_NUM_OBSERVATIONS) , OneClassActiveLearning.FullSplitStrat()),
            :param => Dict(:num_al_iterations => 5,
                           :solver => TEST_SOLVER,
                           :initial_pools => fill(:U, TEST_DATA_NUM_OBSERVATIONS),
                           :adjust_K => true))

        @testset "pool based single query" begin
            exp =  deepcopy(experiment)
            exp[:query_strategy] = Dict(:type => :(OneClassActiveLearning.QueryStrategies.RandomPQs),
                                        :param => Dict{Symbol, Any}())
            exp[:oracle] = Dict(:type => :PoolOracle,
                                :param => Dict{Symbol, Any}())

            expected_experiment = deepcopy(exp)

            res = OneClassActiveLearning.active_learn(exp)

            @test res.status[:exit_code] == :success
            @test length(res.al_history, :query_history) == 5
            @test length(res.al_history, :angle_batch_diversity) == 5
            @test length(res.al_history, :euclidean_batch_diversity) == 5
            @test all(values(res.al_history, :angle_batch_diversity) .≈ 0.0)
            @test all(values(res.al_history, :euclidean_batch_diversity) .≈ 0.0)
            @test length(res.al_history, :time_qs) == 5
            @test all(values(res.al_history, :time_qs) .> 0.0)

            @test length(res.al_history, :time_fit) == 6
            @test all(values(res.al_history, :time_fit) .> 0.0)

            @test length(res.al_history, :query_history) == 5
            @test !isempty(res.worker_info)

            @test length(res.al_history, :cm) == 6
            @test OneClassActiveLearning.cohens_kappa(last(res.al_history[:cm])[2]) ≈ last(res.al_history[:cohens_kappa])[2]

            @test res.experiment[:param][:initial_pools] == expected_experiment[:param][:initial_pools]
        end

        @testset "pool based batch query" begin
            exp =  deepcopy(experiment)
            exp[:query_strategy] = Dict(:type => :(OneClassActiveLearning.QueryStrategies.RandomBatchQs),
                                        :param => Dict{Symbol, Any}(:k => 5))
            exp[:oracle] = Dict(:type => :PoolOracle,
                                :param => Dict{Symbol, Any}())

            expected_experiment = deepcopy(exp)

            res = OneClassActiveLearning.active_learn(exp)

            @test res.status[:exit_code] == :success
            @test length(res.al_history, :query_history) == 5
            @test all(length.(values(res.al_history, :query_history)) .== 5)
            @test length(res.al_history, :angle_batch_diversity) == 5
            @test length(res.al_history, :euclidean_batch_diversity) == 5
            @test all(.~(values(res.al_history, :euclidean_batch_diversity) .≈ 0.0))
            @test length(res.al_history, :time_qs) == 5
            @test all(values(res.al_history, :time_qs) .> 0.0)

            @test length(res.al_history, :time_fit) == 6
            @test all(values(res.al_history, :time_fit) .> 0.0)

            @test length(res.al_history, :query_history) == 5
            @test !isempty(res.worker_info)

            @test length(res.al_history, :cm) == 6
            @test OneClassActiveLearning.cohens_kappa(last(res.al_history[:cm])[2]) ≈ last(res.al_history[:cohens_kappa])[2]

            @test res.experiment[:param][:initial_pools] == expected_experiment[:param][:initial_pools]
        end

        @testset "query synthesis" begin
            exp =  deepcopy(experiment)
            exp[:query_strategy] = Dict(:type => :(OneClassActiveLearning.QueryStrategies.RandomQss),
                                        :param => Dict{Symbol, Any}())
            exp[:oracle] = OneClassActiveLearning.QuerySynthesisFunctionOracle(x -> fill(:inlier, size(x, 2)))

            expected_experiment = deepcopy(exp)

            res = OneClassActiveLearning.active_learn(exp)

            @test length(res.experiment[:split_strategy].train) == TEST_DATA_NUM_OBSERVATIONS + 5
            @test length(res.experiment[:split_strategy].test) == TEST_DATA_NUM_OBSERVATIONS + 5

            @test res.status[:exit_code] == :success
            @test length(res.al_history, :query_history) == 5
            @test length(res.al_history, :angle_batch_diversity) == 5
            @test length(res.al_history, :euclidean_batch_diversity) == 5
            @test all(values(res.al_history, :angle_batch_diversity) .≈ 0.0)
            @test all(values(res.al_history, :euclidean_batch_diversity) .≈ 0.0)
            @test length(res.al_history, :time_qs) == 5
            @test all(values(res.al_history, :time_qs) .> 0.0)

            @test length(res.al_history, :time_fit) == 6
            @test all(values(res.al_history, :time_fit) .> 0.0)

            @test length(values(res.al_history, :query_history)) == 5
            @test size(values(res.al_history, :query_history)[1]) == (TEST_DATA_NUM_DIMENSIONS, 1)
            @test !isempty(res.worker_info)

            @test length(res.al_history, :cm) == 6
            @test OneClassActiveLearning.cohens_kappa(last(res.al_history[:cm])[2]) ≈ last(res.al_history[:cohens_kappa])[2]

            @test res.experiment[:param][:initial_pools] == expected_experiment[:param][:initial_pools]
        end
    end

    @testset "process queries" begin
        @testset "pool based" begin
            qs = OneClassActiveLearning.QueryStrategies.RandomPQs()
            data = rand(2, 4)
            split_strategy = OneClassActiveLearning.DataSplits(trues(4))
            pools = [:U, :U, :Lin, :U]
            labels = [:inlier, :outlier, :inlier, :inlier]
            data, pools, labels = OneClassActiveLearning.process_queries!([1], labels[[1]], SVDD.RandomOCClassifier(data), split_strategy, data, pools, labels)
            @test pools == [:Lin, :U, :Lin, :U]
            data, pools, labels = OneClassActiveLearning.process_queries!([2,4], labels[[2,4]], SVDD.RandomOCClassifier(data), split_strategy, data, pools, labels)
            @test pools == [:Lin, :Lout, :Lin, :Lin]
            @test labels == [:inlier, :outlier, :inlier, :inlier]
        end

        @testset "query synthesis" begin
            qs = OneClassActiveLearning.QueryStrategies.RandomQss()
            data = rand(2, 3)
            split_strategy = OneClassActiveLearning.DataSplits(trues(3))
            pools = [:U, :U, :Lin]
            labels = [:inlier, :outlier, :inlier]
            data, pools, labels = OneClassActiveLearning.process_queries!(rand(2,1), [:inlier], SVDD.RandomOCClassifier(data), split_strategy, data, pools, labels)
            @test size(data) == (2, 4)
            @test labels == [:inlier, :outlier, :inlier, :inlier]
            @test pools == [:U, :U, :Lin, :Lin]
            @test split_strategy.train == [true, true, true, true]
            @test split_strategy.test == [true, true, true, false]
            data, pools, labels = OneClassActiveLearning.process_queries!(rand(2,3), fill(:outlier, 3), SVDD.RandomOCClassifier(data), split_strategy, data, pools, labels)
            @test size(data) == (2, 7)
            @test labels == [:inlier, :outlier, :inlier, :inlier, :outlier, :outlier, :outlier]
            @test pools == [:U, :U, :Lin, :Lin, :Lout, :Lout, :Lout]
            @test split_strategy.train == [true, true, true, true, true, true, true]
            @test split_strategy.test == [true, true, true, false, false, false, false]
        end
    end
end
