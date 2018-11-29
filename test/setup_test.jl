
@testset "setup" begin
    @testset "simple random example" begin
        experiment = Dict{Symbol, Any}(
            :hash => 0,
            :data_file => TEST_DATA_FILE,
            :output_file => "OneClassActiveLearning.jl/data/output/scenarioA/data_qs_model_id.tmp",
            :model => Dict(:type => :(SVDD.RandomOCClassifier),
                           :param => Dict{Symbol, Any}(),
                           :init_strategy => SVDD.FixedParameterInitialization(GaussianKernel(2), 0.5)),
            :split_strategy => OneClassActiveLearning.DataSplits(trues(TEST_DATA_NUM_OBSERVATIONS) , OneClassActiveLearning.FullSplitStrat()),
            :param => Dict(:num_al_iterations => 5,
                           :solver => Dict(:type => TEST_SOLVER.constructor,
                                           :flags => Dict(TEST_SOLVER.kwargs)),
                           :initial_pools => fill(:U, TEST_DATA_NUM_OBSERVATIONS),
                           :adjust_K => true))

        @testset "pool based" begin
            exp =  deepcopy(experiment)
            exp[:query_strategy] = Dict(:type => :(OneClassActiveLearning.QueryStrategies.RandomPQs),
                                        :param => Dict{Symbol, Any}())
            exp[:oracle] = :PoolOracle

            expected_experiment = deepcopy(exp)

            res = OneClassActiveLearning.active_learn(exp)

            @test res.status[:exit_code] == :success
            @test length(res.al_history, :query_history) == 5
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
            exp[:oracle] = OneClassActiveLearning.QuerySynthesisFunctionOracle(_ -> :inlier)

            expected_experiment = deepcopy(exp)

            res = OneClassActiveLearning.active_learn(exp)

            @test length(res.experiment[:split_strategy].train) == TEST_DATA_NUM_OBSERVATIONS + 5
            @test length(res.experiment[:split_strategy].test) == TEST_DATA_NUM_OBSERVATIONS + 5

            @test res.status[:exit_code] == :success
            @test length(res.al_history, :query_history) == 5
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

    @testset "update data and pools" begin
        @testset "pool based" begin
            qs = OneClassActiveLearning.QueryStrategies.RandomPQs()
            data = rand(2, 3)
            split_strategy = OneClassActiveLearning.DataSplits(trues(3))
            pools = [:U, :U, :Lin]
            labels = [:inlier, :outlier, :inlier]
            OneClassActiveLearning.update_data_and_pools!(qs, data, labels, pools, split_strategy, 1, :inlier)
            @test pools == [:Lin, :U, :Lin]
            OneClassActiveLearning.update_data_and_pools!(qs, data, labels, pools, split_strategy, 2, :outlier)
            @test pools == [:Lin, :Lout, :Lin]
        end

        @testset "query synthesis" begin
            qs = OneClassActiveLearning.QueryStrategies.RandomQss()
            data = rand(2, 3)
            split_strategy = OneClassActiveLearning.DataSplits(trues(3))
            pools = [:U, :U, :Lin]
            labels = [:inlier, :outlier, :inlier]
            data = OneClassActiveLearning.update_data_and_pools!(qs, data, labels, pools, split_strategy, rand(2, 1), :inlier)
            @test size(data) == (2, 4)
            @test labels == [:inlier, :outlier, :inlier, :inlier]
            @test pools == [:U, :U, :Lin, :Lin]
            @test split_strategy.train == [true, true, true, true]
            @test split_strategy.test == [true, true, true, false]
            data = OneClassActiveLearning.update_data_and_pools!(qs, data, labels, pools, split_strategy, rand(2, 1), :outlier)
            @test size(data) == (2, 5)
            @test labels == [:inlier, :outlier, :inlier, :inlier, :outlier]
            @test pools == [:U, :U, :Lin, :Lin, :Lout]
            @test split_strategy.train == [true, true, true, true, true]
            @test split_strategy.test == [true, true, true, false, false]
        end
    end
end
