
@testset "setup" begin

    @testset "simple random example" begin
        data_file = "$(@__DIR__)/../example/dummy.csv"
        number_observations = countlines(open(data_file))

        experiment = Dict{Symbol, Any}(
            :hash => 0,
            :data_file => data_file,
            :output_file => "OneClassActiveLearning.jl/data/output/scenarioA/data_qs_model_id.tmp",
            :model => Dict(:type => :(SVDD.RandomOCClassifier),
                           :param => Dict{Symbol, Any}(),
                           :init_strategy => SVDD.FixedParameterInitialization(GaussianKernel(2), 0.5)),
            :query_strategy => Dict(:type => :(OneClassActiveLearning.QueryStrategies.RandomPQs), :param => Dict{Symbol, Any}()),
            :split_strategy => OneClassActiveLearning.DataSplits(trues(number_observations) , OneClassActiveLearning.FullSplitStrat()),
            :oracle => :PoolOracle,
            :param => Dict(:num_al_iterations => 5,
                           :solver => TEST_SOLVER,
                           :initial_pools => fill(:U, number_observations),
                           :adjust_K => true))
        expected_experiment = deepcopy(experiment)

        res = OneClassActiveLearning.active_learn(experiment)

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

    @testset "update pools" begin
        pools = [:U, :U, :Lin]
        labels = [:inlier, :outlier, :inlier]
        OneClassActiveLearning.update_pools!(pools, 1, :inlier)
        @test pools == [:Lin, :U, :Lin]
        OneClassActiveLearning.update_pools!(pools, 2, :outlier)
        @test pools == [:Lin, :Lout, :Lin]
    end
end
