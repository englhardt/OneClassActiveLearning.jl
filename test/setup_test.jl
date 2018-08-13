
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
            :query_strategy => Dict(:type => :(OneClassActiveLearning.QueryStrategies.RandomQs), :param => Dict{Symbol, Any}()),
            :split_strategy => OneClassActiveLearning.DataSplits(trues(number_observations) , OneClassActiveLearning.FullSplitStrat()),
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
        OneClassActiveLearning.update_pools!(pools, 1, labels)
        @test pools == [:Lin, :U, :Lin]
        OneClassActiveLearning.update_pools!(pools, 2, labels)
        @test pools == [:Lin, :Lout, :Lin]
    end

    @testset "get_query_object" begin
        qs = OneClassActiveLearning.TestQs()
        @testset "a" begin
            data = rand(2, 6)
            pools = fill(:U, 6)
            pools[5] = :Lin
            indices = [1, 3, 5, 7, 9, 11]
            history = [7]
            pool_map = MLLabelUtils.labelmap(pools)
            @test scores = OneClassActiveLearning.qs_score(qs, data, MLLabelUtils.labelmap(pools)) == collect(1:6)
            @test_throws ArgumentError OneClassActiveLearning.get_query_object(qs, data, fill(:Lin, 5), indices, history)
            @test OneClassActiveLearning.get_query_object(qs, data, pools, indices, history) == 11
        end

        @testset "b" begin
            data = rand(2, 5)
            pools = fill(:U, 5)
            pools[5] = :Lin
            indices = [1, 2, 4, 7, 9]
            history = [7]
            pool_map = MLLabelUtils.labelmap(pools)
            @test scores = OneClassActiveLearning.qs_score(qs, data, MLLabelUtils.labelmap(pools)) == collect(1:5)
            @test_throws ArgumentError OneClassActiveLearning.get_query_object(qs, data, fill(:Lin, 5), indices, history)
            @test OneClassActiveLearning.get_query_object(qs, data, pools, indices, history) == 4
        end
    end
end
