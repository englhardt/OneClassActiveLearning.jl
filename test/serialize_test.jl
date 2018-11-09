
@testset "serialize" begin

    @testset "MVHistory" begin
        expected = MVHistory()
        push!(expected, :a, 1, 1.2)
        push!(expected, :a, 2, 2.2)
        push!(expected, :b, 1, 3)
        push!(expected, :b, 2, 4)
        push!(expected, :c, 1, OneClassActiveLearning.ConfusionMatrix(1,1,1,1))
        push!(expected, :c, 2, OneClassActiveLearning.ConfusionMatrix(2,2,2,2))

        actual = Unmarshal.unmarshal(MVHistory, JSON.parse(JSON.json(expected)))

        for k in keys(expected)
            @test expected[k] == actual[k]
        end
    end

    @testset "al summary" begin
        expected = Dict(:m1 => Dict(:a => 1.0, :b => 1, :c => [1.0, 2.0], :d => [1, 2]),
                           :m2 => Dict(:a => 1.0, :b => 1, :c => [1.0], :d => [1, 2]))

        actual = OneClassActiveLearning.unmarshal_al_summary(JSON.parse(JSON.json(expected)))

        for (k_metrics, v_metrics) in expected
            for (k_summary, v_summary) in v_metrics
                @test expected[k_metrics][k_summary] == actual[k_metrics][k_summary]
            end
        end
    end

    @testset "Result" begin
        id = 42
        experiment = Dict{Symbol, Any}(
                :hash => id,
                :data_file => "$(@__DIR__)/../data/input/dummy.csv",
                :output_file => "OneClassActiveLearning.jl/data/output/scenarioA/data_qs_model_id.tmp",
                :model => Dict( :type => :(SVDD.RandomOCClassifier),
                                :init_strategy => :(SVDD.FixedParameterInitialization(GaussianKernel(2), 0.5))),
                :query_strategy => Dict(:type => :(OneClassActiveLearning.QueryStrategies.RandomPQs), :param => Dict()),
                :split_strategy => OneClassActiveLearning.DataSplits(trues(123), OneClassActiveLearning.FullSplitStrat()),
                :param => Dict(:num_al_iterations => 5,
                               :solver => :(IpoptSolver),
                               :initial_pools => fill(:U, 123)))

       al_history = MVHistory()
       push!(al_history, :a, 1, 1.2)
       push!(al_history, :a, 2, 2.2)
       push!(al_history, :b, 1, 3)
       push!(al_history, :b, 2, 4)
       push!(al_history, :query_label, 1, true)
       push!(al_history, :query_label, 2, false)
       push!(al_history, :c, 1, OneClassActiveLearning.ConfusionMatrix(1,1,1,1))
       push!(al_history, :c, 2, OneClassActiveLearning.ConfusionMatrix(2,2,2,2))

        al_summary = Dict(:m1 => Dict(:a => 1.0, :b => 1, :c => [1.0, 2.0], :d => [1, 2]),
                   :m2 => Dict(:a => 1.0, :b => 1, :c => [1.0, 2.0], :d => [1, 2]))

        worker_info = OneClassActiveLearning.get_worker_info()
        expected = OneClassActiveLearning.Result(id, experiment, worker_info, DataStats(1, 1, 0.1, 0.1, [1,2,3,4], [5,6,7]), al_history, al_summary)

        actual = Unmarshal.unmarshal(OneClassActiveLearning.Result, JSON.parse(JSON.json(expected)))
        @test actual.id == 42
        @test length(actual.experiment[:param][:initial_pools]) == 123
        @test haskey(actual.worker_info, :hostname)
        @test haskey(actual.al_history, :c)
        @test haskey(actual.al_summary, :m1)
        @test haskey(actual.status, :exit_code)
    end
end
