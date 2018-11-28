
@testset "oracle" begin
    data_file = "$(@__DIR__)/../example/dummy.csv"
    data, labels = OneClassActiveLearning.load_data(data_file)

    @testset "initialize" begin
        @test_throws ErrorException OneClassActiveLearning.initialize_oracle(OneClassActiveLearning, labels)
    end

    @testset "PoolOracle" begin
        oracle = OneClassActiveLearning.initialize_oracle(PoolOracle, labels)
        @test isa(oracle, PoolOracle)
        for i in 1:3
            @test ask_oracle(oracle, i) == labels[i]
        end
    end

    @testset "QuerySynthesisFunctionOracle" begin
        oracle = QuerySynthesisFunctionOracle(_ -> :inlier)
        @test ask_oracle(oracle, 1) == :inlier
    end

    init_strategy = SimpleCombinedStrategy(FixedGammaStrategy(GaussianKernel(2.0)), FixedCStrategy(0.5))
    @testset "QuerySynthesisFunctionOracle" begin
        oracle = QuerySynthesisOCCOracle(SVDD.RandomOCClassifier, init_strategy, data_file, TEST_SOLVER)
        @test ask_oracle(oracle, rand(TEST_DATA_NUM_DIMENSIONS, 1)) ∈ [:inlier, :outlier]
    end

    @testset "QuerySynthesisSVMOracle" begin
        oracle = QuerySynthesisSVMOracle(init_strategy, data_file)
        @test ask_oracle(oracle, rand(TEST_DATA_NUM_DIMENSIONS, 1)) ∈ [:inlier, :outlier]
    end
end
