
@testset "oracle" begin
    data, labels = OneClassActiveLearning.load_data(TEST_DATA_FILE)
    init_strategy = SimpleCombinedStrategy(FixedGammaStrategy(GaussianKernel(2.0)), FixedCStrategy(0.5))

    @testset "initialize" begin
        @test_throws ErrorException OneClassActiveLearning.initialize_oracle(OneClassActiveLearning, data, labels)
        @test_throws ArgumentError OneClassActiveLearning.initialize_oracle(QuerySynthesisOCCOracle, data, labels)
    end

    @testset "PoolOracle" begin
        oracle = OneClassActiveLearning.initialize_oracle(PoolOracle, data, labels)
        @test isa(oracle, PoolOracle)
        for i in 1:3
            @test ask_oracle(oracle, i) == labels[i]
        end
    end

    @testset "QuerySynthesisFunctionOracle" begin
        oracle = QuerySynthesisFunctionOracle(_ -> :inlier)
        @test ask_oracle(oracle, 1) == :inlier
    end

    @testset "QuerySynthesisOCCOracle" begin
        oracle = OneClassActiveLearning.initialize_oracle(QuerySynthesisOCCOracle, data, labels, Dict{Symbol, Any}(
            :classifier_type => SVDD.RandomOCClassifier,
            :init_strategy => init_strategy,
            :solver => TEST_SOLVER
        ))
        @test ask_oracle(oracle, rand(TEST_DATA_NUM_DIMENSIONS, 1)) ∈ [:inlier, :outlier]
    end

    @testset "QuerySynthesisSVMOracle" begin
        oracle = OneClassActiveLearning.initialize_oracle(QuerySynthesisSVMOracle, data, labels, Dict{Symbol, Any}(
            :init_strategy => init_strategy,
        ))
        @test ask_oracle(oracle, rand(TEST_DATA_NUM_DIMENSIONS, 1)) ∈ [:inlier, :outlier]
        oracle = OneClassActiveLearning.initialize_oracle(QuerySynthesisSVMOracle, data, labels, Dict{Symbol, Any}(
            :gamma_search_range_oracle => [0.5, 1.0],
        ))
        @test ask_oracle(oracle, rand(TEST_DATA_NUM_DIMENSIONS, 1)) ∈ [:inlier, :outlier]
    end
end
