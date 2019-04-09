
@testset "oracle" begin
    data, labels = OneClassActiveLearning.load_data(TEST_DATA_FILE)
    init_strategy = SimpleCombinedStrategy(FixedGammaStrategy(GaussianKernel(2.0)), FixedCStrategy(0.5))

    @testset "initialize" begin
        @test_throws ErrorException OneClassActiveLearning.initialize_oracle(OneClassActiveLearning, data, labels)
        @test_throws ArgumentError OneClassActiveLearning.initialize_oracle(QuerySynthesisOCCOracle, data, labels)
        @test_throws ArgumentError OneClassActiveLearning.initialize_oracle(QuerySynthesisGMMOracle, data, labels)
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

    @testset "QuerySynthesisKNNOracle" begin
        @test_throws ArgumentError OneClassActiveLearning.initialize_oracle(QuerySynthesisKNNOracle, data, labels, Dict{Symbol, Any}(:k => 2))
        oracle = OneClassActiveLearning.initialize_oracle(QuerySynthesisKNNOracle, data, labels)
        @test ask_oracle(oracle, data[:, 1:1]) == labels[1]
        @test ask_oracle(oracle, rand(TEST_DATA_NUM_DIMENSIONS, 1)) ∈ [:inlier, :outlier]
    end

    @testset "QuerySynthesisGMMOracle" begin
        gmm = rand(GMM, 1, 2)
        oracle = QuerySynthesisGMMOracle(gmm, 0.1)
        @test ask_oracle(oracle, rand(2, 1)) ∈ [:inlier, :outlier]
        f = open(TEST_OUTPUT_FILE, "w")
        serialize(f, gmm)
        close(f)
        oracle_param = Dict{Symbol, Any}(:file => TEST_OUTPUT_FILE)
        @test_throws ErrorException OneClassActiveLearning.initialize_oracle(QuerySynthesisGMMOracle, data, labels, oracle_param)
        f = open(TEST_OUTPUT_FILE, "w")
        serialize(f, oracle)
        close(f)
        deserialized_oracle = OneClassActiveLearning.initialize_oracle(QuerySynthesisGMMOracle, data, labels, oracle_param)
        @test ask_oracle(deserialized_oracle, rand(2, 1)) ∈ [:inlier, :outlier]
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

    @testset "QuerySynthesisCVWrapperOracle" begin
        @test_throws ArgumentError OneClassActiveLearning.initialize_oracle(QuerySynthesisCVWrapperOracle, data, labels)
        @test_throws ArgumentError OneClassActiveLearning.initialize_oracle(QuerySynthesisCVWrapperOracle, data, labels, Dict{Symbol, Any}(
            :subtype => QuerySynthesisCVWrapperOracle,
        ))
        oracle = OneClassActiveLearning.Oracles.initialize_oracle(QuerySynthesisCVWrapperOracle, data, labels, Dict{Symbol, Any}(
            :subtype => QuerySynthesisSVMOracle,
            :gamma_search_range_oracle => [0.1, 1],
            :num_folds => 2,
        ))
        @test ask_oracle(oracle, rand(TEST_DATA_NUM_DIMENSIONS, 1)) ∈ [:inlier, :outlier]
    end
end
