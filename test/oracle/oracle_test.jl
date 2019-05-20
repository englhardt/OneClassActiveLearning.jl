
@testset "oracle" begin
    data, labels = OneClassActiveLearning.load_data(TEST_DATA_FILE)
    init_strategy = SimpleCombinedStrategy(FixedGammaStrategy(GaussianKernel(2.0)), FixedCStrategy(0.5))
    expected_feedback = Set(OneClassActiveLearning.LABEL_ENCODING.label)

    @testset "initialize" begin
        @test_throws ErrorException OneClassActiveLearning.initialize_oracle(OneClassActiveLearning, data, labels)
        @test_throws ArgumentError OneClassActiveLearning.initialize_oracle(QuerySynthesisOCCOracle, data, labels)
        @test_throws ArgumentError OneClassActiveLearning.initialize_oracle(QuerySynthesisGMMOracle, data, labels)
    end

    @testset "PoolOracle" begin
        oracle = OneClassActiveLearning.initialize_oracle(PoolOracle, data, labels)
        @test isa(oracle, PoolOracle)
        @test ask_oracle(oracle, Array(1:3)) == labels[1:3]
    end

    @testset "QuerySynthesisFunctionOracle" begin
        oracle = QuerySynthesisFunctionOracle(x -> fill(:inlier, size(x, 2)))
        @test ask_oracle(oracle, ones(2, 1)) == [:inlier]
        feedback = ask_oracle(oracle, rand(TEST_DATA_NUM_DIMENSIONS, 3))
        @test length(feedback) == 3
        @test issubset(Set(feedback), expected_feedback)
    end

    @testset "QuerySynthesisKNNOracle" begin
        @test_throws ArgumentError OneClassActiveLearning.initialize_oracle(QuerySynthesisKNNOracle, data, labels, Dict{Symbol, Any}(:k => 2))
        oracle = OneClassActiveLearning.initialize_oracle(QuerySynthesisKNNOracle, data, labels)
        @test ask_oracle(oracle, data[:, 1:1]) == [labels[1]]
        feedback = ask_oracle(oracle, rand(TEST_DATA_NUM_DIMENSIONS, 3))
        @test length(feedback) == 3
        @test issubset(Set(feedback), expected_feedback)
    end

    @testset "QuerySynthesisGMMOracle" begin
        gmm = rand(GMM, 1, 2)
        oracle = QuerySynthesisGMMOracle(gmm, 0.1)
        feedback = ask_oracle(oracle, rand(2, 1))
        @test length(feedback) == 1
        @test issubset(Set(feedback), expected_feedback)
        f = open(TEST_OUTPUT_FILE, "w")
        serialize(f, gmm)
        close(f)
        oracle_param = Dict{Symbol, Any}(:file => TEST_OUTPUT_FILE)
        @test_throws ErrorException OneClassActiveLearning.initialize_oracle(QuerySynthesisGMMOracle, data, labels, oracle_param)
        f = open(TEST_OUTPUT_FILE, "w")
        serialize(f, oracle)
        close(f)
        deserialized_oracle = OneClassActiveLearning.initialize_oracle(QuerySynthesisGMMOracle, data, labels, oracle_param)
        feedback = ask_oracle(deserialized_oracle, rand(2, 3))
        @test length(feedback) == 3
        @test issubset(Set(feedback), expected_feedback)
    end

    @testset "QuerySynthesisOCCOracle" begin
        oracle = OneClassActiveLearning.initialize_oracle(QuerySynthesisOCCOracle, data, labels, Dict{Symbol, Any}(
            :classifier_type => SVDD.RandomOCClassifier,
            :init_strategy => init_strategy,
            :solver => TEST_SOLVER
        ))
        feedback = ask_oracle(oracle, rand(TEST_DATA_NUM_DIMENSIONS, 3))
        @test length(feedback) == 3
        @test issubset(Set(feedback), expected_feedback)
    end

    @testset "QuerySynthesisSVMOracle" begin
        oracle = OneClassActiveLearning.initialize_oracle(QuerySynthesisSVMOracle, data, labels, Dict{Symbol, Any}(
            :init_strategy => init_strategy,
        ))
        feedback = ask_oracle(oracle, rand(TEST_DATA_NUM_DIMENSIONS, 1))
        @test length(feedback) == 1
        @test issubset(Set(feedback), expected_feedback)
        oracle = OneClassActiveLearning.initialize_oracle(QuerySynthesisSVMOracle, data, labels, Dict{Symbol, Any}(
            :gamma_search_range_oracle => [0.5, 1.0],
        ))
        feedback = ask_oracle(oracle, rand(TEST_DATA_NUM_DIMENSIONS, 3))
        @test length(feedback) == 3
        @test issubset(Set(feedback), expected_feedback)
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
        feedback = ask_oracle(oracle, rand(TEST_DATA_NUM_DIMENSIONS, 3))
        @test length(feedback) == 3
        @test issubset(Set(feedback), expected_feedback)
    end
end
