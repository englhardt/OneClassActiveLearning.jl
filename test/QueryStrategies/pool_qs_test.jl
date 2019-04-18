
@testset "QueryStrategies" begin

    dummy_data, _ = load_data(TEST_DATA_FILE)

    @testset "multi kde" begin
        data = hcat(fill([1, 2, 3, 4,], 3)...)
        @test_throws KDEException multi_kde(data)
        data2 = hcat(data, hcat(fill([1, 2, 3, 4,], 2)...))
        @test_throws KDEException multi_kde(data2)
        data3 = hcat(data2, rand(4,20) * 4)
        @test isa(OneClassActiveLearning.multi_kde(data3), PyObject)
    end

    params = Dict{Symbol, Any}()
    @testset "initialize_qs" begin
        @testset "Standard" begin
            qs_types = [RandomPQs]
            qs_objs = map(x -> initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, params), qs_types)
            for qs in qs_objs
                for labels in [labelmap(fill(:U, 10)), labelmap(fill(:Lin, 10)), labelmap(fill(:Lout, 10))]
                    scores = qs_score(qs, dummy_data, labels)
                    @test length(scores) == size(dummy_data, 2)
                end
            end
        end

        qs_types = [MinimumMarginPQs, MinimumLossPQs]
        for x in qs_types
            @test_throws ArgumentError initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, Dict{Symbol, Any}())
            @test_throws ArgumentError initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, Dict(:p_inlier => nothing))
            @test_throws ArgumentError initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, Dict(:p_inlier => 1.5))
            @test_throws ArgumentError initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, Dict(:p_inlier => -1))
        end
        qs_objs = map(x -> initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, Dict(:p_inlier => 0.05)), qs_types)
        for qs in qs_objs
            @testset "DataBasedPQs $(typeof(qs))" begin
                @testset "only :U" begin
                        @test isdefined(qs, :bw_method)
                        @test_throws OneClassActiveLearning.MissingLabelTypeException qs_score(qs, dummy_data, labelmap(fill(:U, 10)))
                end

                @testset "only :Lin" begin
                    # number of :Lin must be sufficient that leave-one-out does not result in singular covariance matrix
                    scores = qs_score(qs, dummy_data, labelmap(fill(:Lin, 15)))
                    @test length(scores) == size(dummy_data, 2)
                end
                @testset "only :Lout" begin
                    @test_throws OneClassActiveLearning.MissingLabelTypeException qs_score(qs, dummy_data, labelmap(fill(:Lout, 10)))
                end
            end
        end
        @testset "DataBasedPQs GaussianKernel" begin
            classifier = SVDD.VanillaSVDD(dummy_data)
            init_strategy = SVDD.SimpleCombinedStrategy(FixedGammaStrategy(MLKernels.GaussianKernel(0.5)), FixedCStrategy(1))
            SVDD.initialize!(classifier, init_strategy)
            qs = initialize_qs(MinimumMarginPQs, classifier, dummy_data, Dict(:p_inlier => 0.05))
            scores = qs_score(qs, dummy_data, labelmap(fill(:Lin, 15)))
            @test length(scores) == size(dummy_data, 2)
        end

        qs_types = [ExpectedMinimumMarginPQs, ExpectedMaximumEntropyPQs]
        qs_objs = map(x -> initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, params), qs_types)
        for qs in qs_objs
            @testset "DataBasedPQs $(typeof(qs))" begin
                @testset "only :U" begin
                        @test isdefined(qs, :bw_method)
                        @test_throws OneClassActiveLearning.MissingLabelTypeException qs_score(qs, dummy_data, labelmap(fill(:U, 10)))
                end

                @testset "only :Lin" begin
                    # number of :Lin must be sufficient that leave-one-out does not result in singular covariance matrix
                    scores = qs_score(qs, dummy_data, labelmap(fill(:Lin, 15)))
                    @test length(scores) == size(dummy_data, 2)
                end
                @testset "only :Lout" begin
                    @test_throws OneClassActiveLearning.MissingLabelTypeException qs_score(qs, dummy_data, labelmap(fill(:Lout, 10)))
                end
            end
        end

        qs_types = [RandomOutlierPQs, HighConfidencePQs, DecisionBoundaryPQs]
        qs_objs = map(x -> initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, params), qs_types)
        for qs in qs_objs
            @testset "ModelBasedPQs $(typeof(qs))" begin
                scores = qs_score(qs, dummy_data, labelmap(fill(:Lin, 10)))
                @test length(scores) == size(dummy_data, 2)
            end
        end

        qs_types = [NeighborhoodBasedPQs, BoundaryNeighborCombinationPQs{SVDD.RandomOCClassifier}]
        qs_objs = map(x -> initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, params), qs_types)
        for qs in qs_objs
            @testset "HybridPQs $(typeof(qs))" begin
                scores = qs_score(qs, dummy_data, labelmap(fill(:U, 10)))
                @test length(scores) == size(dummy_data, 2)
            end
        end

        @testset "Unknown Strategy" begin
            @test_throws ErrorException qs = initialize_qs(Vector{Int}, SVDD.RandomOCClassifier(dummy_data), dummy_data, params)

        @testset "multiple classifiers" begin

            pools = fill(:U, size(dummy_data, 2))
            init_strategy = SVDD.FixedParameterInitialization(MLKernels.GaussianKernel(0.5), 0.5)

            models = []
            model = SVDD.SVDDneg(dummy_data, pools)
            SVDD.initialize!(model, init_strategy)
            SVDD.fit!(model, TEST_SOLVER)
            push!(models, model)

            model = SVDD.VanillaSVDD(dummy_data)
            SVDD.initialize!(model, init_strategy)
            SVDD.fit!(model, TEST_SOLVER)
            push!(models, model)

            model = SVDD.SSAD(dummy_data, pools)
            SVDD.initialize!(model, init_strategy)
            SVDD.fit!(model, TEST_SOLVER)
            push!(models, model)

            model = SVDD.RandomOCClassifier(dummy_data)
            SVDD.initialize!(model, init_strategy)
            SVDD.fit!(model, TEST_SOLVER)
            push!(models, model)

            qs_types = [MinimumMarginPQs, MinimumLossPQs]
            for qst in qs_types
                for m in models
                    @testset "initialize_qs $qst, $(typeof(m))" begin
                        # check if qs is instantiated
                        qs = OneClassActiveLearning.initialize_qs(qst, model, dummy_data, Dict(:p_inlier => 0.05))
                    end
                end
            end

            qs_types = [ExpectedMinimumMarginPQs, ExpectedMaximumEntropyPQs] ∪ subtypes(ModelBasedPQs) ∪ [NeighborhoodBasedPQs, BoundaryNeighborCombinationPQs{SVDD.RandomOCClassifier}]
            for qst in qs_types
                for m in models
                    @testset "initialize_qs $qst, $(typeof(m))" begin
                        # check if qs is instantiated
                        qs = OneClassActiveLearning.initialize_qs(qst, model, dummy_data, params)
                    end
                end
            end
        end
    end

    @testset "get_query_object" begin
        qs = OneClassActiveLearning.TestPQs()
        @testset "a" begin
            data = rand(2, 6)
            pools = fill(:U, 6)
            pools[5] = :Lin
            indices = [1, 3, 5, 7, 9, 11]
            history = [7]
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
            @test scores = OneClassActiveLearning.qs_score(qs, data, MLLabelUtils.labelmap(pools)) == collect(1:5)
            @test_throws ArgumentError OneClassActiveLearning.get_query_object(qs, data, fill(:Lin, 5), indices, history)
            @test OneClassActiveLearning.get_query_object(qs, data, pools, indices, history) == 4
        end
    end

    @testset "HybridQuerySynthesisPQs" begin
        params = Dict{Symbol, Any}(:qss_type => :RandomQss)
        qs = OneClassActiveLearning.initialize_qs(HybridQuerySynthesisPQs, SVDD.RandomOCClassifier(dummy_data), dummy_data, params)
        data = rand(2, 5)
        pools = fill(:U, 5)
        pools[5] = :Lin
        indices = [1, 2, 4, 7, 9]
        history = [7]
        @test_throws ArgumentError OneClassActiveLearning.get_query_object(qs, data, fill(:Lin, 5), indices, history)
        query_index = OneClassActiveLearning.get_query_object(qs, data, pools, indices, history)
        @test query_index ∈ indices
        @test query_index ∉ history
    end
end
