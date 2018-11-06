
@testset "QueryStrategies" begin

    dummy_data, _ = load_data("$(@__DIR__)/../../example/dummy.csv")

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
            qs_types = [RandomQs]
            qs_objs = map(x -> initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data), qs_types, params)
            for qs in qs_objs
                for labels in [labelmap(fill(:U, 10)), labelmap(fill(:Lin, 10)), labelmap(fill(:Lout, 10))]
                    scores = qs_score(qs, dummy_data, labels)
                    @test length(scores) == size(dummy_data, 2)
                end
            end
        end

        qs_types = [MinimumMarginQs, MinimumLossQs]
        for x in qs_types
            @test_throws ArgumentError initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, Dict{Symbol, Any}())
            @test_throws ArgumentError initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, Dict(:p_inlier => nothing))
            @test_throws ArgumentError initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, Dict(:p_inlier => 1.5))
            @test_throws ArgumentError initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, Dict(:p_inlier => -1))
        end
        qs_objs = map(x -> initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, Dict(:p_inlier => 0.05)), qs_types)
        for qs in qs_objs
            @testset "DataBasedQs $(typeof(qs))" begin
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

        qs_types = [ExpectedMinimumMarginQs, ExpectedMaximumEntropyQs]
        qs_objs = map(x -> initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, params), qs_types)
        for qs in qs_objs
            @testset "DataBasedQs $(typeof(qs))" begin
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

        qs_types = [RandomOutlierQs, HighConfidenceQs, DecisionBoundaryQs]
        qs_objs = map(x -> initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, params), qs_types)
        for qs in qs_objs
            @testset "ModelBasedQs $(typeof(qs))" begin
                scores = qs_score(qs, dummy_data, labelmap(fill(:Lin, 10)))
                @test length(scores) == size(dummy_data, 2)
            end
        end

        qs_types = [NeighborhoodBasedQs, BoundaryNeighborCombinationQs]
        qs_objs = map(x -> initialize_qs(x, SVDD.RandomOCClassifier(dummy_data), dummy_data, params), qs_types)
        for qs in qs_objs
            @testset "HybridQs $(typeof(qs))" begin
                scores = qs_score(qs, dummy_data, labelmap(fill(:U, 10)))
                @test length(scores) == size(dummy_data, 2)
            end
        end

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

            qs_types = [MinimumMarginQs, MinimumLossQs]
            for qst in qs_types
                for m in models
                    @testset "initialize_qs $qst, $m" begin
                        # check if qs is instantiated
                        qs = OneClassActiveLearning.initialize_qs(qst, model, dummy_data, Dict(:p_inlier => 0.05))
                end
                end
            end

            qs_types = [ExpectedMinimumMarginQs, ExpectedMaximumEntropyQs] ∪ subtypes(ModelBasedQs) ∪ [NeighborhoodBasedQs, BoundaryNeighborCombinationQs]
            for qst in qs_types
                for m in models
                    @testset "initialize_qs $qst, $m" begin
                        # check if qs is instantiated
                        qs = OneClassActiveLearning.initialize_qs(qst, model, dummy_data, params)
                end
                end
            end
        end
    end
end
