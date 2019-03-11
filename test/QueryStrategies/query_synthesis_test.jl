
@testset "query synthesis" begin
    @testset "initialization" begin
        epsilon = fill(0.1, 2)
        data = ones(2, 10)
        labels = fill(:U, 10)
        history = Array{Float64, 2}[]

        @testset "RandomQss" begin
            @test_throws ArgumentError RandomQss(epsilon=-0.1)
            @test_throws ArgumentError RandomQss(epsilon=-ones(2))
            @test RandomQss(epsilon=1.0) !== nothing
            @test RandomQss(epsilon=ones(2)) !== nothing

        end

        @testset "RandomOutlierQss" begin
            occ = SVDD.RandomOCClassifier(data)
            @test_throws ArgumentError RandomOutlierQss(occ, max_tries=0)
            @test_throws ArgumentError RandomOutlierQss(occ, epsilon=-0.1)
            @test_throws ArgumentError RandomOutlierQss(occ, epsilon=-ones(2))
            @test RandomOutlierQss(occ, epsilon=1.0) !== nothing
            @test RandomOutlierQss(occ, epsilon=ones(2)) !== nothing
        end
    end

    @testset "get query" begin
        data = rand(2, 10)
        labels = fill(:U, size(data, 2))
        history = Vector{Array{Float64, 2}}()
        optimizer = ParticleSwarmOptimization()

        occ = SVDD.VanillaSVDD(data)
        init_strategy = SVDD.SimpleCombinedStrategy(SVDD.FixedGammaStrategy(GaussianKernel(2.0)), SVDD.FixedCStrategy(1))
        SVDD.initialize!(occ, init_strategy)
        SVDD.fit!(occ, TEST_SOLVER)
        for qs_type in [RandomQss, RandomOutlierQss, TestQss, DecisionBoundaryQss, NaiveExplorativeMarginQss]
            @testset "$qs_type" begin
                qs = initialize_qs(qs_type, occ, data, Dict(:optimizer => optimizer))
                query = get_query_object(qs, data, labels, history)
                @test size(query) == (size(data, 1), 1)
                if qs_type == NaiveExplorativeMarginQss
                    labels[end] = :Lout
                    query = get_query_object(qs, data, labels, history)
                    @test size(query) == (size(data, 1), 1)
                end
            end
        end
        @testset "ExplorativeMarginQss" begin
            qs = initialize_qs(ExplorativeMarginQss, occ, data,
                               Dict(:solver => TEST_SOLVER,
                                    :optimizer => optimizer))
            query = get_query_object(qs, data, labels, history)
            @test size(query) == (size(data, 1), 1)
            labels[end] = :Lout
            query = get_query_object(qs, data, labels, history)
            @test size(query) == (size(data, 1), 1)
        end
    end
end
