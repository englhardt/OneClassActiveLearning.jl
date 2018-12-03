
@testset "query synthesis" begin
    @testset "random baselines" begin
        epsilon = fill(0.1, 2)
        data = ones(2, 10)
        labels = fill(:U, 10)
        history = Array{Float64, 2}[]

        @testset "RandomQss" begin
            @test_throws ArgumentError RandomQss(epsilon=-0.1)
            @test_throws ArgumentError RandomQss(epsilon=-ones(2))
            query = get_query_object(RandomQss(), data, labels, history)
            @test size(query) == (2, 1)
        end

        @testset "RandomOutlierQss" begin
            occ = SVDD.RandomOCClassifier(data)
            @test_throws ArgumentError RandomOutlierQss(occ, max_tries=0)
            @test_throws ArgumentError RandomOutlierQss(occ, epsilon=-0.1)
            @test_throws ArgumentError RandomOutlierQss(occ, epsilon=-ones(2))
            query = get_query_object(RandomOutlierQss(occ), data, labels, history)
            @test size(query) == (2, 1)
        end
    end

    @testset "get query" begin
        data = rand(2, 10)
        labels = fill(:U, 10)
        history = Vector{Array{Float64, 2}}()
        optimizer = ParticleSwarmOptimization()

        @testset "TestQss" begin
            qs = TestQss(optimizer=optimizer)
            query = get_query_object(qs, data, labels, history)
            @test size(query) == (2, 1)
        end

        occ = SVDD.VanillaSVDD(data)
        init_strategy = SVDD.SimpleCombinedStrategy(SVDD.FixedGammaStrategy(GaussianKernel(2.0)), SVDD.FixedCStrategy(1))
        SVDD.initialize!(occ, init_strategy)
        SVDD.fit!(occ, TEST_SOLVER)
        for qs_type in [DecisionBoundaryQss, ExplorativeMarginQss]
            @testset "$qs_type" begin
                qs = initialize_qs(qs_type, occ, data, Dict(:optimizer => optimizer))
                query = get_query_object(qs, data, labels, history)
                @test size(query) == (2, 1)
                labels[end] = :Lout
                query = get_query_object(qs, data, labels, history)
                @test size(query) == (2, 1)
            end
        end
    end
end
