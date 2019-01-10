
@testset "query synthesis" begin
    @testset "initialization" begin
        epsilon = fill(0.1, 2)
        data = ones(2, 10)
        labels = fill(:U, 10)
        history = Array{Float64, 2}[]

        @testset "RandomQss" begin
            @test_throws ArgumentError RandomQss(epsilon=-0.1)
            @test_throws ArgumentError RandomQss(epsilon=-ones(2))
            @test_throws MissingLabelTypeException get_query_object(RandomQss(), data, fill(:Lout, 10), history)
            @test RandomQss(epsilon=1.0) != nothing
            @test RandomQss(epsilon=ones(2)) != nothing

        end

        @testset "RandomOutlierQss" begin
            occ = SVDD.RandomOCClassifier(data)
            @test_throws ArgumentError RandomOutlierQss(occ, max_tries=0)
            @test_throws ArgumentError RandomOutlierQss(occ, epsilon=-0.1)
            @test_throws ArgumentError RandomOutlierQss(occ, epsilon=-ones(2))
            @test_throws MissingLabelTypeException get_query_object(RandomOutlierQss(occ), data, fill(:Lout, 10), history)
            @test RandomOutlierQss(occ, epsilon=1.0) != nothing
            @test RandomOutlierQss(occ, epsilon=ones(2)) != nothing
        end
    end

    @testset "get query" begin
        data = rand(2, 10)
        labels = fill(:U, 10)
        history = Vector{Array{Float64, 2}}()
        optimizer = ParticleSwarmOptimization()

        occ = SVDD.VanillaSVDD(data)
        init_strategy = SVDD.SimpleCombinedStrategy(SVDD.FixedGammaStrategy(GaussianKernel(2.0)), SVDD.FixedCStrategy(1))
        SVDD.initialize!(occ, init_strategy)
        SVDD.fit!(occ, TEST_SOLVER)
        for qs_type in [RandomQss, RandomOutlierQss, TestQss, DecisionBoundaryQss, ExplorativeMarginQss]
            @testset "$qs_type" begin
                qs = initialize_qs(qs_type, occ, data, Dict(:optimizer => optimizer))
                query = get_query_object(qs, data, labels, history)
                @test size(query) == (2, 1)
                if qs_type == ExplorativeMarginQss
                    labels[end] = :Lout
                    query = get_query_object(qs, data, labels, history)
                    @test size(query) == (2, 1)
                end
            end
        end
    end
end
