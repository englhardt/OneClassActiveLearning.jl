
@testset "query synthesis" begin
    @testset "random baselines" begin
        limits = ones(2, 2)
        data = ones(2, 10)
        labels = fill(:U, 10)
        history = Array{Float64, 2}[]

        @testset "RandomQss" begin
            @test_throws ArgumentError RandomQss(limits=limits)
            query = get_query_object(RandomQss(), data, labels, history)
            @test size(query) == (2, 1)
        end

        @testset "RandomOutlierQss" begin
            occ = SVDD.RandomOCClassifier(data)
            @test_throws ArgumentError RandomOutlierQss(occ, max_tries=0)
            @test_throws ArgumentError RandomOutlierQss(occ, limits=limits)
            query = get_query_object(RandomOutlierQss(occ), data, labels, history)
            @test size(query) == (2, 1)
        end
    end

    @testset "get query" begin
        data = rand(2, 10)
        labels = fill(:U, 10)
        indices = collect(1:10)
        history = Vector{Array{Float64, 2}}()
        optimizer = ParticleSwarmOptimization(zeros(2))

        @testset "TestQss" begin
            qs = TestQss(optimizer=optimizer)
            query = get_query_object(qs, data, labels, indices, history)
            @test size(query) == (2, 1)
        end
    end
end
