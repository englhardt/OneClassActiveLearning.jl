
@testset "query synthesis optimization" begin
    @testset "particle swarm optimization" begin
        for d in [1, 2, 10]
            @testset "$d dim" begin
                Random.seed!(0)
                f(x) = vec(sum(abs.(x), dims=1))
                x_opt, _ = OneClassActiveLearning.QueryStrategies.pso(f, -ones(d), ones(d))
                @test size(x_opt) == (d,)
                @test all(f(reshape(x_opt, 1, d)) .< 0.015 * d)
            end
        end
    end
end
