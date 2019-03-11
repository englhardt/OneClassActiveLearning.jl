
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
    @testset "query optimizers" begin
        MAX_ITER = 1
        for opt in [EvolutionaryOptimization(:cmaes, μ=1, λ=2, iterations=MAX_ITER),
                    ParticleSwarmOptimization(swarmsize=3, maxiter=MAX_ITER),
                    BlackBoxOptimization(:dxnes, MaxSteps=MAX_ITER)]
            for d in [1, 2, 10]
                @testset "$(typeof(opt)): $d dim" begin
                    Random.seed!(0)
                    f(x) = -vec(sum(abs.(x), dims=1))
                    x_opt = OneClassActiveLearning.QueryStrategies.query_synthesis_optimize(f, opt, randn(d, 10), fill(:U, 10))
                    @test size(x_opt) == (d, 1)
                end
            end
        end
    end
end
