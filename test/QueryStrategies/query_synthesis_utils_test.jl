
@testset "query synthesis utils" begin
    UTILS = OneClassActiveLearning.QueryStrategies.QuerySynthesisStrategies
    @test_throws ArgumentError UTILS.check_epsilon(-1)
    @test UTILS.check_epsilon(0) === nothing
    @test_throws ArgumentError UTILS.check_epsilon([-1])
    @test UTILS.check_epsilon([0]) === nothing
    @test UTILS.check_epsilon([0; 5.0]) === nothing
    @test UTILS.check_epsilon(zeros(5)) === nothing
    @test_throws ArgumentError UTILS.check_data_limits([-1])
    @test_throws ArgumentError UTILS.check_data_limits([[-1] [1 2]])
    @test_throws ArgumentError UTILS.check_data_limits([0 1; 0 0])
    @test UTILS.check_data_limits([0 1; 1 2]) === nothing
    x = [0 1 3; 6 5 3]
    test_minima, test_maxima = UTILS.extrema_arrays(x)
    @test all(test_minima .== [0; 3])
    @test all(test_maxima .== [3; 6])
    other_minima, other_maxima = UTILS.data_boundaries(x)
    @test all(other_minima .== test_minima)
    @test all(other_maxima .== test_maxima)
    other_minima, other_maxima = UTILS.data_boundaries(x, 0.1)
    @test all(other_minima .< test_minima)
    @test all(other_maxima .> test_maxima)
    test_data = UTILS.rand_in_hyper_rect(test_minima, test_maxima)
    @test all(test_minima .<= test_data .<= test_maxima)
    test_data = UTILS.rand_in_hyper_rect(test_minima, test_maxima, -0.1)
    @test all(test_minima .<= test_data .<= test_maxima)
    test_data = UTILS.rand_in_hyper_rect(test_minima, test_maxima, [-0.11; -0.12])
    @test all(test_minima .<= test_data .<= test_maxima)
end
