
@testset "query synthesis utils" begin
    UTILS = OneClassActiveLearning.QueryStrategies
    @test_throws ArgumentError UTILS.check_epsilon(-1)
    @test UTILS.check_epsilon(0) === nothing
    @test_throws ArgumentError UTILS.check_epsilon([-1])
    @test UTILS.check_epsilon([0]) === nothing
    @test UTILS.check_epsilon([0; 5.0]) === nothing
    @test UTILS.check_epsilon(zeros(5)) === nothing
    @test_throws ArgumentError UTILS.check_limits([-1])
    @test_throws ArgumentError UTILS.check_limits([[-1] [1 2]])
    @test_throws ArgumentError UTILS.check_limits([0 1; 0 0])
    test_minima, test_maxima = UTILS.extrema_arrays([0 1 3; 6 5 3])
    @test all(test_minima .== [0; 3])
    @test all(test_maxima .== [3; 6])
    test_data = UTILS.rand_in_hypercube(test_minima, test_maxima)
    @test all(test_minima .<= test_data .<= test_maxima)
    test_data = UTILS.rand_in_hypercube(test_minima, test_maxima, -0.1)
    @test all(test_minima .<= test_data .<= test_maxima)
    test_data = UTILS.rand_in_hypercube(test_minima, test_maxima, [1.0; 1.0])
    @test all(test_minima .<= test_data .<= test_maxima)
end
