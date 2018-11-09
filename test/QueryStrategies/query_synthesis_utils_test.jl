
@testset "query synthesis utils" begin
    @test_throws ArgumentError OneClassActiveLearning.QueryStrategies.check_epsilon([-1])
    @test OneClassActiveLearning.QueryStrategies.check_epsilon([0]) == nothing
    @test OneClassActiveLearning.QueryStrategies.check_epsilon(zeros(5)) == nothing
    @test_throws ArgumentError OneClassActiveLearning.QueryStrategies.check_limits([-1])
    @test_throws ArgumentError OneClassActiveLearning.QueryStrategies.check_limits([[-1] [1 2]])
    @test_throws ArgumentError OneClassActiveLearning.QueryStrategies.check_limits([0 1; 0 0])
    @test OneClassActiveLearning.QueryStrategies.check_epsilon([0 1]) == nothing
    @test OneClassActiveLearning.QueryStrategies.check_epsilon([0 1; 0 1]) == nothing
end
