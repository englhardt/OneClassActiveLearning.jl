using OneClassActiveLearning
using SVDD
using Base.Test
using Ipopt
using MLDataUtils, MLLabelUtils, MLKernels
using ValueHistories
using PyCall

TEST_SOLVER = IpoptSolver(print_level=0)

srand(0)

@testset "OneClassActiveLearning" begin
    include("QueryStrategies/qs_test.jl")
    include("data_util_test.jl")
    include("evaluate_test.jl")
    include("setup_test.jl")
    include("serialize_test.jl")
    include("result_test.jl")
end
