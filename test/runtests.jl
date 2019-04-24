using OneClassActiveLearning
using SVDD
using Test
using Random
using JuMP, Ipopt
using MLDataUtils, MLLabelUtils, MLKernels
using ValueHistories
using PyCall
using Statistics
using LinearAlgebra
using Dates
using JSON, Unmarshal
using InteractiveUtils
using Distributions
using GaussianMixtures
using Serialization

TEST_SOLVER =  with_optimizer(Ipopt.Optimizer, print_level=0)
TEST_DATA_FILE = joinpath(@__DIR__, "test.csv")
TEST_OUTPUT_FILE = joinpath(@__DIR__, "output.tmp")
TEST_DATA_NUM_DIMENSIONS, TEST_DATA_NUM_OBSERVATIONS = size(load_data(TEST_DATA_FILE)[1])

Random.seed!(0)

@testset "OneClassActiveLearning" begin
    include("QueryStrategies/pool_qs_test.jl")
    include("QueryStrategies/batch_qs_test.jl")
    include("QueryStrategies/query_synthesis_utils_test.jl")
    include("QueryStrategies/query_synthesis_optimization_test.jl")
    include("QueryStrategies/query_synthesis_test.jl")
    include("QueryStrategies/subspace_qs_test.jl")
    include("oracle/oracle_test.jl")
    include("data_util_test.jl")
    include("evaluate_test.jl")
    include("setup_test.jl")
    include("serialize_test.jl")
    include("result_test.jl")
end
