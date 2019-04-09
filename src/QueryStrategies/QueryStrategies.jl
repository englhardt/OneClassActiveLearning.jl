module QueryStrategies

using Reexport
using Distances
using MLKernels
using MLLabelUtils
using NearestNeighbors
using Statistics
using LinearAlgebra
using InteractiveUtils
using LIBSVM
using JuMP
using SVDD
using PyCall

const gaussian_kde = PyNULL()

abstract type QueryStrategy end
abstract type PoolQs <: QueryStrategy end

function qs_score end

"""
get_query_object(qs::QueryStrategy, data::Array{T, 2}, pools::Vector{Symbol}, global_indices::Vector{Int}, history::Vector{Int})

# Arguments
- `query_data`: Subset of the full data set
- `pools`: Labels for `query_data`
- `global_indices`: Indices of the observations in `query_data` relative to the full data set.
- `history`: Indices of previous queries
"""
function get_query_object end

include("qs_utils.jl")
include("qs_base.jl")

include("SequentialQueryStrategies/SequentialQueryStrategies.jl")
@reexport using .SequentialQueryStrategies

include("QuerySynthesisStrategies/QuerySynthesisStrategies.jl")
@reexport using .QuerySynthesisStrategies

include("SubspaceQueryStrategies/SubspaceQueryStrategies.jl")
@reexport using .SubspaceQueryStrategies


function __init__()
    copy!(gaussian_kde, pyimport_conda("scipy.stats", "scipy").gaussian_kde)
end

export
    QueryStrategy,
    PoolQs,

    qs_score,
    initialize_qs,
    get_query_object
end
