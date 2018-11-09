
abstract type QueryStrategy end

abstract type PoolQs <: QueryStrategy end
abstract type QuerySynthesisStrategy <: QueryStrategy end

abstract type DataBasedPQs <: PoolQs end
abstract type ModelBasedPQs <: PoolQs end
abstract type HybridPQs <: PoolQs end

abstract type DataBasedQss <: QuerySynthesisStrategy end
abstract type ModelBasedQss <: QuerySynthesisStrategy end
abstract type HybridQss <: QuerySynthesisStrategy end

using MLKernels
using MLLabelUtils
using NearestNeighbors
using Statistics
using LinearAlgebra
using InteractiveUtils
using LIBSVM
using SVDD

function initialize_qs(qs::DataType, model::OCClassifier, data::Array{T, 2}, params::Dict{Symbol, <:Any})::QueryStrategy where T <: Real
    if qs <: HybridPQs || qs <: HybridQss
        return qs(model, data; params...)
    elseif qs <: ModelBasedPQs || qs <: ModelBasedQss
        return qs(model; params...)
    elseif qs <: DataBasedPQs || qs <: DataBasedQss
        kernel = get_kernel(model)
        if typeof(kernel) == GaussianKernel
            return qs(data, bw_method=MLKernels.getvalue(strategy.kernel.alpha); params...)
        else
            return qs(data; params...)
        end
    elseif qs <: QueryStrategy
        return qs(; params...)
    end
    throw(ErrorException("Unknown query strategy of type $(qs)."))
end

"""
get_query_object(qs::QueryStrategy, data::Array{T, 2}, pools::Vector{Symbol}, global_indices::Vector{Int}, history::Vector{Int})

# Arguments
- `query_data`: Subset of the full data set
- `pools`: Labels for `query_data`
- `global_indices`: Indices of the observations in `query_data` relative to the full data set.
- `history`: Indices of previous queries
"""
function get_query_object(qs::PoolQs, query_data::Array{T, 2}, pools::Vector{Symbol}, global_indices::Vector{Int}, history::Vector{Int})::Int where T <: Real
    pool_map = labelmap(pools)
    haskey(pool_map, :U) || throw(ArgumentError("No more points that are unlabeled."))
    scores = qs_score(qs, query_data, pool_map)
    @assert length(scores) == size(query_data, 2)
    candidates = [i for i in pool_map[:U] if global_indices[i] ∉ history]
    local_query_index = candidates[argmax(scores[candidates])]
    return global_indices[local_query_index]
end
