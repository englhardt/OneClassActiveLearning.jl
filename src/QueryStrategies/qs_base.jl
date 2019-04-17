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

function initialize_qs(qs::DataType, model::SVDD.OCClassifier, data::Array{T, 2} where T <: Real, params)::qs
    if qs <: HybridPQs || qs <: HybridQss
        return qs(model, data; params...)
    elseif qs <: ModelBasedPQs || qs <: ModelBasedQss
        return qs(model; params...)
    elseif qs <: DataBasedPQs || qs <: DataBasedQss
        kernel = SVDD.get_kernel(model)
        if typeof(kernel) == MLKernels.GaussianKernel
            return qs(data, bw_method=MLKernels.getvalue(strategy.kernel.alpha); params...)
        else
            return qs(data; params...)
        end
    elseif qs <: SubspaceQs
        return qs(model, data; params...)
    elseif qs <: QueryStrategy
        return qs(; params...)
    end
    throw(ErrorException("Unknown query strategy of type $(qs)."))
end
