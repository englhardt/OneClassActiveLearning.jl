abstract type QueryStrategy end

function qs_score end

"""
get_query_objects(qs::QueryStrategy, data::Array{T, 2}, pools::Vector{Symbol}, global_indices::Vector{Int}, history::Vector{Int})

# Arguments
- `query_data`: Subset of the full data set
- `pools`: Labels for `query_data`
- `global_indices`: Indices of the observations in `query_data` relative to the full data set.
- `history`: Indices of previous queries
"""
function get_query_objects end

function initialize_qs(qs::DataType, model::SVDD.OCClassifier, data::Array{T, 2}, params::Dict{Symbol, <:Any})::QueryStrategy where T <: Real
    if qs <: BatchPQs
        if haskey(params, :SequentialStrategy)
            sequential_strategy_type = eval(params[:SequentialStrategy][:type])
            sequential_strategy_params = params[:SequentialStrategy][:param]
            sequential_strategy = initialize_qs(sequential_strategy_type, model, data, sequential_strategy_params)
            # use all params except SequentialStrategy for batch query initialization
            batch_params = filter(tuple -> first(tuple) != :SequentialStrategy, params)
            if qs <: ExtendingBatchQs
                return qs(sequential_strategy; batch_params...)
            else
                return qs(model, sequential_strategy; batch_params...)
            end
        else
            return qs(; params...)
        end
    elseif qs <: SequentialPQs
        if qs <: HybridPQs
            return qs(model, data; params...)
        elseif qs <: ModelBasedPQs
            return qs(model; params...)
        elseif qs <: DataBasedPQs
            kernel = SVDD.get_kernel(model)
            if typeof(kernel) == MLKernels.GaussianKernel
                return qs(data, bw_method=kernel.alpha.value.x; params...)
            else
                return qs(data; params...)
            end
        else
            return qs(; params...)
        end
    elseif qs <: QuerySynthesisStrategy
        if qs <: HybridQss
            return qs(model, data; params...)
        elseif qs <: ModelBasedQss
            return qs(model; params...)
        elseif qs <: DataBasedQss
            kernel = get_kernel(model)
            if typeof(kernel) == MLKernels.GaussianKernel
                return qs(data, bw_method=kernel.alpha.value.x; params...)
            else
                return qs(data; params...)
            end
        else
            return qs(; params...)
        end
    elseif qs <: SubspaceQueryStrategy
        return qs(model, data; params...)
    elseif qs <: QueryStrategy
        return qs(; params...)
    end
    throw(ErrorException("Unknown query strategy of type $(qs)."))
end
