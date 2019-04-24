abstract type PoolQs <: QueryStrategy end
abstract type SequentialPQs <: PoolQs end

abstract type DataBasedPQs <: SequentialPQs end
abstract type ModelBasedPQs <: SequentialPQs end
abstract type HybridPQs <: SequentialPQs end

function get_query_object(qs::SequentialPQs,
                        query_data::Array{T, 2} where T <: Real,
                        pools::Vector{Symbol},
                        global_indices::Vector{Int},
                        history::Vector{Int})::Int
    pool_map = MLLabelUtils.labelmap(pools)
    haskey(pool_map, :U) || throw(ArgumentError("No more points that are unlabeled."))
    scores = qs_score(qs, query_data, pool_map)
    @assert length(scores) == size(query_data, 2)
    all_history_values = (length(history) > 0) && (collect(Iterators.flatten(history)))
    candidates = [i for i in pool_map[:U] if global_indices[i] ∉ all_history_values]
    @debug "[QS] Selecting from $(length(candidates)) candidates."
    local_query_index = candidates[argmax(scores[candidates])]
    return global_indices[local_query_index]
end
