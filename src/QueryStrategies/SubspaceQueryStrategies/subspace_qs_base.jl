abstract type SubspaceQueryStrategy <: QueryStrategy end

function get_query_objects(qs::SubspaceQueryStrategy,
                        query_data::Array{T, 2} where T <: Real,
                        pools::Vector{Symbol},
                        global_indices::Vector{Int},
                        history::Vector{Vector{Int}})::Vector{Int}
    pool_map = MLLabelUtils.labelmap(pools)
    haskey(pool_map, :U) || throw(ArgumentError("No more points that are unlabeled."))
    scores = qs_score(qs, query_data, pool_map)
    @assert length(scores) == size(query_data, 2)
    all_history_values = (length(history) > 0) && (collect(Iterators.flatten(history)))
    candidates = [i for i in pool_map[:U] if global_indices[i] ∉ all_history_values]
    debug(getlogger(@__MODULE__), "[QS] Selecting from $(length(candidates)) candidates.")
    local_query_index = candidates[argmax(scores[candidates])]
    return [global_indices[local_query_index]]
end
