struct HybridQuerySynthesisPQs <: HybridPQs
    qss::QuerySynthesisStrategy
    dist_func::Function
    function HybridQuerySynthesisPQs(model::SVDD.OCClassifier, data::Array{T, 2}; qss_type=nothing, dist_func=:euclidean, other_params...) where T <: Real
        new(initialize_qs(eval(qss_type), model, data, Dict{Symbol, Any}(other_params)), eval(dist_func))
    end
end

"""
get_query_object(qs::HybridQuerySynthesisPQs, data::Array{T, 2}, pools::Vector{Symbol}, global_indices::Vector{Int}, history::Vector{Int})
This query strategy first performs a query synthesis and then selects the unlabeled
observation closest to the synthetic query as final query.
# Arguments
- `query_data`: Subset of the full data set
- `pools`: Labels for `query_data`
- `global_indices`: Indices of the observations in `query_data` relative to the full data set.
- `history`: Indices of previous queries
"""
function get_query_objects(qs::HybridQuerySynthesisPQs,
                          query_data::Array{T, 2},
                          pools::Vector{Symbol},
                          global_indices::Vector{Int},
                          history::Vector{Vector{Int}})::Vector{Int} where T <: Real
    pool_map = MLLabelUtils.labelmap(pools)
    haskey(pool_map, :U) || throw(ArgumentError("No more points that are unlabeled."))
    # Perform query synthesis
    mask = pools .!= :U
    all_history_values = (length(history) > 0) && (collect(Iterators.flatten(history)))
    query_history = [query_data[:, i:i] for i in indexin(all_history_values, global_indices)]
    qss_object = vec(get_query_objects(qs.qss, query_data[:, mask], pools[mask], query_history))
    @assert length(qss_object) == size(query_data, 1)
    # Choose unlabeled observation closest to synthetic query
    candidates = [i for i in pool_map[:U] if global_indices[i] ∉ history]
    distances = [qs.dist_func(qss_object, query_data[:, i]) for i in candidates]
    local_query_index = candidates[argmin(distances)]
    return [global_indices[local_query_index]]
end
