using NearestNeighbors

function knn_indices(x::Array{T, 2}; k=1::Int)::Array{Int, 2} where T <: Real
    kdtree = KDTree(x)
    return hcat(knn(kdtree, x, k + 1, true)[1]...)[2:end,:]
end

function knn_mean_dist(x::Array{T, 2}; k=1)::Array{Float64,1} where T <: Real
    kdtree = KDTree(x)
    return map(d -> mean(d[2:end]), knn(kdtree, x, k + 1, true)[2])
end

function get_query_object(qs::SequentialPQs,
                        query_data::Array{T, 2} where T <: Real,
                        pools::Vector{Symbol},
                        global_indices::Vector{Int},
                        history::Vector{Int})::Int
    pool_map = labelmap(pools)
    haskey(pool_map, :U) || throw(ArgumentError("No more points that are unlabeled."))
    scores = qs_score(qs, query_data, pool_map)
    @assert length(scores) == size(query_data, 2)
    all_history_values = (length(history) > 0) && (collect(Iterators.flatten(history)))
    candidates = [i for i in pool_map[:U] if global_indices[i] ∉ all_history_values]
    @debug "[QS] Selecting from $(length(candidates)) candidates."
    local_query_index = candidates[argmax(scores[candidates])]
    return global_indices[local_query_index]
end
