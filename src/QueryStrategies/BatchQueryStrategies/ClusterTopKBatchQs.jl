struct ClusterTopKBatchQs <: BatchPQs
    model::SVDD.OCClassifier
    sequentialQs::SequentialPQs
    k::Int

    function ClusterTopKBatchQs(model::SVDD.OCClassifier, sequential_strategy::SequentialPQs; k::Int)::ClusterTopKBatchQs
        (k < 1) && throw(ArgumentError("Invalid batch size k=$(k)."))
        return new(model, sequential_strategy, k)
    end
end

"""
Ensures selection of an informative, representative and diverse batch
by clustering the most informative observations and then using the k
cluster centers as batch query.

Number of most informative observations selected = 10*batch_size,
Number of clusters computed = batch_size
"""
function select_batch(qs::ClusterTopKBatchQs, x::Array{T, 2}, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int})::Vector{Int} where T <: Real
    num_observations = length(candidate_indices)
    if num_observations <= qs.k
        return candidate_indices
    end

    m = min(num_observations, 10 * qs.k)

    candidate_scores = qs_score(qs.sequentialQs, x, labels)[candidate_indices]
    descending_indices = sortperm(candidate_scores; rev=true)
    best_m = x[:, candidate_indices[descending_indices[1:m]]]

    distances = Distances.pairwise(Distances.Euclidean(), best_m, dims=2)
    clustering = kmedoids(distances, qs.k)

    medoid_indices = clustering.medoids

    return return candidate_indices[descending_indices[1:m]][medoid_indices]
end
