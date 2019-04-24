struct ClusterBatchQs <: BatchPQs
    model::SVDD.OCClassifier
    sequentialQs::SequentialPQs
    k::Int
    max_iterations::Int

    function ClusterBatchQs(model::SVDD.OCClassifier, sequential_strategy::SequentialPQs; k::Int = 0, max_iterations::Int = 1000)::ClusterBatchQs
        (k < 1) && throw(ArgumentError("Invalid batch size k=$(k)."))
        return new(model, sequential_strategy, k, max_iterations)
    end
end

"""
For maximum diversity in batch selection.
Compute k-medoids clustering with k=batchsize, return cluster centers
"""
function select_batch(qs::ClusterBatchQs, x::Array{T, 2}, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int})::Vector{Int} where T <: Real
    num_observations = length(candidate_indices)
    if num_observations <= qs.k
        return candidate_indices
    end

    m = min(num_observations, 10*qs.k)

    sequential_strategy = qs.sequentialQs
    candidate_scores = qs_score(sequential_strategy, x, labels)[candidate_indices]
    descending_indices = sortperm(candidate_scores; rev=true)
    best_m = x[:, candidate_indices[descending_indices[1:m]]]

    distances = pairwise(Euclidean(), best_m)
    clustering = kmedoids(distances, qs.k)

    medoid_indices = clustering.medoids

    return candidate_indices[medoid_indices]
end
