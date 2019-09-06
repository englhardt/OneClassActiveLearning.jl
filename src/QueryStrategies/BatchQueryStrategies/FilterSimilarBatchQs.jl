struct FilterSimilarBatchQs <: BatchPQs
    model::SVDD.OCClassifier
    sequential_qs::SequentialPQs
    div_measure::Function
    k::Int

    function FilterSimilarBatchQs(model::SVDD.OCClassifier, sequential_qs::SequentialPQs; diversity::Symbol, k::Int)::TopKBatchQs
        (k < 1) && throw(ArgumentError("Invalid batch size k=$(k)."))
        diversity_measure = get_pairwise_distance(diversity)
        return new(model, sequential_qs, diversity_measure, k)
    end
end

"""
Iteratively find most similar pairs and prune the one with less utility calculated
with a sequential strategy. This process is repeated until batch_size observations
are left.
"""
function select_batch(qs::FilterSimilarBatchQs, x::Array{T, 2}, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int})::Vector{Int} where T <: Real
    num_observations = length(candidate_indices)
    if num_observations <= qs.k
        return candidate_indices
    end

    candidate_scores = qs_score(qs.sequential_qs, x, labels)[candidate_indices]
    distances = qs.div_measure(qs.model, x, candidate_indices)

    mask = trues(num_observations)
    while sum(mask) > qs.k
        a, b = Tuple(findmin(distances[mask, mask])[2])
        eliminate_idx = candidate_scores[a] > candidate_scores[b] ? b : a
        mask[findall(mask)[eliminate_idx]] = false
    end

    batch = candidate_indices[findall(mask)]
    return batch
end
