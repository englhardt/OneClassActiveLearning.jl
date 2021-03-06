struct TopKBatchQs <: ExtendingBatchQs
    sequentialQs::SequentialPQs
    k::Int

    function TopKBatchQs(sequentialQs::SequentialPQs; k::Int)::TopKBatchQs
        (k < 1) && throw(ArgumentError("Invalid batch size k=$(k)."))
        return new(sequentialQs, k)
    end
end

"""
Use sequential strategy, select batch_size observations with highest usefulness.
"""
function select_batch(qs::TopKBatchQs, x::Array{T, 2}, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int})::Vector{Int} where T <: Real
    num_observations = length(candidate_indices)
    if num_observations <= qs.k
        return candidate_indices
    end

    sequential_strategy = qs.sequentialQs
    candidate_scores = qs_score(sequential_strategy, x, labels)[candidate_indices]
    descending_indices = sortperm(candidate_scores; rev=true)
    best_k = candidate_indices[descending_indices[1:qs.k]]
    return best_k
end
