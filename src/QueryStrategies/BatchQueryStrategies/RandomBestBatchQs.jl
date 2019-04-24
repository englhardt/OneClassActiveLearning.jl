struct RandomBestBatchQs <: ExtendingBatchQs
    sequentialQs::SequentialPQs
    k::Int
    m::Int

    function RandomBestBatchQs(sequentialQs::SequentialPQs; k::Int=0, m::Int=0)::RandomBatchQs
        (k < 1) && throw(ArgumentError("Invalid batch size k=$(k)."))
        (m < 0) && throw(ArgumentError("Invalid batch increase m=$(m)."))
        return new(sequentialQs, k, m)
    end
end

"""
candidates are indices in x which need to be considered,
returns indices of best batch in x
"""
function select_batch(qs::RandomBestBatchQs, x::Array{T, 2}, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int})::Vector{Int} where T <: Real
    num_observations = length(candidate_indices)
    if num_observations <= qs.k
        return candidate_indices
    end

    sequential_strategy = qs.sequentialQs
    candidate_scores = qs_score(sequential_strategy, x, labels)[candidate_indices]
    descending_indices = sortperm(candidate_scores; rev=true)
    # find best k+m elements
    best_k_m = candidate_indices[descending_indices[1:min(qs.k + qs.m, end)]]
    # take random sample of size k from best elements
    best_sampled = Random.shuffle(best_k_m)[1:qs.k]
    return best_sampled
end
