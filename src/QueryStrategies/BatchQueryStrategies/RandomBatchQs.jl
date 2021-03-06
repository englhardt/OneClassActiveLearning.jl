struct RandomBatchQs <: BatchPQs
    k::Int

    function RandomBatchQs(; k::Int)::RandomBatchQs
        (k < 1) && throw(ArgumentError("Invalid batch size k=$(k)."))
        return new(k)
    end
end

"""
Select batch by selecting batch_size indices from candidate_indices at random
"""
function select_batch(qs::RandomBatchQs, x::Array{T, 2}, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int})::Vector{Int} where T <: Real
    num_observations = length(candidate_indices)
    if num_observations <= qs.k
        return candidate_indices
    end

    return sample(candidate_indices, qs.k, replace=false)
end
